#!/usr/bin/env python3
"""
Image Upscaler Script - Real-ESRGAN Style

This script provides a local implementation similar to the Cog predict script,
using PyTorch models directly for better compatibility and quality.
Based on Real-ESRGAN/cog_predict.py functionality.

Usage:
    python upscale_image.py --input image.jpg --version 'General - v3' --scale 2 --face_enhance --tile 0
    python upscale_image.py --input_dir ./images --version 'General - v3' --output_dir ./output
    python upscale_image.py --list-versions
"""

import argparse
import os
import sys
import glob
import time
import tempfile
import shutil
import subprocess
from pathlib import Path
from typing import Optional, List, Tuple

import cv2
import numpy as np
from PIL import Image


class ImageUpscaler:
    """PyTorch-based image upscaling utility, similar to Cog predict."""

    def __init__(self, weights_dir: str = "weights"):
        self.weights_dir = Path(weights_dir)
        self.weights_dir.mkdir(exist_ok=True)
        self.realesrgan_dir = Path(__file__).parent / "Real-ESRGAN"

        # Available model versions
        self.available_versions = [
            'General - RealESRGANplus',
            'General - v3',
            'Anime - anime6B',
            'AnimeVideo - v3'
        ]

        # Model name mapping
        self.model_map = {
            'General - RealESRGANplus': 'RealESRGAN_x4plus',
            'General - v3': 'realesr-general-x4v3',
            'Anime - anime6B': 'RealESRGAN_x4plus_anime_6B',
            'AnimeVideo - v3': 'realesr-animevideov3'
        }

    def download_weights(self):
        """Download required weights if not present."""
        weights_to_download = {
            'realesr-general-x4v3.pth': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth',
            'realesr-general-wdn-x4v3.pth': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
            'RealESRGAN_x4plus.pth': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
            'RealESRGAN_x4plus_anime_6B.pth': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth',
            'realesr-animevideov3.pth': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth'
        }

        for weight_name, url in weights_to_download.items():
            weight_path = self.weights_dir / weight_name
            if not weight_path.exists():
                print(f"Downloading {weight_name}...")
                os.system(f'wget {url} -P {self.weights_dir}')

    def process_image(self, img_path: str, version: str, scale: float = 2, face_enhance: bool = False, tile: int = 0) -> tuple:
        """Process a single image using Real-ESRGAN inference script."""
        # Only reset tile if it's None, keep small values as they are valid
        if tile is None:
            tile = 0

        print(f'\n🔄 Processing: {img_path}')
        print(f'   Model: {version}')
        print(f'   Scale: {scale}x')
        print(f'   Face enhance: {face_enhance}')

        # Load input image details early for tile calculation
        input_img = cv2.imread(str(img_path))
        if input_img is not None:
            h, w = input_img.shape[:2]
            print(f'   📐 Input size: {w}x{h} pixels')
            print(f'   📊 Expected output: {int(w*scale)}x{int(h*scale)} pixels')

        # Show tile information with warning for small tiles
        if tile == 0:
            print(f'   🔳 Tile size: {tile} (no tiling - processes entire image at once)')
        else:
            print(f'   🔳 Tile size: {tile}x{tile} pixels per tile')
            if input_img is not None:
                tiles_x = (w + tile - 1) // tile
                tiles_y = (h + tile - 1) // tile
                total_tiles = tiles_x * tiles_y
                print(f'   ⚠️  This will create {total_tiles} tiles ({tiles_x}x{tiles_y})')
                if total_tiles > 1000:
                    print(f'   💡 Suggestion: Use --tile 512 or --tile 0 for better performance')

        # Log GPU availability
        try:
            import torch
            if torch.backends.mps.is_available():
                print(f'   🚀 GPU: Apple Silicon (MPS) - ENABLED')
            elif torch.cuda.is_available():
                print(f'   🚀 GPU: CUDA - ENABLED')
            else:
                print(f'   ⚠️  GPU: Not available - using CPU')
        except ImportError:
            print(f'   ⚠️  PyTorch not available for GPU detection')

        # Get model name
        model_name = self.model_map.get(version)
        if not model_name:
            raise ValueError(f"Unknown version: {version}")

        # Check if model exists
        model_path = self.weights_dir / f"{model_name}.pth"
        if not model_path.exists():
            raise FileNotFoundError(f"Model weight file not found: {model_path}")

        # Create output path
        output_path = Path(img_path).parent / f"temp_output_{Path(img_path).stem}.png"

        # Build command
        cmd = [
            sys.executable, str(self.realesrgan_dir / "inference_realesrgan.py"),
            "-n", model_name,
            "-i", str(img_path),
            "-o", str(output_path.parent),
            "--outscale", str(scale),
            "--tile", str(tile),
            "--gpu-id", "0"  # Enable GPU acceleration by default
        ]

        if face_enhance:
            cmd.extend(["--face_enhance"])

        # Add model path (absolute path)
        cmd.extend(["--model_path", str(model_path.absolute())])

        print(f'   💾 Model file: {model_path.name}')
        print(f'   🔧 Command: {" ".join(cmd[-6:])}')  # Show last few args

        start_time = time.time()
        print(f'   ⏱️  Starting processing...')

        try:
            # Run Real-ESRGAN inference from correct working directory with live output
            print(f'   🔄 Running Real-ESRGAN with live output...')

            # Use Popen for real-time output
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
                cwd=str(Path(__file__).parent)
            )

            # Stream output in real-time
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    # Print Real-ESRGAN progress with indentation
                    print(f'     {output.strip()}')

            return_code = process.poll()
            processing_time = time.time() - start_time

            if return_code != 0:
                print(f"   ❌ Error: Real-ESRGAN failed with return code {return_code}")
                print(f"   🔧 Full command: {' '.join(cmd)}")
                raise RuntimeError(f"Real-ESRGAN failed with return code {return_code}")

            print(f'   ✅ Processing completed in {processing_time:.2f}s')

            # Read output image (Real-ESRGAN saves as {input}_out.{ext})
            input_name = Path(img_path).stem
            input_ext = Path(img_path).suffix
            possible_outputs = [
                Path(img_path).parent / f"{input_name}_out{input_ext}",
                Path(img_path).parent / f"{input_name}_out.png",  # Always saves as PNG for RGBA
                Path(img_path).parent / f"temp_output_{input_name}.png",  # Our fallback
            ]

            print(f'   🔍 Looking for output files...')
            # Small delay to ensure file is written
            time.sleep(0.5)

            output_file = None
            for out_path in possible_outputs:
                if out_path.exists():
                    output_file = out_path
                    break

            if output_file:
                print(f'   📁 Found output: {output_file.name}')
                output_img = cv2.imread(str(output_file))
                if output_img is None:
                    raise ValueError("Could not read output image")

                # Log actual output dimensions
                h, w = output_img.shape[:2]
                print(f'   📐 Actual output size: {w}x{h} pixels')

                # Determine image mode
                img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
                img_mode = 'RGBA' if len(img.shape) == 3 and img.shape[2] == 4 else None

                # Clean up output file
                output_file.unlink(missing_ok=True)

                return output_img, img_mode
            else:
                print(f'   ❌ No output file found in: {[str(p) for p in possible_outputs]}')
                raise FileNotFoundError("Output file not created")

        except Exception as e:
            print(f'Processing error: {e}')
            # Clean up temp file
            output_path.unlink(missing_ok=True)
            raise

    def upscale_image(self, input_path: str, version: str, output_path: str, scale: float = 2, face_enhance: bool = False, tile: int = 0) -> float:
        """Upscale a single image using Real-ESRGAN style processing."""
        start_time = time.time()

        try:
            # Process image
            output, img_mode = self.process_image(input_path, version, scale, face_enhance, tile)

            # Determine output format
            output_path_obj = Path(output_path)
            if img_mode == 'RGBA':
                # RGBA images should be saved in png format
                if output_path_obj.suffix.lower() != '.png':
                    output_path = output_path_obj.with_suffix('.png')

            # Save output
            cv2.imwrite(str(output_path), output)

            processing_time = time.time() - start_time

            # Get file sizes for comparison
            input_size = os.path.getsize(input_path) / 1024  # KB
            output_size = os.path.getsize(output_path) / 1024  # KB

            print(f"\n🎉 UPSCALING COMPLETE!")
            print(f"   📥 Input:  {Path(input_path).name} ({input_size:.1f}KB)")
            print(f"   📤 Output: {Path(output_path).name} ({output_size:.1f}KB)")
            print(f"   ⏱️  Time:   {processing_time:.2f}s")
            print(f"   🚀 Speed:  {input_size/processing_time:.1f}KB/s processed")

            return processing_time

        except Exception as e:
            print(f"✗ Error processing {input_path}: {str(e)}")
            raise

    def upscale_directory(self, input_dir: str, version: str, output_dir: str, scale: float = 2, face_enhance: bool = False, tile: int = 0) -> Tuple[int, float]:
        """Upscale all images in a directory."""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Supported image formats
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

        # Find all images
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))

        if not image_files:
            print(f"No images found in {input_dir}")
            return 0, 0.0

        print(f"Found {len(image_files)} images to process")

        total_time = 0.0
        successful_count = 0

        for img_file in image_files:
            try:
                output_file = output_path / f"upscaled_{img_file.name}"
                processing_time = self.upscale_image(
                    str(img_file), version, str(output_file), scale, face_enhance, tile
                )
                total_time += processing_time
                successful_count += 1
            except Exception as e:
                print(f"Failed to process {img_file}: {str(e)}")
                continue

        avg_time = total_time / successful_count if successful_count > 0 else 0.0
        print(f"\n✓ Processed {successful_count}/{len(image_files)} images successfully")
        print(f"✓ Average processing time: {avg_time:.2f}s per image")

        return successful_count, total_time

    def list_versions(self):
        """List all available model versions."""
        print("Available Real-ESRGAN versions:")
        print("-" * 50)

        version_info = {
            'General - RealESRGANplus': 'General purpose images, enhanced quality',
            'General - v3': 'General purpose images, latest version',
            'Anime - anime6B': 'Specialized for anime and artwork',
            'AnimeVideo - v3': 'Specialized for anime videos'
        }

        for version in self.available_versions:
            description = version_info.get(version, 'Unknown')
            print(f"• {version}")
            print(f"  {description}")
            print()

        print("🏆 RECOMMENDED:")
        print("   • General - v3 - Best for most use cases")
        print("   • General - RealESRGANplus - Enhanced quality")
        print("\n🎨 SPECIALIZED:")
        print("   • Anime - anime6B - Best for anime/artwork")
        print("   • AnimeVideo - v3 - Best for anime videos")

        print("\n⚠️  Face enhancement requires GFPGAN to be installed")
        print("   Install with: pip install gfpgan")

    def setup_environment(self):
        """Setup the environment by downloading required weights."""
        print("Setting up environment...")
        self.download_weights()
        print("✓ Setup complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Upscale images using Real-ESRGAN (PyTorch-based, similar to Cog predict)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upscale single image with default settings
  python upscale_image.py -i photo.jpg -o upscaled.jpg

  # Upscale with specific version and scale
  python upscale_image.py -i photo.jpg --version 'General - v3' --scale 2 -o upscaled.jpg

  # Batch process directory
  python upscale_image.py --input_dir ./photos --version 'General - v3' --output_dir ./output

  # With face enhancement
  python upscale_image.py -i portrait.jpg --version 'General - v3' --face_enhance -o enhanced.jpg

  # List available versions
  python upscale_image.py --list-versions

  # Setup environment (download weights)
  python upscale_image.py --setup
        """
    )

    # Setup option
    parser.add_argument('--setup', action='store_true', help='Setup environment and download weights')

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument('-i', '--input', help='Input image file')
    input_group.add_argument('--input_dir', help='Input directory containing images')
    parser.add_argument('--list-versions', action='store_true', help='List available model versions')

    # Model selection
    parser.add_argument('--version',
                       default='General - v3',
                       choices=['General - RealESRGANplus', 'General - v3', 'Anime - anime6B', 'AnimeVideo - v3'],
                       help='Real-ESRGAN version (default: General - v3)')

    # Processing options
    parser.add_argument('--scale', type=float, default=2, help='Rescaling factor (default: 2)')
    parser.add_argument('--face_enhance', action='store_true', help='Enhance faces with GFPGAN')
    parser.add_argument('--tile', type=int, default=0,
                       help='Tile size in pixels (default: 0=no tiling). Recommended: 512 for 8GB+ RAM, 256 for 4GB RAM. Very small values like 2-64 will be extremely slow!')

    # Output options
    parser.add_argument('-o', '--output', help='Output image file (for single image)')
    parser.add_argument('--output_dir', help='Output directory (for batch processing)')

    # Other options
    parser.add_argument('--weights_dir', default='weights', help='Weights directory path')

    args = parser.parse_args()

    # Initialize upscaler
    upscaler = ImageUpscaler(args.weights_dir)

    # Setup environment if requested
    if args.setup:
        upscaler.setup_environment()
        return

    # List versions if requested
    if args.list_versions:
        upscaler.list_versions()
        return

    # Validate input
    if not args.input and not args.input_dir:
        print("Error: Either --input or --input_dir is required")
        print("Use --list-versions to see available versions")
        return

    try:
        if args.input:
            # Single image processing
            if not args.output:
                # Generate output filename with _upscaled suffix
                input_path = Path(args.input)
                stem = input_path.stem
                suffix = input_path.suffix
                output_name = f"{stem}_upscaled{suffix}"
                args.output = str(input_path.parent / output_name)

            print(f"Upscaling {args.input} using {args.version}...")
            processing_time = upscaler.upscale_image(
                args.input, args.version, args.output, args.scale, args.face_enhance, args.tile
            )
            print(f"✓ Completed in {processing_time:.2f} seconds")

        elif args.input_dir:
            # Directory processing
            if not args.output_dir:
                args.output_dir = f"{args.input_dir}_upscaled"

            print(f"Upscaling images in {args.input_dir} using {args.version}...")
            successful_count, total_time = upscaler.upscale_directory(
                args.input_dir, args.version, args.output_dir, args.scale, args.face_enhance, args.tile
            )
            print(f"✓ Completed {successful_count} images in {total_time:.2f} seconds")

    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
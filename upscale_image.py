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
        if tile <= 100 or tile is None:
            tile = 0

        print(f'Processing: {img_path}, version: {version}, scale: {scale}, face_enhance: {face_enhance}, tile: {tile}')

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
            "--tile", str(tile)
        ]

        if face_enhance:
            cmd.extend(["--face_enhance"])

        # Add model path (absolute path)
        cmd.extend(["--model_path", str(model_path.absolute())])

        try:
            # Run Real-ESRGAN inference from correct working directory
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(Path(__file__).parent))

            if result.returncode != 0:
                print(f"Error running Real-ESRGAN: {result.stderr}")
                print(f"Command: {' '.join(cmd)}")
                raise RuntimeError(f"Real-ESRGAN failed: {result.stderr}")

            # Read output image (Real-ESRGAN saves as {input}_out.{ext})
            input_name = Path(img_path).stem
            input_ext = Path(img_path).suffix
            possible_outputs = [
                Path(img_path).parent / f"{input_name}_out{input_ext}",
                Path(img_path).parent / f"{input_name}_out.png",  # Always saves as PNG for RGBA
                Path(img_path).parent / f"temp_output_{input_name}.png",  # Our fallback
            ]

            # Small delay to ensure file is written
            import time
            time.sleep(0.5)

            output_file = None
            for out_path in possible_outputs:
                if out_path.exists():
                    output_file = out_path
                    break

            if output_file:
                output_img = cv2.imread(str(output_file))
                if output_img is None:
                    raise ValueError("Could not read output image")

                # Determine image mode
                img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
                img_mode = 'RGBA' if len(img.shape) == 3 and img.shape[2] == 4 else None

                # Clean up output file
                output_file.unlink(missing_ok=True)

                return output_img, img_mode
            else:
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
            print(f"✓ Upscaled {input_path} -> {output_path} in {processing_time:.2f}s")

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
    parser.add_argument('--tile', type=int, default=0, help='Tile size for large images (default: 0, no tiling)')

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
                # Generate output filename
                input_path = Path(args.input)
                output_name = f"upscaled_{input_path.name}"
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
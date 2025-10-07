#!/usr/bin/env python3
"""
Quick Upscaler - Streamlined version using pre-downloaded weights

This script provides fast upscaling using the existing weights directory,
bypassing downloads and focusing on performance with proper GPU acceleration.

Usage:
    python quick_upscale.py input.jpg                           # Use default settings
    python quick_upscale.py input.jpg --model ESRGAN_4x         # Specify model
    python quick_upscale.py input.jpg --scale 2 --tile 512      # Custom settings
    python quick_upscale.py --list-models                       # Show available models
"""

import argparse
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

# Suppress FutureWarnings from PyTorch
warnings.filterwarnings("ignore", category=FutureWarning)

# Import our custom inference pipeline
from inference_pipeline import SuperResolutionPipeline, get_optimal_device, estimate_memory_usage


class QuickUpscaler:
    """Fast upscaler using pre-downloaded weights with GPU acceleration."""

    def __init__(self, weights_dir: str = "weights"):
        self.weights_dir = Path(weights_dir)

        # Check if weights directory exists
        if not self.weights_dir.exists():
            print(f"❌ Weights directory '{weights_dir}' not found!")
            print(f"💡 Run './setup_all_models.sh' to download all available models")
            sys.exit(1)

        # Dynamically discover models
        self.models = self._discover_models()

        # Set up recommendations based on available models
        self.recommendations = self._get_recommendations()

    def _detect_model_format(self, model_path: Path) -> Tuple[bool, str, int]:
        """
        Detect if model is compatible with RealESRGAN and determine its properties.
        Returns: (is_compatible, architecture, scale_factor)
        """
        model_name = model_path.stem

        # Known architecture mappings for specific models
        known_architectures = {
            # RRDBNet models (complex architecture with RDB blocks)
            "RealESRGAN_x4plus": ("RRDBNet", 4),
            "RealESRGAN_x4plus_anime_6B": ("RRDBNet", 4),

            # SRVGGNetCompact models (lighter architecture)
            "realesr-general-x4v3": ("SRVGGNetCompact", 4),
            "realesr-general-wdn-x4v3": ("SRVGGNetCompact", 4),
            "realesr-animevideov3": ("SRVGGNetCompact", 4),
        }

        try:
            # Load model to inspect structure
            model_data = torch.load(model_path, map_location='cpu')

            # Check if it has Real-ESRGAN format (params_ema or params keys)
            if isinstance(model_data, dict):
                if 'params_ema' in model_data or 'params' in model_data:
                    # This is Real-ESRGAN format - compatible

                    # Use known architecture if available
                    if model_name in known_architectures:
                        arch, scale = known_architectures[model_name]
                        return True, arch, scale

                    # Fallback: detect from weights structure
                    weights = model_data.get('params_ema', model_data.get('params', {}))

                    if any('rdb' in key.lower() for key in weights.keys()):
                        # RRDBNet has Residual Dense Blocks
                        arch, scale = "RRDBNet", 4
                    elif any(key.startswith('body.') and len(key.split('.')) == 3 for key in weights.keys()):
                        # SRVGGNetCompact has simple body.N.weight structure
                        arch, scale = "SRVGGNetCompact", 4
                    else:
                        # Default fallback
                        arch, scale = "RRDBNet", 4

                    return True, arch, scale

                else:
                    # Raw weights format - needs conversion
                    return False, "RRDBNet", 4
            else:
                # Unknown format
                return False, "Unknown", 4

        except Exception as e:
            print(f"⚠️  Warning: Could not analyze {model_path.name}: {e}")
            return False, "Unknown", 4

    def _discover_models(self) -> Dict[str, Dict]:
        """Dynamically discover all .pth models in the weights directory."""
        models = {}

        # Known model metadata for better descriptions
        model_metadata = {
            # Official Real-ESRGAN models
            "RealESRGAN_x4plus": {
                "description": "📸 Real-ESRGAN+ - Official model, photography",
                "category": "official"
            },
            "RealESRGAN_x4plus_anime_6B": {
                "description": "🎨 Real-ESRGAN Anime 6B - Specialized for anime/artwork",
                "category": "anime"
            },
            "realesr-general-x4v3": {
                "description": "📸 Real-ESRGAN v3 - Latest general purpose",
                "category": "official"
            },
            "realesr-general-wdn-x4v3": {
                "description": "📸 Real-ESRGAN v3 with noise - Noisy image restoration",
                "category": "official"
            },
            "realesr-animevideov3": {
                "description": "🎥 Real-ESRGAN AnimVideo v3 - Video/animation",
                "category": "anime"
            },

            # Custom/Research models (high quality but may need conversion)
            "SwinIR_4x": {
                "description": "🌟 Swin Transformer - Highest quality, research-grade",
                "category": "research"
            },
            "HAT_4x": {
                "description": "🌟 Hybrid Attention Transformer - Cutting-edge quality",
                "category": "research"
            },
            "EDSR_4x": {
                "description": "🌟 Enhanced Deep SR - Research-grade, stable",
                "category": "research"
            },
            "RCAN_4x": {
                "description": "🌟 Residual Channel Attention - Detail-focused",
                "category": "research"
            },
            "ESRGAN_4x": {
                "description": "⚡ Enhanced SRGAN - General purpose",
                "category": "general"
            },
            "ESRGAN_2x": {
                "description": "⚡ Enhanced SRGAN 2x - Fast general purpose",
                "category": "general"
            },
            "RealESRGAN_4x": {
                "description": "⚡ Real-ESRGAN - Real-world images",
                "category": "general"
            },
            "BSRGAN_4x": {
                "description": "🔍 Blind SRGAN - Unknown degradation handling",
                "category": "specialized"
            },
            "SRGAN_4x": {
                "description": "🎭 SRGAN - Perceptual quality, original GAN",
                "category": "specialized"
            }
        }

        # Scan for .pth files
        pth_files = list(self.weights_dir.glob("*.pth"))

        if not pth_files:
            print(f"❌ No .pth model files found in '{self.weights_dir}'!")
            print(f"💡 Run './setup_all_models.sh' to download all available models")
            sys.exit(1)

        print(f"🔍 Scanning {len(pth_files)} model files...")

        for model_path in pth_files:
            model_name = model_path.stem

            # Detect model properties
            is_compatible, arch, scale = self._detect_model_format(model_path)

            # Get metadata or create default
            metadata = model_metadata.get(model_name, {
                "description": f"📁 {model_name} - Auto-detected model",
                "category": "custom"
            })

            models[model_name] = {
                "path": model_path.name,
                "arch": arch,
                "scale": scale,
                "description": metadata["description"],
                "category": metadata["category"],
                "compatible": is_compatible,
                "size_mb": model_path.stat().st_size / (1024 * 1024)
            }

        compatible_count = sum(1 for m in models.values() if m["compatible"])
        total_count = len(models)

        print(f"✅ Found {total_count} models ({compatible_count} ready-to-use, {total_count-compatible_count} need conversion)")

        return models

    def _get_recommendations(self) -> Dict[str, str]:
        """Generate recommendations based on available models."""
        recommendations = {}

        # Priority order for recommendations
        priorities = {
            "general": ["RealESRGAN_x4plus", "realesr-general-x4v3", "ESRGAN_4x"],
            "quality": ["realesr-general-x4v3", "RealESRGAN_x4plus", "SwinIR_4x", "HAT_4x"],
            "photography": ["RealESRGAN_x4plus", "realesr-general-x4v3"],
            "anime": ["RealESRGAN_x4plus_anime_6B", "realesr-animevideov3", "RealESRGAN_anime_4x"],
            "video": ["realesr-animevideov3", "RealESRGAN_x4plus_anime_6B"],
            "fast": ["ESRGAN_2x", "realesr-general-x4v3"]
        }

        for category, model_list in priorities.items():
            for model_name in model_list:
                if model_name in self.models:
                    recommendations[category] = model_name
                    break

        return recommendations

    def get_device(self) -> torch.device:
        """Get the best available device."""
        device = get_optimal_device()

        if device.type == 'cuda':
            print(f"🚀 Using CUDA GPU")
        elif device.type == 'mps':
            print(f"🚀 Using Apple Silicon GPU (MPS)")
        else:
            print(f"⚠️  Using CPU (consider GPU for faster processing)")

        return device

    def list_models(self):
        """List all available models organized by category."""
        print("🔥 AVAILABLE MODELS")
        print("=" * 70)

        # Group models by category
        categories = {
            "official": "📸 REAL-ESRGAN OFFICIAL (Ready to use)",
            "anime": "🎨 ANIME/ARTWORK SPECIALIZED (Ready to use)",
            "general": "⚡ GENERAL PURPOSE",
            "research": "🌟 RESEARCH MODELS (Highest quality)",
            "specialized": "🔍 SPECIALIZED PURPOSE",
            "custom": "📁 CUSTOM/UNKNOWN MODELS"
        }

        for cat_key, cat_name in categories.items():
            models_in_cat = [name for name, info in self.models.items() if info["category"] == cat_key]

            if models_in_cat:
                print(f"\n{cat_name}")
                print("-" * len(cat_name))
                for model_name in models_in_cat:
                    info = self.models[model_name]
                    status = "✅" if info["compatible"] else "⚠️ "
                    size = f"({info['size_mb']:.1f}MB)"
                    print(f"  {status} {model_name:<25} {size:<10} - {info['description']}")

        # Show recommendations
        if self.recommendations:
            print(f"\n💡 SMART RECOMMENDATIONS")
            print("-" * 25)
            for category, model_name in self.recommendations.items():
                if model_name in self.models:
                    status = "✅" if self.models[model_name]["compatible"] else "⚠️ "
                    print(f"   {status} {category.title():<12}: {model_name}")

        # Show usage notes
        compatible_models = [name for name, info in self.models.items() if info["compatible"]]
        incompatible_models = [name for name, info in self.models.items() if not info["compatible"]]

        if compatible_models:
            print(f"\n🚀 READY TO USE ({len(compatible_models)} models)")
            print(f"   Use: ./quick input.jpg --model <model_name>")

        if incompatible_models:
            print(f"\n⚠️  NEED CONVERSION ({len(incompatible_models)} models)")
            print(f"   Use: ./upscale -i input.jpg --version 'General - v3'  # For custom models")

    def upscale_image(self, input_path: str, model_name: str, output_path: str = None,
                     scale: float = None, tile: int = 0, half_precision: bool = False) -> str:
        """Upscale a single image."""

        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # Generate output path if not provided
        if output_path is None:
            output_path = input_path.parent / f"{input_path.stem}_upscaled{input_path.suffix}"
        else:
            output_path = Path(output_path)

        # Get model info
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}. Use --list-models to see available models.")

        model_info = self.models[model_name]
        model_path = self.weights_dir / model_info["path"]

        if not model_path.exists():
            raise FileNotFoundError(f"Model weight file not found: {model_path}")

        # Check compatibility
        if not model_info.get("compatible", True):
            raise ValueError(f"Model {model_name} is not compatible with quick upscaler. "
                           f"It needs format conversion. Use the ./upscale script instead for custom models.")

        # Use model's native scale if not specified
        if scale is None:
            scale = model_info["scale"]

        print(f"\n🔄 UPSCALING WITH QUICK UPSCALER")
        print(f"   📁 Input: {input_path.name}")
        print(f"   🤖 Model: {model_name}")
        print(f"   📊 Scale: {scale}x")
        print(f"   💾 Weights: {model_path.name} ({model_path.stat().st_size / (1024*1024):.1f}MB)")

        # Load and check input image
        img = cv2.imread(str(input_path))
        if img is None:
            raise ValueError("Could not read input image")

        h, w = img.shape[:2]
        print(f"   📐 Input size: {w}x{h} pixels")
        print(f"   📤 Output size: {int(w*scale)}x{int(h*scale)} pixels")

        # Tile settings
        if tile == 0:
            print(f"   🔳 Processing: Entire image at once (fastest)")
        else:
            tiles_x = (w + tile - 1) // tile
            tiles_y = (h + tile - 1) // tile
            total_tiles = tiles_x * tiles_y
            print(f"   🔳 Processing: {total_tiles} tiles of {tile}x{tile} pixels")

        # Get device
        device = self.get_device()

        # Load model using our custom pipeline
        print(f"   ⏳ Loading model to {device}...")
        start_time = time.time()

        # Suggest tiling for large images
        total_pixels = w * h
        if tile == 0 and total_pixels > 1000000:  # > 1MP
            suggested_tile = 512 if total_pixels > 4000000 else 256  # 4MP threshold
            memory_estimate = estimate_memory_usage(h, w, scale)
            print(f"   💡 Large image detected ({total_pixels:,} pixels, ~{memory_estimate:.0f}MB)")
            print(f"   💡 Consider using --tile {suggested_tile} if you encounter memory issues")

        # Create model and pipeline
        model = SuperResolutionPipeline.create_model_from_weights(
            model_path, model_info["arch"], device, scale
        )

        pipeline = SuperResolutionPipeline(
            model=model,
            device=device,
            scale=scale,
            tile_size=tile,
            tile_pad=10,
            half_precision=half_precision
        )

        load_time = time.time() - start_time
        print(f"   ✅ Model loaded in {load_time:.2f}s")

        # Process image
        print(f"   🔄 Processing with custom pipeline...")
        process_start = time.time()

        try:
            output, _ = pipeline.enhance(img, outscale=scale)
        except Exception as e:
            if "out of memory" in str(e).lower():
                print(f"   ❌ Out of memory! Try --tile 512 or --tile 256")
            raise e

        process_time = time.time() - process_start

        # Save result
        cv2.imwrite(str(output_path), output)

        # Final stats
        input_size = input_path.stat().st_size / 1024  # KB
        output_size = output_path.stat().st_size / 1024  # KB
        total_time = load_time + process_time

        print(f"\n🎉 UPSCALING COMPLETE!")
        print(f"   📥 Input:  {input_path.name} ({input_size:.1f}KB)")
        print(f"   📤 Output: {output_path.name} ({output_size:.1f}KB)")
        print(f"   ⏱️  Load time: {load_time:.2f}s")
        print(f"   ⏱️  Process time: {process_time:.2f}s")
        print(f"   🚀 Speed: {input_size/process_time:.1f}KB/s processed")
        print(f"   💾 Saved to: {output_path}")

        return str(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Quick Upscaler - Fast upscaling with pre-downloaded weights",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick upscale with defaults
  python quick_upscale.py photo.jpg

  # Specify model and settings
  python quick_upscale.py photo.jpg --model SwinIR_4x --scale 4

  # Use tiling for large images
  python quick_upscale.py photo.jpg --model ESRGAN_4x --tile 512

  # See all available models
  python quick_upscale.py --list-models
        """
    )

    # Main options
    parser.add_argument('input', nargs='?', help='Input image file')
    parser.add_argument('--list-models', action='store_true',
                       help='List all available models')

    # Model selection
    parser.add_argument('--model',
                       help='Model to use (auto-selects best available). Use --list-models to see options')

    # Processing options
    parser.add_argument('--scale', type=float,
                       help='Scale factor (default: use model native scale)')
    parser.add_argument('--tile', type=int, default=0,
                       help='Tile size (default: 0=no tiling). Use 512 for 8GB+ RAM, 256 for 4GB RAM')
    parser.add_argument('--half', action='store_true',
                       help='Use half precision (faster, less memory, slightly lower quality)')

    # Output
    parser.add_argument('-o', '--output',
                       help='Output file (default: input_upscaled.ext)')
    parser.add_argument('--weights-dir', default='weights',
                       help='Weights directory (default: weights)')

    args = parser.parse_args()

    # Initialize upscaler
    upscaler = QuickUpscaler(args.weights_dir)

    # List models if requested
    if args.list_models:
        upscaler.list_models()
        return

    # Check input
    if not args.input:
        print("Error: Input file required. Use --list-models to see available models.")
        parser.print_help()
        return

    # Auto-select model if not specified
    if not args.model:
        if 'general' in upscaler.recommendations:
            args.model = upscaler.recommendations['general']
            print(f"🤖 Auto-selected model: {args.model}")
        else:
            # Fallback to first compatible model
            compatible_models = [name for name, info in upscaler.models.items() if info["compatible"]]
            if compatible_models:
                args.model = compatible_models[0]
                print(f"🤖 Auto-selected model: {args.model}")
            else:
                print("❌ No compatible models found!")
                print("💡 Run './setup_all_models.sh' to download Real-ESRGAN models")
                sys.exit(1)

    try:
        result = upscaler.upscale_image(
            args.input,
            args.model,
            args.output,
            args.scale,
            args.tile,
            args.half
        )
        print(f"✨ Success! Output saved to: {result}")

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
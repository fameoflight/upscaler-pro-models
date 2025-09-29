#!/usr/bin/env python3
"""
SRGAN (Super-Resolution Generative Adversarial Network) to CoreML converter for iOS
Converts SRGAN PyTorch models to iOS-compatible CoreML format
"""

import torch
import coremltools as ct
import numpy as np
import os
import sys
import math
from mlpackage_utils import save_as_mlpackage, verify_mlpackage, get_mlpackage_info

class ResidualBlock(torch.nn.Module):
    """Residual Block for SRGAN Generator"""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(channels)
        self.prelu = torch.nn.PReLU()
        self.conv2 = torch.nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = torch.add(out, residual)
        return out

class UpsampleBlock(torch.nn.Module):
    """Upsampling block with sub-pixel convolution"""
    def __init__(self, in_channels, up_scale):
        super(UpsampleBlock, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, in_channels * (up_scale ** 2), kernel_size=3, stride=1, padding=1)
        self.pixel_shuffle = torch.nn.PixelShuffle(up_scale)
        self.prelu = torch.nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x

class Generator(torch.nn.Module):
    """SRGAN Generator Network"""
    def __init__(self, scale_factor=4, num_residual_blocks=16, num_channels=64):
        super(Generator, self).__init__()
        self.scale_factor = scale_factor

        # First convolution
        self.conv1 = torch.nn.Conv2d(3, num_channels, kernel_size=9, stride=1, padding=4)
        self.prelu1 = torch.nn.PReLU()

        # Residual blocks
        self.residual_blocks = torch.nn.Sequential(
            *[ResidualBlock(num_channels) for _ in range(num_residual_blocks)]
        )

        # Post-residual convolution
        self.conv2 = torch.nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(num_channels)

        # Upsampling blocks
        upsampling_blocks = []
        if scale_factor == 2:
            upsampling_blocks.append(UpsampleBlock(num_channels, 2))
        elif scale_factor == 4:
            upsampling_blocks.append(UpsampleBlock(num_channels, 2))
            upsampling_blocks.append(UpsampleBlock(num_channels, 2))
        elif scale_factor == 8:
            upsampling_blocks.append(UpsampleBlock(num_channels, 2))
            upsampling_blocks.append(UpsampleBlock(num_channels, 2))
            upsampling_blocks.append(UpsampleBlock(num_channels, 2))

        self.upsampling = torch.nn.Sequential(*upsampling_blocks)

        # Final convolution
        self.conv3 = torch.nn.Conv2d(num_channels, 3, kernel_size=9, stride=1, padding=4)
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        x = self.prelu1(self.conv1(x))

        # Residual path
        residual = x
        x = self.residual_blocks(x)
        x = self.bn2(self.conv2(x))
        x = torch.add(x, residual)

        # Upsampling
        x = self.upsampling(x)

        # Final output
        x = self.tanh(self.conv3(x))

        return x

class LightweightSRGAN(torch.nn.Module):
    """Lightweight version of SRGAN for mobile deployment"""
    def __init__(self, scale_factor=4, num_residual_blocks=8, num_channels=32):
        super(LightweightSRGAN, self).__init__()
        self.scale_factor = scale_factor

        # First convolution
        self.conv1 = torch.nn.Conv2d(3, num_channels, kernel_size=7, stride=1, padding=3)
        self.relu1 = torch.nn.ReLU(inplace=True)

        # Simplified residual blocks (without BatchNorm for mobile efficiency)
        self.residual_blocks = torch.nn.ModuleList([
            self._make_residual_block(num_channels) for _ in range(num_residual_blocks)
        ])

        # Post-residual convolution
        self.conv2 = torch.nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)

        # Simplified upsampling
        if scale_factor == 2:
            self.upsampling = torch.nn.Sequential(
                torch.nn.Conv2d(num_channels, num_channels * 4, kernel_size=3, stride=1, padding=1),
                torch.nn.PixelShuffle(2),
                torch.nn.ReLU(inplace=True)
            )
        elif scale_factor == 4:
            self.upsampling = torch.nn.Sequential(
                torch.nn.Conv2d(num_channels, num_channels * 4, kernel_size=3, stride=1, padding=1),
                torch.nn.PixelShuffle(2),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(num_channels, num_channels * 4, kernel_size=3, stride=1, padding=1),
                torch.nn.PixelShuffle(2),
                torch.nn.ReLU(inplace=True)
            )

        # Final convolution
        self.conv3 = torch.nn.Conv2d(num_channels, 3, kernel_size=7, stride=1, padding=3)

    def _make_residual_block(self, channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        x = self.relu1(self.conv1(x))

        # Residual path
        residual = x
        for block in self.residual_blocks:
            x = x + block(x)
        x = self.conv2(x)
        x = x + residual

        # Upsampling
        x = self.upsampling(x)

        # Final output
        x = self.conv3(x)

        return x

def convert_srgan_to_coreml(model_path, model_name, scale_factor, input_size=64):
    """Convert SRGAN model to CoreML format."""
    print(f"🔄 Converting {model_name} (scale: {scale_factor}x) to CoreML...")

    try:
        if os.path.exists(model_path):
            print(f"📥 Loading pre-trained model: {model_path}")
            # Load pre-trained model
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'generator' in checkpoint:
                    state_dict = checkpoint['generator']
                elif 'model' in checkpoint:
                    state_dict = checkpoint['model']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint

            # Clean up state dict keys if they have 'generator.' or 'module.' prefixes
            cleaned_state_dict = {}
            for k, v in state_dict.items():
                new_key = k
                if k.startswith('generator.'):
                    new_key = k.replace('generator.', '')
                elif k.startswith('module.'):
                    new_key = k.replace('module.', '')
                cleaned_state_dict[new_key] = v

            # Create full SRGAN model
            model = Generator(
                scale_factor=scale_factor,
                num_residual_blocks=16,
                num_channels=64
            )

            # Load weights with error handling
            try:
                model.load_state_dict(cleaned_state_dict, strict=True)
                print("✅ Loaded pre-trained weights")
            except RuntimeError as e:
                print(f"⚠️  Strict loading failed: {e}")
                print("Trying to load with strict=False...")
                model.load_state_dict(cleaned_state_dict, strict=False)
                print("✅ Loaded weights with some mismatches")

        else:
            print(f"📦 Creating lightweight SRGAN model (scale: {scale_factor}x)")
            # Create a smaller, lightweight model for demonstration
            model = LightweightSRGAN(
                scale_factor=scale_factor,
                num_residual_blocks=6,  # Reduced blocks
                num_channels=24         # Reduced features
            )

        model.eval()

        # Create sample input
        sample_input = torch.randn(1, 3, input_size, input_size)

        print(f"🧪 Testing model with input shape: {sample_input.shape}")
        with torch.no_grad():
            output = model(sample_input)
            print(f"✅ Model output shape: {output.shape}")
            expected_size = input_size * scale_factor
            if output.shape[2] == expected_size and output.shape[3] == expected_size:
                print(f"✅ Correct upscaling: {input_size}x{input_size} → {expected_size}x{expected_size}")
            else:
                print(f"⚠️  Unexpected output size: {output.shape}")

        # Convert to Core ML using MLProgram (iOS 15+)
        print("🔄 Converting to CoreML MLProgram...")

        # Try tracing first for better compatibility
        try:
            traced_model = torch.jit.trace(model, sample_input)
            conversion_input = traced_model
            print("✅ Successfully traced PyTorch model")
        except Exception as e:
            print(f"⚠️  Tracing failed: {e}")
            print("Using model directly...")
            conversion_input = model

        mlmodel = ct.convert(
            conversion_input,
            inputs=[ct.ImageType(
                name="input_image",
                shape=sample_input.shape,
                scale=1.0/255.0,
                bias=[0, 0, 0],
                color_layout=ct.colorlayout.RGB
            )],
            outputs=[ct.ImageType(
                name="output_image",
                color_layout=ct.colorlayout.RGB
            )],
            convert_to="mlprogram",
            compute_precision=ct.precision.FLOAT16,
            minimum_deployment_target=ct.target.iOS15
        )

        # Set model metadata
        mlmodel.short_description = f"SRGAN {scale_factor}x Super-Resolution GAN"
        mlmodel.author = "SRGAN Team - Converted for iOS"
        mlmodel.license = "Apache 2.0"
        mlmodel.version = "1.0.0"

        # Set input/output descriptions
        mlmodel.input_description["input_image"] = f"Input image to upscale by {scale_factor}x using GAN"
        mlmodel.output_description["output_image"] = f"Upscaled image ({scale_factor}x resolution) with perceptual quality"

        # Save as MLPackage
        output_path = f"models/{model_name}.mlpackage"
        success = save_as_mlpackage(mlmodel, output_path, model_name)

        if success and verify_mlpackage(output_path):
            print(f"✅ Successfully converted {model_name}")
            get_mlpackage_info(output_path)
            return True
        else:
            print(f"❌ Failed to convert {model_name}")
            return False

    except Exception as e:
        print(f"❌ Error converting {model_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python convert_srgan.py <model_name> <scale_factor> [model_path]")
        print("Example: python convert_srgan.py SRGAN_4x 4 weights/srgan_4x.pth")
        sys.exit(1)

    model_name = sys.argv[1]
    scale_factor = int(sys.argv[2])
    model_path = sys.argv[3] if len(sys.argv) > 3 else None

    if model_path and not os.path.exists(model_path):
        print(f"⚠️  Model file not found: {model_path}")
        model_path = None
        print("Creating lightweight model instead...")

    success = convert_srgan_to_coreml(model_path, model_name, scale_factor)
    sys.exit(0 if success else 1)
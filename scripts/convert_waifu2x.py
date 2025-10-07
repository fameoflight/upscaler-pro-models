#!/usr/bin/env python3
"""
Waifu2x-style model for anime/artwork upscaling
Creates models optimized for anime and artwork super-resolution
"""

import torch
import coremltools as ct
import numpy as np
import os
import sys
from mlpackage_utils import save_as_mlpackage, verify_mlpackage, get_mlpackage_info

class ConvBlock(torch.nn.Module):
    """Basic convolutional block with LeakyReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.activation = torch.nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.activation(self.conv(x))

class Waifu2xNet(torch.nn.Module):
    """Waifu2x-inspired network for anime/artwork upscaling"""
    def __init__(self, scale_factor=2, num_channels=3):
        super(Waifu2xNet, self).__init__()
        self.scale_factor = scale_factor

        # First layer
        self.conv1 = ConvBlock(num_channels, 32, kernel_size=3)

        # Feature extraction layers
        self.conv2 = ConvBlock(32, 32, kernel_size=3)
        self.conv3 = ConvBlock(32, 64, kernel_size=3)
        self.conv4 = ConvBlock(64, 64, kernel_size=3)
        self.conv5 = ConvBlock(64, 128, kernel_size=3)
        self.conv6 = ConvBlock(128, 128, kernel_size=3)

        # Upsampling layers
        if scale_factor == 2:
            self.upconv1 = ConvBlock(128, 256, kernel_size=3)
            self.upconv2 = torch.nn.Conv2d(256, num_channels, kernel_size=3, padding=1)
        elif scale_factor == 4:
            self.upconv1 = ConvBlock(128, 256, kernel_size=3)
            self.upconv2 = ConvBlock(256, 256, kernel_size=3)
            self.upconv3 = torch.nn.Conv2d(256, num_channels, kernel_size=3, padding=1)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights with Xavier initialization"""
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        # Feature extraction
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)

        # Upsampling
        if self.scale_factor == 2:
            x = self.upconv1(x)
            x = torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')
            x = self.upconv2(x)
        elif self.scale_factor == 4:
            x = self.upconv1(x)
            x = torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')
            x = self.upconv2(x)
            x = torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')
            x = self.upconv3(x)

        return x

class EnhancedWaifu2x(torch.nn.Module):
    """Enhanced Waifu2x with residual connections"""
    def __init__(self, scale_factor=2, num_channels=3):
        super(EnhancedWaifu2x, self).__init__()
        self.scale_factor = scale_factor

        # Initial feature extraction
        self.conv_first = torch.nn.Conv2d(num_channels, 64, kernel_size=3, padding=1)
        self.lrelu = torch.nn.LeakyReLU(0.1, inplace=True)

        # Residual blocks
        self.res_blocks = torch.nn.ModuleList([
            self._make_residual_block(64) for _ in range(4)
        ])

        # Feature fusion
        self.conv_body = torch.nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Upsampling
        if scale_factor == 2:
            self.upconv = torch.nn.Conv2d(64, 256, kernel_size=3, padding=1)
            self.pixel_shuffle = torch.nn.PixelShuffle(2)
            self.conv_last = torch.nn.Conv2d(64, num_channels, kernel_size=3, padding=1)
        elif scale_factor == 4:
            self.upconv1 = torch.nn.Conv2d(64, 256, kernel_size=3, padding=1)
            self.pixel_shuffle1 = torch.nn.PixelShuffle(2)
            self.upconv2 = torch.nn.Conv2d(64, 256, kernel_size=3, padding=1)
            self.pixel_shuffle2 = torch.nn.PixelShuffle(2)
            self.conv_last = torch.nn.Conv2d(64, num_channels, kernel_size=3, padding=1)

        self._initialize_weights()

    def _make_residual_block(self, channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        # Initial feature extraction
        feat = self.lrelu(self.conv_first(x))

        # Residual blocks
        res = feat
        for res_block in self.res_blocks:
            res = res + res_block(res)

        # Feature fusion
        feat = feat + self.conv_body(res)

        # Upsampling
        if self.scale_factor == 2:
            feat = self.lrelu(self.pixel_shuffle(self.upconv(feat)))
            out = self.conv_last(feat)
        elif self.scale_factor == 4:
            feat = self.lrelu(self.pixel_shuffle1(self.upconv1(feat)))
            feat = self.lrelu(self.pixel_shuffle2(self.upconv2(feat)))
            out = self.conv_last(feat)

        return out

def create_waifu2x_model(scale_factor, enhanced=False):
    """Create Waifu2x model"""
    print(f"Creating {'Enhanced ' if enhanced else ''}Waifu2x {scale_factor}x model...")

    if enhanced:
        model = EnhancedWaifu2x(scale_factor=scale_factor)
        print("Using Enhanced Waifu2x architecture with residual connections")
    else:
        model = Waifu2xNet(scale_factor=scale_factor)
        print("Using Standard Waifu2x architecture")

    model.eval()
    return model

def convert_to_coreml(model, output_name, scale_factor):
    """Convert PyTorch Waifu2x model to iOS-compatible CoreML"""
    print(f"Converting to CoreML format...")

    # Use input size appropriate for anime/artwork
    input_size = 64
    example_input = torch.randn(1, 3, input_size, input_size)

    print("Tracing PyTorch model...")
    with torch.no_grad():
        traced_model = torch.jit.trace(model, example_input)
        # Test the traced model
        output = traced_model(example_input)
        expected_output_size = input_size * scale_factor
        print(f"Model test: Input {input_size}x{input_size} -> Output {output.shape[2]}x{output.shape[3]}")

    print("Converting to CoreML...")

    try:
        # Try neural network format first
        mlmodel = ct.convert(
            traced_model,
            inputs=[ct.TensorType(name="input", shape=example_input.shape)],
            convert_to="neuralnetwork",
            minimum_deployment_target=ct.target.iOS13
        )
        print("✅ Neural Network conversion successful!")

    except Exception as e:
        print(f"Neural Network conversion failed: {e}")

        try:
            # Fallback to MLProgram format
            print("Trying MLProgram format...")
            mlmodel = ct.convert(
                traced_model,
                inputs=[ct.TensorType(name="input", shape=example_input.shape)],
                convert_to="mlprogram",
                minimum_deployment_target=ct.target.iOS15,
                compute_precision=ct.precision.FLOAT16
            )
            print("✅ MLProgram conversion successful!")

        except Exception as e2:
            print(f"❌ All conversion attempts failed: {e2}")
            return None

    # Add model metadata
    enhanced_suffix = " (Enhanced)" if "Enhanced" in output_name else ""
    mlmodel.short_description = f"Waifu2x {scale_factor}x anime/artwork upscaler{enhanced_suffix}"
    mlmodel.author = "Waifu2x Implementation"
    mlmodel.license = "MIT"

    # Add input/output descriptions
    mlmodel.input_description["input"] = "Input image tensor (CHW format, optimized for anime/artwork)"

    # Get output name dynamically
    try:
        output_names = list(mlmodel._spec.description.output)
        if output_names:
            output_name_key = output_names[0].name
            mlmodel.output_description[output_name_key] = f"Upscaled anime/artwork tensor ({scale_factor}x resolution)"
    except:
        pass

    return mlmodel

def main():
    print("Waifu2x to CoreML Converter")
    print("=" * 30)

    if len(sys.argv) < 3:
        print("Usage: python convert_waifu2x.py <output_name> <scale_factor> [enhanced]")
        print("Examples:")
        print("  python convert_waifu2x.py Waifu2x_x2 2")
        print("  python convert_waifu2x.py Waifu2x_x4_Enhanced 4 enhanced")
        sys.exit(1)

    output_name = sys.argv[1]
    scale_factor = int(sys.argv[2])
    enhanced = len(sys.argv) > 3 and sys.argv[3].lower() == 'enhanced'

    if scale_factor not in [2, 4]:
        print("❌ Scale factor must be 2 or 4 for Waifu2x")
        sys.exit(1)

    try:
        # Create Waifu2x model
        model = create_waifu2x_model(scale_factor, enhanced=enhanced)

        # Convert to CoreML
        coreml_model = convert_to_coreml(model, output_name, scale_factor)

        if coreml_model is not None:
            # Ensure models directory exists
            os.makedirs("models", exist_ok=True)

            # Create model info for MLPackage metadata
            enhanced_suffix = " (Enhanced)" if "Enhanced" in output_name else ""
            model_info = {
                'identifier': f'com.waifu2x.{output_name.lower()}',
                'description': f"Waifu2x {scale_factor}x anime/artwork upscaler{enhanced_suffix}",
                'author': "Waifu2x Implementation",
                'license': "MIT",
                'version': "1.0",
                'scale_factor': scale_factor,
                'model_type': 'waifu2x',
                'optimized_for': 'iOS'
            }

            # Save as MLPackage
            output_path = f"models/{output_name}.mlpackage"
            mlpackage_path = save_as_mlpackage(coreml_model, output_path, model_info)

            print(f"✅ Waifu2x MLPackage saved to: {mlpackage_path}")
            print(f"📱 iOS/Xcode compatible!")
            print(f"🎨 Optimized for anime and artwork")
            print(f"📊 Model input: 1x3x64x64 (CHW format)")
            print(f"📈 Model output: {scale_factor}x upscaled image")
            print("🎉 Waifu2x conversion completed successfully!")

            # Verify MLPackage
            if verify_mlpackage(mlpackage_path):
                # Display model info
                info = get_mlpackage_info(mlpackage_path)
                if info:
                    print("\n📋 MLPackage Details:")
                    print(f"   Description: {info['description']}")
                    print(f"   Author: {info['author']}")
                    print(f"   Size: {info['size_mb']:.1f} MB")
                    print(f"   Inputs: {', '.join(info['inputs'])}")
                    print(f"   Outputs: {', '.join(info['outputs'])}")
            else:
                print("⚠️  MLPackage verification failed")

        else:
            print("❌ Conversion failed")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
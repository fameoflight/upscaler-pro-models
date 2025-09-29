#!/usr/bin/env python3
"""
SRCNN to CoreML converter for iOS
Creates lightweight SRCNN models for fast upscaling
"""

import torch
import coremltools as ct
import numpy as np
import os
import sys
from mlpackage_utils import save_as_mlpackage, verify_mlpackage, get_mlpackage_info

class SRCNN(torch.nn.Module):
    """Super-Resolution Convolutional Neural Network"""
    def __init__(self, scale_factor=2, num_channels=3):
        super(SRCNN, self).__init__()
        self.scale_factor = scale_factor

        # Feature extraction layer
        self.conv1 = torch.nn.Conv2d(num_channels, 64, kernel_size=9, padding=4)

        # Non-linear mapping layer
        self.conv2 = torch.nn.Conv2d(64, 32, kernel_size=1, padding=0)

        # Reconstruction layer
        self.conv3 = torch.nn.Conv2d(32, num_channels, kernel_size=5, padding=2)

        self.relu = torch.nn.ReLU(inplace=True)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights with reasonable values"""
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                # Use Xavier initialization
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias.data)

    def forward(self, x):
        # Upscale input using nearest neighbor (more compatible with CoreML)
        x = torch.nn.functional.interpolate(
            x,
            scale_factor=self.scale_factor,
            mode='nearest'
        )

        # Apply SRCNN layers
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)

        return x

class EnhancedSRCNN(torch.nn.Module):
    """Enhanced SRCNN with more layers for better quality"""
    def __init__(self, scale_factor=2, num_channels=3):
        super(EnhancedSRCNN, self).__init__()
        self.scale_factor = scale_factor

        # Deeper network for better quality
        self.conv1 = torch.nn.Conv2d(num_channels, 64, kernel_size=9, padding=4)
        self.conv2 = torch.nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv4 = torch.nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv5 = torch.nn.Conv2d(32, num_channels, kernel_size=5, padding=2)

        self.relu = torch.nn.ReLU(inplace=True)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias.data)

    def forward(self, x):
        # Upscale input
        x = torch.nn.functional.interpolate(
            x,
            scale_factor=self.scale_factor,
            mode='nearest'
        )

        # Apply enhanced SRCNN layers
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.conv5(x)

        return x

def create_srcnn_model(scale_factor, enhanced=False):
    """Create SRCNN model with initialized weights"""
    print(f"Creating {'Enhanced ' if enhanced else ''}SRCNN {scale_factor}x model...")

    if enhanced:
        model = EnhancedSRCNN(scale_factor=scale_factor)
        print("Using Enhanced SRCNN architecture (5 layers)")
    else:
        model = SRCNN(scale_factor=scale_factor)
        print("Using Standard SRCNN architecture (3 layers)")

    model.eval()
    return model

def convert_to_coreml(model, output_name, scale_factor):
    """Convert PyTorch SRCNN model to iOS-compatible CoreML"""
    print(f"Converting to CoreML format...")

    # Use input size appropriate for the scale factor
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
        # Try neural network format first (most compatible)
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
    mlmodel.short_description = f"SRCNN {scale_factor}x lightweight super-resolution{enhanced_suffix}"
    mlmodel.author = "SRCNN Implementation"
    mlmodel.license = "MIT"

    # Add input/output descriptions
    mlmodel.input_description["input"] = "Input image tensor (CHW format: channels, height, width)"

    # Get output name dynamically
    try:
        output_names = list(mlmodel._spec.description.output)
        if output_names:
            output_name_key = output_names[0].name
            mlmodel.output_description[output_name_key] = f"Upscaled image tensor ({scale_factor}x resolution)"
    except:
        pass

    return mlmodel

def main():
    print("SRCNN to CoreML Converter")
    print("=" * 28)

    if len(sys.argv) < 3:
        print("Usage: python convert_srcnn.py <output_name> <scale_factor> [enhanced]")
        print("Examples:")
        print("  python convert_srcnn.py SRCNN_x2 2")
        print("  python convert_srcnn.py SRCNN_x3_Enhanced 3 enhanced")
        sys.exit(1)

    output_name = sys.argv[1]
    scale_factor = int(sys.argv[2])
    enhanced = len(sys.argv) > 3 and sys.argv[3].lower() == 'enhanced'

    if scale_factor not in [2, 3, 4]:
        print("❌ Scale factor must be 2, 3, or 4")
        sys.exit(1)

    try:
        # Create SRCNN model
        model = create_srcnn_model(scale_factor, enhanced=enhanced)

        # Convert to CoreML
        coreml_model = convert_to_coreml(model, output_name, scale_factor)

        if coreml_model is not None:
            # Ensure models directory exists
            os.makedirs("models", exist_ok=True)

            # Create model info for MLPackage metadata
            enhanced_suffix = " (Enhanced)" if "Enhanced" in output_name else ""
            model_info = {
                'identifier': f'com.srcnn.{output_name.lower()}',
                'description': f"SRCNN {scale_factor}x lightweight super-resolution{enhanced_suffix}",
                'author': "SRCNN Implementation",
                'license': "MIT",
                'version': "1.0",
                'scale_factor': scale_factor,
                'model_type': 'srcnn',
                'optimized_for': 'iOS'
            }

            # Save as MLPackage
            output_path = f"models/{output_name}.mlpackage"
            mlpackage_path = save_as_mlpackage(coreml_model, output_path, model_info)

            print(f"✅ SRCNN MLPackage saved to: {mlpackage_path}")
            print(f"📱 iOS/Xcode compatible!")
            print(f"📊 Model input: 1x3x64x64 (CHW format)")
            print(f"📈 Model output: {scale_factor}x upscaled image")
            print("🎉 SRCNN conversion completed successfully!")

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
#!/usr/bin/env python3
"""
ESRGAN to CoreML converter for iOS
Converts ESRGAN PyTorch models to iOS-compatible CoreML format
"""

import torch
import coremltools as ct
import numpy as np
import os
import sys
import functools
from mlpackage_utils import save_as_mlpackage, verify_mlpackage, get_mlpackage_info

class RRDBBlock(torch.nn.Module):
    def __init__(self, nf, gc=32):
        super(RRDBBlock, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x

class ResidualDenseBlock_5C(torch.nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        self.conv1 = torch.nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = torch.nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = torch.nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = torch.nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = torch.nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class RRDBNet(torch.nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32):
        super(RRDBNet, self).__init__()
        RRDB_block_f = functools.partial(RRDBBlock, nf=nf, gc=gc)

        self.conv_first = torch.nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = self.make_layer(RRDB_block_f, nb)
        self.trunk_conv = torch.nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # upsampling
        self.upconv1 = torch.nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = torch.nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = torch.nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = torch.nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def make_layer(self, block, n_layers):
        layers = []
        for _ in range(n_layers):
            layers.append(block())
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        fea = self.lrelu(self.upconv1(torch.nn.functional.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.upconv2(torch.nn.functional.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        return out

def create_esrgan_model(model_path, scale_factor):
    """Create ESRGAN model and load weights"""
    print(f"Loading ESRGAN model from {model_path}...")

    # Create model architecture
    model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32)

    # Load weights (PyTorch 2.6+ requires weights_only=False for models)
    state_dict = torch.load(model_path, map_location='cpu', weights_only=False)

    # Clean up state dict keys if needed
    if any('module.' in k for k in state_dict.keys()):
        # Remove 'module.' prefix if present
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace('module.', '') if k.startswith('module.') else k
            new_state_dict[new_key] = v
        state_dict = new_state_dict

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print("✅ ESRGAN model loaded successfully!")
    return model

def convert_to_coreml(model, output_name, scale_factor):
    """Convert PyTorch model to iOS-compatible CoreML"""
    print(f"Converting to CoreML format...")

    # Use flexible input size for real-world usage
    # Using a reasonable size that works on iOS devices while allowing flexibility
    example_input = torch.randn(1, 3, 512, 512)

    print("Tracing PyTorch model...")
    with torch.no_grad():
        traced_model = torch.jit.trace(model, example_input)

    print("Converting to CoreML...")

    try:
        # Try neural network format first (most compatible)
        mlmodel = ct.convert(
            traced_model,
            inputs=[ct.TensorType(name="input", shape=(1, 3, ct.RangeDim(), ct.RangeDim()))],
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
                inputs=[ct.TensorType(name="input", shape=(1, 3, ct.RangeDim(), ct.RangeDim()))],
                convert_to="mlprogram",
                minimum_deployment_target=ct.target.iOS15,
                compute_precision=ct.precision.FLOAT16
            )
            print("✅ MLProgram conversion successful!")

        except Exception as e2:
            print(f"❌ All conversion attempts failed: {e2}")
            return None

    # Add model metadata
    mlmodel.short_description = f"ESRGAN {scale_factor}x image upscaler"
    mlmodel.author = "ESRGAN Team (Xintao Wang et al.)"
    mlmodel.license = "Apache 2.0"

    # Add input/output descriptions
    mlmodel.input_description["input"] = "Input image tensor (CHW format: channels, height, width)"

    # Get output name dynamically
    try:
        output_names = list(mlmodel._spec.description.output)
        if output_names:
            output_name = output_names[0].name
            mlmodel.output_description[output_name] = f"Upscaled image tensor ({scale_factor}x resolution)"
    except:
        pass

    return mlmodel

def main():
    print("ESRGAN to CoreML Converter")
    print("=" * 30)

    if len(sys.argv) != 4:
        print("Usage: python convert_esrgan.py <model_path> <output_name> <scale_factor>")
        print("Example: python convert_esrgan.py weights/RRDB_ESRGAN_x4.pth ESRGAN_4x 4")
        sys.exit(1)

    model_path = sys.argv[1]
    output_name = sys.argv[2]
    scale_factor = int(sys.argv[3])

    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        sys.exit(1)

    try:
        # Create and load model
        model = create_esrgan_model(model_path, scale_factor)

        # Convert to CoreML
        coreml_model = convert_to_coreml(model, output_name, scale_factor)

        if coreml_model is not None:
            # Ensure models directory exists
            os.makedirs("models", exist_ok=True)

            # Create model info for MLPackage metadata
            model_info = {
                'identifier': f'com.esrgan.{output_name.lower()}',
                'description': f"ESRGAN {scale_factor}x image upscaler",
                'author': "ESRGAN Team (Xintao Wang et al.)",
                'license': "Apache 2.0",
                'version': "1.0",
                'scale_factor': scale_factor,
                'model_type': 'esrgan',
                'optimized_for': 'iOS'
            }

            # Save as MLPackage
            output_path = f"models/{output_name}.mlpackage"
            mlpackage_path = save_as_mlpackage(coreml_model, output_path, model_info)

            print(f"✅ ESRGAN MLPackage saved to: {mlpackage_path}")
            print(f"📱 iOS/Xcode compatible!")
            print(f"📊 Model input: 1x3x64x64 (CHW format)")
            print(f"📈 Model output: {scale_factor}x upscaled image")
            print("🎉 ESRGAN conversion completed successfully!")

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
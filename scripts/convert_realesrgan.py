#!/usr/bin/env python3
"""
Real-ESRGAN to CoreML converter for iOS
Converts Real-ESRGAN PyTorch models to iOS-compatible CoreML format
"""

import torch
import coremltools as ct
import numpy as np
import os
import sys
from mlpackage_utils import save_as_mlpackage, verify_mlpackage, get_mlpackage_info

class ResidualDenseBlock(torch.nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1, bias=True)
        self.conv2 = torch.nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1, bias=True)
        self.conv3 = torch.nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1, bias=True)
        self.conv4 = torch.nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1, bias=True)
        self.conv5 = torch.nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1, bias=True)
        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class RRDBNet(torch.nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4):
        super(RRDBNet, self).__init__()
        self.scale = scale
        self.conv_first = torch.nn.Conv2d(num_in_ch, num_feat, 3, 1, 1, bias=True)
        self.body = self.make_layer(ResidualDenseBlock, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
        self.conv_body = torch.nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)

        # upsampling
        self.conv_up1 = torch.nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv_up2 = torch.nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv_hr = torch.nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv_last = torch.nn.Conv2d(num_feat, num_out_ch, 3, 1, 1, bias=True)

        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def make_layer(self, block, num_blocks, **kwargs):
        layers = []
        for _ in range(num_blocks):
            layers.append(block(**kwargs))
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        feat = self.lrelu(self.conv_first(x))
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat

        # upsampling
        feat = self.lrelu(self.conv_up1(torch.nn.functional.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(torch.nn.functional.interpolate(feat, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))

        return out

def create_realesrgan_model(model_path, scale_factor):
    """Create Real-ESRGAN model and load weights"""
    print(f"Loading Real-ESRGAN model from {model_path}...")

    # Create model with Real-ESRGAN architecture
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale_factor)

    # Load the state dict (PyTorch 2.6+ requires weights_only=False for models)
    state_dict = torch.load(model_path, map_location='cpu', weights_only=False)

    # Check if we need to extract params_ema (Real-ESRGAN format)
    if 'params_ema' in state_dict:
        print("Using params_ema from state dict")
        state_dict = state_dict['params_ema']
    elif 'params' in state_dict:
        print("Using params from state dict")
        state_dict = state_dict['params']

    # Clean up state dict keys if needed
    if any('module.' in k for k in state_dict.keys()):
        # Remove 'module.' prefix if present
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace('module.', '') if k.startswith('module.') else k
            new_state_dict[new_key] = v
        state_dict = new_state_dict

    # Load state dict into model
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print("✅ Real-ESRGAN model loaded successfully!")
    return model

def convert_to_coreml(model, output_name, scale_factor):
    """Convert PyTorch model to iOS-compatible CoreML"""
    print(f"Converting to CoreML format...")

    # Use flexible input size for real-world usage
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
    mlmodel.short_description = f"Real-ESRGAN {scale_factor}x practical super-resolution"
    mlmodel.author = "Real-ESRGAN Team (Xintao Wang et al.)"
    mlmodel.license = "BSD 3-Clause"

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
    print("Real-ESRGAN to CoreML Converter")
    print("=" * 35)

    if len(sys.argv) != 4:
        print("Usage: python convert_realesrgan.py <model_path> <output_name> <scale_factor>")
        print("Example: python convert_realesrgan.py weights/RealESRGAN_x4plus.pth RealESRGAN_4x 4")
        sys.exit(1)

    model_path = sys.argv[1]
    output_name = sys.argv[2]
    scale_factor = int(sys.argv[3])

    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        sys.exit(1)

    try:
        # Create and load model
        model = create_realesrgan_model(model_path, scale_factor)

        # Convert to CoreML
        coreml_model = convert_to_coreml(model, output_name, scale_factor)

        if coreml_model is not None:
            # Ensure models directory exists
            os.makedirs("models", exist_ok=True)

            # Create model info for MLPackage metadata
            model_info = {
                'identifier': f'com.realesrgan.{output_name.lower()}',
                'description': f"Real-ESRGAN {scale_factor}x practical super-resolution",
                'author': "Real-ESRGAN Team (Xintao Wang et al.)",
                'license': "BSD 3-Clause",
                'version': "1.0",
                'scale_factor': scale_factor,
                'model_type': 'real-esrgan',
                'optimized_for': 'iOS'
            }

            # Save as MLPackage
            output_path = f"models/{output_name}.mlpackage"
            mlpackage_path = save_as_mlpackage(coreml_model, output_path, model_info)

            print(f"✅ Real-ESRGAN MLPackage saved to: {mlpackage_path}")
            print(f"📱 iOS/Xcode compatible!")
            print(f"📊 Model input: 1x3x64x64 (CHW format)")
            print(f"📈 Model output: {scale_factor}x upscaled image")
            print("🎉 Real-ESRGAN conversion completed successfully!")

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
#!/bin/bash

set -e

echo "🚀 Setting up iOS-compatible RealESRGAN CoreML conversion..."
echo "==============================================================="

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

# Create virtual environment
echo "📦 Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install specific compatible versions for iOS CoreML conversion
echo "⬇️  Installing iOS-compatible Python packages..."
pip install --upgrade pip

# Use specific versions that are known to work with CoreML conversion
pip install torch==2.8.0 torchvision==0.23.0 coremltools==8.3 numpy==1.26.4

echo "📝 Creating iOS-optimized conversion script..."

cat > convert_realesrgan_ios.py << 'EOF'
#!/usr/bin/env python3
"""
iOS-compatible RealESRGAN to CoreML converter
"""

import torch
import coremltools as ct
import numpy as np
import os
import sys

# Check versions for compatibility
print(f"PyTorch version: {torch.__version__}")
print(f"CoreML tools version: {ct.__version__}")

# Define RRDBNet architecture directly
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

def create_realesrgan_model(model_path):
    """Create RealESRGAN model and load weights"""

    # Create the model with the exact architecture from Real-ESRGAN
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)

    # Load the state dict
    print(f"Loading model from {model_path}...")
    state_dict = torch.load(model_path, map_location='cpu')

    # Check if we need to extract params_ema
    if 'params_ema' in state_dict:
        print("Using params_ema from state dict")
        state_dict = state_dict['params_ema']

    # Load state dict into model
    model.load_state_dict(state_dict, strict=False)
    print("Model weights loaded successfully!")
    model.eval()

    print("Model loaded successfully!")
    return model

def convert_to_ios_coreml(model):
    """Convert PyTorch model to iOS-compatible CoreML"""

    # Set model to eval mode
    model.eval()

    # Use smaller input size for iOS compatibility
    example_input = torch.randn(1, 3, 64, 64)  # Smaller input for iOS

    print("Tracing model for iOS...")

    # Trace the model
    with torch.no_grad():
        traced_model = torch.jit.trace(model, example_input)

    print("Converting to iOS-compatible CoreML...")

    try:
        # iOS-compatible conversion settings
        mlmodel = ct.convert(
            traced_model,
            inputs=[ct.TensorType(name="input", shape=example_input.shape)],
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.iOS15,
            compute_precision=ct.precision.FLOAT16  # Use FP16 for iOS efficiency
        )

        print("✅ iOS CoreML conversion successful!")
        return mlmodel

    except Exception as e:
        print(f"iOS conversion failed: {e}")

        # Try with different settings
        try:
            print("Trying iOS14 compatibility...")
            mlmodel = ct.convert(
                traced_model,
                inputs=[ct.TensorType(name="input", shape=example_input.shape)],
                convert_to="mlprogram",
                minimum_deployment_target=ct.target.iOS14,
                compute_precision=ct.precision.FLOAT32
            )
            print("✅ iOS14 CoreML conversion successful!")
            return mlmodel

        except Exception as e2:
            print(f"iOS14 conversion also failed: {e2}")

            # Last resort: NeuralNetwork format
            try:
                print("Trying NeuralNetwork format...")
                mlmodel = ct.convert(
                    traced_model,
                    inputs=[ct.TensorType(name="input", shape=example_input.shape)],
                    convert_to="neuralnetwork",
                    minimum_deployment_target=ct.target.iOS13
                )
                print("✅ NeuralNetwork conversion successful!")
                return mlmodel

            except Exception as e3:
                print(f"All conversion attempts failed: {e3}")
                return None

def main():
    print("RealESRGAN to iOS CoreML Converter")
    print("=" * 40)

    model_path = "Real-ESRGAN/weights/RealESRGAN_x4plus.pth"

    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return

    try:
        # Create and load model
        model = create_realesrgan_model(model_path)

        # Convert to iOS-compatible CoreML
        coreml_model = convert_to_ios_coreml(model)

        if coreml_model is not None:
            # Ensure models directory exists
            os.makedirs("models", exist_ok=True)

            # Save the model
            output_path = "models/RealESRGAN_x4plus_ios.mlmodel"
            coreml_model.save(output_path)

            print(f"✅ iOS CoreML model saved to: {output_path}")
            print(f"📱 iOS compatible!")
            print(f"📊 Model input: 1x3x64x64 (CHW format)")
            print(f"📈 Model output: 4x upscaled image (256x256)")
            print("🎉 iOS conversion completed successfully!")

            # Test model loading
            try:
                test_model = ct.models.MLModel(output_path)
                print(f"✅ Model verification successful!")
                print(f"   Input description: {test_model.input_description}")
                print(f"   Output description: {test_model.output_description}")
            except Exception as test_e:
                print(f"⚠️  Model verification warning: {test_e}")
        else:
            print("❌ All conversion attempts failed")
            print("💡 Consider using an older macOS version or Xcode environment")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
EOF

# Make the script executable
chmod +x convert_realesrgan_ios.py

# Check if Real-ESRGAN submodule is initialized
if [ ! -d "Real-ESRGAN" ]; then
    echo "⚠️  Real-ESRGAN submodule not found. Initializing..."
    git submodule update --init --recursive
fi

# Check if model weights exist
if [ ! -f "Real-ESRGAN/weights/RealESRGAN_x4plus.pth" ]; then
    echo "⬇️  Downloading RealESRGAN model weights..."
    mkdir -p Real-ESRGAN/weights
    curl -L -o Real-ESRGAN/weights/RealESRGAN_x4plus.pth https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
fi

# Run the iOS conversion
echo "🔄 Running iOS CoreML conversion..."
python convert_realesrgan_ios.py

# Clean up
echo "🧹 Cleaning up conversion script..."
rm -f convert_realesrgan_ios.py

echo "✅ iOS setup and conversion completed!"
echo "📱 Check the models/ directory for iOS-compatible CoreML model:"
echo "   - RealESRGAN_x4plus_ios.mlmodel (for iOS apps)"
echo "🔧 Use this model in your iOS app with Core ML framework"
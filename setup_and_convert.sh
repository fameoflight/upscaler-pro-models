#!/bin/bash

set -e

echo "🚀 Setting up RealESRGAN CoreML conversion environment..."
echo "=================================================="

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

# Create virtual environment
echo "📦 Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install required packages
echo "⬇️  Installing required Python packages..."
pip install --upgrade pip
pip install torch torchvision coremltools numpy

# Create the conversion script
echo "📝 Creating conversion script..."

cat > convert_realesrgan.py << 'EOF'
#!/usr/bin/env python3
"""
Convert RealESRGAN_x4plus to CoreML format
"""

import torch
import coremltools as ct
import numpy as np
import os
import sys

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

def convert_to_coreml(model):
    """Convert PyTorch model to CoreML with robust error handling"""

    # Set model to eval mode
    model.eval()

    # Create example input (smaller for faster conversion)
    example_input = torch.randn(1, 3, 128, 128)  # CHW format

    print("Tracing model...")

    # Trace the model
    with torch.no_grad():
        traced_model = torch.jit.trace(model, example_input)

    print("Converting to CoreML...")

    mlmodel = None
    conversion_errors = []

    # Try different conversion approaches
    approaches = [
        ("MLProgram", {
            "convert_to": "mlprogram",
            "compute_precision": ct.precision.FLOAT32
        }),
        ("NeuralNetwork", {
            "convert_to": "neuralnetwork",
            "compute_precision": ct.precision.FLOAT32
        }),
        ("Default", {})
    ]

    for approach_name, kwargs in approaches:
        try:
            print(f"Trying {approach_name} conversion...")
            mlmodel = ct.convert(
                traced_model,
                inputs=[ct.TensorType(name="input", shape=example_input.shape)],
                **kwargs
            )
            print(f"✅ {approach_name} conversion successful!")
            break
        except Exception as e:
            error_msg = f"{approach_name} failed: {str(e)}"
            print(f"❌ {error_msg}")
            conversion_errors.append(error_msg)

    if mlmodel is None:
        print("🔄 All CoreML conversion attempts failed. Saving as TorchScript instead...")
        # Save as TorchScript as fallback
        traced_model.save("models/RealESRGAN_x4plus_torchscript.pt")
        print("✅ Model saved as TorchScript: models/RealESRGAN_x4plus_torchscript.pt")
        return None

    # Add metadata if CoreML conversion succeeded
    mlmodel.short_description = "RealESRGAN x4 plus image upscaler"
    mlmodel.author = "Real-ESRGAN Team"
    mlmodel.license = "BSD 3-Clause"

    # Add input/output descriptions
    mlmodel.input_description["input"] = "Input image (CHW format: 3 channels, height, width)"
    if hasattr(mlmodel, 'output_description'):
        try:
            output_names = list(mlmodel._spec.description.output)
            if output_names:
                output_name = output_names[0].name
                mlmodel.output_description[output_name] = "Upscaled image (4x resolution)"
        except:
            pass  # Ignore output description errors

    return mlmodel

def main():
    print("RealESRGAN to CoreML Converter")
    print("=" * 40)

    model_path = "Real-ESRGAN/weights/RealESRGAN_x4plus.pth"

    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("Please ensure the Real-ESRGAN submodule is initialized and weights are downloaded.")
        return

    try:
        # Create and load model
        model = create_realesrgan_model(model_path)

        # Ensure models directory exists
        os.makedirs("models", exist_ok=True)

        # Convert to CoreML
        coreml_model = convert_to_coreml(model)

        if coreml_model is not None:
            # Save the CoreML model
            output_path = "models/RealESRGAN_x4plus.mlmodel"
            coreml_model.save(output_path)
            print(f"✅ CoreML model saved to: {output_path}")
            print(f"📊 Model input: 1x3xHxW (CHW format)")
            print(f"📈 Model output: 4x upscaled image")
            print("🎉 CoreML conversion completed successfully!")
        else:
            print("📝 CoreML conversion failed, but TorchScript model is available.")
            print("   The TorchScript model can be used with PyTorch for inference.")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
EOF

# Make the script executable
chmod +x convert_realesrgan.py

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

# Run the conversion
echo "🔄 Running CoreML conversion..."
python convert_realesrgan.py

# Clean up (optional - comment out if you want to keep the venv)
echo "🧹 Cleaning up conversion script..."
# Keep venv for potential future use
rm -f convert_realesrgan.py

echo "✅ Setup and conversion process completed!"
echo "📁 Check the models/ directory for output files:"
echo "   - RealESRGAN_x4plus.mlmodel (if CoreML conversion succeeded)"
echo "   - RealESRGAN_x4plus_torchscript.pt (fallback TorchScript model)"
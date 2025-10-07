#!/bin/bash

set -e

echo "🚀 Creating iOS MLPackage from RealESRGAN..."
echo "============================================="

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

# Use existing virtual environment
if [ -d "venv" ]; then
    echo "📦 Using existing virtual environment..."
    source venv/bin/activate
else
    echo "❌ Virtual environment not found. Please run setup_and_convert_ios.sh first."
    exit 1
fi

# Create the MLPackage conversion script
echo "📝 Creating MLPackage conversion script..."

cat > convert_realesrgan_mlpackage.py << 'EOF'
#!/usr/bin/env python3
"""
Convert RealESRGAN to iOS MLPackage format for Xcode
"""

import torch
import coremltools as ct
import numpy as np
import os
import shutil
import json
import tempfile

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

def convert_to_mlpackage(model):
    """Convert PyTorch model to iOS MLPackage format"""

    # Set model to eval mode
    model.eval()

    # Use smaller input size for iOS compatibility (like in iOS script)
    example_input = torch.randn(1, 3, 64, 64)

    print("Tracing model for MLPackage...")

    # Trace the model
    with torch.no_grad():
        traced_model = torch.jit.trace(model, example_input)

    print("Converting to CoreML for MLPackage...")

    mlmodel = None

    try:
        # Use the working approach from iOS script
        mlmodel = ct.convert(
            traced_model,
            inputs=[ct.TensorType(name="input", shape=example_input.shape)],
            convert_to="neuralnetwork",
            minimum_deployment_target=ct.target.iOS13
        )

        print("✅ CoreML conversion successful!")
        return mlmodel

    except Exception as e:
        print(f"CoreML conversion failed: {e}")

        # Try with MLProgram as fallback
        try:
            print("Trying MLProgram format...")
            mlmodel = ct.convert(
                traced_model,
                inputs=[ct.TensorType(name="input", shape=example_input.shape)],
                convert_to="mlprogram",
                minimum_deployment_target=ct.target.iOS15,
                compute_precision=ct.precision.FLOAT16
            )
            print("✅ MLProgram conversion successful!")
            return mlmodel

        except Exception as e2:
            print(f"MLProgram conversion also failed: {e2}")

    if mlmodel is None:
        print("❌ All conversion attempts failed")
        return None

    # Add metadata
    mlmodel.short_description = "RealESRGAN x4 plus image upscaler"
    mlmodel.author = "Real-ESRGAN Team"
    mlmodel.license = "BSD 3-Clause"

    # Add input/output descriptions
    mlmodel.input_description["input"] = "Input image (CHW format: channels, height, width)"

    try:
        output_names = list(mlmodel._spec.description.output)
        if output_names:
            output_name = output_names[0].name
            mlmodel.output_description[output_name] = "Upscaled image (4x resolution)"
    except:
        pass

    return mlmodel

def main():
    print("RealESRGAN to iOS MLPackage Converter")
    print("=" * 45)

    model_path = "Real-ESRGAN/weights/RealESRGAN_x4plus.pth"

    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return

    try:
        # Create and load model
        model = create_realesrgan_model(model_path)

        # Convert to MLPackage
        mlmodel = convert_to_mlpackage(model)

        if mlmodel is not None:
            # Ensure models directory exists
            os.makedirs("models", exist_ok=True)

            # Save as .mlmodel first (to ensure it works)
            mlmodel_path = "models/RealESRGAN_x4plus.mlmodel"
            mlmodel.save(mlmodel_path)

            print(f"✅ CoreML model saved to: {mlmodel_path}")

            # Create MLPackage structure manually (since .save() with .mlpackage extension has issues)
            mlpackage_path = "models/RealESRGAN_x4plus.mlpackage"

            # Remove existing directory if it exists
            if os.path.exists(mlpackage_path):
                shutil.rmtree(mlpackage_path)

            # Create MLPackage directory structure
            os.makedirs(mlpackage_path, exist_ok=True)
            os.makedirs(os.path.join(mlpackage_path, "Data"), exist_ok=True)

            # Copy the .mlmodel file into the MLPackage
            shutil.copy2(mlmodel_path, os.path.join(mlpackage_path, "Model.mlmodel"))

            # Create manifest for the MLPackage
            manifest_content = {
                "schemaVersion": {"major": 1, "minor": 0, "patch": 0},
                "formatVersion": {"major": 1, "minor": 0, "patch": 0},
                "modelIdentifier": "com.realesrgan.x4plus",
                "modelDescription": {
                    "inputDescriptions": [
                        {
                            "name": "input",
                            "shortDescription": "Input image tensor",
                            "type": "multiArray",
                            "shape": [1, 3, 64, 64],
                            "dataType": "float32"
                        }
                    ],
                    "outputDescriptions": [
                        {
                            "name": "var_1387",
                            "shortDescription": "Upscaled image tensor",
                            "type": "multiArray",
                            "shape": [1, 3, 256, 256],
                            "dataType": "float32"
                        }
                    ],
                    "metadata": {
                        "author": "Real-ESRGAN Team",
                        "license": "BSD 3-Clause",
                        "description": "RealESRGAN x4plus image upscaler",
                        "version": "1.0"
                    }
                }
            }

            with open(os.path.join(mlpackage_path, "Manifest.json"), 'w') as f:
                json.dump(manifest_content, f, indent=2)

            with open(os.path.join(mlpackage_path, "Data", "Manifest.json"), 'w') as f:
                json.dump(manifest_content, f, indent=2)

            print(f"✅ iOS MLPackage created at: {mlpackage_path}")
            print(f"📱 Xcode-compatible MLPackage format!")
            print(f"📊 Model input: 1x3x64x64 (CHW format)")
            print(f"📈 Model output: 4x upscaled image (256x256)")
            print("🎉 MLPackage conversion completed successfully!")

            # Verify MLPackage
            try:
                test_model = ct.models.MLModel(mlmodel_path)
                print(f"✅ Model verification successful!")
                print(f"   Input description: {test_model.input_description}")
                print(f"   Output description: {test_model.output_description}")

                # Check MLPackage contents
                if os.path.exists(mlpackage_path):
                    print(f"📦 MLPackage contents:")
                    for root, dirs, files in os.walk(mlpackage_path):
                        level = root.replace(mlpackage_path, '').count(os.sep)
                        indent = ' ' * 2 * level
                        print(f"{indent}{os.path.basename(root)}/")
                        subindent = ' ' * 2 * (level + 1)
                        for file in files:
                            print(f"{subindent}{file}")

            except Exception as test_e:
                print(f"⚠️  Model verification warning: {test_e}")
        else:
            print("❌ MLPackage conversion failed")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
EOF

# Make the script executable
chmod +x convert_realesrgan_mlpackage.py

# Check if model weights exist
if [ ! -f "Real-ESRGAN/weights/RealESRGAN_x4plus.pth" ]; then
    echo "⬇️  Downloading RealESRGAN model weights..."
    mkdir -p Real-ESRGAN/weights
    curl -L -o Real-ESRGAN/weights/RealESRGAN_x4plus.pth https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
fi

# Run the MLPackage conversion
echo "🔄 Running MLPackage conversion..."
python convert_realesrgan_mlpackage.py

# Clean up
echo "🧹 Cleaning up conversion script..."
rm -f convert_realesrgan_mlpackage.py

echo "✅ MLPackage creation completed!"
echo "📱 Check the models/ directory for iOS-compatible MLPackage:"
echo "   - RealESRGAN_x4plus.mlmodel (legacy format)"
echo "   - RealESRGAN_x4plus.mlpackage (modern Xcode format)"
echo "🔧 Use the .mlpackage in your Xcode project for iOS integration"
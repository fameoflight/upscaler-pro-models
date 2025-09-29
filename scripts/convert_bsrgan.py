#!/usr/bin/env python3
"""
BSRGAN (Blind Super-Resolution GAN) to CoreML converter for iOS
Converts BSRGAN PyTorch models to iOS-compatible CoreML format
"""

import torch
import coremltools as ct
import numpy as np
import os
import sys
import math
import functools
from mlpackage_utils import save_as_mlpackage, verify_mlpackage, get_mlpackage_info

class ResidualDenseBlock_5C(torch.nn.Module):
    """Residual Dense Block with 5 convolutions"""
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = torch.nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = torch.nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = torch.nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = torch.nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = torch.nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        # mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class RRDBBlock(torch.nn.Module):
    """Residual in Residual Dense Block"""
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

class BSRGANGenerator(torch.nn.Module):
    """BSRGAN Generator Network"""
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=4):
        super(BSRGANGenerator, self).__init__()
        RRDB_block_f = functools.partial(RRDBBlock, nf=nf, gc=gc)
        self.sf = sf

        self.conv_first = torch.nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = self.make_layer(RRDB_block_f, nb)
        self.trunk_conv = torch.nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling
        self.upconv1 = torch.nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        if sf==4:
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
        if self.sf==4:
            fea = self.lrelu(self.upconv2(torch.nn.functional.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        return out

class LightweightBSRGAN(torch.nn.Module):
    """Lightweight version of BSRGAN for mobile deployment"""
    def __init__(self, in_nc=3, out_nc=3, nf=32, nb=12, gc=16, sf=4):
        super(LightweightBSRGAN, self).__init__()
        self.sf = sf

        # First convolution
        self.conv_first = torch.nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)

        # Simplified residual blocks
        self.residual_blocks = torch.nn.ModuleList([
            self._make_residual_block(nf, gc) for _ in range(nb)
        ])

        self.trunk_conv = torch.nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # Upsampling layers
        self.upconv1 = torch.nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        if sf == 4:
            self.upconv2 = torch.nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.HRconv = torch.nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = torch.nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def _make_residual_block(self, nf, gc):
        """Simple residual block without dense connections for efficiency"""
        return torch.nn.Sequential(
            torch.nn.Conv2d(nf, gc, 3, 1, 1, bias=True),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
            torch.nn.Conv2d(gc, nf, 3, 1, 1, bias=True)
        )

    def forward(self, x):
        # First convolution
        fea = self.conv_first(x)

        # Residual blocks
        trunk = fea
        for block in self.residual_blocks:
            trunk = trunk + block(trunk) * 0.2

        trunk = self.trunk_conv(trunk)
        fea = fea + trunk

        # Upsampling
        fea = self.lrelu(self.upconv1(torch.nn.functional.interpolate(fea, scale_factor=2, mode='nearest')))
        if self.sf == 4:
            fea = self.lrelu(self.upconv2(torch.nn.functional.interpolate(fea, scale_factor=2, mode='nearest')))

        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        return out

def convert_bsrgan_to_coreml(model_path, model_name, scale_factor, input_size=64):
    """Convert BSRGAN model to CoreML format."""
    print(f"🔄 Converting {model_name} (scale: {scale_factor}x) to CoreML...")

    try:
        if os.path.exists(model_path):
            print(f"📥 Loading pre-trained model: {model_path}")
            # Load pre-trained model
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'params_ema' in checkpoint:
                    state_dict = checkpoint['params_ema']
                elif 'params' in checkpoint:
                    state_dict = checkpoint['params']
                elif 'model' in checkpoint:
                    state_dict = checkpoint['model']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint

            # Clean up state dict keys if they have prefixes
            cleaned_state_dict = {}
            for k, v in state_dict.items():
                new_key = k
                if k.startswith('generator.'):
                    new_key = k.replace('generator.', '')
                elif k.startswith('module.'):
                    new_key = k.replace('module.', '')
                cleaned_state_dict[new_key] = v

            # Create full BSRGAN model
            model = BSRGANGenerator(
                in_nc=3,
                out_nc=3,
                nf=64,
                nb=23,
                gc=32,
                sf=scale_factor
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
            print(f"📦 Creating lightweight BSRGAN model (scale: {scale_factor}x)")
            # Create a smaller, lightweight model for demonstration
            model = LightweightBSRGAN(
                in_nc=3,
                out_nc=3,
                nf=24,          # Reduced features
                nb=8,           # Reduced blocks
                gc=12,          # Reduced growth channels
                sf=scale_factor
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
        mlmodel.short_description = f"BSRGAN {scale_factor}x Blind Super-Resolution"
        mlmodel.author = "BSRGAN Team - Converted for iOS"
        mlmodel.license = "Apache 2.0"
        mlmodel.version = "1.0.0"

        # Set input/output descriptions
        mlmodel.input_description["input_image"] = f"Input image to upscale by {scale_factor}x (blind SR)"
        mlmodel.output_description["output_image"] = f"Upscaled image ({scale_factor}x resolution) with blind super-resolution"

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
        print("Usage: python convert_bsrgan.py <model_name> <scale_factor> [model_path]")
        print("Example: python convert_bsrgan.py BSRGAN_4x 4 weights/bsrgan_4x.pth")
        sys.exit(1)

    model_name = sys.argv[1]
    scale_factor = int(sys.argv[2])
    model_path = sys.argv[3] if len(sys.argv) > 3 else None

    if model_path and not os.path.exists(model_path):
        print(f"⚠️  Model file not found: {model_path}")
        model_path = None
        print("Creating lightweight model instead...")

    success = convert_bsrgan_to_coreml(model_path, model_name, scale_factor)
    sys.exit(0 if success else 1)
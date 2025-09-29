#!/usr/bin/env python3
"""
EDSR (Enhanced Deep Super-Resolution) to CoreML converter for iOS
Converts EDSR PyTorch models to iOS-compatible CoreML format
"""

import torch
import coremltools as ct
import numpy as np
import os
import sys
import math
from mlpackage_utils import save_as_mlpackage, verify_mlpackage, get_mlpackage_info

class MeanShift(torch.nn.Conv2d):
    """RGB mean subtraction/addition layer"""
    def __init__(self, rgb_range, rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class ResBlock(torch.nn.Module):
    """Residual Block for EDSR"""
    def __init__(self, conv, n_feats, kernel_size, bias=True, bn=False, act=torch.nn.ReLU(True), res_scale=1):
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(torch.nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = torch.nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res

class Upsampler(torch.nn.Sequential):
    """Upsampling block for EDSR"""
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):
        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(torch.nn.PixelShuffle(2))
                if bn:
                    m.append(torch.nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(torch.nn.ReLU(True))
                elif act == 'prelu':
                    m.append(torch.nn.PReLU(n_feats))
        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(torch.nn.PixelShuffle(3))
            if bn:
                m.append(torch.nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(torch.nn.ReLU(True))
            elif act == 'prelu':
                m.append(torch.nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

class EDSR(torch.nn.Module):
    """Enhanced Deep Super-Resolution Network"""
    def __init__(self, args=None, scale=4, n_resblocks=32, n_feats=256, res_scale=0.1, rgb_range=255):
        super(EDSR, self).__init__()

        self.scale = scale
        self.sub_mean = MeanShift(rgb_range)
        self.add_mean = MeanShift(rgb_range, sign=1)

        # Define conv function
        def conv(in_channels, out_channels, kernel_size, bias=True):
            return torch.nn.Conv2d(in_channels, out_channels, kernel_size,
                                   padding=(kernel_size//2), bias=bias)

        # Define head module
        m_head = [conv(3, n_feats, 3)]

        # Define body module
        m_body = [
            ResBlock(conv, n_feats, 3, res_scale=res_scale)
            for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, 3))

        # Define tail module
        m_tail = [
            Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, 3, 3)
        ]

        self.head = torch.nn.Sequential(*m_head)
        self.body = torch.nn.Sequential(*m_body)
        self.tail = torch.nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x

class LightweightEDSR(torch.nn.Module):
    """Lightweight version of EDSR for mobile deployment"""
    def __init__(self, scale=4, n_resblocks=16, n_feats=64, res_scale=0.1):
        super(LightweightEDSR, self).__init__()

        self.scale = scale

        # Define conv function
        def conv(in_channels, out_channels, kernel_size, bias=True):
            return torch.nn.Conv2d(in_channels, out_channels, kernel_size,
                                   padding=(kernel_size//2), bias=bias)

        # Head
        self.head = conv(3, n_feats, 3)

        # Body - Lightweight residual blocks
        self.body = torch.nn.ModuleList([
            ResBlock(conv, n_feats, 3, res_scale=res_scale)
            for _ in range(n_resblocks)
        ])
        self.body_conv = conv(n_feats, n_feats, 3)

        # Tail - Upsampling
        self.upsampler = Upsampler(conv, scale, n_feats, act=False)
        self.final_conv = conv(n_feats, 3, 3)

    def forward(self, x):
        # Head
        x = self.head(x)

        # Body
        res = x
        for block in self.body:
            x = block(x)
        x = self.body_conv(x)
        x = x + res

        # Tail
        x = self.upsampler(x)
        x = self.final_conv(x)

        return x

def convert_edsr_to_coreml(model_path, model_name, scale_factor, input_size=64):
    """Convert EDSR model to CoreML format."""
    print(f"🔄 Converting {model_name} (scale: {scale_factor}x) to CoreML...")

    try:
        if model_path and os.path.exists(model_path):
            print(f"📥 Loading pre-trained model: {model_path}")
            # Load pre-trained model
            state_dict = torch.load(model_path, map_location='cpu', weights_only=False)

            # Handle different state dict formats
            if 'model' in state_dict:
                state_dict = state_dict['model']
            elif 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']

            # Create full EDSR model
            model = EDSR(
                scale=scale_factor,
                n_resblocks=32,
                n_feats=256,
                res_scale=0.1
            )

            # Load weights with error handling
            try:
                model.load_state_dict(state_dict, strict=True)
                print("✅ Loaded pre-trained weights")
            except RuntimeError as e:
                print(f"⚠️  Strict loading failed: {e}")
                print("Trying to load with strict=False...")
                model.load_state_dict(state_dict, strict=False)
                print("✅ Loaded weights with some mismatches")

        else:
            print(f"📦 Creating lightweight EDSR model (scale: {scale_factor}x)")
            # Create a smaller, lightweight model for demonstration
            model = LightweightEDSR(
                scale=scale_factor,
                n_resblocks=8,     # Reduced blocks
                n_feats=32,        # Reduced features
                res_scale=0.2
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
        mlmodel.short_description = f"EDSR {scale_factor}x Super-Resolution"
        mlmodel.author = "EDSR Team - Converted for iOS"
        mlmodel.license = "BSD"
        mlmodel.version = "1.0.0"

        # Set input/output descriptions
        mlmodel.input_description["input_image"] = f"Input image to upscale by {scale_factor}x"
        mlmodel.output_description["output_image"] = f"Upscaled image ({scale_factor}x resolution)"

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
        print("Usage: python convert_edsr.py <model_name> <scale_factor> [model_path]")
        print("Example: python convert_edsr.py EDSR_4x 4 weights/edsr_4x.pth")
        sys.exit(1)

    model_name = sys.argv[1]
    scale_factor = int(sys.argv[2])
    model_path = sys.argv[3] if len(sys.argv) > 3 else None

    if model_path and not os.path.exists(model_path):
        print(f"⚠️  Model file not found: {model_path}")
        model_path = None
        print("Creating lightweight model instead...")

    success = convert_edsr_to_coreml(model_path, model_name, scale_factor)
    sys.exit(0 if success else 1)
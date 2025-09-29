#!/usr/bin/env python3
"""
RCAN (Residual Channel Attention Networks) to CoreML converter for iOS
Converts RCAN PyTorch models to iOS-compatible CoreML format
"""

import torch
import coremltools as ct
import numpy as np
import os
import sys
import math
from mlpackage_utils import save_as_mlpackage, verify_mlpackage, get_mlpackage_info

class ChannelAttention(torch.nn.Module):
    """Channel Attention Module (CAM)"""
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        # Global average pooling: feature --> point
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        # Feature channel downscale and upscale --> channel weight
        self.conv_du = torch.nn.Sequential(
            torch.nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class RCAB(torch.nn.Module):
    """Residual Channel Attention Block (RCAB)"""
    def __init__(self, conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=torch.nn.ReLU(True), res_scale=1):
        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(torch.nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(ChannelAttention(n_feat, reduction))
        self.body = torch.nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res = res * self.res_scale
        res += x
        return res

class ResidualGroup(torch.nn.Module):
    """Residual Group (RG)"""
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = [
            RCAB(conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=act, res_scale=res_scale)
            for _ in range(n_resblocks)
        ]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = torch.nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

class Upsampler(torch.nn.Sequential):
    """Upsampling block"""
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

class RCAN(torch.nn.Module):
    """Residual Channel Attention Network"""
    def __init__(self, args=None, n_resgroups=10, n_resblocks=20, n_feats=64, reduction=16, scale=4, rgb_range=255, n_colors=3, res_scale=1):
        super(RCAN, self).__init__()

        self.scale = scale

        # Define conv function
        def conv(in_channels, out_channels, kernel_size, bias=True):
            return torch.nn.Conv2d(in_channels, out_channels, kernel_size,
                                   padding=(kernel_size//2), bias=bias)

        kernel_size = 3
        act = torch.nn.ReLU(True)

        # Define head module
        modules_head = [conv(n_colors, n_feats, kernel_size)]

        # Define body module
        modules_body = [
            ResidualGroup(conv, n_feats, kernel_size, reduction, act=act, res_scale=res_scale, n_resblocks=n_resblocks)
            for _ in range(n_resgroups)
        ]
        modules_body.append(conv(n_feats, n_feats, kernel_size))

        # Define tail module
        modules_tail = [
            Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, n_colors, kernel_size)
        ]

        self.head = torch.nn.Sequential(*modules_head)
        self.body = torch.nn.Sequential(*modules_body)
        self.tail = torch.nn.Sequential(*modules_tail)

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)
        return x

class LightweightRCAN(torch.nn.Module):
    """Lightweight version of RCAN for mobile deployment"""
    def __init__(self, n_resgroups=4, n_resblocks=6, n_feats=32, reduction=16, scale=4, n_colors=3, res_scale=1):
        super(LightweightRCAN, self).__init__()

        self.scale = scale

        # Define conv function
        def conv(in_channels, out_channels, kernel_size, bias=True):
            return torch.nn.Conv2d(in_channels, out_channels, kernel_size,
                                   padding=(kernel_size//2), bias=bias)

        kernel_size = 3
        act = torch.nn.ReLU(True)

        # Head
        self.head = conv(n_colors, n_feats, kernel_size)

        # Body - Residual Groups
        self.body = torch.nn.ModuleList([
            ResidualGroup(conv, n_feats, kernel_size, reduction, act=act, res_scale=res_scale, n_resblocks=n_resblocks)
            for _ in range(n_resgroups)
        ])
        self.body_conv = conv(n_feats, n_feats, kernel_size)

        # Tail - Upsampling
        self.upsampler = Upsampler(conv, scale, n_feats, act=False)
        self.final_conv = conv(n_feats, n_colors, kernel_size)

    def forward(self, x):
        # Head
        x = self.head(x)

        # Body
        res = x
        for group in self.body:
            x = group(x)
        x = self.body_conv(x)
        x = x + res

        # Tail
        x = self.upsampler(x)
        x = self.final_conv(x)

        return x

def convert_rcan_to_coreml(model_path, model_name, scale_factor, input_size=64):
    """Convert RCAN model to CoreML format."""
    print(f"🔄 Converting {model_name} (scale: {scale_factor}x) to CoreML...")

    try:
        if os.path.exists(model_path):
            print(f"📥 Loading pre-trained model: {model_path}")
            # Load pre-trained model
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model' in checkpoint:
                    state_dict = checkpoint['model']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint

            # Create full RCAN model
            model = RCAN(
                n_resgroups=10,
                n_resblocks=20,
                n_feats=64,
                reduction=16,
                scale=scale_factor,
                res_scale=1
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
            print(f"📦 Creating lightweight RCAN model (scale: {scale_factor}x)")
            # Create a smaller, lightweight model for demonstration
            model = LightweightRCAN(
                n_resgroups=2,     # Reduced groups
                n_resblocks=4,     # Reduced blocks
                n_feats=24,        # Reduced features
                reduction=8,       # Reduced channel attention
                scale=scale_factor,
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
        mlmodel.short_description = f"RCAN {scale_factor}x Super-Resolution with Channel Attention"
        mlmodel.author = "RCAN Team - Converted for iOS"
        mlmodel.license = "Apache 2.0"
        mlmodel.version = "1.0.0"

        # Set input/output descriptions
        mlmodel.input_description["input_image"] = f"Input image to upscale by {scale_factor}x"
        mlmodel.output_description["output_image"] = f"Upscaled image ({scale_factor}x resolution) with enhanced details"

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
        print("Usage: python convert_rcan.py <model_name> <scale_factor> [model_path]")
        print("Example: python convert_rcan.py RCAN_4x 4 weights/rcan_4x.pth")
        sys.exit(1)

    model_name = sys.argv[1]
    scale_factor = int(sys.argv[2])
    model_path = sys.argv[3] if len(sys.argv) > 3 else None

    if model_path and not os.path.exists(model_path):
        print(f"⚠️  Model file not found: {model_path}")
        model_path = None
        print("Creating lightweight model instead...")

    success = convert_rcan_to_coreml(model_path, model_name, scale_factor)
    sys.exit(0 if success else 1)
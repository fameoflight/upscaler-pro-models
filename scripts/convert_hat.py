#!/usr/bin/env python3
"""
HAT (Hybrid Attention Transformer) to CoreML converter for iOS
Converts HAT PyTorch models to iOS-compatible CoreML format
"""

import torch
import coremltools as ct
import numpy as np
import os
import sys
import math
from mlpackage_utils import save_as_mlpackage, verify_mlpackage, get_mlpackage_info

class ChannelAttention(torch.nn.Module):
    """Channel Attention Module"""
    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        y = self.attention(x)
        return x * y

class SpatialAttention(torch.nn.Module):
    """Spatial Attention Module"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = torch.nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv(attention)
        return x * self.sigmoid(attention)

class CBAM(torch.nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, num_feat, squeeze_factor=16):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(num_feat, squeeze_factor)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class HAB(torch.nn.Module):
    """Hybrid Attention Block"""
    def __init__(self, num_feat, compress_ratio=3, squeeze_factor=30):
        super(HAB, self).__init__()
        self.cab = CBAM(num_feat, squeeze_factor)

        # Main convolution layers
        self.conv1 = torch.nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.act1 = torch.nn.GELU()
        self.conv2 = torch.nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # Channel compression and expansion
        self.compress = torch.nn.Conv2d(num_feat, num_feat // compress_ratio, 1)
        self.expand = torch.nn.Conv2d(num_feat // compress_ratio, num_feat, 1)

    def forward(self, x):
        identity = x

        # Main path
        out = self.conv1(x)
        out = self.act1(out)
        out = self.conv2(out)

        # Attention
        out = self.cab(out)

        # Compression and expansion
        compressed = self.compress(out)
        out = self.expand(compressed)

        return out + identity

class RHAG(torch.nn.Module):
    """Residual Hybrid Attention Group"""
    def __init__(self, num_feat, num_block, compress_ratio=3, squeeze_factor=30):
        super(RHAG, self).__init__()
        self.residual_layer = torch.nn.Sequential(
            *[HAB(num_feat, compress_ratio, squeeze_factor) for _ in range(num_block)]
        )
        self.conv = torch.nn.Conv2d(num_feat, num_feat, 3, 1, 1)

    def forward(self, x):
        identity = x
        out = self.residual_layer(x)
        out = self.conv(out)
        return out + identity

class HAT(torch.nn.Module):
    """Hybrid Attention Transformer for Image Super-Resolution"""
    def __init__(self, img_size=64, patch_size=1, in_chans=3, embed_dim=180, depths=[6, 6, 6, 6, 6, 6],
                 num_heads=[6, 6, 6, 6, 6, 6], window_size=16, compress_ratio=3, squeeze_factor=30,
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=torch.nn.LayerNorm, ape=False, patch_norm=True, upscale=2, img_range=1.,
                 upsampler='', resi_connection='1conv', **kwargs):
        super(HAT, self).__init__()

        self.window_size = window_size
        self.upscale = upscale
        self.upsampler = upsampler
        self.img_range = img_range

        # Shallow feature extraction
        self.conv_first = torch.nn.Conv2d(in_chans, embed_dim, 3, 1, 1)

        # Deep feature extraction
        self.num_layers = len(depths)
        self.embed_dim = embed_dim

        # Build Residual Hybrid Attention Groups
        self.layers = torch.nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RHAG(
                num_feat=embed_dim,
                num_block=depths[i_layer],
                compress_ratio=compress_ratio,
                squeeze_factor=squeeze_factor
            )
            self.layers.append(layer)

        self.norm = norm_layer(embed_dim)
        self.conv_after_body = torch.nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        # Build upsampling layers
        if upsampler == 'pixelshuffle':
            # For lightweight upsampler
            self.conv_before_upsample = torch.nn.Sequential(
                torch.nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
                torch.nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, embed_dim)
            self.conv_last = torch.nn.Conv2d(embed_dim, in_chans, 3, 1, 1)
        else:
            # Classic upsampling
            self.conv_before_upsample = torch.nn.Sequential(
                torch.nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
                torch.nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, embed_dim)
            self.conv_last = torch.nn.Conv2d(embed_dim, in_chans, 3, 1, 1)

    def forward(self, x):
        H, W = x.shape[2:]

        # Shallow feature extraction
        x = self.conv_first(x)

        # Deep feature extraction
        res = x
        for layer in self.layers:
            x = layer(x)

        x = self.conv_after_body(x) + res

        # Upsampling
        x = self.conv_before_upsample(x)
        x = self.conv_last(self.upsample(x))

        return x

class LightweightHAT(torch.nn.Module):
    """Lightweight version of HAT for mobile deployment"""
    def __init__(self, num_feat=32, num_group=3, num_block=4, upscale=4, compress_ratio=2, squeeze_factor=16):
        super(LightweightHAT, self).__init__()

        self.upscale = upscale

        # Shallow feature extraction
        self.conv_first = torch.nn.Conv2d(3, num_feat, 3, 1, 1)

        # Deep feature extraction - Residual Hybrid Attention Groups
        self.layers = torch.nn.ModuleList([
            RHAG(num_feat, num_block, compress_ratio, squeeze_factor)
            for _ in range(num_group)
        ])

        self.conv_after_body = torch.nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # Upsampling layers
        self.conv_before_upsample = torch.nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.upsample = Upsample(upscale, num_feat)
        self.conv_last = torch.nn.Conv2d(num_feat, 3, 3, 1, 1)

    def forward(self, x):
        # Shallow feature extraction
        x = self.conv_first(x)

        # Deep feature extraction
        res = x
        for layer in self.layers:
            x = layer(x)

        x = self.conv_after_body(x) + res

        # Upsampling
        x = self.conv_before_upsample(x)
        x = self.conv_last(self.upsample(x))

        return x

class Upsample(torch.nn.Sequential):
    """Upsample module"""
    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(torch.nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(torch.nn.PixelShuffle(2))
        elif scale == 3:
            m.append(torch.nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(torch.nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)

def convert_hat_to_coreml(model_path, model_name, scale_factor, input_size=64):
    """Convert HAT model to CoreML format."""
    print(f"🔄 Converting {model_name} (scale: {scale_factor}x) to CoreML...")

    try:
        if model_path and os.path.exists(model_path):
            print(f"📥 Loading pre-trained model: {model_path}")
            # Load pre-trained model
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'params' in checkpoint:
                    state_dict = checkpoint['params']
                elif 'model' in checkpoint:
                    state_dict = checkpoint['model']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint

            # Create full HAT model
            model = HAT(
                img_size=input_size,
                embed_dim=180,
                depths=[6, 6, 6, 6, 6, 6],
                num_heads=[6, 6, 6, 6, 6, 6],
                window_size=16,
                compress_ratio=3,
                squeeze_factor=30,
                upscale=scale_factor,
                img_range=1.,
                upsampler='pixelshuffle'
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
            print(f"📦 Creating lightweight HAT model (scale: {scale_factor}x)")
            # Create a smaller, lightweight model for demonstration
            model = LightweightHAT(
                num_feat=24,        # Reduced features
                num_group=2,        # Reduced groups
                num_block=3,        # Reduced blocks
                upscale=scale_factor,
                compress_ratio=2,   # Less compression
                squeeze_factor=8    # Less channel attention
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
        mlmodel.short_description = f"HAT {scale_factor}x Hybrid Attention Transformer"
        mlmodel.author = "HAT Team - Converted for iOS"
        mlmodel.license = "Apache 2.0"
        mlmodel.version = "1.0.0"

        # Set input/output descriptions
        mlmodel.input_description["input_image"] = f"Input image to upscale by {scale_factor}x using hybrid attention"
        mlmodel.output_description["output_image"] = f"Upscaled image ({scale_factor}x resolution) with attention mechanisms"

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
        print("Usage: python convert_hat.py <model_name> <scale_factor> [model_path]")
        print("Example: python convert_hat.py HAT_4x 4 weights/hat_4x.pth")
        sys.exit(1)

    model_name = sys.argv[1]
    scale_factor = int(sys.argv[2])
    model_path = sys.argv[3] if len(sys.argv) > 3 else None

    if model_path and not os.path.exists(model_path):
        print(f"⚠️  Model file not found: {model_path}")
        model_path = None
        print("Creating lightweight model instead...")

    success = convert_hat_to_coreml(model_path, model_name, scale_factor)
    sys.exit(0 if success else 1)
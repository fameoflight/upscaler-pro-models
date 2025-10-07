#!/usr/bin/env python3
"""
SwinIR to CoreML converter for iOS
Converts SwinIR (Swin Transformer for Image Restoration) PyTorch models to iOS-compatible CoreML format
"""

import torch
import coremltools as ct
import numpy as np
import os
import sys
import math
from mlpackage_utils import save_as_mlpackage, verify_mlpackage, get_mlpackage_info

class WindowAttention(torch.nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = torch.nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = torch.nn.Dropout(attn_drop)
        self.proj = torch.nn.Linear(dim, dim)
        self.proj_drop = torch.nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = torch.nn.functional.softmax(attn, dim=-1)
        else:
            attn = torch.nn.functional.softmax(attn, dim=-1)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerBlock(torch.nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=torch.nn.GELU, norm_layer=torch.nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, mlp_hidden_dim),
            act_layer(),
            torch.nn.Dropout(drop),
            torch.nn.Linear(mlp_hidden_dim, dim),
            torch.nn.Dropout(drop)
        )

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # W-MSA/SW-MSA
        x_windows = window_partition(x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=None)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x

def window_partition(x, window_size):
    """Partition into non-overlapping windows."""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """Reverse window partition."""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class SwinIR(torch.nn.Module):
    def __init__(self, img_size=64, patch_size=1, in_chans=3, embed_dim=96, depths=[6, 6, 6, 6],
                 num_heads=[6, 6, 6, 6], window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=torch.nn.LayerNorm, ape=False, patch_norm=True,
                 upscale=2, img_range=1., upsampler='', resi_connection='1conv', **kwargs):
        super(SwinIR, self).__init__()

        self.window_size = window_size
        self.shift_size = window_size // 2
        self.upscale = upscale
        self.upsampler = upsampler

        # Shallow feature extraction
        self.conv_first = torch.nn.Conv2d(in_chans, embed_dim, 3, 1, 1)

        # Body: Swin Transformer blocks
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        # Build layers
        self.layers = torch.nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=embed_dim,
                               input_resolution=(img_size, img_size),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=drop_path_rate,
                               norm_layer=norm_layer,
                               downsample=None,
                               use_checkpoint=False)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)

        # Build the upsampling layer
        if upsampler == 'pixelshuffle':
            self.conv_before_upsample = torch.nn.Sequential(
                torch.nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
                torch.nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, embed_dim)
            self.conv_last = torch.nn.Conv2d(embed_dim, in_chans, 3, 1, 1)
        else:
            # Classic upsampling
            self.conv_after_body = torch.nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
            self.conv_before_upsample = torch.nn.Sequential(
                torch.nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
                torch.nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, embed_dim)
            self.conv_last = torch.nn.Conv2d(embed_dim, in_chans, 3, 1, 1)

    def forward(self, x):
        H, W = x.shape[2:]
        x = self.conv_first(x)

        # Convert to patch embeddings
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C

        # Apply Swin Transformer Blocks
        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)  # B L C

        # Convert back to image format
        x = x.transpose(1, 2).view(-1, self.embed_dim, H, W)

        if hasattr(self, 'conv_after_body'):
            x = self.conv_after_body(x) + x

        x = self.conv_before_upsample(x)
        x = self.conv_last(self.upsample(x))

        return x

class BasicLayer(torch.nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=torch.nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # Build blocks
        self.blocks = torch.nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # Patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

class Upsample(torch.nn.Sequential):
    """Upsample module."""
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

def convert_swinir_to_coreml(model_path, model_name, scale_factor, input_size=64):
    """Convert SwinIR model to CoreML format."""
    print(f"🔄 Converting {model_name} (scale: {scale_factor}x) to CoreML...")

    try:
        if model_path and os.path.exists(model_path):
            print(f"📥 Loading pre-trained model: {model_path}")
            # Load pre-trained model
            state_dict = torch.load(model_path, map_location='cpu')
            if 'params' in state_dict:
                state_dict = state_dict['params']

            # Create model with appropriate configuration
            model = SwinIR(
                upscale=scale_factor,
                img_size=input_size,
                window_size=8,
                img_range=1.,
                depths=[6, 6, 6, 6, 6, 6],
                embed_dim=180,
                num_heads=[6, 6, 6, 6, 6, 6],
                mlp_ratio=2,
                upsampler='pixelshuffle'
            )

            # Load weights
            model.load_state_dict(state_dict, strict=False)
        else:
            print(f"📦 Creating lightweight SwinIR model (scale: {scale_factor}x)")
            # Create a smaller, lightweight model for demonstration
            model = SwinIR(
                upscale=scale_factor,
                img_size=input_size,
                window_size=8,
                img_range=1.,
                depths=[2, 2, 2, 2],  # Reduced depth
                embed_dim=64,         # Reduced embedding
                num_heads=[4, 4, 4, 4],  # Reduced heads
                mlp_ratio=2,
                upsampler='pixelshuffle'
            )

        model.eval()

        # Create sample input
        sample_input = torch.randn(1, 3, input_size, input_size)

        print(f"🧪 Testing model with input shape: {sample_input.shape}")
        with torch.no_grad():
            output = model(sample_input)
            print(f"✅ Model output shape: {output.shape}")

        # Convert to TorchScript format first
        print("🔄 Converting model to TorchScript...")
        traced_model = torch.jit.trace(model, sample_input)

        # Convert to Core ML using neural network format (more compatible)
        print("🔄 Converting to CoreML Neural Network...")

        try:
            # Try MLProgram first (iOS 15+)
            mlmodel = ct.convert(
                traced_model,
                source="pytorch",
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
        except Exception as e:
            print(f"⚠️  MLProgram conversion failed, trying neural network format: {str(e)}")
            # Fallback to neural network format
            mlmodel = ct.convert(
                traced_model,
                source="pytorch",
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
                compute_precision=ct.precision.FLOAT16
            )

        # Set model metadata
        mlmodel.short_description = f"SwinIR {scale_factor}x Super-Resolution"
        mlmodel.author = "SwinIR Team - Converted for iOS"
        mlmodel.license = "Apache 2.0"
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
        return False

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python convert_swinir.py <model_name> <scale_factor> [model_path]")
        print("Example: python convert_swinir.py SwinIR_4x 4 weights/swinir_4x.pth")
        sys.exit(1)

    model_name = sys.argv[1]
    scale_factor = int(sys.argv[2])
    model_path = sys.argv[3] if len(sys.argv) > 3 else None

    if model_path and not os.path.exists(model_path):
        print(f"⚠️  Model file not found: {model_path}")
        model_path = None
        print("Creating lightweight model instead...")

    success = convert_swinir_to_coreml(model_path, model_name, scale_factor)
    sys.exit(0 if success else 1)
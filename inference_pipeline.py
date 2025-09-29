#!/usr/bin/env python3
"""
Custom Inference Pipeline for Super-Resolution Models

This module provides a self-contained inference pipeline that replaces RealESRGANer,
giving us full control over the upscaling process without external dependencies.

Features:
- Direct model loading and inference
- Custom tiling for memory management
- GPU/CPU device handling
- Image preprocessing/postprocessing
- Progress tracking
"""

import math
import time
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================
# Model Architectures
# ===========================

class ResidualDenseBlock(nn.Module):
    """Residual Dense Block for RRDBNet."""

    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # Initialization
        for m in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # Empirical scaling factor 0.2
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block for RRDBNet."""

    def __init__(self, num_feat, num_grow_ch=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        # Empirical scaling factor 0.2
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """Networks consisting of Residual in Residual Dense Block (RRDB)."""

    def __init__(self, num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_block=23, num_grow_ch=32):
        super(RRDBNet, self).__init__()
        self.scale = scale
        num_upsample = int(math.log(scale, 2))

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # Upsample
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        feat = self.conv_first(x)
        trunk = self.conv_body(self.body(feat))
        feat = feat + trunk

        # Upsample
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))

        return out


class SRVGGNetCompact(nn.Module):
    """Compact SRVGG Network for lightweight super-resolution."""

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu'):
        super(SRVGGNetCompact, self).__init__()
        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.num_feat = num_feat
        self.num_conv = num_conv
        self.upscale = upscale
        self.act_type = act_type

        body = []
        for i in range(num_conv):
            in_ch = num_in_ch if i == 0 else num_feat
            out_ch = num_feat
            body.append(nn.Conv2d(in_ch, out_ch, 3, 1, 1))
            if act_type == 'relu':
                body.append(nn.ReLU(inplace=True))
            elif act_type == 'prelu':
                body.append(nn.PReLU(num_parameters=out_ch))
            elif act_type == 'leakyrelu':
                body.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))

        self.body = nn.Sequential(*body)

        # Upsample layers
        self.upsampler = nn.Sequential(
            nn.Conv2d(num_feat, num_out_ch * (upscale ** 2), 3, 1, 1),
            nn.PixelShuffle(upscale)
        )

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        out = self.body(x)
        out = self.upsampler(out)
        return out


class SuperResolutionPipeline:
    """Custom inference pipeline for super-resolution models."""

    def __init__(self,
                 model: torch.nn.Module,
                 device: torch.device,
                 scale: int = 4,
                 tile_size: int = 0,
                 tile_pad: int = 10,
                 half_precision: bool = False):
        """
        Initialize the inference pipeline.

        Args:
            model: PyTorch model (RRDBNet, SRVGGNetCompact, etc.)
            device: Target device (cuda/mps/cpu)
            scale: Upscaling factor
            tile_size: Tile size for processing (0 = no tiling)
            tile_pad: Padding around tiles
            half_precision: Use FP16 for faster inference
        """
        self.model = model.to(device)
        self.device = device
        self.scale = scale
        self.tile_size = tile_size
        self.tile_pad = tile_pad
        self.half_precision = half_precision

        # Set model to evaluation mode
        self.model.eval()

        # Convert to half precision if requested
        if half_precision:
            self.model = self.model.half()

    def preprocess_image(self, img: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for model input.

        Args:
            img: Input image as numpy array (H, W, C) in BGR format, [0, 255]

        Returns:
            Preprocessed tensor (1, C, H, W) in RGB format, [0, 1]
        """
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0

        # Convert to tensor and add batch dimension
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)

        # Move to device and convert to half precision if needed
        img_tensor = img_tensor.to(self.device)
        if self.half_precision:
            img_tensor = img_tensor.half()

        return img_tensor

    def postprocess_tensor(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Postprocess model output tensor to image.

        Args:
            tensor: Model output tensor (1, C, H, W) in [0, 1] range

        Returns:
            Image as numpy array (H, W, C) in BGR format, [0, 255]
        """
        # Convert to CPU and remove batch dimension
        img = tensor.squeeze(0).cpu().float()

        # Clamp values to [0, 1]
        img = torch.clamp(img, 0, 1)

        # Convert to numpy and change to HWC format
        img = img.permute(1, 2, 0).numpy()

        # Convert to [0, 255] and uint8
        img = (img * 255.0).round().astype(np.uint8)

        # Convert RGB back to BGR
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        return img

    def inference_single(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """
        Run inference on a single image tensor.

        Args:
            img_tensor: Preprocessed image tensor (1, C, H, W)

        Returns:
            Output tensor (1, C, H*scale, W*scale)
        """
        with torch.no_grad():
            try:
                output = self.model(img_tensor)
                return output
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    raise RuntimeError(
                        f"GPU out of memory! Try using smaller --tile size (e.g., --tile 256 or --tile 128). "
                        f"Current tile size: {self.tile_size if self.tile_size > 0 else 'no tiling'}"
                    )
                else:
                    raise e

    def inference_tiled(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """
        Run inference with tiling for memory efficiency.

        Args:
            img_tensor: Preprocessed image tensor (1, C, H, W)

        Returns:
            Output tensor (1, C, H*scale, W*scale)
        """
        batch, channel, height, width = img_tensor.shape
        output_height = height * self.scale
        output_width = width * self.scale

        # Initialize output tensor
        output_shape = (batch, channel, output_height, output_width)
        output = torch.zeros(output_shape, dtype=img_tensor.dtype, device=self.device)

        tiles_x = math.ceil(width / self.tile_size)
        tiles_y = math.ceil(height / self.tile_size)
        total_tiles = tiles_x * tiles_y

        print(f"   🔳 Processing {total_tiles} tiles ({tiles_x}x{tiles_y})...")

        for y in range(tiles_y):
            for x in range(tiles_x):
                # Calculate tile boundaries
                start_x = x * self.tile_size
                start_y = y * self.tile_size
                end_x = min(start_x + self.tile_size, width)
                end_y = min(start_y + self.tile_size, height)

                # Add padding
                padded_start_x = max(start_x - self.tile_pad, 0)
                padded_start_y = max(start_y - self.tile_pad, 0)
                padded_end_x = min(end_x + self.tile_pad, width)
                padded_end_y = min(end_y + self.tile_pad, height)

                # Extract tile with padding
                tile = img_tensor[:, :, padded_start_y:padded_end_y, padded_start_x:padded_end_x]

                # Run inference on tile
                with torch.no_grad():
                    tile_output = self.model(tile)

                # Calculate output positions
                output_start_x = start_x * self.scale
                output_start_y = start_y * self.scale
                output_end_x = end_x * self.scale
                output_end_y = end_y * self.scale

                # Calculate positions in the tile output (accounting for padding)
                tile_start_x = (start_x - padded_start_x) * self.scale
                tile_start_y = (start_y - padded_start_y) * self.scale
                tile_end_x = tile_start_x + (output_end_x - output_start_x)
                tile_end_y = tile_start_y + (output_end_y - output_start_y)

                # Place tile in output
                output[:, :, output_start_y:output_end_y, output_start_x:output_end_x] = \
                    tile_output[:, :, tile_start_y:tile_end_y, tile_start_x:tile_end_x]

                # Progress update
                current_tile = y * tiles_x + x + 1
                if current_tile % max(1, total_tiles // 10) == 0 or current_tile == total_tiles:
                    progress = (current_tile / total_tiles) * 100
                    print(f"     Progress: {current_tile}/{total_tiles} tiles ({progress:.1f}%)")

        return output

    def enhance(self, img: np.ndarray, outscale: Optional[float] = None) -> Tuple[np.ndarray, str]:
        """
        Enhance (upscale) an image.

        Args:
            img: Input image as numpy array (H, W, C) in BGR format
            outscale: Output scale factor (if different from model scale)

        Returns:
            Tuple of (enhanced_image, mode)
            - enhanced_image: Upscaled image as numpy array
            - mode: Image mode (for compatibility)
        """
        if outscale is None:
            outscale = self.scale

        # Preprocess image
        img_tensor = self.preprocess_image(img)

        # Run inference (tiled or single)
        if self.tile_size > 0:
            output_tensor = self.inference_tiled(img_tensor)
        else:
            output_tensor = self.inference_single(img_tensor)

        # Postprocess output
        output_img = self.postprocess_tensor(output_tensor)

        # Handle different output scales
        if outscale != self.scale:
            h, w = output_img.shape[:2]
            target_h = int(h * outscale / self.scale)
            target_w = int(w * outscale / self.scale)
            output_img = cv2.resize(output_img, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

        return output_img, None  # Mode is None for compatibility

    @classmethod
    def create_model_from_weights(cls,
                                 model_path: Union[str, Path],
                                 architecture: str,
                                 device: torch.device,
                                 scale: int = 4) -> torch.nn.Module:
        """
        Create and load a model from weights file using our built-in architectures.

        Args:
            model_path: Path to the .pth weights file
            architecture: Model architecture ("RRDBNet" or "SRVGGNetCompact")
            device: Target device
            scale: Model scale factor

        Returns:
            Loaded PyTorch model
        """
        # Create model with our built-in architectures
        if architecture == "SRVGGNetCompact":
            model = SRVGGNetCompact(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_conv=32,
                upscale=scale,
                act_type='prelu'
            )
        elif architecture == "RRDBNet":
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=scale
            )
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

        # Load weights
        model_data = torch.load(model_path, map_location='cpu')

        if isinstance(model_data, dict) and ('params_ema' in model_data or 'params' in model_data):
            # Real-ESRGAN format
            weights = model_data.get('params_ema', model_data.get('params'))
        else:
            # Raw weights format
            weights = model_data

        # Load state dict
        model.load_state_dict(weights, strict=True)

        return model.to(device)


def get_optimal_device() -> torch.device:
    """Get the best available device for inference."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def estimate_memory_usage(height: int, width: int, scale: int,
                         channels: int = 3, dtype_size: int = 4) -> float:
    """
    Estimate GPU memory usage for an image.

    Args:
        height, width: Image dimensions
        scale: Upscaling factor
        channels: Number of channels (usually 3)
        dtype_size: Size of data type in bytes (4 for float32, 2 for float16)

    Returns:
        Estimated memory usage in MB
    """
    input_size = height * width * channels * dtype_size
    output_size = (height * scale) * (width * scale) * channels * dtype_size
    # Add some overhead for model weights and intermediate computations
    total_size = (input_size + output_size) * 2
    return total_size / (1024 * 1024)  # Convert to MB
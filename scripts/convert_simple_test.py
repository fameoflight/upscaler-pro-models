#!/usr/bin/env python3
"""
Simple test model to verify the conversion pipeline works
"""

import torch
import coremltools as ct
import numpy as np
import os
import sys

class SimpleUpscaler(torch.nn.Module):
    """Simple test upscaler that just does nearest neighbor interpolation"""
    def __init__(self, scale_factor=2):
        super(SimpleUpscaler, self).__init__()
        self.scale_factor = scale_factor

        # Simple conv layer to verify the pipeline
        self.conv = torch.nn.Conv2d(3, 3, kernel_size=3, padding=1)

    def forward(self, x):
        # Simple convolution
        x = self.conv(x)

        # Use nearest neighbor instead of bicubic for compatibility
        x = torch.nn.functional.interpolate(
            x,
            scale_factor=self.scale_factor,
            mode='nearest'
        )

        return x

def main():
    print("Simple Upscaler Test")
    print("=" * 20)

    # Create simple model
    model = SimpleUpscaler(scale_factor=2)
    model.eval()

    # Test input
    example_input = torch.randn(1, 3, 64, 64)

    print("Testing model...")
    with torch.no_grad():
        output = model(example_input)
        print(f"Input shape: {example_input.shape}")
        print(f"Output shape: {output.shape}")

    # Convert to CoreML
    print("Converting to CoreML...")
    try:
        traced_model = torch.jit.trace(model, example_input)

        mlmodel = ct.convert(
            traced_model,
            inputs=[ct.TensorType(name="input", shape=example_input.shape)],
            convert_to="neuralnetwork",
            minimum_deployment_target=ct.target.iOS13
        )

        # Save model
        os.makedirs("models", exist_ok=True)
        output_path = "models/SimpleTest_2x.mlmodel"
        mlmodel.save(output_path)

        print(f"✅ Test model saved to: {output_path}")
        print("🎉 Conversion pipeline is working!")

        # Verify model
        test_model = ct.models.MLModel(output_path)
        print(f"✅ Model verification successful!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
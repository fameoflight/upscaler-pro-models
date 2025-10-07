#!/usr/bin/env python3
"""
PyTorch vs CoreML Model Comparison
Generates upscaled images using both PyTorch and CoreML models for visual comparison
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
from typing import Tuple, Optional

try:
    import torch
    import torch.nn.functional as F
    import torchvision.transforms as transforms
    from PIL import Image
    import coremltools as ct
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
except ImportError as e:
    print(f"❌ Required package not found: {e}")
    print("Please install: pip install torch torchvision coremltools pillow matplotlib")
    sys.exit(1)

def get_optimal_device() -> torch.device:
    """Get the best available device for inference"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def load_test_image(image_path: str, model_name: str = None) -> np.ndarray:
    """Load and preprocess test image without resizing"""
    try:
        # For ESRGAN and Real-ESRGAN models, use BGR format like the official scripts
        if "ESRGAN" in model_name or "RealESRGAN" in model_name:
            # Use cv2 like the official Real-ESRGAN scripts
            import cv2
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Could not load image with cv2: {image_path}")

            # Convert BGR to RGB for display (models were trained on RGB but expect BGR input)
            img_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            print(f"📸 Loaded test image with cv2: {img_display.shape} (original size, converted BGR->RGB)")

            # Convert to numpy array (C, H, W format for PyTorch) - keep BGR for model input
            img_array = img.transpose(2, 0, 1).astype(np.float32) / 255.0  # Keep BGR order for model
            print(f"📐 Preprocessed image shape (BGR order): {img_array.shape}")

            return img_array, Image.fromarray(img_display)  # Return RGB for display
        else:
            # For other models, use PIL RGB
            img = Image.open(image_path).convert('RGB')
            print(f"📸 Loaded test image: {img.size} (original size)")

            # Convert to numpy array (C, H, W format for PyTorch)
            img_array = np.array(img).transpose(2, 0, 1).astype(np.float32) / 255.0
            print(f"📐 Preprocessed image shape: {img_array.shape}")

            return img_array, img

    except Exception as e:
        print(f"❌ Failed to load image {image_path}: {e}")
        return None, None

def load_pytorch_model(model_name: str, weights_path: Optional[str] = None) -> torch.nn.Module:
    """Load PyTorch model"""
    try:
        # Import conversion functions
        sys.path.append('.')

        if "SRCNN" in model_name:
            from convert_srcnn import create_srcnn_model
            scale = int(model_name.split("_x")[1]) if "_x" in model_name else 2
            model = create_srcnn_model(scale)

        elif "RealESRGAN" in model_name:
            from convert_realesrgan import create_realesrgan_model
            scale = int(model_name.split("_x")[1]) if "_x" in model_name else 4
            model = create_realesrgan_model(weights_path if weights_path else None, scale)

        elif "ESRGAN" in model_name:
            from convert_esrgan import create_esrgan_model
            scale = int(model_name.split("_x")[1]) if "_x" in model_name else 4
            model = create_esrgan_model(weights_path if weights_path else None, scale)

        else:
            # Fallback to simple model
            print(f"⚠️  Unsupported model type: {model_name}, using simple test model")

            class SimpleUpscaler(torch.nn.Module):
                def __init__(self, scale_factor=2):
                    super().__init__()
                    self.scale_factor = scale_factor
                    self.conv = torch.nn.Conv2d(3, 3, 3, padding=1)

                def forward(self, x):
                    x = self.conv(x)
                    x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
                    return x

            scale = int(model_name.split("_x")[1]) if "_x" in model_name else 2
            model = SimpleUpscaler(scale)

        # Load weights if available (skip for ESRGAN/RealESRGAN as they're handled in create function)
        if weights_path and os.path.exists(weights_path) and "ESRGAN" not in model_name and "RealESRGAN" not in model_name:
            try:
                state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
                model.load_state_dict(state_dict, strict=False)
                print(f"✅ Loaded PyTorch weights: {weights_path}")
            except Exception as e:
                print(f"⚠️  Could not load weights, using generated model: {e}")

        model.eval()
        return model

    except Exception as e:
        print(f"❌ Failed to create PyTorch model: {e}")
        return None

def load_coreml_model(model_path: str) -> ct.models.MLModel:
    """Load CoreML model"""
    try:
        model = ct.models.MLModel(model_path)
        print(f"✅ Loaded CoreML model: {model_path}")
        return model
    except Exception as e:
        print(f"❌ Failed to load CoreML model {model_path}: {e}")
        return None

def run_pytorch_inference(model: torch.nn.Module, input_array: np.ndarray) -> np.ndarray:
    """Run PyTorch inference"""
    try:
        # Convert to tensor
        input_tensor = torch.from_numpy(input_array).unsqueeze(0)  # Add batch dimension

        print(f"🧪 PyTorch input shape: {input_tensor.shape}")

        # Move to optimal device for performance
        device = get_optimal_device()
        model = model.to(device)
        input_tensor = input_tensor.to(device)
        print(f"🚀 Using device: {device}")

        # Run inference
        with torch.no_grad():
            start_time = time.time()
            output_tensor = model(input_tensor)
            pytorch_time = time.time() - start_time

        # Convert back to numpy
        output_array = output_tensor.squeeze(0).detach().cpu().numpy()

        # Ensure values are in valid range [0, 1]
        output_array = np.clip(output_array, 0, 1)

        print(f"✅ PyTorch output shape: {output_array.shape}")
        print(f"⏱️  PyTorch inference time: {pytorch_time:.4f}s")

        return output_array, pytorch_time

    except Exception as e:
        print(f"❌ PyTorch inference failed: {e}")
        return None, 0

def run_coreml_inference(model: ct.models.MLModel, input_array: np.ndarray) -> np.ndarray:
    """Run CoreML inference"""
    try:
        # Prepare input - add batch dimension if needed
        if len(input_array.shape) == 3:
            input_array = np.expand_dims(input_array, axis=0)

        input_dict = {"input": input_array}

        print(f"🧪 CoreML input shape: {input_array.shape}")

        # Run inference
        start_time = time.time()
        output_dict = model.predict(input_dict)
        coreml_time = time.time() - start_time

        # Extract output
        if len(output_dict) == 1:
            output_array = list(output_dict.values())[0]
        else:
            # Try common output names
            for name in ["output", "result", "upscaled", "prediction"]:
                if name in output_dict:
                    output_array = output_dict[name]
                    break
            else:
                output_array = list(output_dict.values())[0]

        # Remove batch dimension if present
        if len(output_array.shape) == 4 and output_array.shape[0] == 1:
            output_array = output_array.squeeze(0)

        # Ensure values are in valid range [0, 1]
        output_array = np.clip(output_array, 0, 1)

        print(f"✅ CoreML output shape: {output_array.shape}")
        print(f"⏱️  CoreML inference time: {coreml_time:.4f}s")

        return output_array, coreml_time

    except Exception as e:
        print(f"❌ CoreML inference failed: {e}")
        return None, 0

def save_output_image(output_array: np.ndarray, output_path: str, model_name: str = None) -> bool:
    """Save output array as image"""
    try:
        # Convert from (C, H, W) to (H, W, C)
        if len(output_array.shape) == 3 and output_array.shape[0] == 3:
            output_array = output_array.transpose(1, 2, 0)

        # Convert to 0-255 range
        output_array = (output_array * 255).astype(np.uint8)

        # Handle color space conversion for ESRGAN models
        if "ESRGAN" in model_name or "RealESRGAN" in model_name:
            # Model output is in RGB format (since models were trained on RGB)
            # But the array might be in BGR order from the model input processing
            # Convert BGR to RGB if needed
            import cv2
            output_array = cv2.cvtColor(output_array, cv2.COLOR_BGR2RGB)
            print(f"🎨 Converted output BGR->RGB for {model_name}")

        # Save image
        img = Image.fromarray(output_array)
        img.save(output_path)
        print(f"💾 Saved image: {output_path}")
        return True

    except Exception as e:
        print(f"❌ Failed to save image {output_path}: {e}")
        return False

def create_comparison_plot(original_img: Image.Image, pytorch_output: np.ndarray, coreml_output: np.ndarray,
                          model_name: str, comparison_results: dict, output_dir: str) -> str:
    """Create a matplotlib comparison plot"""
    try:
        # Convert outputs to displayable format
        pytorch_display = pytorch_output.transpose(1, 2, 0) if len(pytorch_output.shape) == 3 else pytorch_output
        coreml_display = coreml_output.transpose(1, 2, 0) if len(coreml_output.shape) == 3 else coreml_output

        # Handle color space conversion for ESRGAN models in display
        if "ESRGAN" in model_name or "RealESRGAN" in model_name:
            # Convert BGR to RGB for display if needed
            import cv2
            if pytorch_display.shape[2] == 3:  # Ensure it's 3-channel
                pytorch_display = cv2.cvtColor(pytorch_display, cv2.COLOR_BGR2RGB)
            if coreml_display.shape[2] == 3:  # Ensure it's 3-channel
                coreml_display = cv2.cvtColor(coreml_display, cv2.COLOR_BGR2RGB)
            print(f"🎨 Converted display outputs BGR->RGB for {model_name}")

        # Calculate difference
        diff = np.abs(pytorch_display - coreml_display)

        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'PyTorch vs CoreML Comparison - {model_name}', fontsize=16, fontweight='bold')

        # Top row: Original and upscaled images
        axes[0, 0].imshow(original_img)
        axes[0, 0].set_title('Original Input', fontweight='bold')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(np.clip(pytorch_display, 0, 1))
        axes[0, 1].set_title('PyTorch Output', fontweight='bold', color='green')
        axes[0, 1].axis('off')

        axes[0, 2].imshow(np.clip(coreml_display, 0, 1))
        axes[0, 2].set_title('CoreML Output', fontweight='bold', color='blue')
        axes[0, 2].axis('off')

        # Bottom row: Difference maps
        # Overall difference
        diff_overall = np.mean(diff, axis=2)  # Average across channels
        im1 = axes[1, 0].imshow(diff_overall, cmap='hot', vmin=0, vmax=0.1)
        axes[1, 0].set_title('Overall Difference\n(Averaged)', fontweight='bold')
        axes[1, 0].axis('off')
        plt.colorbar(im1, ax=axes[1, 0], fraction=0.046, pad=0.04)

        # Red channel difference
        im2 = axes[1, 1].imshow(diff[:, :, 0], cmap='Reds', vmin=0, vmax=0.1)
        axes[1, 1].set_title('Red Channel Difference', fontweight='bold')
        axes[1, 1].axis('off')
        plt.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)

        # Green channel difference
        im3 = axes[1, 2].imshow(diff[:, :, 1], cmap='Greens', vmin=0, vmax=0.1)
        axes[1, 2].set_title('Green Channel Difference', fontweight='bold')
        axes[1, 2].axis('off')
        plt.colorbar(im3, ax=axes[1, 2], fraction=0.046, pad=0.04)

        # Add comparison statistics as text
        stats_text = f"""Max Difference: {comparison_results.get('max_diff', 0):.6f}
Mean Difference: {comparison_results.get('mean_diff', 0):.6f}
PSNR: {comparison_results.get('psnr', 0):.2f} dB
SSIM: {comparison_results.get('ssim', 0):.4f}"""

        fig.text(0.02, 0.5, stats_text, transform=fig.transFigure,
                fontsize=10, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))

        # Adjust layout
        plt.tight_layout()

        # Save plot
        plot_path = f"{output_dir}/comparison_{model_name}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"📊 Saved comparison plot: {plot_path}")
        return plot_path

    except Exception as e:
        print(f"❌ Failed to create comparison plot: {e}")
        return None

def create_difference_histogram(pytorch_output: np.ndarray, coreml_output: np.ndarray,
                               model_name: str, output_dir: str) -> str:
    """Create histogram of pixel differences"""
    try:
        # Calculate difference
        diff = np.abs(pytorch_output - coreml_output).flatten()

        # Create histogram
        plt.figure(figsize=(10, 6))

        # Plot histogram
        plt.hist(diff, bins=100, alpha=0.7, color='skyblue', edgecolor='black')

        # Add statistics
        mean_diff = np.mean(diff)
        max_diff = np.max(diff)
        std_diff = np.std(diff)

        plt.axvline(mean_diff, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_diff:.6f}')
        plt.axvline(mean_diff + std_diff, color='orange', linestyle='--', alpha=0.7, label=f'Mean+STD: {mean_diff + std_diff:.6f}')

        plt.title(f'Pixel Difference Distribution - {model_name}', fontsize=14, fontweight='bold')
        plt.xlabel('Absolute Difference', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Use log scale if many small differences
        if np.sum(diff < 0.001) / len(diff) > 0.9:
            plt.yscale('log')
            plt.title(f'Pixel Difference Distribution (log scale) - {model_name}', fontsize=14, fontweight='bold')

        # Save histogram
        hist_path = f"{output_dir}/difference_histogram_{model_name}.png"
        plt.savefig(hist_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"📊 Saved difference histogram: {hist_path}")
        return hist_path

    except Exception as e:
        print(f"❌ Failed to create difference histogram: {e}")
        return None

def compare_outputs(pytorch_output: np.ndarray, coreml_output: np.ndarray, model_name: str) -> dict:
    """Compare PyTorch and CoreML outputs"""
    try:
        # Ensure same shape
        if pytorch_output.shape != coreml_output.shape:
            print(f"❌ Shape mismatch: PyTorch {pytorch_output.shape} vs CoreML {coreml_output.shape}")
            return {"match": False, "error": "Shape mismatch"}

        # Calculate differences
        abs_diff = np.abs(pytorch_output - coreml_output)
        max_diff = np.max(abs_diff)
        mean_diff = np.mean(abs_diff)

        # Calculate PSNR
        mse = np.mean((pytorch_output - coreml_output) ** 2)
        if mse == 0:
            psnr = float('inf')
        else:
            psnr = 20 * np.log10(1.0 / np.sqrt(mse))

        # Calculate SSIM (simplified)
        def ssim_simple(img1, img2):
            mu1, mu2 = np.mean(img1), np.mean(img2)
            sigma1, sigma2 = np.std(img1), np.std(img2)
            sigma12 = np.mean((img1 - mu1) * (img2 - mu2))

            c1, c2 = 0.01**2, 0.03**2
            ssim = ((2*mu1*mu2 + c1) * (2*sigma12 + c2)) / ((mu1**2 + mu2**2 + c1) * (sigma1**2 + sigma2**2 + c2))
            return ssim

        ssim_value = ssim_simple(pytorch_output.flatten(), coreml_output.flatten())

        results = {
            "match": True,
            "max_diff": float(max_diff),
            "mean_diff": float(mean_diff),
            "psnr": float(psnr),
            "ssim": float(ssim_value),
            "shape": pytorch_output.shape
        }

        print(f"📊 Comparison Results for {model_name}:")
        print(f"   Max difference: {max_diff:.6f}")
        print(f"   Mean difference: {mean_diff:.6f}")
        print(f"   PSNR: {psnr:.2f} dB")
        print(f"   SSIM: {ssim_value:.4f}")

        # Determine quality of match
        if max_diff < 1e-4:
            print(f"   ✅ Excellent match (virtually identical)")
        elif max_diff < 1e-3:
            print(f"   ✅ Good match (minor differences)")
        elif max_diff < 1e-2:
            print(f"   ⚠️  Fair match (noticeable differences)")
        else:
            print(f"   ❌ Poor match (significant differences)")

        return results

    except Exception as e:
        print(f"❌ Error comparing outputs: {e}")
        return {"match": False, "error": str(e)}

def compare_model(model_name: str, test_image_path: str, output_dir: str) -> dict:
    """Compare a single model between PyTorch and CoreML"""

    print(f"\n🔬 Comparing model: {model_name}")
    print("=" * 60)

    # Find model files
    weights_path = None
    for ext in ['.pth', '.pt', '.npz']:
        path = f"weights/{model_name}{ext}"
        if os.path.exists(path):
            weights_path = path
            break

    coreml_path = None
    for ext in ['.mlpackage', '.mlmodel']:
        path = f"models/{model_name}{ext}"
        if os.path.exists(path):
            coreml_path = path
            break

    if not coreml_path:
        print(f"❌ CoreML model not found: {model_name}")
        return {"success": False, "error": "CoreML model not found"}

    # Load test image with appropriate size for the model
    input_array, original_img = load_test_image(test_image_path, model_name)
    if input_array is None:
        return {"success": False, "error": "Failed to load test image"}

    # Load PyTorch model
    print("\n📥 Loading PyTorch model...")
    pytorch_model = load_pytorch_model(model_name, weights_path)
    if pytorch_model is None:
        return {"success": False, "error": "Failed to load PyTorch model"}

    # Load CoreML model
    print("\n📥 Loading CoreML model...")
    coreml_model = load_coreml_model(coreml_path)
    if coreml_model is None:
        return {"success": False, "error": "Failed to load CoreML model"}

    # Run PyTorch inference
    print("\n🧪 Running PyTorch inference...")
    pytorch_output, pytorch_time = run_pytorch_inference(pytorch_model, input_array)
    if pytorch_output is None:
        return {"success": False, "error": "PyTorch inference failed"}

    # Run CoreML inference
    print("\n🧪 Running CoreML inference...")
    coreml_output, coreml_time = run_coreml_inference(coreml_model, input_array)
    if coreml_output is None:
        return {"success": False, "error": "CoreML inference failed"}

    # Save output images
    print(f"\n💾 Saving comparison images...")
    base_name = Path(test_image_path).stem

    pytorch_output_path = f"{output_dir}/{base_name}_{model_name}_pytorch.png"
    coreml_output_path = f"{output_dir}/{base_name}_{model_name}_coreml.png"

    pytorch_saved = save_output_image(pytorch_output, pytorch_output_path, model_name)
    coreml_saved = save_output_image(coreml_output, coreml_output_path, model_name)

    # Compare outputs
    print(f"\n📊 Comparing outputs...")
    comparison_results = compare_outputs(pytorch_output, coreml_output, model_name)

    # Generate visual comparison plots
    print(f"\n📈 Generating visual comparisons...")
    plot_path = create_comparison_plot(original_img, pytorch_output, coreml_output, model_name, comparison_results, output_dir)
    hist_path = create_difference_histogram(pytorch_output, coreml_output, model_name, output_dir)

    # Prepare final results
    results = {
        "model_name": model_name,
        "success": True,
        "pytorch_time": pytorch_time,
        "coreml_time": coreml_time,
        "speedup": pytorch_time / coreml_time if coreml_time > 0 else float('inf'),
        "pytorch_output_saved": pytorch_saved,
        "coreml_output_saved": coreml_saved,
        "pytorch_path": pytorch_output_path if pytorch_saved else None,
        "coreml_path": coreml_output_path if coreml_saved else None,
        "comparison": comparison_results,
        "plot_path": plot_path,
        "histogram_path": hist_path
    }

    print(f"\n📋 Summary for {model_name}:")
    print(f"   PyTorch time: {pytorch_time:.4f}s")
    print(f"   CoreML time: {coreml_time:.4f}s")
    print(f"   Speedup: {results['speedup']:.2f}x")
    print(f"   Images saved: {output_dir}/")
    if plot_path:
        print(f"   📊 Comparison plot: {plot_path}")
    if hist_path:
        print(f"   📊 Difference histogram: {hist_path}")

    return results

def main():
    """Main comparison function"""

    print("🔬 PyTorch vs CoreML Model Comparison")
    print("=" * 50)

    # Change to project directory
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    os.chdir(project_dir)

    # Parse arguments
    if len(sys.argv) < 2:
        print("Usage: python compare_pytorch_coreml.py <test_image_path> [model_name]")
        print("Example: python compare_pytorch_coreml.py test-data/1.jpg SRCNN_x2")
        print("If no model specified, will test all available models")
        sys.exit(1)

    test_image_path = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else None

    # Check test image
    if not os.path.exists(test_image_path):
        print(f"❌ Test image not found: {test_image_path}")
        sys.exit(1)

    # Create output directory
    output_dir = "comparison_outputs"
    Path(output_dir).mkdir(exist_ok=True)
    print(f"📁 Output directory: {output_dir}/")

    # Determine models to test
    if model_name:
        models_to_test = [model_name]
    else:
        # Find available models
        models_dir = Path("models")
        available_models = []
        for model_file in models_dir.glob("*.mlpackage"):
            available_models.append(model_file.stem)
        for model_file in models_dir.glob("*.mlmodel"):
            if model_file.stem not in available_models:
                available_models.append(model_file.stem)

        # Filter to Real-ESRGAN models that handle flexible input sizes
        preferred_models = ["RealESRGAN_4x", "RealESRGAN_anime_4x", "ESRGAN_4x"]
        models_to_test = []
        for model in preferred_models:
            if model in available_models:
                models_to_test.append(model)

        # If no preferred models found, use first few available
        if not models_to_test and available_models:
            models_to_test = available_models[:2]

    if not models_to_test:
        print("❌ No models found for comparison")
        sys.exit(1)

    print(f"🎯 Testing models: {', '.join(models_to_test)}")
    print(f"📸 Test image: {test_image_path}")

    # Run comparisons
    all_results = []
    successful_comparisons = 0

    for model in models_to_test:
        result = compare_model(model, test_image_path, output_dir)
        all_results.append(result)

        if result.get("success", False):
            successful_comparisons += 1

        print("\n" + "="*60)

    # Final summary
    print(f"\n🎯 COMPARISON SUMMARY")
    print("=" * 40)
    print(f"Models tested: {len(models_to_test)}")
    print(f"Successful comparisons: {successful_comparisons}")
    print(f"Failed comparisons: {len(models_to_test) - successful_comparisons}")

    if successful_comparisons > 0:
        print(f"\n📁 Comparison images saved in: {output_dir}/")
        print(f"💡 You can now compare the PyTorch and CoreML outputs visually")

        # Show average speedup
        speedups = [r["speedup"] for r in all_results if r.get("success") and r.get("speedup", float('inf')) != float('inf')]
        if speedups:
            avg_speedup = np.mean(speedups)
            print(f"⚡ Average CoreML speedup: {avg_speedup:.2f}x")

    return successful_comparisons > 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
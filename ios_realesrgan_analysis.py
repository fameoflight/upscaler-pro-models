#!/usr/bin/env python3
"""
Real-ESRGAN iOS Analysis Script
Comprehensive analysis of Real-ESRGAN performance and iOS compatibility
"""

import os
import numpy as np
from PIL import Image, ImageFilter
import glob

def calculate_psnr(img1, img2):
    """Calculate PSNR between two images"""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def calculate_ssim(img1, img2):
    """Calculate SSIM between two images (simplified version)"""
    # Convert to grayscale
    gray1 = np.mean(img1, axis=2)
    gray2 = np.mean(img2, axis=2)

    # Calculate means
    mu1 = np.mean(gray1)
    mu2 = np.mean(gray2)

    # Calculate variances
    sigma1_sq = np.var(gray1)
    sigma2_sq = np.var(gray2)
    sigma12 = np.cov(gray1.flatten(), gray2.flatten())[0, 1]

    # Constants
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    # SSIM calculation
    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)

    ssim = numerator / denominator
    return ssim

def analyze_sharpness(image):
    """Analyze image sharpness using multiple metrics"""
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = np.mean(image, axis=2)
    else:
        gray = image

    # Laplacian variance (higher = sharper)
    laplacian_var = np.var(Image.fromarray(gray.astype(np.uint8)).filter(ImageFilter.FIND_EDGES))

    # Gradient magnitude
    grad_x = np.gradient(gray, axis=1)
    grad_y = np.gradient(gray, axis=0)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    mean_gradient = np.mean(gradient_magnitude)

    return {
        'laplacian_variance': laplacian_var,
        'mean_gradient': mean_gradient
    }

def analyze_enhancement_results():
    """Analyze Real-ESRGAN enhancement results"""
    print("🔍 Real-ESRGAN iOS Enhancement Analysis")
    print("=" * 60)

    # Find all comparison results
    comparison_files = glob.glob("enhancement_results/*_comparison_*.png")
    if not comparison_files:
        print("❌ No enhancement results found!")
        return

    print(f"📊 Analyzing {len(comparison_files)} test cases...")

    results_summary = []

    for comp_file in comparison_files:
        # Extract test name
        base_name = os.path.basename(comp_file).replace('_comparison_', '')
        test_name = '_'.join(base_name.split('_')[:-3])  # Remove timestamp

        # Find corresponding images
        original_path = comp_file.replace('_comparison_', '_original_')
        low_quality_path = comp_file.replace('_comparison_', '_low_quality_')
        enhanced_path = comp_file.replace('_comparison_', '_enhanced_')

        if not all(os.path.exists(p) for p in [original_path, low_quality_path, enhanced_path]):
            continue

        print(f"\n🖼️  Analyzing: {test_name}")
        print("-" * 50)

        # Load images
        try:
            original_img = np.array(Image.open(original_path))
            low_quality_img = np.array(Image.open(low_quality_path))
            enhanced_img = np.array(Image.open(enhanced_path))

            # Resize enhanced to match original for comparison
            if enhanced_img.shape != original_img.shape:
                enhanced_pil = Image.fromarray(enhanced_img)
                enhanced_pil = enhanced_pil.resize((original_img.shape[1], original_img.shape[0]), Image.Resampling.LANCZOS)
                enhanced_img = np.array(enhanced_pil)

            # Calculate quality metrics
            # PSNR (higher = better)
            psnr_low = calculate_psnr(original_img, low_quality_img)
            psnr_enhanced = calculate_psnr(original_img, enhanced_img)

            # SSIM (higher = better, max 1.0)
            ssim_low = calculate_ssim(original_img, low_quality_img)
            ssim_enhanced = calculate_ssim(original_img, enhanced_img)

            # Sharpness analysis
            original_sharpness = analyze_sharpness(original_img)
            low_sharpness = analyze_sharpness(low_quality_img)
            enhanced_sharpness = analyze_sharpness(enhanced_img)

            print(f"📈 Quality Metrics:")
            print(f"   PSNR - Low Quality: {psnr_low:.2f} dB, Enhanced: {psnr_enhanced:.2f} dB")
            print(f"   SSIM - Low Quality: {ssim_low:.4f}, Enhanced: {ssim_enhanced:.4f}")
            print(f"   Sharpness (Laplacian) - Original: {original_sharpness['laplacian_variance']:.1f}")
            print(f"   Sharpness (Laplacian) - Low Quality: {low_sharpness['laplacian_variance']:.1f}")
            print(f"   Sharpness (Laplacian) - Enhanced: {enhanced_sharpness['laplacian_variance']:.1f}")

            # Calculate improvement ratios
            psnr_improvement = psnr_enhanced - psnr_low
            ssim_improvement = ssim_enhanced - ssim_low
            sharpness_improvement = enhanced_sharpness['laplacian_variance'] / low_sharpness['laplacian_variance'] if low_sharpness['laplacian_variance'] > 0 else 1

            print(f"📊 Improvement Analysis:")
            print(f"   PSNR Improvement: {psnr_improvement:+.2f} dB")
            print(f"   SSIM Improvement: {ssim_improvement:+.4f}")
            print(f"   Sharpness Ratio: {sharpness_improvement:.2f}x")

            # Color analysis
            enhanced_gray_corr = analyze_color_grayscale_tendency(enhanced_img)
            print(f"   Color Correlation: {enhanced_gray_corr['avg_channel_correlation']:.3f}")

            if enhanced_gray_corr['avg_channel_correlation'] > 0.9:
                print(f"   ⚠️  High channel correlation detected (potential gray issue)")

            # Store results
            results_summary.append({
                'test_name': test_name,
                'psnr_improvement': psnr_improvement,
                'ssim_improvement': ssim_improvement,
                'sharpness_ratio': sharpness_improvement,
                'color_correlation': enhanced_gray_corr['avg_channel_correlation'],
                'enhanced_path': enhanced_path,
                'is_effective': psnr_improvement > 1 and ssim_improvement > 0.01
            })

        except Exception as e:
            print(f"❌ Error analyzing {test_name}: {e}")

    # Generate iOS recommendations
    print(f"\n" + "=" * 60)
    print("📱 iOS Compatibility & Performance Analysis")
    print("=" * 60)

    if results_summary:
        effective_tests = [r for r in results_summary if r['is_effective']]
        gray_issue_tests = [r for r in results_summary if r['color_correlation'] > 0.9]

        print(f"📈 Overall Results:")
        print(f"   • {len(effective_tests)}/{len(results_summary)} tests showed effective enhancement")
        print(f"   • {len(gray_issue_tests)}/{len(results_summary)} tests show potential gray output issues")

        if effective_tests:
            avg_psnr_improvement = np.mean([r['psnr_improvement'] for r in effective_tests])
            avg_ssim_improvement = np.mean([r['ssim_improvement'] for r in effective_tests])
            avg_sharpness_ratio = np.mean([r['sharpness_ratio'] for r in effective_tests])

            print(f"   • Average PSNR improvement: {avg_psnr_improvement:+.2f} dB")
            print(f"   • Average SSIM improvement: {avg_ssim_improvement:+.4f}")
            print(f"   • Average sharpness improvement: {avg_sharpness_ratio:.2f}x")

        # iOS-specific recommendations
        print(f"\n🔧 iOS-Specific Recommendations:")

        if gray_issue_tests:
            print(f"\n🚨 Gray Output Issues Detected:")
            print(f"   Both RealESRGAN models show high channel correlation (>0.9)")
            print(f"   This indicates potential problems with:")
            print(f"   1. **Model Conversion**: CoreML conversion may have altered color processing")
            print(f"   2. **Input Normalization**: iOS app may use different preprocessing")
            print(f"   3. **Data Types**: Float precision differences between platforms")

            print(f"\n💡 Recommended Fixes:")
            print(f"   1. **Reconvert with Different Parameters**:")
            print(f"      - Try MLProgram format instead of NeuralNetwork")
            print(f"      - Use FLOAT32 precision instead of FLOAT16")
            print(f"      - Adjust input normalization in conversion script")

            print(f"   2. **iOS App Integration**:")
            print(f"      - Verify CVPixelBuffer to tensor conversion")
            print(f"      - Check RGB vs BGR channel order")
            print(f"      - Ensure proper [0, 1] normalization")
            print(f"      - Test with different input sizes")

            print(f"   3. **Alternative Models**:")
            print(f"      - Try ESRGAN or other super-resolution models")
            print(f"      - Test with different scale factors (2x vs 4x)")

        else:
            print(f"✅ Models appear to be working correctly")
            print(f"💡 iOS integration tips:")
            print(f"   - Ensure proper memory management for large tensors")
            print(f"   - Test on actual iOS devices (not just simulator)")
            print(f"   - Consider using smaller input sizes for better performance")

    print(f"\n📁 Enhancement results saved in: enhancement_results/")
    print(f"🔍 Check comparison images to visually assess enhancement quality")

def analyze_color_grayscale_tendency(image_array):
    """Analyze if image tends toward grayscale"""
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        r_channel = image_array[:, :, 0]
        g_channel = image_array[:, :, 1]
        b_channel = image_array[:, :, 2]

        # Calculate correlation between channels
        rg_corr = np.corrcoef(r_channel.flatten(), g_channel.flatten())[0, 1]
        rb_corr = np.corrcoef(r_channel.flatten(), b_channel.flatten())[0, 1]
        gb_corr = np.corrcoef(g_channel.flatten(), b_channel.flatten())[0, 1]

        avg_corr = (abs(rg_corr) + abs(rb_corr) + abs(gb_corr)) / 3

        return {
            'avg_channel_correlation': avg_corr,
            'rg_corr': rg_corr,
            'rb_corr': rb_corr,
            'gb_corr': gb_corr,
            'is_grayscale_like': avg_corr > 0.9
        }
    return {'avg_channel_correlation': 0, 'is_grayscale_like': False}

def main():
    analyze_enhancement_results()

if __name__ == "__main__":
    main()
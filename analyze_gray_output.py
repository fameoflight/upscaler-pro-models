#!/usr/bin/env python3
"""
Gray Output Analysis Script
Analyzes Real-ESRGAN test results to identify gray output issues on iOS
"""

import os
import numpy as np
from PIL import Image
import glob

def detect_gray_image(image_path, threshold=0.1):
    """Detect if an image is grayscale or has low contrast"""
    try:
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)

        # Convert to grayscale
        gray_array = np.mean(img_array, axis=2)

        # Calculate standard deviation of grayscale values
        gray_std = np.std(gray_array)

        # Check if standard deviation is below threshold (indicating low contrast/gray image)
        is_gray = gray_std < threshold

        # Additional check: calculate color channel correlation
        r_channel = img_array[:, :, 0]
        g_channel = img_array[:, :, 1]
        b_channel = img_array[:, :, 2]

        # Calculate correlation between channels
        rg_corr = np.corrcoef(r_channel.flatten(), g_channel.flatten())[0, 1]
        rb_corr = np.corrcoef(r_channel.flatten(), b_channel.flatten())[0, 1]
        gb_corr = np.corrcoef(g_channel.flatten(), b_channel.flatten())[0, 1]

        # High correlation between channels indicates grayscale-like behavior
        avg_corr = (abs(rg_corr) + abs(rb_corr) + abs(gb_corr)) / 3

        return {
            'is_gray': is_gray,
            'gray_std': gray_std,
            'avg_channel_correlation': avg_corr,
            'rg_corr': rg_corr,
            'rb_corr': rb_corr,
            'gb_corr': gb_corr,
            'shape': img_array.shape
        }
    except Exception as e:
        print(f"Error analyzing {image_path}: {e}")
        return None

def analyze_test_results():
    """Analyze all test results and provide iOS debugging recommendations"""
    print("🔍 Real-ESRGAN Gray Output Analysis")
    print("=" * 50)

    # Find all output images
    output_images = glob.glob("test_results/*_output_*.png")
    input_images = glob.glob("test_results/*_input_*.png")

    if not output_images:
        print("❌ No output images found in test_results/")
        return

    print(f"📊 Found {len(output_images)} output images to analyze")
    print()

    gray_issues = []
    potential_issues = []

    for output_path in output_images:
        # Get corresponding input image
        base_name = os.path.basename(output_path).replace('_output_', '_input_')
        input_path = os.path.join(os.path.dirname(output_path), base_name)

        if not os.path.exists(input_path):
            print(f"⚠️  No input image found for {output_path}")
            continue

        print(f"🖼️  Analyzing: {os.path.basename(output_path)}")

        # Analyze output image
        output_analysis = detect_gray_image(output_path)
        input_analysis = detect_gray_image(input_path)

        if output_analysis is None or input_analysis is None:
            continue

        # Compare with input
        input_std = input_analysis['gray_std']
        output_std = output_analysis['gray_std']
        std_ratio = output_std / input_std if input_std > 0 else 0

        print(f"   Input gray std: {input_std:.4f}")
        print(f"   Output gray std: {output_std:.4f}")
        print(f"   Std ratio (output/input): {std_ratio:.4f}")
        print(f"   Channel correlation: {output_analysis['avg_channel_correlation']:.4f}")

        # Determine if there's a gray output issue
        issue_detected = False
        issue_reasons = []

        if output_analysis['is_gray']:
            issue_detected = True
            issue_reasons.append("Output is grayscale (low contrast)")

        if std_ratio < 0.5:
            issue_detected = True
            issue_reasons.append(f"Low contrast ratio ({std_ratio:.2f})")

        if output_analysis['avg_channel_correlation'] > 0.9:
            issue_detected = True
            issue_reasons.append("High channel correlation (grayscale-like)")

        if issue_detected:
            gray_issues.append({
                'output_path': output_path,
                'input_path': input_path,
                'output_analysis': output_analysis,
                'input_analysis': input_analysis,
                'std_ratio': std_ratio,
                'reasons': issue_reasons
            })
            print(f"   ⚠️  GRAY OUTPUT DETECTED: {', '.join(issue_reasons)}")
        elif std_ratio < 0.7:
            potential_issues.append({
                'output_path': output_path,
                'input_path': input_path,
                'std_ratio': std_ratio
            })
            print(f"   🔍 POTENTIAL ISSUE: Moderate contrast reduction")
        else:
            print(f"   ✅ Normal output")

        print()

    # Generate iOS debugging recommendations
    print("📱 iOS Debugging Recommendations")
    print("=" * 50)

    if gray_issues:
        print(f"🚨 Found {len(gray_issues)} images with confirmed gray output issues:")
        for issue in gray_issues:
            print(f"   • {os.path.basename(issue['output_path'])}")
            print(f"     Reasons: {', '.join(issue['reasons'])}")
            print(f"     Contrast ratio: {issue['std_ratio']:.3f}")
        print()

        print("🔧 Recommended iOS Fixes:")
        print("1. **Check Input Normalization**")
        print("   - Ensure input images are normalized to [0, 1] range")
        print("   - Verify RGB vs BGR channel order")
        print("   - Check if model expects different preprocessing")

        print("\n2. **Model Input/Output Handling**")
        print("   - Verify tensor shapes match model expectations")
        print("   - Check if model expects different data types (float16 vs float32)")
        print("   - Ensure proper tensor format (CHW vs HWC)")

        print("\n3. **iOS-Specific Issues**")
        print("   - Test with smaller input sizes (64x64 works)")
        print("   - Check memory constraints on device")
        print("   - Verify CoreML model compatibility with iOS version")

        print("\n4. **Model Conversion Issues**")
        print("   - Reconvert model with proper preprocessing")
        print("   - Try different CoreML conversion parameters")
        print("   - Consider using MLProgram format vs NeuralNetwork")

        print("\n5. **Swift Code Check**")
        print("   - Review MLModelRegistry.swift integration")
        print("   - Verify image preprocessing in iOS app")
        print("   - Check CVPixelBuffer to tensor conversion")

    elif potential_issues:
        print(f"🔍 Found {len(potential_issues)} images with potential issues:")
        for issue in potential_issues:
            print(f"   • {os.path.basename(issue['output_path'])} (contrast ratio: {issue['std_ratio']:.3f})")
        print("\n💡 Monitor these images for gray output on iOS")

    else:
        print("✅ No obvious gray output issues detected in current tests")
        print("💡 If iOS still shows gray output, check:")
        print("   - iOS-specific preprocessing differences")
        print("   - Model loading/runtime errors on device")
        print("   - Different image formats on iOS vs test environment")

    print(f"\n📁 Test results saved in: test_results/")
    print("🔍 Check comparison images for visual verification")

def main():
    analyze_test_results()

if __name__ == "__main__":
    main()
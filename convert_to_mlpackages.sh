#!/bin/bash

set -e

echo "🔄 Converting all existing models to MLPackage format..."
echo "======================================================"

# Activate virtual environment
source venv/bin/activate

# Clean up old .mlmodel files and create fresh MLPackages
echo "🧹 Cleaning up old .mlmodel files..."
find models/ -name "*.mlmodel" -not -path "*/Data/*" -exec rm -f {} \;

echo "📦 Creating MLPackage models from scratch..."

# Create all lightweight models (no weights needed)
echo "Creating SRCNN models..."
python scripts/convert_srcnn.py SRCNN_x2 2
python scripts/convert_srcnn.py SRCNN_x3 3

echo "Creating Waifu2x models..."
python scripts/convert_waifu2x.py Waifu2x_x2 2
python scripts/convert_waifu2x.py Waifu2x_x4 4

# Create enhanced versions
echo "Creating enhanced models..."
python scripts/convert_srcnn.py SRCNN_x2_Enhanced 2 enhanced
python scripts/convert_waifu2x.py Waifu2x_x4_Enhanced 4 enhanced

# Convert pre-trained models if they exist
if [ -f "weights/RealESRGAN_x4plus.pth" ]; then
    echo "Converting Real-ESRGAN model..."
    python scripts/convert_realesrgan.py "weights/RealESRGAN_x4plus.pth" "RealESRGAN_4x" 4
fi

if [ -f "weights/RRDB_ESRGAN_x4.pth" ]; then
    echo "Converting ESRGAN 4x model..."
    python scripts/convert_esrgan.py "weights/RRDB_ESRGAN_x4.pth" "ESRGAN_4x" 4
fi

if [ -f "weights/RRDB_ESRGAN_x2.pth" ]; then
    echo "Converting ESRGAN 2x model..."
    python scripts/convert_esrgan.py "weights/RRDB_ESRGAN_x2.pth" "ESRGAN_2x" 2
fi

echo ""
echo "✅ MLPackage conversion completed!"
echo "📦 Available MLPackage models:"
echo ""

# List all MLPackage files with details
for package in models/*.mlpackage; do
    if [ -d "$package" ]; then
        package_name=$(basename "$package")
        size=$(du -sh "$package" | cut -f1)
        echo "   📱 $package_name ($size)"
    fi
done

echo ""
echo "🎯 All models are now in MLPackage format!"
echo "   - Drag and drop directly into Xcode"
echo "   - iOS 13+ compatible"
echo "   - Optimized for Core ML framework"
echo ""
echo "🔧 Next steps:"
echo "   1. Add MLPackage files to your Xcode project"
echo "   2. Update MLModelRegistry.swift with new paths"
echo "   3. Test models in your iOS app"
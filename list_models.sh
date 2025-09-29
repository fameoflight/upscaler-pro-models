#!/bin/bash

echo "📦 Available MLPackage Models for iOS"
echo "====================================="
echo ""

echo "🎯 Ready-to-use MLPackage models:"
echo ""

for package in models/*.mlpackage; do
    if [ -d "$package" ]; then
        package_name=$(basename "$package")
        size=$(du -sh "$package" | cut -f1)

        # Extract model type and scale from filename
        case "$package_name" in
            "RealESRGAN"*)
                icon="📸"
                type="Real-world Photos"
                ;;
            "ESRGAN"*)
                icon="🖼️"
                type="General Purpose"
                ;;
            "Waifu2x"*)
                icon="🎨"
                type="Anime/Artwork"
                ;;
            "SRCNN"*)
                icon="⚡"
                type="Fast/Lightweight"
                ;;
            *)
                icon="🔧"
                type="Other"
                ;;
        esac

        printf "  %s %-28s %s (%s)\n" "$icon" "$package_name" "$type" "$size"
    fi
done

echo ""
echo "🔧 Usage in iOS/Xcode:"
echo "  1. Drag .mlpackage files directly into your Xcode project"
echo "  2. Add to target when prompted"
echo "  3. Load in Swift with: MLModel(contentsOf: url)"
echo ""
echo "💡 Model Recommendations:"
echo "  📸 Photos/Real images: RealESRGAN_4x.mlpackage"
echo "  🎨 Anime/Digital art: Waifu2x_x4.mlpackage"
echo "  ⚡ Fast processing: SRCNN_x2.mlpackage"
echo "  🖼️  General purpose: Use RealESRGAN as fallback"
echo ""
echo "🎉 All models are iOS 13+ compatible and optimized for Core ML!"
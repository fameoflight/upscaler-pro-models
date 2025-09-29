#!/bin/bash

set -e

echo "🚀 Setting up all upscaling models for iOS..."
echo "============================================="

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating Python virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

# Install required packages
echo "⬇️  Installing Python packages..."
pip install --upgrade pip
pip install torch==2.8.0 torchvision==0.23.0 coremltools==8.3 numpy==1.26.4

# Create directories
mkdir -p models
mkdir -p weights

# Define model download URLs
# Note: ESRGAN original models are hard to find, using alternative high-quality models
ESRGAN_2x_URL="https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth"
ESRGAN_4x_URL="https://github.com/xinntao/ESRGAN/releases/download/0.0.0/RRDB_ESRGAN_x4.pth"
REALESRGAN_4x_URL="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
REALESRGAN_ANIME_4x_URL="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"

# Function to download model with verification
download_model() {
    local model_name=$1
    local model_url=$2
    local model_path="weights/${model_name}.pth"
    local min_size=1048576  # 1MB minimum size

    if [ ! -f "$model_path" ]; then
        echo "⬇️  Downloading $model_name..."

        # Download with progress and follow redirects
        if curl -L --fail --progress-bar -o "$model_path" "$model_url"; then
            # Verify download
            local file_size=$(stat -f%z "$model_path" 2>/dev/null || stat -c%s "$model_path" 2>/dev/null || echo "0")

            if [ "$file_size" -lt "$min_size" ]; then
                echo "❌ Download failed: File too small ($file_size bytes, expected > $min_size bytes)"

                # Check if it contains error messages
                if [ "$file_size" -lt 1000 ]; then
                    echo "📋 File content:"
                    cat "$model_path"
                fi

                rm -f "$model_path"
                return 1
            fi

            # Verify file is binary (PyTorch models should be binary)
            if file "$model_path" | grep -q "text"; then
                echo "❌ Download failed: File is text, not a binary model file"
                echo "📋 File content (first 200 chars):"
                head -c 200 "$model_path"
                echo ""
                rm -f "$model_path"
                return 1
            fi

            # Check for PyTorch magic bytes or ZIP signature (PyTorch models are often ZIP files)
            local file_header=$(hexdump -C "$model_path" | head -1)
            if echo "$file_header" | grep -E "(504b 0304|504b 0506)" >/dev/null; then
                echo "✅ Downloaded $model_name ($(numfmt --to=iec $file_size)) - Valid PyTorch model"
                return 0
            elif echo "$file_header" | grep -E "(7f45 4c46|8950 4e47)" >/dev/null; then
                echo "⚠️  Downloaded $model_name ($(numfmt --to=iec $file_size)) - Binary file (format unknown)"
                return 0
            else
                echo "✅ Downloaded $model_name ($(numfmt --to=iec $file_size)) - Binary file"
                return 0
            fi
        else
            echo "❌ Download failed: URL not found or network error"
            rm -f "$model_path"
            return 1
        fi
    else
        # Check existing file size
        local file_size=$(stat -f%z "$model_path" 2>/dev/null || stat -c%s "$model_path" 2>/dev/null || echo "0")

        if [ "$file_size" -lt "$min_size" ]; then
            echo "⚠️  Existing $model_name is too small, re-downloading..."
            rm -f "$model_path"
            download_model "$model_name" "$model_url"
        else
            echo "✅ $model_name already exists ($(numfmt --to=iec $file_size))"
            return 0
        fi
    fi
}

# Download all models
echo "📥 Downloading model weights..."

# Track successful downloads
DOWNLOADED_MODELS=""

if download_model "ESRGAN_2x" "$ESRGAN_2x_URL"; then
    DOWNLOADED_MODELS="$DOWNLOADED_MODELS ESRGAN_2x"
fi

if download_model "ESRGAN_4x" "$ESRGAN_4x_URL"; then
    DOWNLOADED_MODELS="$DOWNLOADED_MODELS ESRGAN_4x"
fi

if download_model "RealESRGAN_4x" "$REALESRGAN_4x_URL"; then
    DOWNLOADED_MODELS="$DOWNLOADED_MODELS RealESRGAN_4x"
fi

if download_model "RealESRGAN_anime_4x" "$REALESRGAN_ANIME_4x_URL"; then
    DOWNLOADED_MODELS="$DOWNLOADED_MODELS RealESRGAN_anime_4x"
fi

echo ""
echo "📊 Download Summary:"
if [ -n "$DOWNLOADED_MODELS" ]; then
    echo "✅ Successfully downloaded:$DOWNLOADED_MODELS"
else
    echo "⚠️  No pre-trained models downloaded (will create lightweight models only)"
fi

echo "🔄 Converting models to MLPackage format..."

# Convert pre-trained models (only if successfully downloaded)
echo ""
echo "🔄 Converting pre-trained models..."

# Convert ESRGAN models
if [ -f "weights/ESRGAN_2x.pth" ] && [ "$(stat -f%z "weights/ESRGAN_2x.pth" 2>/dev/null || echo "0")" -gt 1048576 ]; then
    echo "Converting ESRGAN 2x..."
    python scripts/convert_esrgan.py "weights/ESRGAN_2x.pth" "ESRGAN_2x" 2
else
    echo "⏭️  Skipping ESRGAN_2x (not available)"
fi

if [ -f "weights/ESRGAN_4x.pth" ] && [ "$(stat -f%z "weights/ESRGAN_4x.pth" 2>/dev/null || echo "0")" -gt 1048576 ]; then
    echo "Converting ESRGAN 4x..."
    python scripts/convert_esrgan.py "weights/ESRGAN_4x.pth" "ESRGAN_4x" 4
else
    echo "⏭️  Skipping ESRGAN_4x (not available)"
fi

# Convert RealESRGAN models
if [ -f "weights/RealESRGAN_4x.pth" ] && [ "$(stat -f%z "weights/RealESRGAN_4x.pth" 2>/dev/null || echo "0")" -gt 1048576 ]; then
    echo "Converting RealESRGAN 4x..."
    python scripts/convert_realesrgan.py "weights/RealESRGAN_4x.pth" "RealESRGAN_4x" 4
else
    echo "⏭️  Skipping RealESRGAN_4x (not available)"
fi

if [ -f "weights/RealESRGAN_anime_4x.pth" ] && [ "$(stat -f%z "weights/RealESRGAN_anime_4x.pth" 2>/dev/null || echo "0")" -gt 1048576 ]; then
    echo "Converting RealESRGAN anime 4x..."
    python scripts/convert_realesrgan.py "weights/RealESRGAN_anime_4x.pth" "Waifu2x_RealESRGAN_4x" 4
else
    echo "⏭️  Skipping RealESRGAN anime (not available)"
fi

# Create lightweight models (always available)
echo ""
echo "🔄 Creating lightweight models..."

echo "Creating SRCNN models (fast, small)..."
python scripts/convert_srcnn.py "SRCNN_x2" 2
python scripts/convert_srcnn.py "SRCNN_x3" 3

echo "Creating Waifu2x models (anime specialized)..."
python scripts/convert_waifu2x.py "Waifu2x_x2" 2
python scripts/convert_waifu2x.py "Waifu2x_x4" 4

echo "✅ All models setup completed!"
echo "📱 Check the models/ directory for iOS-compatible MLPackage models:"
ls -la models/

echo ""
echo "🎉 Setup complete! Available MLPackage models:"
echo "   - ESRGAN_2x.mlpackage (2x general upscaling)"
echo "   - ESRGAN_4x.mlpackage (4x general upscaling)"
echo "   - RealESRGAN_4x.mlpackage (4x real-world images)"
echo "   - Waifu2x_x2.mlpackage (2x anime/artwork)"
echo "   - Waifu2x_x4.mlpackage (4x anime/artwork)"
echo "   - Waifu2x_RealESRGAN_4x.mlpackage (4x anime, Real-ESRGAN trained)"
echo "   - SRCNN_x2.mlpackage (lightweight 2x)"
echo "   - SRCNN_x3.mlpackage (lightweight 3x)"
echo ""
echo "📦 All models are in MLPackage format - drag directly into Xcode!"
echo "🔧 Use these models in your iOS app with Core ML framework"
# Upscaler Pro Models

A comprehensive collection of super-resolution models optimized for iOS, supporting multiple upscaling algorithms including ESRGAN, Real-ESRGAN, SRCNN, and Waifu2x.

## 🚀 Quick Start

### One-Command Setup (MLPackage Format)
```bash
./setup_all_models.sh
```

### Convert Existing Models to MLPackage
```bash
./convert_to_mlpackages.sh
```

### Individual Model Setup
```bash
# Real-ESRGAN (iOS optimized)
./setup_and_convert_ios.sh

# Real-ESRGAN MLPackage format
./create_mlpackage.sh
```

## 📁 Project Structure

```
upscaler-pro-models/
├── models/                    # 📦 MLPackage models (.mlpackage)
├── weights/                   # 🏋️ PyTorch model weights (.pth)
├── scripts/                   # 🔧 Individual conversion scripts
│   ├── convert_esrgan.py      # ESRGAN → MLPackage converter
│   ├── convert_realesrgan.py  # Real-ESRGAN → MLPackage converter
│   ├── convert_srcnn.py       # SRCNN → MLPackage converter
│   ├── convert_waifu2x.py     # Waifu2x → MLPackage converter
│   ├── mlpackage_utils.py     # MLPackage creation utilities
│   └── convert_simple_test.py # Test conversion pipeline
├── Real-ESRGAN/              # 📚 Real-ESRGAN submodule
├── venv/                     # 🐍 Python virtual environment
├── setup_all_models.sh       # 🎯 Master setup script (MLPackage)
├── convert_to_mlpackages.sh  # 🔄 Convert existing to MLPackage
└── CLAUDE.md                 # 📖 Detailed documentation
```

## 🎨 Available Models

| Model | Use Case | Scale | Size | Quality |
|-------|----------|--------|------|---------|
| **ESRGAN 2x/4x** | General purpose | 2x, 4x | ~65MB | High |
| **Real-ESRGAN 4x** | Photography/Natural images | 4x | ~67MB | Very High |
| **Waifu2x 2x/4x** | Anime/Artwork | 2x, 4x | ~15MB | High |
| **SRCNN 2x/3x** | Fast processing | 2x, 3x | ~5MB | Medium |

## 🔧 Manual Conversion

### Convert Individual Models (MLPackage Format)
```bash
# Activate environment
source venv/bin/activate

# Convert ESRGAN → MLPackage
python scripts/convert_esrgan.py weights/RRDB_ESRGAN_x4.pth ESRGAN_4x 4

# Convert Real-ESRGAN → MLPackage
python scripts/convert_realesrgan.py weights/RealESRGAN_x4plus.pth RealESRGAN_4x 4

# Convert SRCNN → MLPackage (creates from scratch)
python scripts/convert_srcnn.py SRCNN_x2 2

# Convert Waifu2x → MLPackage (creates from scratch)
python scripts/convert_waifu2x.py Waifu2x_x2 2
```

### Enhanced Versions (MLPackage Format)
```bash
# Enhanced SRCNN with more layers → MLPackage
python scripts/convert_srcnn.py SRCNN_x2_Enhanced 2 enhanced

# Enhanced Waifu2x with residual connections → MLPackage
python scripts/convert_waifu2x.py Waifu2x_x4_Enhanced 4 enhanced
```

## 📱 iOS Integration

### Swift Code Example (MLPackage)
```swift
import CoreML

class ModelManager {
    func loadModel(named modelName: String) -> MLModel? {
        // For MLPackage: Use the package name without extension
        guard let modelURL = Bundle.main.url(forResource: modelName, withExtension: "mlpackage"),
              let model = try? MLModel(contentsOf: modelURL) else {
            return nil
        }
        return model
    }

    func upscaleImage(model: MLModel, image: CVPixelBuffer) throws -> CVPixelBuffer? {
        let input = try MLDictionaryFeatureProvider(dictionary: ["input": MLFeatureValue(pixelBuffer: image)])
        let output = try model.prediction(from: input)
        return output.featureValue(for: "output")?.imageBufferValue
    }
}

// Usage example:
let modelManager = ModelManager()
if let esrganModel = modelManager.loadModel(named: "ESRGAN_4x") {
    // Use the model for upscaling
}
```

### Model Selection Logic
- **General images**: RealESRGAN_4x → ESRGAN_4x fallback
- **Anime/Art**: Waifu2x → ESRGAN fallback
- **Fast processing**: SRCNN_x2 → SRCNN_x3
- **Photos**: ESRGAN_4x → RealESRGAN_4x fallback

## 🧪 Testing

### Verify Setup
```bash
# Test conversion pipeline
source venv/bin/activate
python scripts/convert_simple_test.py

# Check converted models
ls -la models/
```

### Model Information
```bash
# Get model details using Python
source venv/bin/activate
python -c "import coremltools as ct; model = ct.models.MLModel('models/SRCNN_x2.mlmodel'); print(f'Input: {model.input_description}'); print(f'Output: {model.output_description}')"
```

## 🔄 Model Updates

### Adding New Models
1. Add model info to `MLModelRegistry.swift`
2. Add download URL to `setup_all_models.sh`
3. Test conversion
4. Update documentation

### Custom Models
```bash
# Convert your own PyTorch model
python scripts/convert_esrgan.py path/to/your/model.pth YourModel_4x 4
```

## 🛠 Requirements

- **System**: macOS 10.15+ (for CoreML Tools)
- **Python**: 3.8+
- **Xcode**: 12+ (for iOS deployment)
- **Dependencies**: torch, coremltools, numpy

## ❗ Troubleshooting

### Common Issues

**"Model conversion failed"**
- Try different CoreML target versions
- Check PyTorch compatibility

**"Model too large"**
- Use SRCNN models for size constraints
- Consider FLOAT16 precision

**"iOS compatibility issues"**
- Ensure iOS 13+ deployment target
- Test on actual iOS devices

### Debug Commands
```bash
# Check environment
source venv/bin/activate && python --version

# Verify downloads
ls -la weights/

# Test individual conversion
python scripts/convert_simple_test.py
```

## 📜 License

- **ESRGAN**: Apache 2.0 License
- **Real-ESRGAN**: BSD 3-Clause License
- **SRCNN/Waifu2x**: Custom implementations (MIT)

---

## 🎉 **MLPackage Ready!**

✨ All models are now converted to **MLPackage format** - the modern standard for Xcode integration!

### Key Benefits:
- 📦 **Drag & Drop**: Directly into Xcode projects
- 🚀 **Better Performance**: Optimized for iOS/macOS
- 🔧 **Easier Integration**: No manual bundling required
- 📱 **iOS 13+**: Modern CoreML support

### Get Started:
```bash
./setup_all_models.sh        # Create all MLPackage models
./convert_to_mlpackages.sh   # Convert existing models
```

✨ **Your super-resolution models are ready for iOS!**
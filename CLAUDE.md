# Upscaler Pro Models - Project Documentation

## Overview
This repository contains machine learning models and conversion scripts for the Upscaler Pro iOS app. It provides multiple super-resolution models optimized for different use cases and converted to iOS-compatible CoreML format.

## Available Models

### ESRGAN Models
- **ESRGAN_2x**: Enhanced Super-Resolution GAN for 2x upscaling
  - Use case: General purpose, balanced quality/performance
  - Scale factor: 2x
  - Model size: ~65MB

- **ESRGAN_4x**: Enhanced Super-Resolution GAN for 4x upscaling
  - Use case: High-quality general image upscaling
  - Scale factor: 4x
  - Model size: ~65MB

### Real-ESRGAN Models
- **RealESRGAN_4x**: Practical super-resolution for real-world images
  - Use case: Photography, natural images
  - Scale factor: 4x
  - Model size: ~67MB

- **Waifu2x_x4**: Specialized for anime/artwork upscaling
  - Use case: Anime, illustrations, digital art
  - Scale factor: 4x
  - Model size: ~67MB

### SRCNN Models (Lightweight)
- **SRCNN_x2**: Super-Resolution Convolutional Neural Network
  - Use case: Fast processing, lower quality acceptable
  - Scale factor: 2x
  - Model size: ~5MB

- **SRCNN_x3**: Super-Resolution Convolutional Neural Network
  - Use case: 3x upscaling with fast processing
  - Scale factor: 3x
  - Model size: ~5MB

## Quick Start

### Setup All Models
Run the master setup script to download and convert all models:

```bash
./setup_all_models.sh
```

### Setup Individual Models
For specific model types:

```bash
# Real-ESRGAN only (iOS optimized)
./setup_and_convert_ios.sh

# Real-ESRGAN MLPackage format
./create_mlpackage.sh

# Legacy setup script
./setup_and_convert.sh
```

### Manual Model Conversion
If you have custom PyTorch weights:

```bash
# Activate environment
source venv/bin/activate

# Convert ESRGAN model
python scripts/convert_esrgan.py weights/your_model.pth YourModel_4x 4

# Convert Real-ESRGAN model
python scripts/convert_realesrgan.py weights/your_model.pth YourModel_4x 4
```

## Project Structure

```
upscaler-pro-models/
├── models/                 # Converted CoreML models (.mlmodel files)
├── weights/               # Downloaded PyTorch model weights (.pth files)
├── scripts/               # Individual conversion scripts
├── Real-ESRGAN/          # Real-ESRGAN submodule
├── venv/                 # Python virtual environment
├── setup_all_models.sh   # Master setup script
├── setup_and_convert_ios.sh  # iOS-specific Real-ESRGAN setup
└── create_mlpackage.sh   # MLPackage format creation
```

## Model Integration in iOS

### Swift Integration Example
```swift
import CoreML

// Load a model
guard let model = try? MLModel(contentsOf: modelURL) else { return }

// Create prediction
let input = try MLDictionaryFeatureProvider(dictionary: ["input": inputImage])
let output = try model.prediction(from: input)
```

### Model Selection Logic
The models are designed to work with the `MLModelRegistry.swift` system:

- **General images**: RealESRGAN_4x → ESRGAN_4x fallback
- **Anime/Art**: Waifu2x_x4 → ESRGAN_4x fallback
- **Fast processing**: SRCNN_x2 → SRCNN_x3
- **Photos**: ESRGAN_4x → RealESRGAN_4x fallback

## Development Workflow

### Adding New Models
1. Add model info to `MLModelRegistry.swift`
2. Add download URL and conversion logic to `setup_all_models.sh`
3. Test conversion: `python scripts/convert_<type>.py`
4. Verify in iOS: Load and test model

### Testing Models
```bash
# Check model file integrity
ls -la models/

# Verify CoreML model format
python -c "import coremltools as ct; print(ct.models.MLModel('models/YourModel.mlmodel'))"
```

### Updating Models
1. Update download URLs in setup scripts
2. Re-run setup: `./setup_all_models.sh`
3. Test in iOS app
4. Update version numbers in `MLModelRegistry.swift`

## Requirements

### System Requirements
- macOS 10.15+ (for CoreML Tools)
- Python 3.8+
- Xcode 12+ (for iOS deployment)

### Python Dependencies
- torch==2.3.0
- torchvision==0.18.0
- coremltools==8.0
- numpy==1.26.4

## Troubleshooting

### Common Issues
1. **"Model conversion failed"**: Try different CoreML target versions
2. **"Model too large"**: Use SRCNN models for size constraints
3. **"iOS compatibility issues"**: Ensure iOS 13+ deployment target

### Debug Commands
```bash
# Check Python environment
source venv/bin/activate && python --version

# Verify model downloads
ls -la weights/

# Test individual conversion
python scripts/convert_realesrgan.py weights/RealESRGAN_x4plus.pth TestModel 4
```

### Performance Optimization
- Use FLOAT16 precision for smaller models
- Consider input size limitations (64x64 recommended)
- Test on actual iOS devices for performance validation

## Contributing

### Adding New Model Types
1. Create conversion script in `scripts/`
2. Add to `setup_all_models.sh`
3. Update `MLModelRegistry.swift` with model info
4. Add documentation here

### Model Quality Guidelines
- Test on variety of image types
- Benchmark against existing models
- Ensure iOS compatibility
- Document use cases and limitations

## Model Sources & Licenses

- **ESRGAN**: BSD 3-Clause License
- **Real-ESRGAN**: BSD 3-Clause License
- **SRCNN**: Custom implementation for demonstration

See individual model repositories for detailed licensing information.
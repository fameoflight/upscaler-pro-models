#!/usr/bin/env python3
"""
Generate FeatureDescriptions.json for MLModel packages
"""

import json
import os
import coremltools as ct
from pathlib import Path


def generate_feature_descriptions(model_path, model_name=None):
    """
    Generate FeatureDescriptions.json for a CoreML model
    """
    if model_name is None:
        model_name = Path(model_path).stem

    # Load the model to inspect its inputs/outputs
    model = ct.models.MLModel(model_path)

    # Get model metadata
    spec = model.get_spec()
    input_desc = {}
    output_desc = {}

    # Extract input descriptions
    for input in spec.description.input:
        input_desc[input.name] = {
            "shortDescription": f"Input image tensor {input.name}",
            "metadata": {}
        }

    # Extract output descriptions
    for output in spec.description.output:
        output_desc[output.name] = {
            "shortDescription": f"Output upscaled image {output.name}",
            "metadata": {}
        }

    # Get metadata with safe access
    metadata = spec.description.metadata
    short_desc = f"{model_name} super-resolution model"
    author = "Unknown"
    license = "Unknown"
    version = "1.0"

    # Safely extract metadata fields
    try:
        if hasattr(metadata, 'shortDescription') and metadata.shortDescription:
            short_desc = metadata.shortDescription
    except:
        pass

    try:
        if hasattr(metadata, 'author') and metadata.author:
            author = metadata.author
    except:
        pass

    try:
        if hasattr(metadata, 'license') and metadata.license:
            license = metadata.license
    except:
        pass

    try:
        if hasattr(metadata, 'version') and metadata.version:
            version = str(metadata.version)
    except:
        pass

    # Create FeatureDescriptions.json structure
    feature_descriptions = {
        "inputDescriptions": input_desc,
        "outputDescriptions": output_desc,
        "metadata": {
            "modelDescription": {
                "shortDescription": short_desc,
                "author": author,
                "license": license,
                "version": version
            }
        }
    }

    return feature_descriptions


def process_mlpackage(mlpackage_path):
    """
    Process an MLModel package and add FeatureDescriptions.json
    """
    mlpackage_path = Path(mlpackage_path)
    if not mlpackage_path.exists():
        print(f"Error: MLModel package not found at {mlpackage_path}")
        return False

    # Find the .mlmodel file
    mlmodel_files = list(mlpackage_path.rglob("*.mlmodel"))
    if not mlmodel_files:
        print(f"Error: No .mlmodel file found in {mlpackage_path}")
        return False

    mlmodel_path = mlmodel_files[0]
    model_name = mlmodel_path.stem

    # Generate feature descriptions
    try:
        feature_descriptions = generate_feature_descriptions(str(mlmodel_path), model_name)
    except Exception as e:
        print(f"Error generating feature descriptions for {model_name}: {e}")
        return False

    # Determine where to save the FeatureDescriptions.json
    # Check if it's the old format (Data/com.apple.CoreML/) or new format
    coreml_dir = mlmodel_path.parent
    if coreml_dir.name == "com.apple.CoreML" and coreml_dir.parent.name == "Data":
        # Old format - save in com.apple.CoreML directory
        output_path = coreml_dir / "FeatureDescriptions.json"
    else:
        # New format - save in package root
        output_path = mlpackage_path / "FeatureDescriptions.json"

    # Save the JSON file
    with open(output_path, 'w') as f:
        json.dump(feature_descriptions, f, indent=2)

    print(f"Generated FeatureDescriptions.json for {model_name}")
    print(f"Saved to: {output_path}")
    return True


def process_all_models():
    """
    Process all MLModel packages in the models directory
    """
    models_dir = Path("models")
    if not models_dir.exists():
        print("Error: models directory not found")
        return

    mlpackages = list(models_dir.glob("*.mlpackage"))
    if not mlpackages:
        print("No MLModel packages found in models directory")
        return

    print(f"Processing {len(mlpackages)} MLModel packages...")

    success_count = 0
    for mlpackage in mlpackages:
        if process_mlpackage(mlpackage):
            success_count += 1
        print()

    print(f"Completed: {success_count}/{len(mlpackages)} packages processed successfully")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate FeatureDescriptions.json for MLModel packages")
    parser.add_argument("model", nargs="?", help="Path to specific MLModel package (optional)")
    parser.add_argument("--all", action="store_true", help="Process all models in models directory")

    args = parser.parse_args()

    if args.all:
        process_all_models()
    elif args.model:
        process_mlpackage(args.model)
    else:
        print("Usage: python generate_feature_descriptions.py [--all] [model_path]")
        print("  --all    Process all models in models directory")
        print("  model    Path to specific MLModel package")
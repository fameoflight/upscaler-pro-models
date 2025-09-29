#!/usr/bin/env python3
"""
MLPackage utilities for creating iOS-compatible MLPackage format
"""

import os
import shutil
import json
import coremltools as ct

def save_as_mlpackage(mlmodel, output_path, model_info):
    """
    Save a CoreML model as MLPackage format with proper metadata

    Args:
        mlmodel: The CoreML model object
        output_path: Path where to save the MLPackage (should end with .mlpackage)
        model_info: Dictionary with model metadata
    """

    # Ensure output path ends with .mlpackage
    if not output_path.endswith('.mlpackage'):
        output_path = output_path.replace('.mlmodel', '.mlpackage')

    # Remove existing directory if it exists
    if os.path.exists(output_path):
        shutil.rmtree(output_path)

    print(f"Creating MLPackage at: {output_path}")

    try:
        # Try direct MLPackage save first (newer CoreML tools)
        mlmodel.save(output_path)
        print("✅ Direct MLPackage save successful!")

        # Add custom metadata if the directory structure allows
        try:
            _add_custom_metadata(output_path, model_info)
        except Exception as metadata_e:
            print(f"⚠️  Custom metadata addition failed: {metadata_e}")

        return output_path

    except Exception as e:
        print(f"Direct MLPackage save failed: {e}")
        print("Falling back to manual MLPackage creation...")

        # Fallback: Save as .mlmodel first, then create MLPackage structure
        temp_mlmodel_path = output_path.replace('.mlpackage', '_temp.mlmodel')
        mlmodel.save(temp_mlmodel_path)

        # Create MLPackage structure manually
        _create_mlpackage_structure(temp_mlmodel_path, output_path, model_info)

        # Clean up temp file
        if os.path.exists(temp_mlmodel_path):
            os.remove(temp_mlmodel_path)

        return output_path

def _create_mlpackage_structure(mlmodel_path, mlpackage_path, model_info):
    """Create MLPackage directory structure manually"""

    # Create MLPackage directory structure
    os.makedirs(mlpackage_path, exist_ok=True)
    data_dir = os.path.join(mlpackage_path, "Data")
    os.makedirs(data_dir, exist_ok=True)
    coreml_dir = os.path.join(data_dir, "com.apple.CoreML")
    os.makedirs(coreml_dir, exist_ok=True)

    # Copy the .mlmodel file into the MLPackage
    model_name = os.path.basename(mlpackage_path).replace('.mlpackage', '.mlmodel')
    target_model_path = os.path.join(coreml_dir, model_name)
    shutil.copy2(mlmodel_path, target_model_path)

    # Create Manifest.json for the MLPackage root
    root_manifest = _create_root_manifest(model_info)
    with open(os.path.join(mlpackage_path, "Manifest.json"), 'w') as f:
        json.dump(root_manifest, f, indent=2)

    # Create Manifest.json for the Data directory
    data_manifest = _create_data_manifest(model_info, model_name)
    with open(os.path.join(data_dir, "Manifest.json"), 'w') as f:
        json.dump(data_manifest, f, indent=2)

    print(f"✅ Manual MLPackage structure created at: {mlpackage_path}")

def _create_root_manifest(model_info):
    """Create root-level Manifest.json"""
    return {
        "fileFormatVersion": "1.0.0",
        "itemInfoEntries": {
            "Data": {
                "path": "Data",
                "digest": "placeholder_digest",
                "isDirectory": True
            }
        }
    }

def _create_data_manifest(model_info, model_filename):
    """Create Data-level Manifest.json"""
    return {
        "fileFormatVersion": "1.0.0",
        "itemInfoEntries": {
            "com.apple.CoreML": {
                "path": "com.apple.CoreML",
                "digest": "placeholder_digest",
                "isDirectory": True
            }
        },
        "rootModelIdentifier": f"com.apple.CoreML/{model_filename}",
        "modelInfo": {
            "modelIdentifier": model_info.get('identifier', 'com.example.model'),
            "modelDescription": {
                "shortDescription": model_info.get('description', 'Super-resolution model'),
                "metadata": {
                    "author": model_info.get('author', 'Unknown'),
                    "license": model_info.get('license', 'Unknown'),
                    "version": model_info.get('version', '1.0'),
                    "scaleFactor": model_info.get('scale_factor', 4)
                }
            }
        }
    }

def _add_custom_metadata(mlpackage_path, model_info):
    """Add custom metadata to existing MLPackage"""
    manifest_path = os.path.join(mlpackage_path, "Manifest.json")

    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)

            # Add custom metadata
            if 'metadata' not in manifest:
                manifest['metadata'] = {}

            manifest['metadata'].update({
                'author': model_info.get('author', 'Unknown'),
                'license': model_info.get('license', 'Unknown'),
                'version': model_info.get('version', '1.0'),
                'scaleFactor': model_info.get('scale_factor', 4),
                'modelType': model_info.get('model_type', 'super-resolution'),
                'optimizedFor': model_info.get('optimized_for', 'iOS')
            })

            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)

            print("✅ Custom metadata added to MLPackage")

        except Exception as e:
            print(f"⚠️  Failed to add custom metadata: {e}")

def verify_mlpackage(mlpackage_path):
    """Verify that the MLPackage can be loaded"""
    try:
        model = ct.models.MLModel(mlpackage_path)
        print(f"✅ MLPackage verification successful: {mlpackage_path}")

        # Print package contents
        print(f"📦 MLPackage contents:")
        for root, dirs, files in os.walk(mlpackage_path):
            level = root.replace(mlpackage_path, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                print(f"{subindent}{file}")

        return True

    except Exception as e:
        print(f"❌ MLPackage verification failed: {e}")
        return False

def get_mlpackage_info(mlpackage_path):
    """Get information about an MLPackage"""
    try:
        model = ct.models.MLModel(mlpackage_path)

        info = {
            'description': getattr(model, 'short_description', 'N/A'),
            'author': getattr(model, 'author', 'N/A'),
            'license': getattr(model, 'license', 'N/A')
        }

        # Get input/output info
        try:
            info['inputs'] = list(model.input_description.keys())
            info['outputs'] = list(model.output_description.keys())
        except:
            info['inputs'] = ['Unknown']
            info['outputs'] = ['Unknown']

        # Get file size
        total_size = 0
        for root, dirs, files in os.walk(mlpackage_path):
            for file in files:
                total_size += os.path.getsize(os.path.join(root, file))
        info['size_mb'] = total_size / (1024 * 1024)

        return info

    except Exception as e:
        print(f"Failed to get MLPackage info: {e}")
        return None

if __name__ == "__main__":
    print("MLPackage Utilities")
    print("This module provides utilities for creating and managing MLPackage files.")
    print("Import this module in your conversion scripts to use these functions.")
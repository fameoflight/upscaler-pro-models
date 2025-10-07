# Upscaler Pro Models - Makefile
# Provides convenient targets for model setup, conversion, and verification

.PHONY: help setup clean clean-all clean-models clean-weights clean-venv
.PHONY: verify-requirements verify-models verify-weights test-conversion
.PHONY: list-models list-weights list-scripts
.PHONY: setup-esrgan setup-realesrgan setup-all
.PHONY: convert-models convert-esrgan convert-realesrgan convert-all
.PHONY: create-mlpackages test-models
.PHONY: install-deps update-deps

# Default target
.DEFAULT_GOAL := help

# Configuration
PYTHON := python3
VENV := venv
ACTIVATE := $(VENV)/bin/activate
MODELS_DIR := models
WEIGHTS_DIR := weights
SCRIPTS_DIR := scripts
REQUIREMENTS := requirements.txt

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
PURPLE := \033[0;35m
CYAN := \033[0;36m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(CYAN)Upscaler Pro Models - Makefile Targets$(NC)"
	@echo "$(YELLOW)==========================================$(NC)"
	@echo ""
	@echo "$(GREEN)Setup Targets:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / && /setup/ {printf "  $(BLUE)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(GREEN)Conversion Targets:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / && /convert|create/ {printf "  $(BLUE)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(GREEN)Verification Targets:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / && /verify|test|list/ {printf "  $(BLUE)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(GREEN)Utility Targets:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / && !/setup|convert|verify|test|list/ {printf "  $(BLUE)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""

# Setup Targets

setup: setup-all ## Setup all models and dependencies

setup-all: ## Setup all models and dependencies
	@echo "$(PURPLE)🚀 Setting up all upscaling models for iOS...$(NC)"
	@echo "$(YELLOW)=============================================$(NC)"
	./setup_all_models.sh

setup-esrgan: ## Setup ESRGAN models only
	@echo "$(PURPLE)📦 Setting up ESRGAN models...$(NC)"
	@echo "$(YELLOW)=============================$(NC)"
	@if [ -f "setup_and_convert.sh" ]; then \
		./setup_and_convert.sh; \
	else \
		echo "$(RED)❌ setup_and_convert.sh not found$(NC)"; \
		exit 1; \
	fi

setup-realesrgan: ## Setup Real-ESRGAN models (iOS optimized)
	@echo "$(PURPLE)📱 Setting up Real-ESRGAN models for iOS...$(NC)"
	@echo "$(YELLOW)=========================================$(NC)"
	@if [ -f "setup_and_convert_ios.sh" ]; then \
		./setup_and_convert_ios.sh; \
	else \
		echo "$(RED)❌ setup_and_convert_ios.sh not found$(NC)"; \
		exit 1; \
	fi

# Dependencies

install-deps: ## Install Python dependencies
	@echo "$(PURPLE)⬇️  Installing Python packages...$(NC)"
	@if [ ! -d "$(VENV)" ]; then \
		echo "$(BLUE)📦 Creating virtual environment...$(NC)"; \
		$(PYTHON) -m venv $(VENV); \
	fi
	@source $(ACTIVATE) && pip install --upgrade pip && pip install -r $(REQUIREMENTS)

update-deps: ## Update Python dependencies
	@echo "$(PURPLE)🔄 Updating Python packages...$(NC)"
	@if [ -d "$(VENV)" ]; then \
		source $(ACTIVATE) && pip install --upgrade -r $(REQUIREMENTS); \
	else \
		echo "$(RED)❌ Virtual environment not found. Run 'make install-deps' first.$(NC)"; \
		exit 1; \
	fi

# Conversion Targets

convert-all: convert-models ## Convert all downloaded models

convert-models: ## Convert all downloaded models to CoreML
	@echo "$(PURPLE)🔄 Converting all models to CoreML...$(NC)"
	@echo "$(YELLOW)===================================_$(NC)"
	@if [ -d "$(WEIGHTS_DIR)" ]; then \
		for weight_file in $(WEIGHTS_DIR)/*.pth; do \
			if [ -f "$$weight_file" ]; then \
				echo "$(BLUE)Converting $$(basename $$weight_file)$(NC)"; \
				source $(ACTIVATE) && python $(SCRIPTS_DIR)/convert_esrgan.py "$$weight_file" "$$(basename $$weight_file .pth)" 4; \
			fi; \
		done; \
	else \
		echo "$(RED)❌ Weights directory not found. Run setup first.$(NC)"; \
		exit 1; \
	fi

convert-esrgan: ## Convert ESRGAN models
	@echo "$(PURPLE)🔄 Converting ESRGAN models...$(NC)"
	@echo "$(YELLOW)=============================_$(NC)"
	@source $(ACTIVATE) && for model in $(WEIGHTS_DIR)/*ESRGAN*.pth; do \
		if [ -f "$$model" ]; then \
			python $(SCRIPTS_DIR)/convert_esrgan.py "$$model" "$$(basename $$model .pth)" 4; \
		fi; \
	done

convert-realesrgan: ## Convert Real-ESRGAN models
	@echo "$(PURPLE)🔄 Converting Real-ESRGAN models...$(NC)"
	@echo "$(YELLOW)=================================_$(NC)"
	@source $(ACTIVATE) && for model in $(WEIGHTS_DIR)/*RealESRGAN*.pth; do \
		if [ -f "$$model" ]; then \
			python $(SCRIPTS_DIR)/convert_realesrgan.py "$$model" "$$(basename $$model .pth)" 4; \
		fi; \
	done

create-mlpackages: ## Create MLPackage format models
	@echo "$(PURPLE)📦 Creating MLPackage format models...$(NC)"
	@echo "$(YELLOW)=====================================$(NC)"
	@if [ -f "create_mlpackage.sh" ]; then \
		./create_mlpackage.sh; \
	else \
		echo "$(RED)❌ create_mlpackage.sh not found$(NC)"; \
		exit 1; \
	fi

# Verification Targets

verify-requirements: ## Verify Python requirements are installed
	@echo "$(PURPLE)🔍 Verifying Python requirements...$(NC)"
	@echo "$(YELLOW)====================================$(NC)"
	@if [ ! -d "$(VENV)" ]; then \
		echo "$(RED)❌ Virtual environment not found$(NC)"; \
		exit 1; \
	fi
	@source $(ACTIVATE) && python -c "import torch, torchvision, coremltools, numpy; print('$(GREEN)✅ All requirements satisfied$(NC)')"

verify-models: ## Verify converted CoreML models
	@echo "$(PURPLE)🔍 Verifying CoreML models...$(NC)"
	@echo "$(YELLOW)=============================$(NC)"
	@if [ ! -d "$(MODELS_DIR)" ]; then \
		echo "$(RED)❌ Models directory not found$(NC)"; \
		exit 1; \
	fi
	@cd $(MODELS_DIR) && for model in *.mlpackage *.mlmodel; do \
		if [ -d "$$model" ] || [ -f "$$model" ]; then \
			echo "$(BLUE)✓ Verifying $$model$(NC)"; \
			cd .. && source $(ACTIVATE) && python -c "import coremltools as ct; m = ct.models.MLModel('$(MODELS_DIR)/$$model'); print('$(GREEN)  ✓ Model loaded successfully$(NC)')" && cd $(MODELS_DIR) || true; \
		fi; \
	done

verify-weights: ## Verify downloaded model weights
	@echo "$(PURPLE)🔍 Verifying model weights...$(NC)"
	@echo "$(YELLOW)=============================$(NC)"
	@if [ ! -d "$(WEIGHTS_DIR)" ]; then \
		echo "$(RED)❌ Weights directory not found$(NC)"; \
		exit 1; \
	fi
	@cd $(WEIGHTS_DIR) && for weight in *.pth; do \
		if [ -f "$$weight" ]; then \
			echo "$(GREEN)✓ Found $$weight ($(shell stat -f%z "$$weight" 2>/dev/null || stat -c%s "$$weight" 2>/dev/null || echo "unknown size") bytes)$(NC)"; \
		fi; \
	done

test-conversion: ## Test model conversion process
	@echo "$(PURPLE)🧪 Testing model conversion...$(NC)"
	@echo "$(YELLOW)=============================$(NC)"
	@source $(ACTIVATE) && python $(SCRIPTS_DIR)/convert_simple_test.py

verify-comparison: ## Comprehensive PyTorch vs CoreML verification
	@echo "$(PURPLE)🔍 Running comprehensive model verification...$(NC)"
	@echo "$(YELLOW)=========================================_$(NC)"
	@source $(ACTIVATE) && python $(SCRIPTS_DIR)/verify_model_comparison.py

verify-pipeline: ## Test inference pipeline functionality
	@echo "$(PURPLE)🧪 Testing inference pipeline...$(NC)"
	@echo "$(YELLOW)=================================$(NC)"
	@if [ -d "test-data" ]; then \
		echo "📁 Found test-data directory with test images"; \
		ls test-data/; \
		source $(ACTIVATE) && python -c "from inference_pipeline import get_optimal_device; print(f'✅ Optimal device: {get_optimal_device()}')"; \
	else \
		echo "⚠️  No test-data directory found"; \
		source $(ACTIVATE) && python -c "from inference_pipeline import get_optimal_device; print(f'✅ Optimal device: {get_optimal_device()}')"; \
	fi

test-with-data: ## Test models using test-data images and save results
	@echo "$(PURPLE)🧪 Testing models with test data...$(NC)"
	@echo "$(YELLOW)=================================$(NC)"
	@if [ -d "test-data" ] && [ -f "upscale_image.py" ]; then \
		mkdir -p test_outputs; \
		echo "$(BLUE)📁 Output will be saved to test_outputs/ directory$(NC)"; \
		for test_image in test-data/*; do \
			if [ -f "$$test_image" ]; then \
				echo "$(BLUE)🧪 Testing with $$(basename $$test_image)$(NC)"; \
				source $(ACTIVATE) && python upscale_image.py "$$test_image" --model RealESRGAN_4x --output "test_outputs/upscaled_$$(basename $$test_image)" || echo "$(YELLOW)⚠️  Test failed for $$test_image$(NC)"; \
			fi; \
		done; \
		echo "$(GREEN)✅ Test images saved to test_outputs/$(NC)"; \
		ls -la test_outputs/; \
	else \
		echo "$(RED)❌ test-data directory or upscale_image.py not found$(NC)"; \
	fi

demo-upscale: ## Create demo upscaled images using all available models
	@echo "$(PURPLE)🎨 Creating demo upscaled images...$(NC)"
	@echo "$(YELLOW)=================================$(NC)"
	@if [ -d "test-data" ] && [ -f "upscale_image.py" ]; then \
		mkdir -p demo_outputs; \
		test_image=$$(ls test-data/* | head -1); \
		if [ -n "$$test_image" ]; then \
			echo "$(BLUE)📸 Using test image: $$(basename $$test_image)$(NC)"; \
			echo "$(BLUE)🎯 Testing with different models:$(NC)"; \
			for model in SRCNN_x2 ESRGAN_2x RealESRGAN_4x; do \
				echo "$(BLUE)   → Upscaling with $$model$(NC)"; \
				source $(ACTIVATE) && python upscale_image.py "$$test_image" --model $$model --output "demo_outputs/$$(basename $$test_image .jpg)_$${model}.png" 2>/dev/null || echo "$(YELLOW)     ⚠️  Model $$model not available$(NC)"; \
			done; \
			echo "$(GREEN)✅ Demo images saved to demo_outputs/$(NC)"; \
			ls -la demo_outputs/; \
		else \
			echo "$(RED)❌ No test images found$(NC)"; \
		fi; \
	else \
		echo "$(RED)❌ test-data directory or upscale_image.py not found$(NC)"; \
	fi

compare-models: ## Compare outputs from different models on the same image
	@echo "$(PURPLE)🔬 Comparing model outputs...$(NC)"
	@echo "$(YELLOW)=============================$(NC)"
	@if [ -d "test-data" ] && [ -f "upscale_image.py" ]; then \
		mkdir -p comparison_outputs; \
		test_image=$$(ls test-data/* | head -1); \
		if [ -n "$$test_image" ]; then \
			echo "$(BLUE)📸 Comparing models using: $$(basename $$test_image)$(NC)"; \
			models="SRCNN_x2 ESRGAN_2x RealESRGAN_4x"; \
			for model in $$models; do \
				echo "$(BLUE)   → Processing with $$model$(NC)"; \
				source $(ACTIVATE) && python upscale_image.py "$$test_image" --model $$model --output "comparison_outputs/$(basename $$test_image .jpg)_$${model}.png" 2>/dev/null && echo "$(GREEN)     ✅ Success$(NC)" || echo "$(YELLOW)     ⚠️  Failed$(NC)"; \
			done; \
			echo ""; \
			echo "$(GREEN)📊 Comparison complete! Files in comparison_outputs/:$(NC)"; \
			ls -la comparison_outputs/; \
			echo ""; \
			echo "$(BLUE)💡 You can now compare the upscaled images manually$(NC)"; \
		else \
			echo "$(RED)❌ No test images found$(NC)"; \
		fi; \
	else \
		echo "$(RED)❌ test-data directory or upscale_image.py not found$(NC)"; \
	fi

compare-pytorch-coreml: ## Compare PyTorch vs CoreML model outputs with visual results
	@echo "$(PURPLE)🔬 PyTorch vs CoreML Visual Comparison$(NC)"
	@echo "$(YELLOW)=====================================$(NC)"
	@if [ -d "test-data" ]; then \
		test_image=$$(ls test-data/* | head -1); \
		if [ -n "$$test_image" ]; then \
			echo "$(BLUE)📸 Using test image: $$(basename $$test_image)$(NC)"; \
			echo "$(BLUE)🧪 Running PyTorch vs CoreML comparison...$(NC)"; \
			source $(ACTIVATE) && python $(SCRIPTS_DIR)/compare_pytorch_coreml.py "$$test_image"; \
		else \
			echo "$(RED)❌ No test images found$(NC)"; \
		fi; \
	else \
		echo "$(RED)❌ test-data directory not found$(NC)"; \
	fi

compare-single-model: ## Compare specific model between PyTorch and CoreML (usage: make compare-single-model MODEL=SRCNN_x2)
	@if [ -z "$(MODEL)" ]; then \
		echo "$(RED)❌ MODEL not specified$(NC)"; \
		echo "$(YELLOW)Usage: make compare-single-model MODEL=SRCNN_x2$(NC)"; \
		exit 1; \
	fi
	@if [ -d "test-data" ]; then \
		test_image=$$(ls test-data/* | head -1); \
		if [ -n "$$test_image" ]; then \
			echo "$(PURPLE)🔬 Comparing $(MODEL) between PyTorch and CoreML$(NC)"; \
			echo "$(YELLOW)=========================================$(NC)"; \
			echo "$(BLUE)📸 Using test image: $$(basename $$test_image)$(NC)"; \
			source $(ACTIVATE) && python $(SCRIPTS_DIR)/compare_pytorch_coreml.py "$$test_image" $(MODEL); \
		else \
			echo "$(RED)❌ No test images found$(NC)"; \
		fi; \
	else \
		echo "$(RED)❌ test-data directory not found$(NC)"; \
	fi

test-models: ## Test converted models with sample data
	@echo "$(PURPLE)🧪 Testing converted models...$(NC)"
	@echo "$(YELLOW)===============================_$(NC)"
	@if [ -f "upscale_image.py" ]; then \
		source $(ACTIVATE) && python upscale_image.py --test; \
	else \
		echo "$(YELLOW)⚠️  upscale_image.py not found, skipping model tests$(NC)"; \
	fi

# Listing Targets

list-models: ## List available converted models
	@echo "$(PURPLE)📋 Available CoreML models:$(NC)"
	@echo "$(YELLOW)===========================$(NC)"
	@if [ -d "$(MODELS_DIR)" ]; then \
		cd $(MODELS_DIR) && for model in *.mlpackage *.mlmodel; do \
			if [ -d "$$model" ] || [ -f "$$model" ]; then \
				size=$$(du -sh "$$model" 2>/dev/null | cut -f1); \
				echo "$(GREEN)✓ $$model ($$size)$(NC)"; \
			fi; \
		done; \
	else \
		echo "$(RED)❌ Models directory not found$(NC)"; \
	fi

list-weights: ## List downloaded model weights
	@echo "$(PURPLE)📋 Downloaded model weights:$(NC)"
	@echo "$(YELLOW)============================_$(NC)"
	@if [ -d "$(WEIGHTS_DIR)" ]; then \
		cd $(WEIGHTS_DIR) && for weight in *.pth; do \
			if [ -f "$$weight" ]; then \
				size=$$(du -sh "$$weight" 2>/dev/null | cut -f1); \
				echo "$(GREEN)✓ $$weight ($$size)$(NC)"; \
			fi; \
		done; \
	else \
		echo "$(RED)❌ Weights directory not found$(NC)"; \
	fi

list-scripts: ## List available conversion scripts
	@echo "$(PURPLE)📋 Available conversion scripts:$(NC)"
	@echo "$(YELLOW)================================_$(NC)"
	@if [ -d "$(SCRIPTS_DIR)" ]; then \
		cd $(SCRIPTS_DIR) && for script in *.py; do \
			if [ -f "$$script" ]; then \
				echo "$(GREEN)✓ $$script$(NC)"; \
			fi; \
		done; \
	else \
		echo "$(RED)❌ Scripts directory not found$(NC)"; \
	fi

# Utility Targets

clean: clean-models ## Clean converted models

clean-models: ## Remove converted CoreML models
	@echo "$(PURPLE)🗑️  Cleaning converted models...$(NC)"
	@if [ -d "$(MODELS_DIR)" ]; then \
		rm -rf $(MODELS_DIR)/*.mlpackage $(MODELS_DIR)/*.mlmodel; \
		echo "$(GREEN)✅ Converted models cleaned$(NC)"; \
	else \
		echo "$(YELLOW)⚠️  Models directory does not exist$(NC)"; \
	fi

clean-weights: ## Remove downloaded model weights
	@echo "$(PURPLE)🗑️  Cleaning model weights...$(NC)"
	@if [ -d "$(WEIGHTS_DIR)" ]; then \
		rm -rf $(WEIGHTS_DIR)/*.pth; \
		echo "$(GREEN)✅ Model weights cleaned$(NC)"; \
	else \
		echo "$(YELLOW)⚠️  Weights directory does not exist$(NC)"; \
	fi

clean-venv: ## Remove Python virtual environment
	@echo "$(PURPLE)🗑️  Cleaning virtual environment...$(NC)"
	@if [ -d "$(VENV)" ]; then \
		rm -rf $(VENV); \
		echo "$(GREEN)✅ Virtual environment cleaned$(NC)"; \
	else \
		echo "$(YELLOW)⚠️  Virtual environment does not exist$(NC)"; \
	fi

clean-all: clean-models clean-weights clean-venv ## Clean all generated files

# Status and Information

status: ## Show project status
	@echo "$(CYAN)Upscaler Pro Models - Status$(NC)"
	@echo "$(YELLOW)============================$(NC)"
	@echo ""
	@echo "$(BLUE)Virtual Environment:$(NC)"
	@if [ -d "$(VENV)" ]; then \
		echo "$(GREEN)✓ Present$(NC)"; \
	else \
		echo "$(RED)✗ Missing$(NC)"; \
	fi
	@echo ""
	@echo "$(BLUE)Model Weights:$(NC)"
	@if [ -d "$(WEIGHTS_DIR)" ]; then \
		count=$$(ls $(WEIGHTS_DIR)/*.pth 2>/dev/null | wc -l); \
		echo "$(GREEN)✓ $$count files found$(NC)"; \
	else \
		echo "$(RED)✗ Directory missing$(NC)"; \
	fi
	@echo ""
	@echo "$(BLUE)CoreML Models:$(NC)"
	@if [ -d "$(MODELS_DIR)" ]; then \
		count=$$(ls $(MODELS_DIR)/*.mlpackage $(MODELS_DIR)/*.mlmodel 2>/dev/null | wc -l); \
		echo "$(GREEN)✓ $$count models found$(NC)"; \
	else \
		echo "$(RED)✗ Directory missing$(NC)"; \
	fi

info: ## Show detailed project information
	@echo "$(CYAN)Upscaler Pro Models - Project Info$(NC)"
	@echo "$(YELLOW)=================================$(NC)"
	@echo ""
	@echo "$(BLUE)Python Version:$(NC) $$($(PYTHON) --version 2>&1)"
	@echo "$(BLUE)Working Directory:$(NC) $$(pwd)"
	@echo "$(BLUE)Models Directory:$(NC) $(MODELS_DIR)"
	@echo "$(BLUE)Weights Directory:$(NC) $(WEIGHTS_DIR)"
	@echo "$(BLUE)Scripts Directory:$(NC) $(SCRIPTS_DIR)"
	@echo ""
	@echo "$(BLUE)Available Models:$(NC)"
	@$(MAKE) list-models
	@echo ""
	@echo "$(BLUE)Available Weights:$(NC)"
	@$(MAKE) list-weights

# Quick commands (for common workflows)

quick: install-deps setup-realesrgan ## Quick setup for Real-ESRGAN only
quick-all: install-deps setup-all ## Quick setup for all models
quick-test: verify-requirements test-conversion ## Quick verification test

quick-verify: verify-requirements verify-comparison ## Quick comprehensive verification

# Advanced targets

batch-convert: ## Batch convert specific model type (usage: make batch-convert MODEL_TYPE=esrgan)
	@if [ -z "$(MODEL_TYPE)" ]; then \
		echo "$(RED)❌ MODEL_TYPE not specified$(NC)"; \
		echo "$(YELLOW)Usage: make batch-convert MODEL_TYPE=esrgan|realesrgan|hat|edsr|rcan|swinir|bsrgan|srgan|waifu2x|srcnn$(NC)"; \
		exit 1; \
	fi
	@echo "$(PURPLE)🔄 Batch converting $(MODEL_TYPE) models...$(NC)"
	@echo "$(YELLOW)======================================_$(NC)"
	@source $(ACTIVATE) && python $(SCRIPTS_DIR)/convert_$(MODEL_TYPE).py

check-health: ## Check system health and dependencies
	@echo "$(PURPLE)🏥 System Health Check$(NC)"
	@echo "$(YELLOW)======================_$(NC)"
	@echo "$(BLUE)Python:$(NC) $$($(PYTHON) --version 2>&1)"
	@echo "$(BLUE)pip:$(NC) $$($(PYTHON) -m pip --version 2>&1)"
	@echo "$(BLUE)Git:$(NC) $$(git --version 2>/dev/null || echo 'Not installed')"
	@echo "$(BLUE)Available disk space:$(NC) $$(df -h . | tail -1 | awk '{print $$4}')"
	@if [ -d "$(VENV)" ]; then \
		echo "$(GREEN)✓ Virtual environment exists$(NC)"; \
		source $(ACTIVATE) && pip list | grep -E "(torch|coremltools|numpy)" || echo "$(YELLOW)⚠️  Some packages may be missing$(NC)"; \
	else \
		echo "$(RED)✗ Virtual environment missing$(NC)"; \
	fi

# Development targets

dev-setup: install-deps ## Development setup with all dependencies
	@echo "$(PURPLE)🛠️  Development environment setup complete$(NC)"
	@echo "$(YELLOW)====================================_$(NC)"

dev-test: dev-setup verify-requirements verify-comparison verify-models verify-pipeline ## Full development test suite
	@echo "$(GREEN)🎉 All development tests passed!$(NC)"

# CI/CD helpers

ci-install: install-deps ## CI/CD: Install dependencies
ci-verify: verify-requirements verify-comparison verify-models ## CI/CD: Verify all requirements and models
ci-test: test-conversion verify-pipeline ## CI/CD: Run conversion tests
ci-build: convert-all create-mlpackages ## CI/CD: Build all models
ci-full: ci-install ci-verify ci-test ci-build ## CI/CD: Full pipeline

# Documentation generation

docs: ## Generate model documentation
	@echo "$(PURPLE)📚 Generating model documentation...$(NC)"
	@echo "$(YELLOW)====================================_$(NC)"
	@$(MAKE) list-models > MODEL_LIST.txt
	@$(MAKE) list-weights >> MODEL_LIST.txt
	@echo "$(GREEN)✓ Documentation generated in MODEL_LIST.txt$(NC)"

# Version information
reconvert-esrgan: ## Reconvert ESRGAN models with flexible input sizes
	@echo "$(PURPLE)🔄 Re-converting ESRGAN models with flexible input sizes...$(NC)"
	@echo "$(YELLOW)==============================================_$(NC)"
	@source $(ACTIVATE) && \
	for weight in weights/*ESRGAN*.pth; do \
		if [ -f "$$weight" ]; then \
			model_name=$$(basename "$$weight" .pth); \
			scale=$$(echo "$$model_name" | grep -o "x[0-9]" | tail -1 | sed 's/x//'); \
			scale=$${scale:-4}; \
			echo "$(BLUE)Reconverting $$model_name (scale: $$scale)$(NC)"; \
			python scripts/convert_esrgan.py "$$weight" "$$model_name" "$$scale"; \
		fi; \
	done

reconvert-realesrgan: ## Reconvert Real-ESRGAN models with flexible input sizes
	@echo "$(PURPLE)🔄 Re-converting Real-ESRGAN models with flexible input sizes...$(NC)"
	@echo "$(YELLOW)===================================================$(NC)"
	@source $(ACTIVATE) && \
	for weight in weights/*RealESRGAN*.pth; do \
		if [ -f "$$weight" ]; then \
			model_name=$$(basename "$$weight" .pth); \
			scale=$$(echo "$$model_name" | grep -o "x[0-9]" | tail -1 | sed 's/x//'); \
			scale=$${scale:-4}; \
			echo "$(BLUE)Reconverting $$model_name (scale: $$scale)$(NC)"; \
			python scripts/convert_realesrgan.py "$$weight" "$$model_name" "$$scale"; \
		fi; \
	done

reconvert-all-flexible: ## Reconvert all ESRGAN/Real-ESRGAN models with flexible sizes
	@echo "$(PURPLE)🔄 Re-converting all ESRGAN models with flexible input sizes...$(NC)"
	@echo "$(YELLOW)=====================================================$(NC)"
	@$(MAKE) reconvert-esrgan
	@$(MAKE) reconvert-realesrgan
	@echo "$(GREEN)✅ All ESRGAN models reconverted with flexible input sizes$(NC)"

version: ## Show version information
	@echo "$(CYAN)Upscaler Pro Models$(NC)"
	@echo "$(YELLOW)Version: 1.0.0$(NC)"
	@echo "$(BLUE)Makefile Version: 1.0.0$(NC)"
	@echo "$(BLUE)Last Updated: $(shell date)$(NC)"
#!/bin/bash
# train_coding.sh - Script to run MLX-optimized training on Apple Silicon specifically for the coding model

# Check for Apple Silicon
if [[ "$(uname -m)" != "arm64" ]]; then
    echo "⚠️ Warning: This script is optimized for Apple Silicon. You're running on $(uname -m)."
    echo "MLX may not work correctly on this architecture."
    read -p "Continue anyway? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Exiting..."
        exit 1
    fi
fi

# Check if MLX virtual environment exists and activate it
if [[ -d "mlx_venv" ]]; then
    echo "🔄 Activating MLX virtual environment..."
    source mlx_venv/bin/activate || { echo "❌ Failed to activate virtual environment"; exit 1; }
    echo "✅ MLX virtual environment activated"
else
    echo "⚠️ No MLX virtual environment found."
    echo "Run ./fine_tuning/fixed_setup_mlx.sh first to set up the MLX environment."
    exit 1
fi

# Ensure MLX is installed
if ! python3 -c "import mlx" &> /dev/null; then
    echo "❌ MLX not found. Please install it first with:"
    echo "./fine_tuning/fixed_setup_mlx.sh"
    exit 1
fi

echo "✅ MLX detected successfully!"

# Set environment variables to optimize MLX and prevent PyTorch/MPS conflicts
export MLX_USE_METAL=1                     # Force MLX to use Metal
export MLX_METAL_PROGRESS=1                # Show Metal compilation progress
export PYTORCH_ENABLE_MPS_FALLBACK=0       # Disable MPS fallback in PyTorch
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 # Prevent MPS from allocating memory
export PYTORCH_ALLOCATOR_CONF=max_split_size_mb:32 # Limit PyTorch memory splits

# Clear any existing CUDA device visibility
export CUDA_VISIBLE_DEVICES=""

# Check that config file exists
if [[ -f "domain_configs.yaml" ]]; then
    CONFIG_ARG="--config domain_configs.yaml"
    echo "📝 Using configuration file: domain_configs.yaml"
else
    echo "❌ Configuration file domain_configs.yaml not found."
    echo "Please ensure the config file exists in the project root."
    exit 1
fi

echo "🍏 Running with MLX optimization for Apple Silicon..."
echo "Environment configured:"
echo "- MLX_USE_METAL=1"
echo "- MLX_METAL_PROGRESS=1"
echo "- PYTORCH_ENABLE_MPS_FALLBACK=0"
echo "- PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0"

# Run the training script with MLX enabled, specifically for the coding domain
echo "▶️ Starting training for CODING domain with MLX..."
python3 training.py --use-mlx $CONFIG_ARG --domains coding "$@"

# Check exit status
if [[ $? -ne 0 ]]; then
    echo "❌ Training failed. Please check the error messages above."
    exit 1
else
    echo "✅ Training completed successfully!"
fi

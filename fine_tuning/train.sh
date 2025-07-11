#!/bin/bash
# train.sh - Script to run MLX-optimized training on Apple Silicon

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

# Check if virtual environment exists and activate it
if [[ -d "venv" ]]; then
    echo "🔄 Activating virtual environment..."
    source venv/bin/activate || { echo "❌ Failed to activate virtual environment"; exit 1; }
    echo "✅ Virtual environment activated"
else
    echo "⚠️ No virtual environment found. Installing dependencies may be required."
    echo "Run ./fine_tuning/setup_mlx.sh first to set up the environment."
    
    read -p "Continue without virtual environment? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Exiting. Please run ./fine_tuning/setup_mlx.sh first."
        exit 1
    fi
fi

# Ensure MLX is installed
if ! python3 -c "import mlx" &> /dev/null; then
    echo "❌ MLX not found. Please install it first with:"
    echo "./fine_tuning/setup_mlx.sh"
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
if [[ "$*" != *"--config"* ]]; then
    if [[ -f "domain_configs.yaml" ]]; then
        CONFIG_ARG="--config domain_configs.yaml"
        echo "📝 Using default configuration file: domain_configs.yaml"
    else
        echo "⚠️ No configuration file specified and default domain_configs.yaml not found."
        echo "Please specify a configuration file with --config."
        exit 1
    fi
else
    CONFIG_ARG=""
fi

echo "🍏 Running with MLX optimization for Apple Silicon..."
echo "Environment configured:"
echo "- MLX_USE_METAL=1"
echo "- MLX_METAL_PROGRESS=1"
echo "- PYTORCH_ENABLE_MPS_FALLBACK=0"
echo "- PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0"

# Run the training script with MLX enabled
echo "▶️ Starting training with MLX..."
python3 fine_tuning/training.py --use-mlx $CONFIG_ARG "$@"

# Check exit status
if [[ $? -ne 0 ]]; then
    echo "❌ Training failed. Please check the error messages above."
    exit 1
else
    echo "✅ Training completed successfully!"
fi

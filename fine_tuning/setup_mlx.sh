#!/bin/bash
# setup_mlx.sh - Script to install MLX and its dependencies for Apple Silicon
# MLX is Apple's Machine Learning framework optimized for Apple Silicon

echo "🍏 Setting up MLX for Apple Silicon..."
echo "This script will install MLX and its dependencies."

# Check if running on macOS
if [ "$(uname)" != "Darwin" ]; then
    echo "❌ Error: MLX is only supported on macOS with Apple Silicon."
    exit 1
fi

# Check for Apple Silicon
if [ "$(uname -m)" != "arm64" ]; then
    echo "❌ Error: MLX requires Apple Silicon (M1/M2/M3) hardware."
    exit 1
fi

# Check for Python 3
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is required but not found."
    echo "Please install Python 3 first."
    exit 1
fi

# Create and activate a virtual environment (optional but recommended)
echo "📦 Creating a Python virtual environment..."
if [ -d "venv" ]; then
    echo "Virtual environment already exists. Using existing environment."
else
    python3 -m venv venv
fi

# Activate the virtual environment
source venv/bin/activate

# Clean up any potential corrupted installation
echo "🧹 Cleaning up any previous installations..."
rm -rf venv/lib/python3.10/site-packages/numpy* || true
rm -rf venv/lib/python3.10/site-packages/torch* || true
rm -rf venv/lib/python3.10/site-packages/mlx* || true

# Install MLX and dependencies - one by one with error checking
echo "📚 Installing MLX and dependencies..."
pip install --upgrade pip || { echo "❌ Failed to upgrade pip"; exit 1; }
pip install mlx || { echo "❌ Failed to install MLX"; exit 1; }

# Test MLX before proceeding
echo "🔍 Testing MLX installation..."
if python3 -c "import mlx; print(f'✅ MLX {mlx.__version__} installed successfully!')" &> /dev/null; then
    python3 -c "import mlx; print(f'✅ MLX {mlx.__version__} installed successfully!')"
else
    echo "❌ MLX installation failed."
    exit 1
fi

# Continue with other dependencies
echo "📦 Installing PyTorch (CPU version for MLX compatibility)..."
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu || { echo "❌ Failed to install PyTorch"; exit 1; }

echo "📦 Installing transformers and related packages..."
pip install transformers || { echo "❌ Failed to install transformers"; exit 1; }
pip install datasets || { echo "❌ Failed to install datasets"; exit 1; }
pip install huggingface_hub || { echo "❌ Failed to install huggingface_hub"; exit 1; }

echo "📦 Installing fine-tuning packages..."
pip install accelerate || { echo "❌ Failed to install accelerate"; exit 1; }
pip install peft || { echo "❌ Failed to install peft"; exit 1; }

echo "📦 Installing utility packages..."
pip install wandb tqdm pyyaml || { echo "❌ Failed to install utility packages"; exit 1; }

# Instructions
echo ""
echo "🎉 MLX setup complete!"
echo ""
echo "To use the virtual environment:"
echo "  source venv/bin/activate"
echo ""
echo "To train with MLX:"
echo "  ./train.sh --config domain_configs.yaml"
echo ""
echo "For optimal performance:"
echo "  - Make sure your Mac is plugged in"
echo "  - Close other GPU-intensive applications"
echo "  - Consider using smaller batch sizes for larger models"
echo ""
echo "Enjoy faster training on your Apple Silicon Mac! 🚀"

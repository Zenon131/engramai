#!/bin/bash
# fixed_setup_mlx.sh - Improved MLX setup script for Apple Silicon

echo "🍏 Setting up MLX for Apple Silicon..."

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

# Create fresh virtual environment
echo "📦 Creating a fresh virtual environment..."
rm -rf mlx_venv
python3 -m venv mlx_venv
source mlx_venv/bin/activate

# Install MLX first
echo "📚 Installing MLX..."
pip install --upgrade pip
pip install mlx

# Test MLX (without version check)
echo "🔍 Testing MLX installation..."
if python3 -c "import mlx; print('✅ MLX installed successfully!')" &> /dev/null; then
    python3 -c "import mlx; print('✅ MLX installed successfully!')"
else
    echo "❌ MLX installation failed."
    exit 1
fi

# Install PyTorch CPU version (to avoid MPS issues)
echo "📦 Installing PyTorch (CPU version)..."
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies one by one
echo "📦 Installing transformers..."
pip install transformers

echo "📦 Installing datasets..."
pip install datasets

echo "📦 Installing huggingface_hub..."
pip install huggingface_hub

echo "📦 Installing accelerate..."
pip install accelerate

echo "📦 Installing peft..."
pip install peft

echo "📦 Installing utility packages..."
pip install wandb tqdm pyyaml

echo ""
echo "🎉 MLX setup complete!"
echo ""
echo "To use the virtual environment:"
echo "  source mlx_venv/bin/activate"
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

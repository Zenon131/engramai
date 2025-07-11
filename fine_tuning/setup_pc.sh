#!/bin/bash
# setup_pc.sh - Script to install dependencies for PC (Linux/Windows) with CUDA/ROCm

echo "🚀 Setting up dependencies for PC with GPU..."
echo "This script will install necessary Python packages using pip."

# Check for Python 3
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is required but not found."
    echo "Please install Python 3 first."
    exit 1
fi

# Create and activate a virtual environment (optional but recommended)
echo "📦 Creating a Python virtual environment..."
if [ -d "venv_pc" ]; then
    echo "Virtual environment 'venv_pc' already exists. Using existing environment."
else
    python3 -m venv venv_pc
fi

# Activate the virtual environment
source venv_pc/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip || { echo "❌ Failed to upgrade pip"; exit 1; }

# Install PyTorch with CUDA support
# Note: For ROCm (AMD GPUs), you might need a different command.
# Consult PyTorch installation guide: https://pytorch.org/get-started/locally/
echo "📦 Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 || { echo "❌ Failed to install PyTorch with CUDA"; exit 1; }

# Install other dependencies
echo "📦 Installing transformers, datasets, peft, accelerate, and other packages..."
pip install transformers datasets peft accelerate huggingface_hub wandb tqdm pyyaml bitsandbytes || { echo "❌ Failed to install core dependencies"; exit 1; }

echo ""
echo "🎉 PC setup complete!"
echo ""
echo "To use the virtual environment:"
echo "  source venv_pc/bin/activate"
echo ""
echo "To run training:"
echo "  ./fine_tuning/train_pc.sh --config domain_configs.yaml"
echo ""
echo "Note for Windows users: Use Git Bash or WSL for running this script."
echo "For AMD (ROCm) GPUs, you may need to adjust the PyTorch installation command."
echo ""
echo "Happy training! 💪"
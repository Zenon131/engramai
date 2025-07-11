#!/bin/bash
# train_pc.sh - Script to run training on PC (Linux/Windows) with GPU

echo "🚀 Running training on PC with GPU..."

# Check if virtual environment exists and activate it
if [[ -d "venv_pc" ]]; then
    echo "🔄 Activating virtual environment 'venv_pc'..."
    source venv_pc/bin/activate || { echo "❌ Failed to activate virtual environment 'venv_pc'"; exit 1; }
    echo "✅ Virtual environment activated"
else
    echo "⚠️ No virtual environment 'venv_pc' found."
    echo "Run ./fine_tuning/setup_pc.sh first to set up the environment."

    read -p "Continue without virtual environment? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Exiting. Please run ./fine_tuning/setup_pc.sh first."
        exit 1
    fi
fi

# Check if PyTorch is installed and can detect a GPU
if ! python3 -c "import torch; print(torch.cuda.is_available() or torch.backends.mps.is_available())" | grep -q "True"; then
    echo "❌ No CUDA or MPS GPU detected by PyTorch."
    echo "Please ensure you have a compatible GPU and drivers installed, and PyTorch is installed correctly."
    echo "Run ./fine_tuning/setup_pc.sh to install dependencies."
    exit 1
fi

echo "✅ PyTorch detected a GPU."

# Clear any MLX/MPS specific environment variables
unset MLX_USE_METAL
unset MLX_METAL_PROGRESS
unset PYTORCH_ENABLE_MPS_FALLBACK
unset PYTORCH_MPS_HIGH_WATERMARK_RATIO
unset PYTORCH_ALLOCATOR_CONF

# Ensure CUDA_VISIBLE_DEVICES is not explicitly set to hide devices unless intended
# If you need to specify which GPU to use, set CUDA_VISIBLE_DEVICES before running this script.
# export CUDA_VISIBLE_DEVICES=0 # Example: use only the first GPU

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

echo "▶️ Starting training..."
# Run the training script
python3 fine_tuning/training.py $CONFIG_ARG "$@"

# Check exit status
if [[ $? -ne 0 ]]; then
    echo "❌ Training failed. Please check the error messages above."
    exit 1
else
    echo "✅ Training completed successfully!"
    echo "Model saved to ./claire_models"
fi
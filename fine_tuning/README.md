# EngramAI Fine-Tuning

This directory contains scripts for fine-tuning Llama 3.2 models for specialized domains.

## Quick Start

1. Create a sample configuration:

   ```bash
   python training.py --create-sample-config
   ```

2. Edit the `domain_configs.yaml` file to customize your domains, datasets, and training parameters.

3. Run training for all domains:

   ```bash
   python training.py
   ```

## Apple Silicon Optimization with MLX

For Apple Silicon (M1/M2/M3) Macs, we recommend using MLX which is Apple's machine learning framework optimized for Apple Silicon. This provides better performance and memory efficiency than PyTorch's MPS backend.

1. Set up MLX using the provided script:

   ```bash
   ./setup_mlx.sh
   ```

2. Run training with MLX acceleration using the simplified script:

   ```bash
   ./train.sh
   ```

   This sets all the necessary environment variables to optimize MLX performance and prevent memory conflicts with PyTorch.

3. Train specific domains with MLX:

   ```bash
   ./train.sh --domains law coding
   ```

4. Use custom configuration file:

   ```bash
   ./train.sh --config my_custom_config.yaml
   ```

### Memory Management with MLX

MLX provides significant memory advantages over PyTorch's MPS backend:

- **Better Memory Efficiency**: MLX is specifically designed for Apple Silicon
- **Reduced Memory Errors**: Prevents "MPS backend out of memory" errors
- **Efficient Memory Allocation**: Uses Apple's Metal framework directly

If you still encounter memory issues with MLX:

- Try reducing batch size in your configuration file
- Use a smaller base model (e.g., Llama-3.2-1B instead of larger models)
- Consider closing other memory-intensive applications

## PC (Linux/Windows) with Dedicated GPU (CUDA/ROCm)

To run training on a PC with an NVIDIA (CUDA) or AMD (ROCm) GPU, use the dedicated setup and training scripts.

1. Set up dependencies using the provided script:

   ```bash
   ./fine_tuning/setup_pc.sh
   ```

   - This script will create a virtual environment `venv_pc` and install necessary packages, including PyTorch with CUDA support.
   - **Note for ROCm users:** You may need to modify the PyTorch installation command in `setup_pc.sh` according to the official PyTorch documentation for ROCm support.
   - **Note for Windows users:** Run this script using Git Bash or Windows Subsystem for Linux (WSL).

2. Activate the virtual environment:

   ```bash
   source venv_pc/bin/activate
   ```

3. Run training using the PC training script:

   ```bash
   ./fine_tuning/train_pc.sh
   ```

   - This script ensures that MLX/MPS specific environment variables are unset, allowing PyTorch to utilize the available CUDA or ROCm device.

4. Train specific domains or use a custom configuration file as described in the "Command Line Arguments" section below.

## Command Line Arguments

- `--config`: Path to domain configuration YAML (default: domain_configs.yaml if it exists)
- `--domains`: Specific domains to train (default: all domains in config)
- `--output-dir`: Output directory for trained models (default: ./claire_models)
- `--hf-token`: HuggingFace token for model access and uploading
- `--hf-username`: HuggingFace username for repo names when pushing models
- `--wandb-project`: Weights & Biases project name for logging
- `--push-to-hf`: Push trained models to HuggingFace
- `--create-ollama`: Create Ollama Modelfiles for local deployment
- `--create-sample-config`: Create a sample configuration file
- `--disable-quantization`: Disable 4-bit quantization (useful for debugging)
- `--check-mps`: Check Apple Silicon MPS availability
- `--use-mlx`: Force use of MLX for Apple Silicon (recommended for M1/M2/M3)

## Examples

Check hardware capabilities on Apple Silicon:

```bash
python training.py --check-mps
```

Use MLX acceleration on Apple Silicon (recommended):

```bash
python training.py --use-mlx
```

Create Ollama model files after training:

```bash
python training.py --create-ollama
```

Push models to HuggingFace:

```bash
python training.py --push-to-hf --hf-username your-username
```

## Configuration Format

The configuration YAML contains domain-specific settings:

```yaml
law:
  description: "Legal domain specialist"
  dataset_paths: ["legal_qa.json", "case_law.jsonl"]
  system_prompt: "You are Claire, a legal AI assistant..."
  max_length: 2048
  learning_rate: 2e-4
  batch_size: 4
  epochs: 3
```

## Dataset Formats

Supported formats:

- HuggingFace datasets
- Local JSON/JSONL files with:
  - Instruction/response pairs
  - Input/output pairs
  - Question/answer pairs
  - Chat message format

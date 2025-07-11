#!/usr/bin/env python3
"""
Claire Domain-Specific Fine-Tuning Script
Fine-tunes Meta Llama 3.2 1B for specialized domains with robust training pipeline.

Supports CUDA, MPS (Apple Silicon), and CPU training with automatic device detection.

Requirements:
pip install torch transformers datasets peft accelerate
pip install huggingface_hub wandb tqdm pyyaml numpy

For Apple Silicon (M1/M2/M3):
- BitsAndBytesConfig is disabled (CUDA-only)
- Uses fp32 training for stability
- Smaller batch sizes recommended
- Consider MLX for better Apple Silicon performance: https://github.com/ml-explore/mlx

For CUDA:
pip install bitsandbytes  # For 4-bit quantization

Optional:
pip install trl  # For SFTTrainer (enhanced training)
"""

import os
import json
import argparse
import logging
import platform
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import yaml

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer,
    DataCollatorForLanguageModeling, EarlyStoppingCallback
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import BitsAndBytesConfig - not available on all platforms
try:
    from transformers import BitsAndBytesConfig
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False
    logger.warning("BitsAndBytesConfig not available. Quantization will be disabled.")

from datasets import Dataset as HFDataset, load_dataset
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from accelerate import Accelerator
from huggingface_hub import HfApi, upload_folder
import wandb
from tqdm import tqdm
import numpy as np

# Try to import MLX - Apple's ML framework for Apple Silicon
MLX_AVAILABLE = False
try:
    import mlx
    import mlx.core as mx
    import mlx.nn as nn_mlx
    MLX_AVAILABLE = True
    logger.info("MLX is available! Will use MLX for Apple Silicon acceleration")
except ImportError:
    logger.warning("MLX not available. Consider installing for better performance on Apple Silicon.")
    logger.warning("Install with: pip install mlx")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DomainConfig:
    """Configuration for domain-specific fine-tuning."""
    name: str
    description: str
    dataset_paths: List[str]
    system_prompt: str
    max_length: int = 2048
    learning_rate: float = 2e-4
    batch_size: int = 4
    epochs: int = 3
    warmup_steps: int = 100
    weight_decay: float = 0.01
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])

class DomainDataset(Dataset):
    """Custom dataset for domain-specific training."""
    
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 2048):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Format conversation
        if "messages" in item:
            # Chat format
            formatted = self.tokenizer.apply_chat_template(
                item["messages"], 
                tokenize=False, 
                add_generation_prompt=False
            )
        else:
            # Instruction format
            formatted = self._format_instruction(item)
            
        # Tokenize
        encoding = self.tokenizer(
            formatted,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": encoding["input_ids"].flatten()
        }
    
    def _format_instruction(self, item: Dict) -> str:
        """Format instruction-response pairs."""
        if "instruction" in item and "response" in item:
            return f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item['response']}"
        elif "input" in item and "output" in item:
            return f"### Input:\n{item['input']}\n\n### Output:\n{item['output']}"
        elif "question" in item and "answer" in item:
            return f"### Question:\n{item['question']}\n\n### Answer:\n{item['answer']}"
        else:
            return str(item)

class ClaireFinetuner:
    """Main fine-tuning class for Claire models."""
    
    def __init__(self, 
                 base_model: str = "meta-llama/Llama-3.2-1B",
                 output_dir: str = "./claire_models",
                 hf_token: Optional[str] = os.getenv("HF_TOKEN"),
                 wandb_project: Optional[str] = None,
                 use_4bit: bool = True):
        
        self.base_model = base_model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.wandb_project = wandb_project
        
        # Detect device and adjust quantization
        self.device = self._detect_device()
        
        # Ensure environment variables are set properly if using MLX
        if self.device == "mlx":
            # These environment variables should be set regardless of how MLX was activated
            os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"  # Disable memory limits
            os.environ["MLX_USE_METAL"] = "1"                       # Force Metal usage
            logger.info("🍏 MLX device selected. Environment configured for Apple Silicon.")
        
        self.use_4bit = use_4bit and self.device == "cuda" and BITSANDBYTES_AVAILABLE
        
        if self.device == "mps" and use_4bit:
            logger.warning("4-bit quantization not supported on MPS. Disabling quantization.")
            logger.info("For better M1/M2/M3 performance, consider using MLX: https://github.com/ml-explore/mlx")
            
            # Safety check - MLX is available but we're using MPS
            if MLX_AVAILABLE:
                logger.warning("⚠️ MLX is available but not being used. Add --use-mlx flag for better performance.")
        
        if use_4bit and not BITSANDBYTES_AVAILABLE:
            logger.warning("BitsAndBytesConfig not available. Disabling quantization.")
            logger.info("Install with: pip install bitsandbytes")
        
        # Initialize accelerator - skip for MLX as we'll handle training directly
        if self.device != "mlx":
            self.accelerator = Accelerator()
        else:
            # For MLX, we'll handle acceleration differently
            self.accelerator = None
            logger.info("Skipping PyTorch accelerator initialization for MLX.")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model, 
            token=self.hf_token,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        logger.info(f"Initialized with base model: {base_model}")
        logger.info(f"Device: {self.device}, Quantization: {self.use_4bit}")
    
    def _detect_device(self) -> str:
        """Detect the best available device with proper prioritization."""
        # Check if MLX is explicitly requested via environment variable
        if os.environ.get("USE_MLX", "").lower() in ("1", "true", "yes") and MLX_AVAILABLE:
            logger.info("MLX usage explicitly requested via USE_MLX environment variable")
            return "mlx"
            
        # Check for CUDA first
        if torch.cuda.is_available():
            logger.info(f"CUDA detected with {torch.cuda.device_count()} GPU(s)")
            return "cuda"
            
        # Check for Apple Silicon options
        if MLX_AVAILABLE:
            # On Apple Silicon, prefer MLX over MPS
            is_apple_silicon = platform.processor() == 'arm' or 'Apple M' in platform.processor()
            if is_apple_silicon:
                logger.info("MLX detected on Apple Silicon - preferred over MPS")
                return "mlx"
            else:
                logger.info("MLX available but not on Apple Silicon - checking other options")
        
        # Check for MPS (Apple Silicon PyTorch backend)
        if torch.backends.mps.is_available():
            logger.info("MPS (Apple Silicon) detected")
            
            # Warn if MLX is also available - MLX would have been better
            if MLX_AVAILABLE:
                logger.warning("⚠️ Both MLX and MPS are available. MLX would provide better performance.")
                logger.warning("Consider using --use-mlx flag for better performance.")
            
            return "mps"
            
        # Fallback to CPU
        logger.info("No GPU detected. Using CPU.")
        return "cpu"
    
    def load_domain_configs(self, config_path: str) -> Dict[str, DomainConfig]:
        """Load domain configurations from YAML file."""
        with open(config_path, 'r') as f:
            configs = yaml.safe_load(f)
            
        domain_configs = {}
        for name, config in configs.items():
            domain_configs[name] = DomainConfig(name=name, **config)
            
        return domain_configs
    
    def prepare_dataset(self, config: DomainConfig) -> Tuple[DomainDataset, DomainDataset]:
        """Load and prepare datasets for training."""
        all_data = []
        
        for dataset_path in config.dataset_paths:
            logger.info(f"Loading dataset: {dataset_path}")
            
            if dataset_path.startswith("http"):
                # HuggingFace dataset
                try:
                    dataset = load_dataset(dataset_path, split="train")
                    all_data.extend(dataset.to_list())
                except Exception as e:
                    logger.error(f"Failed to load HF dataset {dataset_path}: {e}")
                    continue
            else:
                # Local file - resolve the path correctly
                path = Path(dataset_path)
                
                # Check if path is absolute or relative
                if not path.is_absolute():
                    # Check for datasets in multiple locations with priority
                    possible_paths = [
                        Path(dataset_path),  # Current directory
                        Path("fine_tuning/datasets") / path.name,  # fine_tuning/datasets/ directory
                        Path("datasets") / path.name,  # datasets/ directory
                        Path(self.output_dir.parent) / "datasets" / path.name  # Adjacent to output directory
                    ]
                    
                    # Use the first path that exists
                    path = next((p for p in possible_paths if p.exists()), path)
                    
                # Now try to load the file
                if path.exists():
                    if path.suffix == ".json":
                        with open(path, 'r') as f:
                            data = json.load(f)
                            all_data.extend(data if isinstance(data, list) else [data])
                    elif path.suffix == ".jsonl":
                        with open(path, 'r') as f:
                            for line in f:
                                all_data.append(json.loads(line))
                else:
                    logger.warning(f"Dataset path not found: {dataset_path} (tried {path})")
        
        # Add system prompt to each example
        if config.system_prompt:
            for item in all_data:
                if "messages" in item:
                    item["messages"].insert(0, {"role": "system", "content": config.system_prompt})
                elif "instruction" in item:
                    item["instruction"] = f"{config.system_prompt}\n\n{item['instruction']}"
        
        # Split train/validation
        np.random.shuffle(all_data)
        split_idx = int(0.9 * len(all_data))
        train_data = all_data[:split_idx]
        val_data = all_data[split_idx:]
        
        logger.info(f"Prepared {len(train_data)} train, {len(val_data)} validation samples")
        
        return (
            DomainDataset(train_data, self.tokenizer, config.max_length),
            DomainDataset(val_data, self.tokenizer, config.max_length)
        )
    
    def setup_model(self, config: DomainConfig) -> nn.Module:
        """Setup model with LoRA configuration."""
        # Handle MLX separately - it has its own workflow
        if self.device == "mlx":
            logger.info("Setting up MLX model for Apple Silicon...")
            # Force CPU to avoid MPS memory issues during load
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"  # Disable MPS fallback
            
            # MLX requires we use CPU for initial loading
            # Turn off CUDA/MPS temporarily during model loading
            original_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Hide CUDA devices
            
            logger.info("Loading model on CPU before MLX conversion...")
            
            # Use float32 for maximum compatibility during transfer
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                torch_dtype=torch.float32,
                device_map=None,  # Force CPU
                low_cpu_mem_usage=True,  # Optimize memory during loading
                trust_remote_code=True,
                token=self.hf_token
            )
            
            # Apply lightweight LoRA for efficient training
            target_modules = ["q_proj", "v_proj"]  # Simplified for MLX
            
            lora_config = LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=target_modules,
                task_type=TaskType.CAUSAL_LM,
                bias="none"
            )
            
            # Apply LoRA
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
            
            # Restore original CUDA visibility
            if original_cuda_visible_devices:
                os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible_devices
                
            logger.info("Model prepared for MLX acceleration!")
            return model
            
        # Regular PyTorch setup for other devices
        # Configure quantization and device mapping based on available hardware
        bnb_config = None
        device_map = None
        torch_dtype = torch.bfloat16
        
        if self.device == "cuda" and self.use_4bit and BITSANDBYTES_AVAILABLE:
            # 4-bit quantization for CUDA only
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            device_map = "auto"
        elif self.device == "mps":
            # MPS-specific configuration
            device_map = None  # Let MPS handle device placement
            torch_dtype = torch.float16  # MPS works better with float16
        elif self.device == "cuda":
            # CUDA without quantization
            device_map = "auto"
        else:
            # CPU fallback
            device_map = None
            torch_dtype = torch.float32
        
        # Load base model
        model_kwargs = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": True,
            "token": self.hf_token
        }
        
        if bnb_config is not None:
            model_kwargs["quantization_config"] = bnb_config
        
        if device_map is not None:
            model_kwargs["device_map"] = device_map
        
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            **model_kwargs
        )
        
        # Move to MPS device if needed
        if self.device == "mps":
            model = model.to("mps")
        
        # Prepare for k-bit training if using quantization
        if self.use_4bit and self.device == "cuda" and BITSANDBYTES_AVAILABLE:
            model = prepare_model_for_kbit_training(model)
        
        # LoRA configuration - adjust for device limitations
        target_modules = config.target_modules
        if self.device == "mps":
            # Some modules might not work well with MPS, use conservative targets
            target_modules = ["q_proj", "v_proj"]
        
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=target_modules,
            task_type=TaskType.CAUSAL_LM,
            bias="none"
        )
        
        # Apply LoRA
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        return model
    
    def train_domain_model(self, domain_name: str, config: DomainConfig) -> str:
        """Train a model for specific domain."""
        logger.info(f"Starting training for domain: {domain_name}")
        
        # If using MLX, reconfirm environment settings
        if self.device == "mlx":
            # Double-check that PyTorch doesn't try to use MPS
            if torch.backends.mps.is_available():
                logger.info("Ensuring PyTorch doesn't use MPS when MLX is active")
                # These help prevent MPS from being used when MLX is selected
                os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
                os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
        
        # Setup wandb
        if self.wandb_project:
            wandb.init(
                project=self.wandb_project,
                name=f"claire-{domain_name}",
                config=config.__dict__
            )
        
        # Prepare datasets
        train_dataset, val_dataset = self.prepare_dataset(config)
        
        # Setup model
        model = self.setup_model(config)
        
        # Output directory for this domain
        domain_output_dir = self.output_dir / f"claire-{domain_name}"
        domain_output_dir.mkdir(exist_ok=True)
        
        # Training arguments - adjust for device capabilities
        training_kwargs = {
            "output_dir": str(domain_output_dir),
            "num_train_epochs": config.epochs,
            "per_device_train_batch_size": config.batch_size,
            "per_device_eval_batch_size": config.batch_size,
            "gradient_accumulation_steps": 16,
            "warmup_steps": config.warmup_steps,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "logging_steps": 10,
            "eval_steps": 500,
            "save_steps": 500,
            "eval_strategy": "steps",
            "save_strategy": "steps",
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
            "report_to": "wandb" if self.wandb_project else None,
            "run_name": f"claire-{domain_name}",
            "remove_unused_columns": False,
            "dataloader_pin_memory": False,
            "optim": "adamw_torch",
            "lr_scheduler_type": "cosine",
            "save_total_limit": 3,
        }
        
        # Device-specific training configurations
        if self.device == "mlx":
            training_kwargs.update({
                "fp16": False,  # MLX manages precision internally
                "bf16": False,
                "dataloader_num_workers": 0,  # Let MLX manage parallelism
                "gradient_accumulation_steps": 8,  # Good balance for MLX
                "per_device_train_batch_size": config.batch_size,  # MLX can handle full batch size
                "per_device_eval_batch_size": config.batch_size,
                "optim": "adamw_torch",  # MLX will optimize this internally
            })
            logger.info("🍎 MLX acceleration detected! Using optimized settings for Apple Silicon.")
        elif self.device == "cuda":
            training_kwargs.update({
                "bf16": True,
                "dataloader_num_workers": 4,
            })
        elif self.device == "mps":
            training_kwargs.update({
                "fp16": False,  # MPS can be unstable with fp16
                "bf16": False,  # MPS doesn't support bf16 well
                "dataloader_num_workers": 0,  # MPS works better with 0 workers
                "gradient_accumulation_steps": 8,  # Increase to handle smaller batches
                "per_device_train_batch_size": max(1, config.batch_size // 2),  # Smaller batches for MPS
                "per_device_eval_batch_size": max(1, config.batch_size // 2),
            })
            logger.warning("MPS training detected. Using conservative settings for stability.")
        else:
            # CPU fallback
            training_kwargs.update({
                "fp16": False,
                "bf16": False,
                "dataloader_num_workers": 0,
                "per_device_train_batch_size": 1,
                "per_device_eval_batch_size": 1,
                "gradient_accumulation_steps": 16,
            })
        
        training_args = TrainingArguments(**training_kwargs)
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )
        
        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Train
        trainer.train()
        
        # Save model
        trainer.save_model()
        self.tokenizer.save_pretrained(domain_output_dir)
        
        # Save config
        with open(domain_output_dir / "domain_config.json", 'w') as f:
            json.dump(config.__dict__, f, indent=2)
        
        logger.info(f"Training completed for {domain_name}")
        
        if self.wandb_project:
            wandb.finish()
            
        return str(domain_output_dir)
    
    def push_to_huggingface(self, model_path: str, repo_name: str, private: bool = False):
        """Push trained model to HuggingFace Hub."""
        if not self.hf_token:
            logger.error("HuggingFace token not provided")
            return
        
        try:
            api = HfApi(token=self.hf_token)
            
            # Create repo if it doesn't exist
            try:
                api.create_repo(repo_name, private=private, exist_ok=True)
            except Exception as e:
                logger.warning(f"Repo creation warning: {e}")
            
            # Upload model
            upload_folder(
                folder_path=model_path,
                repo_id=repo_name,
                repo_type="model",
                token=self.hf_token
            )
            
            logger.info(f"Model pushed to HuggingFace: {repo_name}")
            
        except Exception as e:
            logger.error(f"Failed to push to HuggingFace: {e}")
    
    def create_ollama_modelfile(self, model_path: str, domain_name: str, config: DomainConfig):
        """Create Ollama Modelfile for local deployment."""
        # Note: Ollama expects GGUF format, this creates a template
        modelfile_content = f"""# Ollama Modelfile for Claire-{domain_name}
# Note: You'll need to convert the model to GGUF format first
# Use: python -m llama_cpp.convert --outtype f16 {model_path}

FROM ./claire-{domain_name}.gguf

TEMPLATE \"\"\"<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{config.system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{{{{ .Prompt }}}}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

\"\"\"

PARAMETER stop "<|eot_id|>"
PARAMETER stop "<|end_of_text|>"
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
"""
        
        modelfile_path = Path(model_path) / "Modelfile"
        with open(modelfile_path, 'w') as f:
            f.write(modelfile_content)
        
        logger.info(f"Ollama Modelfile created at: {modelfile_path}")
        
        # Create conversion instructions
        instructions = f"""
To use this model with Ollama:

1. Install Ollama: https://ollama.ai
2. Convert model to GGUF format:
   pip install llama-cpp-python[server]
   python -m llama_cpp.convert --outtype f16 {model_path}
3. Move the .gguf file to the model directory
4. Run: ollama create claire-{domain_name} -f {modelfile_path}
5. Run: ollama run claire-{domain_name}

Alternative: Use HuggingFace hub integration:
1. Push model to HF hub first
2. Use: ollama pull your-username/claire-{domain_name}

Example usage:
ollama run claire-{domain_name} "Your domain-specific question here"
"""
        
        with open(Path(model_path) / "ollama_instructions.txt", 'w') as f:
            f.write(instructions)
    
    def print_device_info(self):
        """Print device information and optimization tips."""
        logger.info(f"Device detected: {self.device}")
        
        if self.device == "mlx":
            logger.info("🍏 Apple Silicon with MLX detected!")
            logger.info("Tips for optimal performance with MLX:")
            logger.info("  - MLX is optimized specifically for Apple Silicon")
            logger.info("  - Training will use MLX's native optimizations")
            logger.info("  - Memory management is handled efficiently by MLX")
            logger.info("  - You can use larger batch sizes than with MPS")
            
        elif self.device == "mps":
            logger.info("🍎 Apple Silicon (MPS) detected!")
            logger.info("Tips for optimal performance:")
            logger.info("  - Training will use fp32 for stability")
            logger.info("  - Quantization is disabled (not supported on MPS)")
            logger.info("  - Consider using smaller batch sizes")
            logger.info("  - For faster training, consider MLX: https://github.com/ml-explore/mlx")
            
        elif self.device == "cuda":
            logger.info("🚀 CUDA GPU detected!")
            logger.info("  - Using 4-bit quantization for memory efficiency")
            logger.info("  - bf16 training enabled for speed")
            logger.info("  - Also supports ROCm for AMD GPUs (ensure correct PyTorch installation)")

        elif self.device == "cpu":
            logger.info("💻 CPU training (slow but works)")
            logger.info("  - Consider using smaller models or cloud GPU")
            
        logger.info(f"Memory usage will be {'low' if self.use_4bit or self.device == 'mlx' else 'high'}")
        
    @staticmethod
    def check_mps_requirements():
        """Check MPS requirements and provide installation help."""
        # Check MLX first - the preferred option for Apple Silicon
        if MLX_AVAILABLE:
            logger.info("✅ MLX is available and ready!")
            logger.info("MLX is the preferred option for Apple Silicon - will use it for best performance.")
        
        # Check MPS as fallback
        if not torch.backends.mps.is_available():
            if torch.backends.mps.is_built():
                logger.warning("MPS not available because the current PyTorch install was not built with MPS enabled.")
            else:
                logger.warning("MPS not available because PyTorch was not built with MPS support.")
                logger.info("To install PyTorch with MPS support:")
                logger.info("  pip3 install torch torchvision torchaudio")
        else:
            logger.info("✅ MPS is available as fallback!")
            if not MLX_AVAILABLE:
                logger.info("For better performance on Apple Silicon, consider installing MLX:")
                logger.info("  pip install mlx")
            return True
        
        if not MLX_AVAILABLE and not torch.backends.mps.is_available():
            logger.warning("Neither MLX nor MPS are available. Will use CPU (slow).")
            return False
        
        return True

def create_sample_config():
    """Create sample domain configuration file."""
    # Ensure the fine_tuning/datasets directory exists
    datasets_dir = Path("datasets")
    if not datasets_dir.exists():
        logger.warning(f"Directory {datasets_dir} does not exist. Creating it...")
        datasets_dir.mkdir(parents=True, exist_ok=True)
    else:
        logger.info(f"Using existing directory: {datasets_dir}")
    
    sample_config = {
        "coding": {
            "description": "Programming and software development specialist",
            "dataset_paths": ["code_qa.json", "programming_tutorials.jsonl"],  # Will look in fine_tuning/datasets/
            "system_prompt": "You are Claire, a programming AI assistant skilled in multiple languages including Python, JavaScript, Java, and Swift. Provide clean, efficient code with explanations.",
            "max_length": 2048,
            "learning_rate": 2e-4,
            "batch_size": 4,
            "epochs": 3
        },
        "law": {
            "description": "Legal domain specialist",
            "dataset_paths": ["legal_qa.json", "case_law.jsonl"],
            "system_prompt": "You are Claire, a legal AI assistant specialized in providing accurate legal information and analysis. Always remind users to consult with qualified legal professionals for specific legal advice.",
            "max_length": 2048,
            "learning_rate": 2e-4,
            "batch_size": 4,
            "epochs": 3
        },
        "biology": {
            "description": "Biology and life sciences specialist", 
            "dataset_paths": ["biology_textbook.json", "research_papers.jsonl"],
            "system_prompt": "You are Claire, a biology AI assistant with expertise in life sciences, genetics, ecology, and molecular biology. Provide scientifically accurate information with proper citations when possible.",
            "max_length": 2048,
            "learning_rate": 1.5e-4,
            "batch_size": 4,
            "epochs": 4
        }
    }
    
    with open("domain_configs.yaml", 'w') as f:
        yaml.dump(sample_config, f, default_flow_style=False, indent=2)
    
    # Check which dataset files already exist
    existing_files = []
    missing_files = []
    
    for domain, config in sample_config.items():
        for dataset_name in config["dataset_paths"]:
            file_path = datasets_dir / dataset_name
            if file_path.exists():
                existing_files.append(str(file_path))
                logger.info(f"✓ Found existing dataset: {file_path}")
            else:
                missing_files.append(str(file_path))
                logger.warning(f"✗ Missing dataset: {file_path}")
    
    # Only create placeholder files for missing datasets
    if missing_files:
        logger.info("Creating placeholder files for missing datasets...")
        
        for domain, config in sample_config.items():
            for dataset_name in config["dataset_paths"]:
                file_path = datasets_dir / dataset_name
                
                # Only create if file doesn't exist
                if not file_path.exists():
                    sample_data = []
                    if file_path.suffix == ".json":
                        # Create sample dataset with 3 examples
                        if "legal" in file_path.name:
                            sample_data = [
                                {"instruction": "What is a contract?", "response": "A contract is a legally binding agreement between two or more parties."},
                                {"instruction": "What is tort law?", "response": "Tort law deals with civil wrongs and injuries, providing remedies and compensation."}
                            ]
                        elif "biology" in file_path.name:
                            sample_data = [
                                {"instruction": "What is DNA?", "response": "DNA (deoxyribonucleic acid) is a molecule composed of two polynucleotide chains that stores genetic information."},
                                {"instruction": "What is photosynthesis?", "response": "Photosynthesis is the process used by plants to convert light energy into chemical energy."}
                            ]
                        else:  # coding - will use existing files you created
                            sample_data = [
                                {"instruction": "How do you sort a list in Python?", "response": "You can sort a list in Python using the sorted() function or the list.sort() method."},
                                {"instruction": "What is a closure in JavaScript?", "response": "A closure is a function that has access to its own scope, the outer function's variables, and global variables."}
                            ]
                        
                        with open(file_path, 'w', encoding='utf-8') as f:
                            json.dump(sample_data, f, indent=2, ensure_ascii=False)
                            logger.info(f"Created placeholder: {file_path}")
                            
                    elif file_path.suffix == ".jsonl":
                        # Create sample JSONL file
                        with open(file_path, 'w', encoding='utf-8') as f:
                            if "case_law" in file_path.name:
                                f.write(json.dumps({"instruction": "Summarize this case", "response": "This is a sample case summary."}) + "\n")
                                f.write(json.dumps({"instruction": "Explain legal precedent", "response": "Legal precedent refers to previous court decisions that guide future rulings."}) + "\n")
                            elif "research" in file_path.name:
                                f.write(json.dumps({"instruction": "Explain CRISPR", "response": "CRISPR is a gene editing technology."}) + "\n")
                                f.write(json.dumps({"instruction": "Describe cellular respiration", "response": "Cellular respiration is the process cells use to produce energy."}) + "\n")
                            else:  # programming - will use existing files
                                f.write(json.dumps({"instruction": "Write a Python function to check if a string is a palindrome", "response": "def is_palindrome(s):\n    s = s.lower().replace(' ', '')\n    return s == s[::-1]"}) + "\n")
                                f.write(json.dumps({"instruction": "Explain big O notation", "response": "Big O notation is used to describe the performance of an algorithm."}) + "\n")
                        logger.info(f"Created placeholder: {file_path}")
    
    print("\n" + "="*60)
    print("Configuration file created: domain_configs.yaml")
    print(f"Using datasets directory: {datasets_dir}")
    
    if existing_files:
        print(f"\n✓ Found {len(existing_files)} existing dataset files:")
        for file in existing_files:
            print(f"  - {file}")
    
    if missing_files:
        print(f"\n📝 Created {len(missing_files)} placeholder files:")
        for file in missing_files:
            print(f"  - {file}")
        print("\nReplace placeholder files with your actual training data.")
    
    print("\nTo start training:")
    print("  python claire_finetune.py --domains coding")
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Claire models for specific domains")
    parser.add_argument("--config", type=str, help="Path to domain configuration YAML")
    parser.add_argument("--domains", nargs="+", help="Specific domains to train (default: all)")
    parser.add_argument("--output-dir", type=str, default="./claire_models", help="Output directory")
    parser.add_argument("--hf-token", type=str, help="HuggingFace token")
    parser.add_argument("--hf-username", type=str, help="HuggingFace username for repo names")
    parser.add_argument("--wandb-project", type=str, help="Weights & Biases project name")
    parser.add_argument("--push-to-hf", action="store_true", help="Push models to HuggingFace")
    parser.add_argument("--create-ollama", action="store_true", help="Create Ollama Modelfiles")
    parser.add_argument("--create-sample-config", action="store_true", help="Create sample config file")
    parser.add_argument("--disable-quantization", action="store_true", help="Disable 4-bit quantization")
    parser.add_argument("--check-mps", action="store_true", help="Check MPS availability")
    parser.add_argument("--use-mlx", action="store_true", help="Force use MLX for Apple Silicon (recommended)")
    
    args = parser.parse_args()
    
    if args.create_sample_config:
        create_sample_config()
        return
    
    if args.check_mps:
        ClaireFinetuner.check_mps_requirements()
        return
    
    # Check for MLX if user requested it
    if args.use_mlx and not MLX_AVAILABLE:
        logger.warning("--use-mlx flag was passed but MLX is not available.")
        logger.warning("Please install MLX with: pip install mlx")
        logger.warning("Will continue with the best available device...")

    # Initialize fine-tuner
    finetuner = ClaireFinetuner(
        output_dir=args.output_dir,
        hf_token=args.hf_token,
        wandb_project=args.wandb_project,
        use_4bit=not args.disable_quantization
    )
    
    # Force MLX if requested and available
    if args.use_mlx and MLX_AVAILABLE:
        finetuner.device = "mlx"
        logger.info("Forcing MLX use as requested")
    
    # Print device info and tips
    finetuner.print_device_info()
    
    # Ensure datasets directory exists
    datasets_dir = Path("fine_tuning/datasets")
    datasets_dir.mkdir(parents=True, exist_ok=True)
    
    # Handle missing config
    if not args.config:
        if os.path.exists("domain_configs.yaml"):
            logger.info("Using default configuration file: domain_configs.yaml")
            args.config = "domain_configs.yaml"
        else:
            logger.info("No configuration file specified. Creating a sample config.")
            create_sample_config()
            logger.info("Please edit domain_configs.yaml and run the script again with specific domains.")
            return
    
    # Load domain configurations
    domain_configs = finetuner.load_domain_configs(args.config)
    
    # Filter domains if specified
    if args.domains:
        domain_configs = {k: v for k, v in domain_configs.items() if k in args.domains}
    
    # Train each domain
    trained_models = {}
    for domain_name, config in domain_configs.items():
        try:
            # Adjust config for MPS if needed
            if finetuner.device == "mps":
                config.batch_size = max(1, config.batch_size // 2)
                logger.info(f"Adjusted batch size for MPS: {config.batch_size}")
            
            model_path = finetuner.train_domain_model(domain_name, config)
            trained_models[domain_name] = model_path
            
            # Create Ollama Modelfile
            if args.create_ollama:
                finetuner.create_ollama_modelfile(model_path, domain_name, config)
            
            # Push to HuggingFace
            if args.push_to_hf and args.hf_username:
                repo_name = f"{args.hf_username}/claire-{domain_name}"
                finetuner.push_to_huggingface(model_path, repo_name)
                
        except Exception as e:
            logger.error(f"Failed to train {domain_name}: {e}")
            continue
    
    # Summary
    logger.info("Training completed!")
    logger.info(f"Device used: {finetuner.device}")
    logger.info(f"Trained models: {list(trained_models.keys())}")
    for domain, path in trained_models.items():
        logger.info(f"  {domain}: {path}")
    
    # Device-specific tips
    if finetuner.device == "mps":
        logger.info("\n🍎 MPS Training Tips:")
        logger.info("  - Training on Apple Silicon is experimental")
        logger.info("  - For production use, consider cloud GPU or MLX")
        logger.info("  - Models trained on MPS should work on any device")

if __name__ == "__main__":
    main()
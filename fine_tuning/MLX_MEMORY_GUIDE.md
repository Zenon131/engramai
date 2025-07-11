# Memory Optimization for EngramAI on Apple Silicon

## Summary of Changes

To address the "MPS backend out of memory" errors on Apple Silicon, we've implemented comprehensive changes that focus on using MLX (Apple's Machine Learning framework) instead of the standard PyTorch MPS backend.

## Files Created/Modified

1. **train.sh** - New script to run training with MLX and optimal environment settings
   - Sets critical environment variables to prevent memory conflicts
   - Ensures PyTorch doesn't interfere with MLX memory allocation

2. **setup_mlx.sh** - New script to install MLX and its dependencies
   - Verifies Apple Silicon hardware
   - Creates virtual environment
   - Installs MLX and required Python packages
   - Tests the installation

3. **training.py** - Improved with better MLX support
   - Enhanced `_detect_device()` method for smarter device selection
   - Modified `setup_model()` to force CPU loading before MLX conversion
   - Added environment variable controls to prevent PyTorch MPS interference

4. **README.md** - Updated with improved MLX usage instructions
   - Added new sections explaining memory management
   - Simplified usage instructions for train.sh and setup_mlx.sh

## Key Memory Optimizations

1. **Environment Variables**
   - `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`: Prevents MPS from allocating memory
   - `PYTORCH_ENABLE_MPS_FALLBACK=0`: Disables MPS fallback when MLX is active
   - `MLX_USE_METAL=1`: Ensures MLX uses Metal for GPU acceleration
   - `PYTORCH_ALLOCATOR_CONF=max_split_size_mb:32`: Limits PyTorch memory splits

2. **Device Isolation**
   - Temporarily hide CUDA devices during model loading
   - Force CPU loading for initial model creation
   - Prevent MPS from being used when MLX is specified

3. **Efficient Model Loading**
   - Use `low_cpu_mem_usage=True` for PyTorch model loading
   - Proper use of `device_map=None` to prevent automatic device assignment

## Usage Instructions

1. Install MLX:

   ```bash
   ./setup_mlx.sh
   ```

2. Run training with MLX:

   ```bash
   ./train.sh --config domain_configs.yaml
   ```

## Troubleshooting

If memory issues persist:

1. Reduce batch size in your domain configuration
2. Try a smaller model variant
3. Check for other memory-intensive applications running
4. Verify that MLX is correctly installed and being used

## References

- [MLX Documentation](https://github.com/ml-explore/mlx)
- [Apple Metal Performance Shaders](https://developer.apple.com/documentation/metalperformanceshaders)

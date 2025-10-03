# ColabLink Examples

Example scripts demonstrating how to use ColabLink.

## Prerequisites

1. Setup Colab runtime (see main README)
2. Connect from local machine:
   ```bash
   colablink init '...'
   ```

## Examples

### 1. MNIST Training (`train_mnist.py`)

Train a simple CNN on MNIST dataset using Colab GPU.

**Run from local terminal:**
```bash
colablink exec python examples/train_mnist.py
```

**What happens:**
- Script runs on Colab GPU
- Downloads MNIST dataset
- Trains for 3 epochs
- Shows real-time progress in your local terminal
- Saves model locally

**Output:**
```
Using device: cuda
GPU: Tesla T4
Memory: 15.36 GB

Loading MNIST dataset...
Creating model...

Starting training...
Epoch 1/3, Batch 0/938, Loss: 2.3045, Acc: 10.94%
Epoch 1/3, Batch 100/938, Loss: 0.3421, Acc: 87.32%
...
Model saved to mnist_model.pt
Training complete!
```

### 2. GPU Information (`gpu_info.py`)

Check GPU availability and specifications.

```bash
colablink exec python examples/gpu_info.py
```

### 3. Jupyter Notebook

Run Jupyter on Colab, access from local browser:

```bash
# Terminal 1: Start Jupyter on Colab
colablink exec "jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser"

# Terminal 2: Forward port
colablink forward 8888

# Browser: Open http://localhost:8888
```

### 4. TensorBoard

Train with TensorBoard logging:

```bash
# Terminal 1: Run training with TensorBoard logging
colablink exec python examples/train_with_tensorboard.py

# Terminal 2: Start TensorBoard
colablink exec "tensorboard --logdir=./runs --port=6006"

# Terminal 3: Forward port
colablink forward 6006

# Browser: Open http://localhost:6006
```

### 5. Interactive Python

Start interactive Python session on Colab:

```bash
colablink shell

# Now you're in Colab's shell
colablink@colab:~$ python
>>> import torch
>>> torch.cuda.is_available()
True
>>> torch.cuda.get_device_name(0)
'Tesla T4'
```

## Tips

### Installing Requirements

Create `requirements.txt` locally:
```txt
torch
torchvision
tensorboard
transformers
```

Install on Colab:
```bash
colablink exec pip install -r requirements.txt
```

### File Locations

All files are read from and written to your local machine:
```python
# Reads from local ./data/
dataset = load_data('./data/dataset.csv')

# Writes to local ./models/
torch.save(model, './models/model.pt')
```

### Environment Variables

Set environment variables on Colab:
```bash
colablink exec "export CUDA_VISIBLE_DEVICES=0"
colablink exec "export TRANSFORMERS_CACHE=~/cache"
```

### Checking GPU Usage

Monitor GPU in real-time:
```bash
colablink exec "watch -n 1 nvidia-smi"
```

Or one-time check:
```bash
colablink exec nvidia-smi
```

## Troubleshooting

### Import Errors

Install missing packages on Colab:
```bash
colablink exec pip install <package_name>
```

### Out of Memory

Check GPU memory:
```bash
colablink exec nvidia-smi
```

Reduce batch size in training scripts.

### Connection Lost

If Colab disconnects, rerun setup cell and reconnect:
```bash
colablink init '...'
```

## More Examples

Check out these real-world use cases:
- Fine-tuning transformers
- Training GANs
- Running Stable Diffusion
- Large dataset processing
- Multi-GPU training

See full examples in the `examples/` directory.

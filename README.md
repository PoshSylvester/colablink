# ColabLink

[![license](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
![python version](https://img.shields.io/badge/python-3.6%2B-blue?logo=python)
[![PyPI version](https://img.shields.io/badge/pypi-v1.0.0-blue)](https://pypi.org/project/colablink/)

Connect your local IDE to Google Colab's GPU runtime. Work entirely locally with your files and terminal while executing on Colab's powerful GPUs.

## Why ColabLink?

- **Automatic Bidirectional Sync** - Files sync instantly in both directions
- **Local Development** - Use your favorite IDE (VS Code, Cursor, PyCharm)
- **Real-time Streaming** - See output as it happens, no buffering
- **Free GPU Access** - Tesla T4/P100/V100 GPUs from Google Colab
- **Zero Config** - One-line setup, works in minutes

```
Local Machine (Your IDE)  ←→  SSH Tunnel  ←→  Google Colab (Free GPU)
    Files & Terminal                             Execute & Stream Results
```

## Quick Start

### 1. Install

```bash
pip install colablink
```

On first run, ColabLink will guide you through installing any missing system dependencies (sshpass, sshfs).

### 2. Setup Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PoshSylvester/colablink/blob/main/colab_setup.ipynb)

```python
!pip install colablink

from colablink import ColabRuntime
runtime = ColabRuntime(password="your_password")
runtime.setup()
runtime.keep_alive()
```

### 3. Connect & Use

Copy the connection command from Colab output and run it:

```bash
colablink init '{"host": "...", "port": "...", "password": "..."}'

# Optional: Specify custom mount directory
colablink init '{"host": "...", "port": "...", "password": "..."}' --mount-dir /custom/path
```

Your files now sync automatically! Colab files appear in `./colab-workspace/` by default.

```bash
# Run commands on Colab GPU:
colablink exec nvidia-smi
colablink exec python train.py

# Generated files appear in: ./colab-workspace/
ls colab-workspace/  # See your models, outputs, etc.
```

---

## Command Reference

### Core Commands

| Command | Purpose | Example |
|---------|---------|---------|
| `init` | Connect to Colab | `colablink init '{...}'` |
| `exec` | Run command on GPU | `colablink exec python train.py` |
| `shell` | Interactive session | `colablink shell` |
| `status` | Check connection | `colablink status` |
| `disconnect` | Close connection | `colablink disconnect` |

### File Management (Manual Mode)

| Command | Purpose | Example |
|---------|---------|---------|
| `upload` | Push to Colab | `colablink upload data/` |
| `download` | Pull from Colab | `colablink download /content/model.pt` |
| `sync` | Sync directory | `colablink sync` |

### Utilities

| Command | Purpose | Example |
|---------|---------|---------|
| `forward` | Port forwarding | `colablink forward 8888` |

### Detailed Usage

<details>
<summary><b>colablink init</b> - Initialize connection</summary>

```bash
colablink init '{"host": "0.tcp.ngrok.io", "port": "12345", "password": "xxx"}'

# With custom mount directory
colablink init '{"host": "...", "port": "...", "password": "..."}' --mount-dir ./my-colab-files
```

Establishes SSH connection, sets up bidirectional file sync, and saves configuration.
Colab files mount to `./colab-workspace/` by default (customizable with `--mount-dir`).

</details>

<details>
<summary><b>colablink exec</b> - Execute commands</summary>

```bash
colablink exec nvidia-smi
colablink exec python train.py
colablink exec "pip install torch torchvision"
```

Features: Real-time streaming, GPU environment auto-configured, keyboard interrupt support.

</details>

<details>
<summary><b>colablink upload / download</b> - File transfer</summary>

```bash
# Upload (auto-detects directories)
colablink upload train.py
colablink upload data/                    # Auto-recursive
colablink upload model.py -d /content/models/

# Download (auto-detects directories)
colablink download /content/model.pt
colablink download /content/output/       # Auto-recursive
colablink download /content/data/ -d ./local_data/
```

Smart detection: Automatically handles files vs directories.

</details>

<details>
<summary><b>colablink sync</b> - Sync entire directory</summary>

```bash
colablink sync                    # Sync current directory
colablink sync -d /path/to/project
```

Automatically excludes: `.git`, `__pycache__`, `venv`, `node_modules`, `*.pyc`, `dist`, `build`

</details>

<details>
<summary><b>colablink forward</b> - Port forwarding</summary>

```bash
colablink forward 8888              # Jupyter
colablink forward 6006              # TensorBoard
colablink forward 5000 --local-port 3000
```

Access forwarded services at `http://localhost:PORT`

</details>

---

## File Synchronization

### Automatic Mode (Default)

When you run `colablink init`, files sync **automatically** in both directions:

- **Local → Colab**: Your local files are accessible on Colab
- **Colab → Local**: Files appear in `./colab-workspace/` (current directory)

No manual commands needed! Works via SSHFS mounting.

### Manual Mode

If automatic sync is unavailable (sshfs not installed), use manual commands:

```bash
colablink upload data/                    # Push to Colab
colablink exec python train.py           # Run
colablink download /content/model.pt     # Pull results
```

---

## Examples

### MNIST Training

```bash
# Sync project
colablink sync

# Train on Colab GPU
colablink exec python examples/train_mnist.py

# Model appears in: ./colab-workspace/mnist_model.pt
```

### Interactive Development

```bash
# Start shell on Colab
colablink shell

# Now you're on Colab with GPU access:
root@colab:~# nvidia-smi
root@colab:~# python
>>> import torch
>>> torch.cuda.is_available()
True
```

### Jupyter Integration

```bash
# Start Jupyter on Colab
colablink exec "jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root" &

# Forward port
colablink forward 8888

# Open: http://localhost:8888
```

---

## Architecture

**Simple 3-step flow:**

1. **Colab Runtime** (`ColabRuntime`) - Sets up SSH server and ngrok tunnel
2. **SSH Tunnel** - Encrypted connection via ngrok
3. **Local Client** (`LocalClient`) - Executes commands and streams output

**Security:** Password-based SSH authentication over TLS-encrypted ngrok tunnel.

**File Access:** SSHFS bidirectional mounting (automatic) or manual upload/download (fallback).

---

## Troubleshooting

### Connection Issues

```bash
# Check status
colablink status

# Reconnect if Colab disconnected
# 1. Rerun Colab setup cell
# 2. Copy new init command
# 3. Run: colablink init '...'
```

### Missing Dependencies

ColabLink will prompt you to install missing dependencies on first run:

**Linux/WSL:**
```bash
sudo apt-get install sshpass sshfs
```

**macOS:**
```bash
brew install hudochenkov/sshpass/sshpass
brew install macfuse sshfs
```

### Command Hangs

Press `Ctrl+C` to cancel. Check connection with `colablink status`.

### Permission Errors

```bash
# Install in user directory
pip install --user colablink

# Or use virtual environment (recommended)
python -m venv venv
source venv/bin/activate
pip install colablink
```

---

## Advanced

### Python API

```python
from colablink import LocalClient

client = LocalClient()
client.initialize(config)
client.execute("python train.py")
client.download("/content/model.pt", "./models/")
```

### VS Code Remote-SSH

After `colablink init`, use VS Code's Remote-SSH extension:

1. Install "Remote - SSH" extension
2. Press `F1` → "Remote-SSH: Connect to Host"
3. Select `colablink` from the list

Configuration is automatically added to `~/.ssh/config`.

### Custom ngrok Token (Recommended)

For stable connections:

1. Get free token at https://ngrok.com
2. Use in Colab:

```python
runtime = ColabRuntime(
    password="your_password",
    ngrok_token="your_ngrok_token"
)
```

---

## Changelog

### [1.0.0] - 2025-10-01 (Beta)

**Core Features:**
- SSH-based connection to Google Colab runtime (GPU or CPU)
- Real-time command execution with unbuffered output streaming
- Automatic GPU/CUDA environment configuration
- Interactive shell mode
- Connection status monitoring

**File Synchronization:**
- Automatic bidirectional file sync via SSHFS
- `colablink upload` - Push files/directories to Colab
- `colablink download` - Pull files/directories from Colab
- `colablink sync` - Sync entire directories with smart exclusions
- Auto-detection of files vs directories
- Tar-based efficient compression

**User Experience:**
- Automatic dependency checking on first run
- Platform-specific installation guidance (Linux/macOS)
- Configurable local mount directory (default: ./colab-workspace)
- Comprehensive CLI help with examples
- Clean text-only output (no glyphs)

**Utilities:**
- Port forwarding for Jupyter, TensorBoard, etc.
- VS Code Remote-SSH integration
- Python API for programmatic use

[Full changelog →](https://github.com/PoshSylvester/colablink/releases)

---

## Contributing

Contributions welcome! Submit issues and pull requests on [GitHub](https://github.com/PoshSylvester/colablink).

**Development:**
```bash
git clone https://github.com/PoshSylvester/colablink.git
cd colablink
pip install -e ".[dev]"
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

**Made for developers who want local development with cloud GPU power**

**Version:** 1.0.0 | **Status:** Beta | **License:** MIT


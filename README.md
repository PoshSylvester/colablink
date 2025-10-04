# ColabLink

[![license](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
![python version](https://img.shields.io/badge/python-3.6%2B-blue?logo=python)
[![PyPI version](https://img.shields.io/badge/pypi-v1.0.0-blue)](https://pypi.org/project/colablink/)

Execute code on Google Colab's powerful GPUs while working entirely from your local development environment. ColabLink provides instant bidirectional file synchronization and seamless command execution through a secure SSH tunnel.

## Key Features

- **Instant Sync** - File changes sync in both directions
- **Local Development** - Use your preferred IDE (VS Code, Cursor, PyCharm, etc.)
- **Secure Connection** - Encrypted SSH tunnel with dedicated user account
- **Multiple Connections** - Connect multiple local sessions to the same runtime
- **Smart Directory Management** - Clean separation of source files and outputs
- **Zero Configuration** - Automatic dependency installation and setup

## Architecture Overview

```
Local Machine (Your IDE)  ←→  SSH Tunnel  ←→  Google Colab Runtime
    Source Files                                Execute & Generate Outputs
    Connection Dirs                             Connection Dirs
```

**Directory Structure:**
```
my_project/
├── train.py              # Source files (synced to Colab)
├── data/                 # Source files (synced to Colab)  
├── connection_abc123/    # Outputs from connection 1
└── connection_xyz789/    # Outputs from connection 2

/content/                 # On Colab
├── train.py              # Synced from local
├── data/                 # Synced from local
├── connection_abc123/    # Outputs for connection 1
└── connection_xyz789/    # Outputs for connection 2
```

## Quick Start

### 1. Install ColabLink

```bash
pip install colablink
```

ColabLink automatically installs required dependencies (`sshpass`, `rsync`, `sshfs`) on first run.

### 2. Setup Colab Runtime

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PoshSylvester/colablink/blob/main/colab_setup.ipynb)

In a new Colab notebook:

```python
!pip install colablink

from getpass import getpass
from colablink import ColabRuntime

# Get your free ngrok token from https://ngrok.com
ngrok_token = getpass("Enter your ngrok authtoken: ")

runtime = ColabRuntime(
    password="secure_password_123",
    ngrok_token=ngrok_token,
    username="colablink"  # Optional: customize username
)

# Setup and display connection info
connection_info = runtime.setup()

# Keep the runtime alive
runtime.keep_alive()
```

### 3. Connect from Local Machine

Copy the connection command from Colab output:

```bash
# Basic connection (auto-generates connection_xxxxxx directory)
colablink init '{"host": "0.tcp.ngrok.io", "port": "12345", "password": "secure_password_123", "username": "colablink"}'

# Full options
colablink init '{"host": "...", "port": "...", "password": "...", "username": "..."}' \
  --remote-dir training \           # Directory name on Colab for outputs (default: connection_xxxxxx)
  --local-dir train_outputs \       # Local directory name for outputs (default: same as remote-dir)  
  --remote-root /content \          # Base directory on Colab (default: /content)
  --profile experiment1             # Profile name for this connection (default: default)
```

### 4. Start Using

```bash
# Execute commands on Colab GPU
colablink exec nvidia-smi
colablink exec python train.py

# Start interactive shell
colablink shell

# Check connection status
colablink status

# Generated files appear instantly in your connection directory
ls connection_abc123/  # See models, outputs, logs, etc.
```

## Command Reference

### Core Commands

| Command | Description | Example |
|---------|-------------|---------|
| `init` | Initialize connection to Colab | `colablink init '{...}'` |
| `exec` | Execute command on Colab | `colablink exec python train.py` |
| `shell` | Start interactive shell | `colablink shell` |
| `status` | Check connection status | `colablink status` |
| `watch` | Run sync agent | `colablink watch` |
| `disconnect` | Close connection | `colablink disconnect` |

### File Management

| Command | Description | Example |
|---------|-------------|---------|
| `upload` | Upload files to Colab | `colablink upload data/` |
| `download` | Download files from Colab | `colablink download model.pt` |
| `sync` | Sync entire directory | `colablink sync` |

### Utilities

| Command | Description | Example |
|---------|-------------|---------|
| `forward` | Port forwarding | `colablink forward 8888` |
| `--profile` | Use specific connection | `colablink exec --profile exp1 python train.py` |

## Multiple Connections

Connect multiple local sessions to the same Colab runtime with isolated output directories:

```bash
# Connection 1: Training experiments
colablink init '{...}' --remote-dir training --local-dir train_outputs --profile exp1

# Connection 2: Data processing  
colablink init '{...}' --remote-dir processing --local-dir proc_outputs --profile exp2

# Use specific connections
colablink exec --profile exp1 python train.py
colablink exec --profile exp2 python process_data.py
colablink watch --profile exp1  # Keep exp1 synced
```

Each connection maintains its own:
- Local output directory (`train_outputs/`, `proc_outputs/`)
- Remote output directory (`/content/training/`, `/content/processing/`)
- SSH configuration and sync state

## File Synchronization

ColabLink uses **dual-path synchronization** to prevent file conflicts:

### Path 1: Source Sync (Local → Colab)
- **What**: Your source files (`.py`, data, configs)
- **Direction**: One-way (local → `/content/`)
- **Speed**: ~0.1 seconds (configurable)
- **Method**: `watchdog` + `rsync`
- **Excludes**: Connection directories (output-only)

### Path 2: Output Sync (Colab → Local)
- **What**: Files created by your code on Colab
- **Direction**: One-way (`/content/connection_*/` → `./connection_*/`)
- **Speed**: Instant (`sshfs`) or ~0.5s (`rsync` fallback)
- **Method**: `sshfs` mount or `rsync` polling
- **Isolation**: Each connection has separate directories

### Sync Configuration

Customize sync speed vs CPU usage:

```bash
# Default: balanced performance
colablink watch

# Near-instant sync (higher CPU usage)
colablink watch --debounce 0.05 --interval 0.2

# Slower sync (lower CPU usage)
colablink watch --debounce 0.5 --interval 2.0
```

## Examples

### Machine Learning Training

```bash
# Setup
colablink init '{...}' --remote-dir ml_training --profile train

# Train model on GPU
colablink exec --profile train python train_model.py

# Model files appear locally instantly
ls ml_training/
# checkpoints/  logs/  model.pt  metrics.json
```

### Jupyter Integration

```bash
# Start Jupyter on Colab
colablink exec "jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root" &

# Forward to local
colablink forward 8888

# Access at http://localhost:8888
```

### Interactive Development

```bash
# Start remote shell
colablink shell

# Now running on Colab with GPU access
$ nvidia-smi
$ python -c "import torch; print(torch.cuda.is_available())"
True
$ exit

# Back to local terminal
```

## Advanced Usage

### Python API

```python
from colablink import LocalClient

# Programmatic usage
client = LocalClient(profile="experiment1")
client.initialize(
    connection_info,
    remote_dir="outputs",
    local_dir="results"
)

# Execute commands
result = client.execute(["python", "train.py"])
client.upload("data/", destination="/content/data/")
client.download("/content/outputs/model.pt", "./models/")
```

### VS Code Remote-SSH

After running `colablink init`, VS Code can connect directly:

1. Install "Remote - SSH" extension
2. Press `F1` → "Remote-SSH: Connect to Host"
3. Select `colablink-<profile>` from the list

SSH configuration is automatically added to `~/.ssh/config`.

### Custom Remote Root

By default, files sync to `/content/` on Colab. You can customize this:

```bash
colablink init '{...}' --remote-root /content/workspace --remote-dir experiment1
# Files sync to: /content/workspace/experiment1/
```

## Troubleshooting

### Connection Issues

```bash
# Check status
colablink status

# Reconnect if Colab session expired
# 1. Rerun the Colab setup cell
# 2. Copy the new connection command
# 3. Run: colablink init '{new_connection_info}'
```

### Missing Dependencies

ColabLink attempts automatic installation. If it fails:

**Linux/WSL:**
```bash
sudo apt-get install sshpass rsync sshfs
```

**macOS:**
```bash
brew install hudochenkov/sshpass/sshpass
brew install --cask macfuse
brew install gromgit/fuse/sshfs-mac
brew install rsync
```

### Sync Issues

```bash
# Manual sync if automatic sync fails
colablink sync                    # Push local changes
colablink download /content/      # Pull all remote files

# Restart sync agent
colablink disconnect
colablink watch
```

### Permission Errors

```bash
# Use virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # or: venv\Scripts\activate on Windows
pip install colablink
```

## Requirements

- **Local**: Python 3.6+, SSH client
- **System Dependencies**: `sshpass`, `rsync`, `sshfs` (auto-installed)
- **Colab**: Standard Google Colab environment
- **Network**: ngrok account (free) for tunnel creation

## Security

- **Encrypted Connection**: All traffic goes through TLS-encrypted ngrok tunnel
- **Dedicated User**: Non-root SSH user created on Colab runtime
- **Password Authentication**: Secure random passwords generated per session
- **Isolated Environment**: Each connection runs in its own directory space

## Contributing

Contributions welcome! Please check the [GitHub repository](https://github.com/PoshSylvester/colablink) for issues and development guidelines.

```bash
# Development setup
git clone https://github.com/PoshSylvester/colablink.git
cd colablink
pip install -e ".[dev]"
```

## License

MIT License - see [LICENSE](LICENSE) for details.

---

**ColabLink v1.0.0** - Bringing local development to cloud computing.
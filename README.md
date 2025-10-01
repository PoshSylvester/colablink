# ColabLink

[![license](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
![python version](https://img.shields.io/badge/python-3.6%2C3.7%2C3.8%2C3.9%2C3.10-blue?logo=python)

Connect your local IDE (VS Code, Cursor, or any terminal) to Google Colab's GPU runtime. Work entirely locally with your files, code, and terminal while executing on Colab's powerful GPUs.

## Overview

ColabLink lets you work on your local machine (with your files and terminal) while executing code on Google Colab's free GPU.

```
Your Local Computer          →  Command  →     Google Colab
  Code files (local)                              GPU Execution
  Terminal (shows logs)      ←  Output  ←        Real-time Results
```

**Key Benefits:**
- **Automatic Bidirectional Sync** - Files sync automatically in both directions via SSHFS
- **Local Terminal Execution** - Run commands in your local terminal, execute on Colab GPU
- **Seamless File Access** - Generated files appear locally instantly, no manual downloads
- **Real-time Streaming** - See logs and output in real-time as your code runs
- **GPU Access** - Full access to Colab's Tesla T4/P100/V100 GPUs
- **IDE Integration** - Works with VS Code, Cursor, PyCharm, or any terminal
- **Transparent** - Feels like working on a local machine with a powerful GPU

## Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [Architecture](#architecture)
- [Troubleshooting](#troubleshooting)
- [Changelog](#changelog)
- [Contributing](#contributing)
- [License](#license)

---

## Quick Start

Get started in 5 minutes!

### Prerequisites

- Python 3.6+ on local machine
- Google account (for Colab)
- Terminal access

### Step 1: Install Locally

```bash
pip install colablink

# Install required system dependencies
# Ubuntu/Debian:
sudo apt-get install sshpass sshfs

# macOS:
brew install hudochenkov/sshpass/sshpass
brew install macfuse sshfs
```

**What are these for?**
- `sshpass`: SSH authentication with password
- `sshfs`: Automatic bidirectional file sync (mount Colab directory locally)

### Step 2: Setup Colab Runtime

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PoshSylvester/colablink/blob/main/colab_setup.ipynb)

Open Google Colab: https://colab.research.google.com/

Or use the pre-configured notebook above, then run:

```python
# Install
!pip install colablink

# Setup and run
from colablink import ColabRuntime

runtime = ColabRuntime(password="choose_a_strong_password")
runtime.setup()
runtime.keep_alive()  # Keep this running!
```

After a moment, you'll see output with connection details.

### Step 3: Connect from Local Terminal

Copy the `colablink init` command from the Colab output and paste it in your local terminal:

```bash
colablink init '{"host": "0.tcp.ngrok.io", "port": "12345", "password": "your_password", "mount_point": "/mnt/local"}'
```

### Step 4: Work with Your Files

**Your files are now automatically synced!**

- **Local → Colab**: Files you create/edit locally are accessible on Colab
- **Colab → Local**: Files generated on Colab appear in `~/.colablink/colab_workspace/`

No manual sync needed! The connection uses SSHFS for bidirectional file access.

Alternatively, you can manually sync if needed:

```bash
# Manual sync (optional - only if automatic sync fails)
colablink sync

# Or upload specific files/directories
colablink upload train.py
colablink upload data/
```

### Step 5: Use Colab GPU from Local Terminal

Now you can run commands locally that execute on Colab:

```bash
# Check GPU
colablink exec nvidia-smi

# Run Python script (on Colab GPU) - file must be synced first
colablink exec python YourProjectName/train.py

# Install packages on Colab
colablink exec pip install torch torchvision

# Interactive shell
colablink shell
```

---

## Installation

### Local Machine Installation

#### Option A: Install from PyPI (Recommended)

```bash
pip install colablink
```

#### Option B: Install from Source

```bash
# Clone the repository
git clone https://github.com/PoshSylvester/colablink.git
cd colablink

# Install
pip install .
```

#### Option C: Install from GitHub

```bash
pip install git+https://github.com/yourusername/colablink.git
```

### Install sshpass (Required)

ColabLink uses `sshpass` for password-based SSH authentication.

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install sshpass
```

**macOS:**
```bash
brew install hudochenkov/sshpass/sshpass
```

**Windows (WSL):**
```bash
# In WSL terminal
sudo apt-get update
sudo apt-get install sshpass
```

### Verify Installation

```bash
# Check colablink is installed
colablink --help

# Check sshpass is installed
which sshpass
```

### Optional: ngrok Authentication

For more stable connections, get a free ngrok account:

1. Sign up at https://ngrok.com (free)
2. Get your authtoken from dashboard
3. Use token in Colab setup:

```python
runtime = ColabRuntime(
    password="your_password",
    ngrok_token="your_ngrok_token"  # Add this
)
runtime.setup()
```

Benefits:
- More stable connections
- Longer tunnel lifetime
- Better connection reliability

### Virtual Environment (Recommended)

```bash
# Create venv
python -m venv colablink-env

# Activate
source colablink-env/bin/activate  # Linux/macOS
# or
colablink-env\Scripts\activate  # Windows

# Install
pip install colablink

# When done
deactivate
```

---

## Usage

### Basic Commands

```bash
# Execute any command on Colab
colablink exec <command>

# Interactive shell (like SSH)
colablink shell

# Check connection status and GPU info
colablink status

# Forward ports (Jupyter, TensorBoard, etc.)
colablink forward 8888  # Jupyter
colablink forward 6006  # TensorBoard

# Disconnect
colablink disconnect
```

### File Synchronization

**By default, ColabLink automatically syncs files bidirectionally using SSHFS.** Files appear instantly in both directions without manual commands!

If automatic sync is unavailable or disabled, ColabLink provides manual file management commands:

#### Download Files from Colab

```bash
# Download a single file
colablink download /content/model.pt

# Download to specific location
colablink download /content/model.pt --destination ./models/

# Download entire directory
colablink download /content/output/ --recursive
```

#### Sync Entire Directory

```bash
# Sync current directory to Colab
colablink sync

# Sync specific directory
colablink sync --directory /path/to/project
```

This will:
- Upload your entire project directory to `/content/YourProjectName/` on Colab
- Automatically exclude common files (`.git`, `__pycache__`, `node_modules`, etc.)
- Use efficient compression for faster transfer

#### Upload Specific Files or Directories

```bash
# Upload a single file
colablink upload train.py

# Upload a directory
colablink upload data/

# Upload with custom destination
colablink upload model.py --destination /content/models/

# Upload directory recursively
colablink upload myproject/ --recursive
```

#### After Syncing

Once files are synced, you can execute them:

```bash
# If you synced current directory named "myproject"
colablink exec python myproject/train.py

# If you uploaded a single file
colablink exec python train.py
```

**Note:** Files need to be synced before execution. Any changes made locally require re-syncing.

### Example Workflow: Training a Model

**Local Machine Setup:**

```bash
# Create project
mkdir my-ml-project
cd my-ml-project

# Create a training script
cat > train.py << 'EOF'
import torch
import torch.nn as nn

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Simple model
model = nn.Linear(10, 1).cuda()
print("Model created on GPU!")

# Dummy training
optimizer = torch.optim.Adam(model.parameters())
for epoch in range(10):
    loss = model(torch.randn(32, 10).cuda()).sum()
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}: loss = {loss.item():.4f}")

print("Training complete!")
EOF
```

**Sync and Run on Colab GPU:**

```bash
# Sync project to Colab
colablink sync

# Execute on Colab (output streams to your terminal)
colablink exec python my-ml-project/train.py
```

**Output:**
```
CUDA available: True
GPU: Tesla T4
Model created on GPU!
Epoch 0: loss = -2.3456
Epoch 1: loss = -2.1234
...
Training complete!
```

### VS Code / Cursor Integration

#### Option 1: Integrated Terminal

1. Open VS Code/Cursor
2. Open your project folder
3. Open integrated terminal (Ctrl+`)
4. Run commands with `colablink exec`

All logs appear in VS Code's terminal panel!

#### Option 2: Remote-SSH Extension

After running `colablink init`, you can use VS Code's Remote-SSH:

1. Install "Remote - SSH" extension
2. Press `F1` → "Remote-SSH: Connect to Host"
3. Select `colablink` from the list
4. VS Code connects to Colab runtime

Configuration is automatically added to `~/.ssh/config`.

### Port Forwarding

Forward Colab services to local machine:

```bash
# Start Jupyter on Colab
colablink exec "jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root"

# In another terminal, forward port
colablink forward 8888

# Access in browser: http://localhost:8888
```

Same for TensorBoard:
```bash
colablink forward 6006
colablink exec "tensorboard --logdir=./runs --port=6006"
# Access at http://localhost:6006
```

### Installing Requirements

```bash
# Create requirements.txt locally
cat > requirements.txt << EOF
torch
torchvision
transformers
pandas
EOF

# Install on Colab
colablink exec pip install -r requirements.txt
```

### Working with Data

Your local data files are accessible via commands:

```python
# In train.py
import pandas as pd

# This reads from your LOCAL machine
df = pd.read_csv('./data/dataset.csv')

# Process with Colab GPU
model.train(df)

# Save back to LOCAL machine
model.save('./models/trained_model.pt')
```

Files are transferred on-demand via SSH.

---

## Examples

### Example 1: MNIST Training

Train a CNN on MNIST using Colab GPU. See `examples/train_mnist.py`.

```bash
colablink exec python examples/train_mnist.py
```

### Example 2: GPU Information

Check GPU availability and specifications:

```bash
colablink exec python examples/gpu_info.py
```

### Example 3: Interactive Python

```bash
colablink shell

root@colab:~# python
>>> import torch
>>> torch.cuda.is_available()
True
>>> torch.cuda.get_device_name(0)
'Tesla T4'
```

### More Examples

Check the `examples/` directory for:
- MNIST training with real-time output
- GPU information scripts
- Jupyter integration
- TensorBoard integration

---

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     LOCAL MACHINE                           │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  User's Terminal / IDE (VS Code, Cursor)             │  │
│  │  - Source code files (local)                         │  │
│  │  - Data files (local)                                │  │
│  │  - Terminal shows real-time output                   │  │
│  └───────────────────┬──────────────────────────────────┘  │
│                      │                                      │
│  ┌───────────────────▼──────────────────────────────────┐  │
│  │  LocalClient (colablink/client.py)                │  │
│  │  - Connection management                             │  │
│  │  - SSH command execution                             │  │
│  │  - Output streaming                                  │  │
│  │  - Port forwarding                                   │  │
│  └───────────────────┬──────────────────────────────────┘  │
│                      │                                      │
└──────────────────────┼──────────────────────────────────────┘
                       │
                       │ SSH over ngrok tunnel
                       │ (Encrypted, Authenticated)
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                   GOOGLE COLAB                              │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  ColabRuntime (colablink/runtime.py)              │  │
│  │  - SSH server setup                                  │  │
│  │  - ngrok tunnel creation                             │  │
│  │  - Connection management                             │  │
│  │  - Keep-alive mechanism                              │  │
│  └───────────────────┬──────────────────────────────────┘  │
│                      │                                      │
│  ┌───────────────────▼──────────────────────────────────┐  │
│  │  Execution Environment                               │  │
│  │  - GPU: Tesla T4 / P100 / V100                       │  │
│  │  - Python runtime                                    │  │
│  │  - CUDA libraries                                    │  │
│  │  - Executes code                                     │  │
│  │  - Streams output back                               │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Components

#### 1. ColabRuntime (Colab-side)

**Location:** `colablink/runtime.py`

**Responsibilities:**
- Install and configure OpenSSH server
- Set up authentication (password-based)
- Create ngrok tunnel for SSH access
- Keep session alive
- Display connection information

**Configuration:**
```python
runtime = ColabRuntime(
    password="secure_password",    # SSH authentication
    ngrok_token="optional_token",  # Stable tunnel (recommended)
    mount_point="/mnt/local"       # Where local files appear
)
```

#### 2. LocalClient (Local machine)

**Location:** `colablink/client.py`

**Responsibilities:**
- Manage connection to Colab
- Execute commands via SSH
- Stream output in real-time
- Forward ports for services
- Handle file access via SSH

**Configuration stored in:** `~/.colablink/config.json`

#### 3. CLI Tool

**Location:** `colablink/cli.py`

**Commands:**
- `colablink init` - Initialize connection
- `colablink exec` - Execute command on Colab
- `colablink shell` - Interactive shell
- `colablink status` - Connection status
- `colablink forward` - Port forwarding
- `colablink disconnect` - Disconnect

### Connection Flow

**Setup Phase:**
1. User opens Colab notebook
2. Runs `ColabRuntime().setup()`
   - Installs OpenSSH server
   - Configures authentication
   - Creates ngrok tunnel
   - Displays connection string
3. User copies connection command
4. Runs locally: `colablink init '...'`
   - Saves configuration
   - Creates SSH config
   - Tests connection
   - Ready to use

**Execution Flow:**
1. User runs: `colablink exec python train.py`
2. LocalClient.execute() is called
   - Builds SSH command with auth
   - Executes via SSH tunnel
3. On Colab:
   - SSH server receives command
   - Executes in Python runtime
   - GPU resources available
   - Output streams back
4. LocalClient receives output
   - stdout → local terminal
   - stderr → local terminal
   - Real-time streaming
5. Execution completes

### Security

- **Password-based SSH authentication**
- **ngrok tunnel**: TLS encrypted
- **SSH**: End-to-end encryption
- **No public key exposure**
- **Session-based credentials**

### Performance Considerations

**Network Latency:**
- RTT: ~100-300ms typical
- Impact: File I/O operations
- Mitigation: Cache data, minimize file reads

**GPU Utilization:**
- Goal: Keep GPU busy
- Issue: I/O bottlenecks
- Solution: Async data loading, prefetching

---

## Troubleshooting

### Common Issues

#### "sshpass not found"

**Solution:**
```bash
# Ubuntu/Debian
sudo apt-get install sshpass

# macOS
brew install hudochenkov/sshpass/sshpass
```

#### "Connection failed"

**Solutions:**
1. Make sure Colab cell is still running
2. Check firewall settings
3. Try with ngrok authtoken
4. Verify password is correct

#### Colab Disconnected

Colab free tier disconnects after inactivity. Simply:
1. Rerun the Colab setup cell
2. Copy new connection command
3. Run `colablink init` again

#### Command Hangs

Press Ctrl+C to cancel. Check connection:
```bash
colablink status
```

#### Permission Denied During Install

**Solution:**
```bash
# Install with user flag
pip install --user colablink

# Or use virtual environment
python -m venv venv
source venv/bin/activate
pip install colablink
```

#### Command Not Found After Installation

**Solution:**
```bash
# Find installation location
pip show colablink

# Add to PATH (add to ~/.bashrc or ~/.zshrc)
export PATH="$HOME/.local/bin:$PATH"

# Reload shell
source ~/.bashrc
```

### Verification

After installation, verify everything works:

```bash
# 1. Check CLI is available
colablink --help

# 2. Check Python import
python -c "from colablink import ColabRuntime, LocalClient; print('OK')"

# 3. Check sshpass
which sshpass

# 4. Setup Colab and test connection
colablink status
```

---

## Changelog

### [1.2.0] - 2025-10-01

#### Added

**Automatic Bidirectional File Sync:**
- SSHFS-based automatic mounting of Colab directory locally
- Files generated on Colab appear instantly in `~/.colablink/colab_workspace/`
- No manual download commands needed
- Real-time file synchronization in both directions
- Automatic reconnection on network interruptions
- Configurable: can enable/disable auto-sync via `auto_sync` parameter

**Manual File Management:**
- `colablink download` - Download files/directories from Colab to local
- Manual sync mode with clear warnings when auto-sync is disabled
- Explicit upload/download/sync commands for fine-grained control

**Improvements:**
- Seamless workflow - write code locally, see results locally
- Trained models and outputs automatically available locally
- Better developer experience with transparent file access
- Graceful fallback to manual sync if SSHFS unavailable
- Bold warnings when manual file management is required

**Requirements:**
- Added `sshfs` as optional but recommended dependency
- Enhanced installation documentation

### [1.1.0] - 2025-10-01

#### Added

**File Synchronization:**
- `colablink sync` - Sync entire directory to Colab with smart exclusions
- `colablink upload` - Upload specific files or directories
- Automatic exclusion of common files (.git, __pycache__, venv, etc.)
- Efficient tar-based compression for faster transfers
- Custom destination support for uploads

**Improvements:**
- Better file management workflow
- No need for manual scp commands
- Streamlined development experience
- Updated documentation with file sync examples

**CLI Enhancements:**
- Added `upload` subcommand with recursive support
- Added `sync` subcommand with directory selection
- Enhanced help text with sync examples

### [1.0.0] - 2025-09-30

#### Complete Rewrite

This is a complete rewrite of the original colabcode package with a new architecture and focus.

#### Added

**Core Features:**
- Local Terminal Execution: Execute commands from local terminal with output streaming to local
- Local File Management: All files (code and data) stay on local machine
- Real-time Output Streaming: See logs and output in real-time
- GPU Access: Full access to Google Colab's GPU (Tesla T4/P100/V100)

**Components:**
- `ColabRuntime` class for Colab-side setup
- `LocalClient` class for local-side connection
- `colablink` CLI tool
- SSH-based secure connection over ngrok tunnel
- Port forwarding for Jupyter, TensorBoard, etc.

**Commands:**
- `colablink init` - Initialize connection
- `colablink exec` - Execute command on Colab
- `colablink shell` - Interactive SSH shell
- `colablink status` - Check connection and GPU
- `colablink forward` - Forward ports
- `colablink disconnect` - Disconnect

#### Changed

**Architecture:**
- Changed from web-based code-server to SSH-based execution
- Moved from browser IDE to local IDE integration
- Changed file access from manual upload to on-demand SSH access
- Improved security with password-based SSH authentication

**User Experience:**
- Simplified setup to single command on each side
- Improved output streaming for real-time feedback
- Added status monitoring and GPU information display

#### Removed

**Legacy Features (from colabcode v0.x):**
- Web-based code-server interface (replaced with local IDE)
- Browser-based VS Code (replaced with local VS Code/Cursor)
- JupyterLab support (can still use via port forwarding)

#### Migration from colabcode 0.x

**Old (colabcode 0.x):**
```python
from colabcode import ColabCode
ColabCode(port=10000, password="password")
# Opens browser-based VS Code
```

**New (ColabLink 2.0):**
```python
# On Colab:
from colablink import ColabRuntime
runtime = ColabRuntime(password="password")
runtime.setup()

# On Local:
colablink init '...'
colablink exec python train.py
```

#### Technical Details

**Dependencies:**
- `pyngrok>=5.0.5` - For tunnel creation
- `sshpass` - For password-based SSH (system package)
- OpenSSH server - Installed automatically on Colab

**Python Support:**
- Python 3.6+
- Tested on Python 3.7, 3.8, 3.9, 3.10

**Platform Support:**
- Linux (native)
- macOS (native)
- Windows (via WSL)

#### Known Issues

1. **File I/O Performance**: Network latency for file access
2. **Colab Timeout**: 12-hour session limit on free tier
3. **Single Connection**: Only one Colab session at a time
4. **sshpass Requirement**: Requires manual installation

#### Future Roadmap

**Version 2.1 (Planned):**
- Reverse SSHFS mounting for faster file access
- File caching system
- Improved error handling

**Version 2.2 (Planned):**
- SSH key authentication support
- Multiple simultaneous connections
- Auto-reconnection on disconnect

**Version 3.0 (Future):**
- VS Code extension
- Cursor native integration
- Remote debugging support

---

## Contributing

Contributions welcome! Please feel free to submit issues and pull requests.

### Development Setup

```bash
# Clone repository
git clone https://github.com/yourusername/colablink.git
cd colablink

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install in editable mode
pip install -e .

# Run tests
pytest tests/
```

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Inspired by colabcode and colab-ssh projects
- Uses pyngrok for tunnel creation
- Built for the ML/AI community

## Support

For issues and questions:
- GitHub Issues: Report problems and request features
- Examples: Check `examples/` directory for sample code
- Documentation: This README contains comprehensive documentation

---

**Made for developers who want local development experience with cloud GPU power**

**Version:** 1.2.0  
**Status:** Stable  
**License:** MIT

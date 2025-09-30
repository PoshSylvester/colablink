"""
ColabRuntime - Runs on Google Colab to set up the execution environment.

This module handles:
- SSH server setup
- Tunnel creation (ngrok)
- Port forwarding
- Keep-alive mechanism
"""

import os
import subprocess
import json
import time
import threading
from pathlib import Path


class ColabRuntime:
    """Sets up Colab runtime to accept connections from local machine."""
    
    def __init__(self, password=None, ngrok_token=None, mount_point="/mnt/local"):
        """
        Initialize Colab runtime.
        
        Args:
            password: SSH password for connection (required)
            ngrok_token: ngrok authtoken for tunnel creation (optional but recommended)
            mount_point: Where to mount local filesystem on Colab
        """
        self.password = password or self._generate_password()
        self.ngrok_token = ngrok_token
        self.mount_point = mount_point
        self.connection_info = {}
        self.ssh_port = 22
        self.sshfs_port = 2222
        self.keep_alive_thread = None
        
    def setup(self):
        """
        Main setup method - run this in Colab notebook.
        
        This will:
        1. Install required packages
        2. Setup SSH server
        3. Create ngrok tunnel
        4. Display connection instructions
        """
        print("="*70)
        print("ColabLink - Setting up GPU runtime...")
        print("="*70)
        
        try:
            # Check if we're in Colab
            self._check_colab_environment()
            
            # Show GPU info
            self._display_gpu_info()
            
            # Install dependencies
            print("\n[1/5] Installing dependencies...")
            self._install_dependencies()
            
            # Setup SSH server
            print("\n[2/5] Configuring SSH server...")
            self._setup_ssh_server()
            
            # Create ngrok tunnel
            print("\n[3/5] Creating secure tunnel...")
            self._create_tunnel()
            
            # Setup mount point
            print("\n[4/5] Preparing filesystem mount point...")
            self._prepare_mount_point()
            
            # Display connection info
            print("\n[5/5] Setup complete!")
            self._display_connection_info()
            
            # Start keep-alive
            self._start_keep_alive()
            
            return self.connection_info
            
        except Exception as e:
            print(f"\nError during setup: {e}")
            raise
    
    def keep_alive(self):
        """
        Keep the Colab session alive.
        Call this to prevent disconnection or just keep the cell running.
        """
        print("\nRuntime is active. Keep this cell running to maintain connection.")
        print("Press Ctrl+C to stop (will disconnect local client).")
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            print("\nShutting down runtime...")
    
    def _check_colab_environment(self):
        """Check if running in Google Colab."""
        try:
            import google.colab
            return True
        except ImportError:
            print("\nWarning: Not running in Google Colab environment.")
            print("This tool is designed for Colab but will attempt to continue...")
            return False
    
    def _display_gpu_info(self):
        """Display available GPU information."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                gpu_info = result.stdout.strip()
                print(f"\nGPU Available: {gpu_info}")
            else:
                print("\nNo GPU detected. Code will run on CPU.")
        except FileNotFoundError:
            print("\nNo GPU detected. Code will run on CPU.")
    
    def _install_dependencies(self):
        """Install required system packages."""
        packages = [
            "openssh-server",
            "sshfs",
            "fuse",
        ]
        
        # Fix any broken packages first
        print("   Fixing broken packages...")
        subprocess.run(
            ["apt-get", "-f", "install", "-y"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Update package list
        print("   Updating package list...")
        result = subprocess.run(
            ["apt-get", "update", "-qq"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.returncode != 0:
            print(f"   Warning: apt-get update had issues: {result.stderr[:200]}")
        
        # Install packages one by one for better error handling
        print("   Installing SSH server and dependencies...")
        for package in packages:
            result = subprocess.run(
                ["apt-get", "install", "-qq", "-y", "--fix-missing", package],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            if result.returncode != 0:
                # Try with dpkg configure first
                subprocess.run(
                    ["dpkg", "--configure", "-a"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                # Retry installation
                result = subprocess.run(
                    ["apt-get", "install", "-qq", "-y", "--fix-broken", package],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                if result.returncode != 0:
                    print(f"   Warning: Failed to install {package}: {result.stderr[:200]}")
        
        # Verify sshd exists
        if not os.path.exists("/usr/sbin/sshd"):
            raise FileNotFoundError(
                "SSH server not found after installation.\n"
                "Please run these commands manually in a Colab cell:\n"
                "  !apt-get update\n"
                "  !dpkg --configure -a\n"
                "  !apt-get install -y openssh-server\n"
                "Then retry the setup."
            )
        
        # Install pyngrok if not already installed
        subprocess.run(
            ["pip", "install", "-q", "pyngrok"],
            stdout=subprocess.DEVNULL
        )
        
        print("   Dependencies installed successfully")
    
    def _setup_ssh_server(self):
        """Configure and start SSH server."""
        # Set root password
        print("   Setting root password...")
        result = subprocess.run(
            ["bash", "-c", f"echo 'root:{self.password}' | chpasswd"],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to set password: {result.stderr}")
        
        # Configure SSH
        print("   Configuring SSH server...")
        ssh_config = """
PermitRootLogin yes
PasswordAuthentication yes
PubkeyAuthentication yes
"""
        try:
            with open("/etc/ssh/sshd_config", "a") as f:
                f.write(ssh_config)
        except Exception as e:
            raise RuntimeError(f"Failed to configure SSH: {e}")
        
        # Create necessary directories
        os.makedirs("/var/run/sshd", exist_ok=True)
        
        # Setup environment for SSH sessions (GPU/CUDA access)
        print("   Configuring environment for GPU access...")
        env_setup = """
# ColabLink environment setup
export LD_LIBRARY_PATH=/usr/lib64-nvidia:/usr/local/cuda/lib64:/usr/local/cuda/lib64/stubs:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
"""
        try:
            # Add to root's .bashrc
            with open("/root/.bashrc", "a") as f:
                f.write(env_setup)
            
            # Also create .bash_profile that sources .bashrc
            with open("/root/.bash_profile", "w") as f:
                f.write("if [ -f ~/.bashrc ]; then source ~/.bashrc; fi\n")
        except Exception as e:
            print(f"   Warning: Could not setup environment: {e}")
        
        # Start SSH service
        print("   Starting SSH daemon...")
        result = subprocess.run(
            ["/usr/sbin/sshd"],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to start SSH server: {result.stderr}")
        
        print("   SSH server running on port 22")
    
    def _create_tunnel(self):
        """Create ngrok tunnel for SSH access."""
        from pyngrok import ngrok
        
        # Set auth token if provided
        if self.ngrok_token:
            ngrok.set_auth_token(self.ngrok_token)
        
        # Kill any existing tunnels
        ngrok.kill()
        
        # Create SSH tunnel
        tunnel = ngrok.connect(self.ssh_port, "tcp")
        
        # Parse tunnel URL: tcp://0.tcp.ngrok.io:12345
        tunnel_url = tunnel.public_url.replace("tcp://", "")
        host, port = tunnel_url.split(":")
        
        self.connection_info = {
            "host": host,
            "port": port,
            "password": self.password,
            "mount_point": self.mount_point
        }
        
        print(f"   Tunnel created: {host}:{port}")
    
    def _prepare_mount_point(self):
        """Create mount point for local filesystem."""
        os.makedirs(self.mount_point, exist_ok=True)
        print(f"   Mount point ready: {self.mount_point}")
    
    def _generate_password(self):
        """Generate a random secure password."""
        import random
        import string
        chars = string.ascii_letters + string.digits
        return ''.join(random.choice(chars) for _ in range(16))
    
    def _display_connection_info(self):
        """Display connection instructions for local machine."""
        config_json = json.dumps(self.connection_info)
        
        print("\n" + "="*70)
        print("SETUP COMPLETE - Runtime Ready!")
        print("="*70)
        print("\nConnection Details:")
        print(f"  Host: {self.connection_info['host']}")
        print(f"  Port: {self.connection_info['port']}")
        print(f"  Password: {self.connection_info['password']}")
        
        print("\n" + "-"*70)
        print("CONNECT FROM YOUR LOCAL MACHINE:")
        print("-"*70)
        
        print("\n1. Install colablink on your local machine:")
        print("   pip install colablink")
        
        print("\n2. Initialize connection (copy-paste this command):")
        print(f"\n   colablink init '{config_json}'")
        
        print("\n3. Execute commands on Colab GPU from your local terminal:")
        print("   colablink exec python train.py")
        print("   colablink exec nvidia-smi")
        
        print("\n4. Or use shell wrapper for transparent execution:")
        print("   colablink shell")
        print("   python train.py  # Runs on Colab GPU automatically")
        
        print("\n" + "="*70)
        print("\nKeep this cell running to maintain the connection!")
        print("="*70)
    
    def _start_keep_alive(self):
        """Start background thread to keep session alive."""
        def keep_alive_task():
            while True:
                time.sleep(300)  # Every 5 minutes
                # Trigger some activity to prevent disconnect
                subprocess.run(["echo", "keep-alive"], stdout=subprocess.DEVNULL)
        
        self.keep_alive_thread = threading.Thread(target=keep_alive_task, daemon=True)
        self.keep_alive_thread.start()


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
import uuid
from pathlib import Path


class ColabRuntime:
    """Sets up Colab runtime to accept connections from local machine."""

    def __init__(self, password=None, ngrok_token=None, username: str = "colablink", remote_root: str = "/content"):
        """
        Initialize Colab runtime.

        Args:
            password: SSH password for connection (required)
            ngrok_token: ngrok authtoken for tunnel creation (required)
            username: SSH user created on the runtime (default: "colablink")
            remote_root: Base directory on Colab for all connections (default: "/content")
        """
        self.password = password or self._generate_password()
        self.ngrok_token = ngrok_token
        self.connection_info = {}
        self.ssh_port = 22
        self.sshfs_port = 2222
        self.keep_alive_thread = None
        self.username = self._sanitize_username(username)
        self.session_id = None
        self.remote_root = remote_root

    def setup(self):
        """
        Main setup method - run this in Colab notebook.

        This will:
        1. Install required packages
        2. Setup SSH server
        3. Create ngrok tunnel
        4. Display connection instructions
        """
        print("=" * 70)
        print("ColabLink - Setting up runtime...")
        print("=" * 70)

        try:
            # Check if we're in Colab
            self._check_colab_environment()

            # Show GPU info
            self._display_gpu_info()

            # Install dependencies
            print("\n[1/4] Installing dependencies...")
            self._install_dependencies()

            # Setup SSH server
            print("\n[2/4] Configuring SSH server...")
            self._setup_ssh_server()

            # Create ngrok tunnel
            print("\n[3/4] Creating secure tunnel...")
            self._create_tunnel()

            # Display connection info
            print("\n[4/4] Setup complete!")
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
            from importlib.util import find_spec

            if find_spec("google.colab") is None:
                raise ImportError
            return True
        except ImportError:
            print("\nWarning: Not running in Google Colab environment.")
            print("This tool is designed for Colab but will attempt to continue...")
            return False

    def _display_gpu_info(self):
        """Display available GPU information."""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
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
            "rsync",
        ]

        # Fix any broken packages first
        print("   Fixing broken packages...")
        subprocess.run(
            ["apt-get", "-f", "install", "-y"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Update package list
        print("   Updating package list...")
        result = subprocess.run(
            ["apt-get", "update", "-qq"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
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
                text=True,
            )
            if result.returncode != 0:
                # Try with dpkg configure first
                subprocess.run(
                    ["dpkg", "--configure", "-a"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                # Retry installation
                result = subprocess.run(
                    ["apt-get", "install", "-qq", "-y", "--fix-broken", package],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                if result.returncode != 0:
                    print(
                        f"   Warning: Failed to install {package}: {result.stderr[:200]}"
                    )
                    print(
                        "   Try running manually in a Colab cell: "
                        f"!apt-get install -y {package}"
                    )

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
        subprocess.run(["pip", "install", "-q", "pyngrok"], stdout=subprocess.DEVNULL)

        print("   Dependencies installed successfully")

    def _setup_ssh_server(self):
        """Configure and start SSH server."""
        self.session_id = uuid.uuid4().hex[:8]

        self._ensure_user()
        self._configure_ssh()
        # NOTE: No workspace creation - connection directories are created by client during init

        # Setup environment for SSH sessions (GPU/CUDA access)
        print("   Configuring environment for GPU access...")
        env_setup = """
# ColabLink environment setup
export LD_LIBRARY_PATH=/usr/lib64-nvidia:/usr/local/cuda/lib64:/usr/local/cuda/lib64/stubs:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
"""
        try:
            user_home = Path("/home") / self.username
            bashrc_path = user_home / ".bashrc"
            bashrc_path.parent.mkdir(parents=True, exist_ok=True)
            with open(bashrc_path, "a") as f:
                f.write(env_setup)

            bash_profile_path = user_home / ".bash_profile"
            with open(bash_profile_path, "w") as f:
                f.write("if [ -f ~/.bashrc ]; then source ~/.bashrc; fi\n")
        except Exception as e:
            print(f"   Warning: Could not setup environment: {e}")

        # Start SSH service
        print("   Starting SSH daemon...")
        result = subprocess.run(["/usr/sbin/sshd"], capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to start SSH server: {result.stderr}")

        # Grant colablink user write access to /content
        print("   Configuring /content permissions...")
        try:
            # Change ownership of /content to allow colablink user to write
            subprocess.run(
                ["chown", "-R", f"{self.username}:root", "/content"],
                capture_output=True, text=True
            )
            # Ensure /content is writable by the user
            subprocess.run(
                ["chmod", "755", "/content"],
                capture_output=True, text=True
            )
        except Exception as e:
            print(f"   Warning: Could not set /content permissions: {e}")

        print("   SSH server running on port 22")

    def _ensure_user(self):
        """Create dedicated user account for ColabLink access."""
        print(f"   Creating user '{self.username}'...")

        # Create user if missing
        user_home = Path("/home") / self.username
        if not user_home.exists():
            result = subprocess.run(
                ["useradd", "-m", "-s", "/bin/bash", self.username],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Failed to create user: {result.stderr}")

        # Set password for the user
        result = subprocess.run(
            ["bash", "-c", f"echo '{self.username}:{self.password}' | chpasswd"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to set password: {result.stderr}")

    def _configure_ssh(self):
        """Harden SSH configuration for ColabLink."""
        print("   Configuring SSH server...")

        os.makedirs("/var/run/sshd", exist_ok=True)

        base_config_path = Path("/etc/ssh/sshd_config")
        backup_path = base_config_path.with_suffix(".colablink.bak")
        try:
            if not backup_path.exists():
                backup_path.write_text(base_config_path.read_text())
        except Exception:
            # Backups are best-effort and should not stop execution
            pass

        desired_settings = {
            "PermitRootLogin": "no",
            "PasswordAuthentication": "yes",
            "PubkeyAuthentication": "yes",
        }

        config_lines = base_config_path.read_text().splitlines()
        updated_lines = []
        for line in config_lines:
            key = line.split()[0] if line.strip() else ""
            if key in desired_settings:
                updated_lines.append(f"{key} {desired_settings.pop(key)}")
            else:
                updated_lines.append(line)

        for key, value in desired_settings.items():
            updated_lines.append(f"{key} {value}")

        base_config_path.write_text("\n".join(updated_lines) + "\n")

    def _create_tunnel(self):
        """Create ngrok tunnel for SSH access."""
        if not self.ngrok_token:
            raise RuntimeError(
                "ngrok authtoken required. Set ngrok_token when constructing ColabRuntime."
            )

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
            "username": self.username,
            "remote_root": self.remote_root,
        }

        print(f"   Tunnel created: {host}:{port}")

    def _generate_password(self):
        """Generate a random secure password."""
        import random
        import string

        chars = string.ascii_letters + string.digits
        return "".join(random.choice(chars) for _ in range(16))

    def _sanitize_username(self, username: str) -> str:
        """Ensure provided username is safe for useradd."""
        import re

        if not username:
            return "colablink"

        candidate = re.sub(r"[^a-z0-9_-]", "", username.lower())
        if not candidate:
            return "colablink"

        # useradd generally expects <= 32 characters
        return candidate[:32]

    def _display_connection_info(self):
        """Display connection instructions for local machine."""
        config_json = json.dumps(self.connection_info)

        print("\n" + "=" * 70)
        print("SETUP COMPLETE - Runtime Ready!")
        print("=" * 70)
        print("\nConnection Details:")
        print(f"  Host: {self.connection_info['host']}")
        print(f"  Port: {self.connection_info['port']}")
        print(f"  User: {self.connection_info['username']}")
        print(f"  Password: {self.connection_info['password']}")

        print("\n" + "-" * 70)
        print("CONNECT FROM YOUR LOCAL MACHINE:")
        print("-" * 70)

        print("\n1. Install colablink on your local machine:")
        print("   pip install colablink")
        print("   # or: pip install git+https://github.com/PoshSylvester/colablink.git")

        print("\n2. Initialize connection (copy-paste this command):")
        print(f"\n   colablink init '{config_json}'")
        print("\n   # Optional: specify custom directories and profiles")
        print(f"   colablink --profile train init '{config_json}' --remote-dir training --local-dir train_outputs")

        print("\n3. Execute commands on Colab runtime from your local terminal:")
        print("   colablink exec python train.py")
        print("   colablink --profile train exec nvidia-smi")

        print("\n4. Or use shell wrapper for transparent execution:")
        print("   colablink shell")
        print("   python train.py  # Runs on Colab runtime automatically")

        print("\nOptional flags:")
        print("   --remote-dir training                   # Directory name on Colab for outputs")
        print("   --local-dir train_outputs              # Local directory name for outputs")
        print("   --remote-root /content                 # Base directory on Colab (default)")
        print("   --profile train                        # Profile name for this connection")

        print("\n" + "=" * 70)
        print("\nKeep this cell running to maintain the connection!")
        print("=" * 70)

    def _start_keep_alive(self):
        """Start background thread to keep session alive."""

        def keep_alive_task():
            while True:
                time.sleep(300)  # Every 5 minutes
                # Trigger some activity to prevent disconnect
                subprocess.run(["echo", "keep-alive"], stdout=subprocess.DEVNULL)

        self.keep_alive_thread = threading.Thread(target=keep_alive_task, daemon=True)
        self.keep_alive_thread.start()

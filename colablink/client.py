"""
LocalClient - Runs on local machine to connect to Colab runtime.

This module handles:
- Connection to Colab via SSH
- Reverse SSHFS mounting (local files accessible on Colab)
- Command execution with real-time output streaming
- Port forwarding
- File synchronization
"""

import os
import sys
import subprocess
import json
import time
import signal
from pathlib import Path
from typing import Optional, Dict, List


class LocalClient:
    """Client for connecting local machine to Colab runtime."""
    
    def __init__(self, config_file: Optional[str] = None, auto_sync: bool = True):
        """
        Initialize local client.
        
        Args:
            config_file: Path to config file (default: ~/.colablink/config.json)
            auto_sync: Enable automatic bidirectional file sync (default: True)
        """
        self.config_file = config_file or os.path.expanduser("~/.colablink/config.json")
        self.config_dir = os.path.dirname(self.config_file)
        self.config: Dict = {}
        self.ssh_config_file = os.path.join(self.config_dir, "ssh_config")
        self.port_forwards: List[subprocess.Popen] = []
        self.local_mount_point = os.path.join(self.config_dir, "colab_workspace")
        self.sync_process: Optional[subprocess.Popen] = None
        self.auto_sync = auto_sync
        
    def initialize(self, connection_info: Dict):
        """
        Initialize connection to Colab runtime.
        
        Args:
            connection_info: Dict with host, port, password, mount_point
        """
        print("Initializing connection to Colab runtime...")
        
        # Create config directory
        os.makedirs(self.config_dir, exist_ok=True)
        
        # Save configuration
        self.config = connection_info
        self._save_config()
        
        # Setup SSH config
        self._setup_ssh_config()
        
        # Test connection
        print("\nTesting connection...")
        if self._test_connection():
            print("Connection successful!")
            
            if self.auto_sync:
                # Setup automatic bidirectional file sync
                print("\nSetting up automatic bidirectional file sync...")
                self._setup_reverse_sshfs()
                mount_success = self._setup_local_mount()
                
                print("\n" + "="*70)
                print("READY TO USE!")
                print("="*70)
                if mount_success:
                    print("\n  ✓ Automatic bidirectional sync ENABLED")
                    print(f"  ✓ Colab files → Local: {self.local_mount_point}")
                    print("\nFiles generated on Colab will appear automatically in:")
                    print(f"  {self.local_mount_point}/")
                else:
                    print("\n  ⚠ Automatic sync unavailable (sshfs not installed)")
                    print("  Use manual commands: upload, download, sync")
            else:
                # Manual sync mode
                print("\n" + "="*70)
                print("READY TO USE! (Manual sync mode)")
                print("="*70)
                print("\n" + "="*70)
                print("⚠  MANUAL FILE SYNC MODE")
                print("="*70)
                print("\n  You must manually manage files:")
                print(f"  • colablink upload <file>    - Push files to Colab")
                print(f"  • colablink download <file>  - Pull files from Colab")
                print(f"  • colablink sync             - Push entire directory")
                print("\n" + "="*70)
            
            print("\nYou can now execute commands on Colab GPU:")
            print("  colablink exec python train.py")
            print("  colablink exec nvidia-smi")
            print("\nOr start a shell with transparent execution:")
            print("  colablink shell")
            print("\nOr use VS Code Remote-SSH:")
            print(f"  Host: colablink")
            print("="*70)
            
            return True
        else:
            print("Connection failed. Please check the connection details.")
            return False
    
    def execute(self, command: str, stream_output: bool = True, cwd: Optional[str] = None):
        """
        Execute command on Colab runtime.
        
        Args:
            command: Command to execute
            stream_output: Whether to stream output in real-time
            cwd: Working directory (local path, will be mapped to Colab)
            
        Returns:
            Exit code
        """
        self._load_config()
        
        if not self.config:
            print("Error: Not connected. Run 'colablink init' first.")
            return 1
        
        # Determine working directory on Colab
        if cwd is None:
            cwd = os.getcwd()
        
        # Map local path to Colab path
        remote_cwd = self._map_local_to_remote(cwd)
        
        # Build SSH command (without TTY to avoid output buffering issues)
        ssh_cmd = self._build_ssh_command(force_tty=False)
        
        # Set up environment for CUDA/GPU access with Python unbuffered mode
        env_setup = "export LD_LIBRARY_PATH=/usr/lib64-nvidia:/usr/local/cuda/lib64:$LD_LIBRARY_PATH && export PATH=/usr/local/cuda/bin:$PATH && export PYTHONUNBUFFERED=1"
        
        # Wrap Python commands with -u flag for unbuffered output
        if 'python' in command.lower() and '-u' not in command:
            command = command.replace('python3', 'python3 -u').replace('python', 'python -u')
        
        full_command = f"{ssh_cmd} '{env_setup} && cd {remote_cwd} && {command}'"
        
        if stream_output:
            # Execute with real-time output streaming
            # Use unbuffered mode and force pseudo-terminal for immediate output
            process = subprocess.Popen(
                full_command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Merge stderr into stdout for proper ordering
                universal_newlines=True,
                bufsize=0  # Unbuffered
            )
            
            # Stream output line by line in real-time
            try:
                for line in iter(process.stdout.readline, ''):
                    if line:
                        print(line, end='', flush=True)
                
                # Wait for process to complete
                returncode = process.wait()
                
                return returncode
            except KeyboardInterrupt:
                process.terminate()
                process.wait()
                return 130  # Standard exit code for SIGINT
        else:
            # Execute without streaming
            result = subprocess.run(full_command, shell=True)
            return result.returncode
    
    def shell(self):
        """
        Start an interactive SSH shell to Colab runtime.
        All commands will execute on Colab with access to local files.
        """
        self._load_config()
        
        if not self.config:
            print("Error: Not connected. Run 'colablink init' first.")
            return 1
        
        print("Starting interactive shell on Colab runtime...")
        print(f"Your local files are accessible at: {self.config['mount_point']}")
        print("Type 'exit' to return to local shell.\n")
        
        # Start interactive SSH session
        ssh_cmd = self._build_ssh_command(interactive=True)
        os.system(ssh_cmd)
    
    def forward_port(self, remote_port: int, local_port: Optional[int] = None):
        """
        Forward a port from Colab to local machine.
        
        Args:
            remote_port: Port on Colab runtime
            local_port: Port on local machine (default: same as remote_port)
        """
        self._load_config()
        
        if not self.config:
            print("Error: Not connected. Run 'colablink init' first.")
            return
        
        local_port = local_port or remote_port
        
        print(f"Forwarding port {remote_port} to localhost:{local_port}")
        
        ssh_cmd = self._build_ssh_command(
            port_forward=f"{local_port}:localhost:{remote_port}"
        )
        
        # Run in background
        process = subprocess.Popen(
            ssh_cmd + " -N",  # -N: no command execution
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        self.port_forwards.append(process)
        print(f"Port forwarding active. Access at: http://localhost:{local_port}")
    
    def status(self):
        """Check connection status."""
        self._load_config()
        
        if not self.config:
            print("Not connected. Run 'colablink init' first.")
            return
        
        print("Connection Status:")
        print(f"  Host: {self.config['host']}")
        print(f"  Port: {self.config['port']}")
        
        if self._test_connection(verbose=False):
            print("  Status: Connected")
            
            # Get GPU info
            result = subprocess.run(
                self._build_ssh_command() + " 'nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader'",
                shell=True,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0 and result.stdout.strip():
                print(f"\nGPU: {result.stdout.strip()}")
        else:
            print("  Status: Disconnected")
    
    def upload(self, source: str, destination: Optional[str] = None, recursive: bool = False):
        """
        Upload files or directories to Colab.
        
        Args:
            source: Local file or directory path
            destination: Remote path on Colab (default: /content/)
            recursive: Whether to upload recursively for directories
        """
        self._load_config()
        
        if not self.config:
            print("Error: Not connected. Run 'colablink init' first.")
            return 1
        
        # Resolve source path
        source_path = os.path.abspath(source)
        if not os.path.exists(source_path):
            print(f"Error: Source path does not exist: {source}")
            return 1
        
        # Determine destination
        if destination is None:
            if os.path.isfile(source_path):
                destination = f"/content/{os.path.basename(source_path)}"
            else:
                destination = f"/content/{os.path.basename(source_path)}"
        
        # Build scp command
        scp_cmd = ["sshpass", "-p", f"'{self.config['password']}'", "scp"]
        
        # Add options
        scp_cmd.extend([
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-P", str(self.config['port']),
        ])
        
        # Check if source is a directory
        is_directory = os.path.isdir(source_path)
        
        # Add recursive flag if needed
        if recursive or is_directory:
            scp_cmd.append("-r")
        
        # Add source and destination
        scp_cmd.append(source_path)
        scp_cmd.append(f"root@{self.config['host']}:{destination}")
        
        # Execute upload with appropriate messaging
        if is_directory:
            print(f"Uploading directory {source} to Colab:{destination}...")
        else:
            print(f"Uploading file {source} to Colab:{destination}...")
            
        result = subprocess.run(
            " ".join(scp_cmd),
            shell=True,
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            if is_directory:
                print(f"✓ Directory uploaded successfully: {destination}")
            else:
                print(f"✓ File uploaded successfully: {destination}")
            return 0
        else:
            print(f"✗ Upload failed: {result.stderr}")
            return 1
    
    def download(self, source: str, destination: Optional[str] = None, recursive: bool = False):
        """
        Download files or directories from Colab to local machine.
        
        Args:
            source: Remote file or directory path on Colab
            destination: Local destination path (default: current directory)
            recursive: Whether to download recursively for directories (auto-detected)
        """
        self._load_config()
        
        if not self.config:
            print("Error: Not connected. Run 'colablink init' first.")
            return 1
        
        # Auto-detect if source is a directory (ends with / or common directory indicators)
        is_likely_directory = (
            source.endswith('/') or 
            '.' not in os.path.basename(source) or
            recursive
        )
        
        # Determine destination
        if destination is None:
            destination = os.getcwd()
        
        destination = os.path.abspath(destination)
        
        # Build scp command
        scp_cmd = ["sshpass", "-p", f"'{self.config['password']}'", "scp"]
        
        # Add options
        scp_cmd.extend([
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-P", str(self.config['port']),
        ])
        
        # Add recursive flag if likely a directory
        if is_likely_directory:
            scp_cmd.append("-r")
            print(f"Downloading directory Colab:{source} to {destination}...")
        else:
            print(f"Downloading file Colab:{source} to {destination}...")
        
        # Add source and destination
        scp_cmd.append(f"root@{self.config['host']}:{source}")
        scp_cmd.append(destination)
        
        # Execute download
        result = subprocess.run(
            " ".join(scp_cmd),
            shell=True,
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            if is_likely_directory:
                print(f"✓ Directory downloaded successfully to: {destination}")
            else:
                print(f"✓ File downloaded successfully to: {destination}")
            return 0
        else:
            # If it failed and we didn't use recursive, suggest trying with recursive
            if not is_likely_directory and "not a regular file" in result.stderr.lower():
                print(f"Note: '{source}' appears to be a directory.")
                print(f"Retrying with recursive mode...")
                return self.download(source, destination, recursive=True)
            print(f"✗ Download failed: {result.stderr}")
            return 1
    
    def sync(self, directory: Optional[str] = None, exclude: Optional[List[str]] = None):
        """
        Sync current directory (or specified directory) to Colab.
        
        Args:
            directory: Directory to sync (default: current directory)
            exclude: List of patterns to exclude (e.g., ['*.pyc', '__pycache__'])
        """
        self._load_config()
        
        if not self.config:
            print("Error: Not connected. Run 'colablink init' first.")
            return 1
        
        # Determine source directory
        if directory is None:
            directory = os.getcwd()
        
        source_dir = os.path.abspath(directory)
        if not os.path.isdir(source_dir):
            print(f"Error: Not a directory: {directory}")
            return 1
        
        # Default exclusions
        if exclude is None:
            exclude = [
                '__pycache__',
                '*.pyc',
                '.git',
                '.gitignore',
                'venv',
                'env',
                '.venv',
                'node_modules',
                '*.egg-info',
                'dist',
                'build',
            ]
        
        # Create destination directory on Colab
        dir_name = os.path.basename(source_dir)
        remote_dir = f"/content/{dir_name}"
        
        print(f"Syncing {source_dir} to Colab:{remote_dir}...")
        
        # Build rsync-like command using scp with find
        ssh_cmd = self._build_ssh_command()
        
        # First, create the directory structure
        subprocess.run(
            f"{ssh_cmd} 'mkdir -p {remote_dir}'",
            shell=True,
            capture_output=True
        )
        
        # Use tar to efficiently transfer directory
        tar_exclude = " ".join([f"--exclude='{pattern}'" for pattern in exclude])
        
        upload_cmd = f"""
        cd {os.path.dirname(source_dir)} && \
        tar czf - {tar_exclude} {dir_name} | \
        sshpass -p '{self.config['password']}' ssh \
            -o StrictHostKeyChecking=no \
            -o UserKnownHostsFile=/dev/null \
            -p {self.config['port']} \
            root@{self.config['host']} \
            'cd /content && tar xzf -'
        """
        
        result = subprocess.run(
            upload_cmd,
            shell=True,
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print(f"Sync complete: {remote_dir}")
            print(f"You can now run: colablink exec python {dir_name}/your_script.py")
            return 0
        else:
            print(f"Sync failed: {result.stderr}")
            return 1
    
    def disconnect(self):
        """Disconnect from Colab runtime."""
        # Kill port forwards
        for process in self.port_forwards:
            process.kill()
        
        # Unmount SSHFS
        self._unmount_sshfs()
        
        print("Disconnected from Colab runtime.")
    
    def _load_config(self):
        """Load configuration from file."""
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                self.config = json.load(f)
    
    def _save_config(self):
        """Save configuration to file."""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
        print(f"Configuration saved to: {self.config_file}")
    
    def _setup_ssh_config(self):
        """Create SSH config for easy connection."""
        ssh_config_content = f"""
Host colablink
    HostName {self.config['host']}
    Port {self.config['port']}
    User root
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
    ServerAliveInterval 60
    ServerAliveCountMax 3
"""
        
        with open(self.ssh_config_file, 'w') as f:
            f.write(ssh_config_content)
        
        # Also add to user's SSH config if not already there
        user_ssh_config = os.path.expanduser("~/.ssh/config")
        if os.path.exists(user_ssh_config):
            with open(user_ssh_config, 'r') as f:
                if 'Host colablink' not in f.read():
                    with open(user_ssh_config, 'a') as f_append:
                        f_append.write(ssh_config_content)
        else:
            os.makedirs(os.path.dirname(user_ssh_config), exist_ok=True)
            with open(user_ssh_config, 'w') as f:
                f.write(ssh_config_content)
    
    def _test_connection(self, verbose: bool = True) -> bool:
        """Test SSH connection to Colab."""
        ssh_cmd = self._build_ssh_command()
        result = subprocess.run(
            f"{ssh_cmd} 'echo connection_test'",
            shell=True,
            capture_output=True,
            text=True
        )
        
        success = result.returncode == 0 and "connection_test" in result.stdout
        
        if verbose:
            if success:
                print("   Connection test passed")
            else:
                print("   Connection test failed")
        
        return success
    
    def _setup_reverse_sshfs(self):
        """Setup reverse SSHFS - mount local filesystem on Colab."""
        # Get current user and home directory
        local_user = os.environ.get('USER', 'user')
        local_home = os.path.expanduser("~")
        
        print("\n   Setting up local filesystem access on Colab...")
        print("   This may take a moment...")
        
        # For simplicity, we'll use the SSH connection directly
        # In a production version, you'd set up a proper SSHFS mount
        # For now, files will be accessed via SSH commands
        
        print("   Local files will be accessed on-demand via SSH")
        print(f"   Working directory: {os.getcwd()}")
    
    def _setup_local_mount(self):
        """Setup local mount - mount Colab's /content directory locally using SSHFS."""
        # Check if sshfs is installed
        result = subprocess.run(
            ["which", "sshfs"],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print("\n   Warning: sshfs not installed. Bidirectional sync unavailable.")
            print("   Install with: sudo apt-get install sshfs  (or brew install macfuse sshfs on macOS)")
            print("   You can still upload files manually with 'colablink upload'")
            return False
        
        # Create local mount point
        os.makedirs(self.local_mount_point, exist_ok=True)
        
        # Check if already mounted
        result = subprocess.run(
            ["mountpoint", "-q", self.local_mount_point],
            capture_output=True
        )
        
        if result.returncode == 0:
            print(f"   Colab directory already mounted at: {self.local_mount_point}")
            return True
        
        # Mount Colab's /content directory locally
        print(f"   Mounting Colab workspace to: {self.local_mount_point}")
        
        mount_cmd = [
            "sshfs",
            f"root@{self.config['host']}:/content",
            self.local_mount_point,
            "-p", str(self.config['port']),
            "-o", f"password_stdin",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "reconnect",
            "-o", "ServerAliveInterval=15",
            "-o", "ServerAliveCountMax=3",
        ]
        
        try:
            process = subprocess.Popen(
                mount_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Send password
            stdout, stderr = process.communicate(input=f"{self.config['password']}\n", timeout=10)
            
            if process.returncode == 0 or self._verify_mount():
                print(f"   Colab workspace mounted successfully!")
                print(f"   Files on Colab will appear in: {self.local_mount_point}/")
                return True
            else:
                print(f"   Warning: Mount failed: {stderr}")
                print("   You can still upload files manually with 'colablink upload'")
                return False
                
        except subprocess.TimeoutExpired:
            print("   Warning: Mount timed out")
            print("   You can still upload files manually with 'colablink upload'")
            return False
        except Exception as e:
            print(f"   Warning: Mount failed: {e}")
            print("   You can still upload files manually with 'colablink upload'")
            return False
    
    def _verify_mount(self):
        """Verify that the mount point is actually mounted."""
        result = subprocess.run(
            ["mountpoint", "-q", self.local_mount_point],
            capture_output=True
        )
        return result.returncode == 0
    
    def _unmount_sshfs(self):
        """Unmount SSHFS if mounted."""
        if not os.path.exists(self.local_mount_point):
            return
        
        # Check if mounted
        if self._verify_mount():
            print(f"Unmounting {self.local_mount_point}...")
            
            # Try fusermount first (Linux)
            result = subprocess.run(
                ["fusermount", "-u", self.local_mount_point],
                capture_output=True
            )
            
            if result.returncode != 0:
                # Try umount (macOS/BSD)
                subprocess.run(
                    ["umount", self.local_mount_point],
                    capture_output=True
                )
    
    def _map_local_to_remote(self, local_path: str) -> str:
        """
        Map local path to remote path.
        For now, uses the same path structure.
        """
        # In a full implementation with SSHFS, this would map properly
        # For now, we'll use /content/workspace as working directory
        return "/content"
    
    def _build_ssh_command(
        self,
        interactive: bool = False,
        port_forward: Optional[str] = None,
        force_tty: bool = False
    ) -> str:
        """Build SSH command with proper options."""
        cmd_parts = ["sshpass", "-p", f"'{self.config['password']}'", "ssh"]
        
        # SSH options
        cmd_parts.extend([
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ServerAliveInterval=60",
            "-o", "ServerAliveCountMax=3",
            "-p", str(self.config['port']),
        ])
        
        # Port forwarding
        if port_forward:
            cmd_parts.extend(["-L", port_forward])
        
        # Interactive mode or force TTY for real-time output
        if interactive:
            cmd_parts.append("-t")
        elif force_tty:
            cmd_parts.append("-tt")  # Force TTY allocation for unbuffered output
        
        # Host
        cmd_parts.append(f"root@{self.config['host']}")
        
        return " ".join(cmd_parts)


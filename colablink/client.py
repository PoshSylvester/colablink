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
import shlex
import shutil
from pathlib import Path, PurePosixPath
from typing import Optional, Dict, List, Sequence, Union


class LocalClient:
    """Client for connecting local machine to Colab runtime."""

    def __init__(
        self,
        config_file: Optional[str] = None,
        auto_sync: bool = True,
        local_mount_dir: Optional[str] = None,
    ):
        """
        Initialize local client.

        Args:
            config_file: Path to config file (default: ~/.colablink/config.json)
            auto_sync: Enable automatic bidirectional file sync (default: True)
            local_mount_dir: Directory to mount Colab files (default: ./colab-workspace)
        """
        self.config_file = config_file or os.path.expanduser("~/.colablink/config.json")
        self.config_dir = os.path.dirname(self.config_file)
        self.config: Dict = {}
        self.ssh_config_file = os.path.join(self.config_dir, "ssh_config")
        self.port_forwards: List[subprocess.Popen] = []
        self.project_root: Optional[Path] = None
        self.remote_workspace: Optional[str] = None

        # Default local mount to ./colab-workspace in current directory
        if local_mount_dir is None:
            self.local_mount_point = os.path.join(os.getcwd(), "colab-workspace")
        else:
            self.local_mount_point = os.path.abspath(local_mount_dir)

        self.sync_process: Optional[subprocess.Popen] = None
        self.auto_sync = auto_sync
        self.default_excludes = [
            "__pycache__",
            "*.pyc",
            ".git",
            ".gitignore",
            ".venv",
            "venv",
            "env",
            "node_modules",
            "*.egg-info",
            "dist",
            "build",
        ]
        self.has_rsync = True

    def initialize(self, connection_info: Dict):
        """Initialize connection to Colab runtime."""
        print("Initializing connection to Colab runtime...")

        # Check dependencies early so we can gracefully degrade if needed
        self._check_dependencies()

        # Establish paths
        os.makedirs(self.config_dir, exist_ok=True)
        os.makedirs(self.local_mount_point, exist_ok=True)
        self.project_root = Path(os.getcwd()).resolve()

        username = connection_info.get("username", "root")
        remote_workspace = connection_info.get("workspace")
        if remote_workspace is None:
            remote_workspace = f"/content/{self.project_root.name}"

        # Persist configuration enriched with local metadata
        self.config = {
            **connection_info,
            "username": username,
            "project_root": str(self.project_root),
            "remote_workspace": remote_workspace,
            "auto_sync": self.auto_sync,
        }
        self.remote_workspace = remote_workspace

        self._save_config()
        self._setup_ssh_config()
        self._ensure_remote_workspace()

        # Test connection
        print("\nTesting connection...")
        if not self._test_connection():
            print("Connection failed. Please check the connection details.")
            return False

        print("Connection successful!")

        if self.auto_sync:
            print("\nSetting up automatic bidirectional file sync...")
            initial_push = self._sync_local_to_remote(initial=True)
            initial_pull = self._sync_remote_to_local(initial=True)
            mount_success = self._setup_local_mount()

            print("\n" + "=" * 70)
            print("READY TO USE!")
            print("=" * 70)
            if mount_success:
                print("\n  [OK] Colab workspace mounted at:")
                print(f"      {self.local_mount_point}")
            else:
                print("\n  [INFO] Local mount skipped (sshfs unavailable)")

            if initial_push == 0:
                print("\n  [OK] Local project synced to Colab")
            if initial_pull == 0:
                print("  [OK] Colab workspace synced back to local")

            print(f"\nRemote workspace: {self.remote_workspace}")

            print(
                "\nFiles created on Colab will flow back automatically after commands run."
            )
        else:
            print("\n" + "=" * 70)
            print("READY TO USE! (Manual sync mode)")
            print("=" * 70)
            print("\n  Manage files manually with:")
            print("    colablink upload <path>")
            print("    colablink download <path>")
            print("    colablink sync")
            print(f"\nRemote workspace: {self.remote_workspace}")

        print("\nYou can now execute commands on Colab runtime:")
        print("  colablink exec python train.py")
        print("  colablink exec nvidia-smi")
        print("\nOr start a shell with transparent execution:")
        print("  colablink shell")
        print("\nOr use VS Code Remote-SSH:")
        print("  Host: colablink")
        print("=" * 70)

        return True

    def execute(
        self,
        command: Union[str, Sequence[Union[str, Path]]],
        stream_output: bool = True,
        cwd: Optional[Union[str, Path]] = None,
    ) -> int:
        """Execute a command on the Colab runtime.

        Args:
            command: Command string or token sequence. Sequences are treated as
                literal arguments and passed through without shell parsing.
            stream_output: When True, stream remote stdout/stderr in real-time.
            cwd: Optional local working directory whose contents map to the
                remote workspace. Accepts ``str`` or ``pathlib.Path``.

        Returns:
            Exit code reported by the remote process.
        """
        self._load_config()

        if not self.config:
            print("Error: Not connected. Run 'colablink init' first.")
            return 1

        self.project_root = Path(self.config.get("project_root", os.getcwd())).resolve()
        self.remote_workspace = self.config.get("remote_workspace", "/content")

        # Normalize command tokens and add python -u when appropriate
        cmd_tokens = self._normalize_command(command)
        command_str = self._join_command_tokens(cmd_tokens)

        # Determine working directory mapping
        local_cwd = Path(cwd or os.getcwd()).resolve()
        remote_cwd = self._map_local_to_remote(str(local_cwd))

        if self.auto_sync:
            self._sync_local_to_remote()

        self._ensure_remote_directory(remote_cwd)

        ssh_cmd = self._build_ssh_command(force_tty=False)

        env_setup = (
            "export PYTHONUNBUFFERED=1 && "
            "if [ -d /usr/lib64-nvidia ]; then "
            "export LD_LIBRARY_PATH=/usr/lib64-nvidia:/usr/local/cuda/lib64:$LD_LIBRARY_PATH; "
            "export PATH=/usr/local/cuda/bin:$PATH; "
            "fi"
        )

        remote_script = f"{env_setup} && cd {shlex.quote(remote_cwd)} && {command_str}"
        full_cmd = ssh_cmd + ["bash", "-lc", remote_script]

        if stream_output:
            process = subprocess.Popen(
                full_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            try:
                assert process.stdout is not None
                for line in process.stdout:
                    print(line, end="", flush=True)
                returncode = process.wait()
            except KeyboardInterrupt:
                process.terminate()
                process.wait()
                returncode = 130
        else:
            result = subprocess.run(full_cmd)
            returncode = result.returncode

        if self.auto_sync:
            self._sync_remote_to_local()

        return returncode

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
        print("Type 'exit' to return to local shell.\n")

        # Start interactive SSH session
        ssh_cmd = self._build_ssh_command(interactive=True)
        subprocess.call(ssh_cmd)

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

        self._clean_port_forwards()

        ssh_cmd = self._build_ssh_command(
            port_forward=f"{local_port}:localhost:{remote_port}",
            extra_args=["-N"],
        )

        # Run in background
        process = subprocess.Popen(
            ssh_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
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
            self._clean_port_forwards()

            # Get GPU info
            result = subprocess.run(
                self._build_ssh_command()
                + [
                    "bash",
                    "-lc",
                    "nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader",
                ],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0 and result.stdout.strip():
                print(f"\nGPU: {result.stdout.strip()}")
        else:
            print("  Status: Disconnected")

    def upload(
        self, source: str, destination: Optional[str] = None, recursive: bool = False
    ):
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

        is_directory = os.path.isdir(source_path)

        remote_base = self.remote_workspace or self.config.get("workspace", "/content")
        if destination is None:
            if is_directory:
                target_path = remote_base
            else:
                target_path = str(
                    PurePosixPath(remote_base) / os.path.basename(source_path)
                )
        else:
            target_path = self._resolve_remote_path(destination)

        if (
            destination is not None
            and not is_directory
            and destination.rstrip().endswith("/")
        ):
            print(
                "Note: Destination ends with '/'; the file will be placed inside "
                f"{target_path}."
            )

        if is_directory or (destination and destination.endswith("/")):
            self._ensure_remote_directory(target_path)
        else:
            parent = str(PurePosixPath(target_path).parent)
            self._ensure_remote_directory(parent)

        scp_cmd: List[str] = [
            "sshpass",
            "-p",
            self.config["password"],
            "scp",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-P",
            str(self.config["port"]),
        ]

        if recursive or is_directory:
            scp_cmd.append("-r")

        scp_cmd.append(source_path)
        scp_cmd.append(f"{self._ssh_target()}:{target_path}")

        if is_directory:
            print(f"Uploading directory {source} to Colab:{target_path}...")
        else:
            print(f"Uploading file {source} to Colab:{target_path}...")

        result = subprocess.run(scp_cmd, capture_output=True, text=True)

        if result.returncode == 0:
            message = "Directory" if is_directory else "File"
            print(f"[OK] {message} uploaded successfully: {target_path}")
            return 0

        print(f"[ERROR] Upload failed: {result.stderr.strip()}")
        return 1

    def download(
        self,
        source: str,
        destination: Optional[str] = None,
        recursive: bool = False,
        overwrite: bool = False,
    ) -> int:
        """
        Download files or directories from Colab to local machine.

        Args:
            source: Remote file or directory path on Colab
            destination: Local destination path (default: current directory)
            recursive: Whether to download recursively for directories (auto-detected)
            overwrite: When False, refuse to overwrite existing files.
        """
        self._load_config()

        if not self.config:
            print("Error: Not connected. Run 'colablink init' first.")
            return 1

        # Auto-detect if source is a directory (ends with / or common directory indicators)
        is_likely_directory = (
            source.endswith("/") or "." not in os.path.basename(source) or recursive
        )

        # Determine destination
        if destination is None:
            destination = os.getcwd()

        destination_input = destination
        destination_path = Path(destination_input).expanduser().resolve(strict=False)

        if is_likely_directory or (
            destination_input and destination_input.rstrip().endswith(os.sep)
        ):
            destination_path.mkdir(parents=True, exist_ok=True)
            local_target = str(destination_path)
        else:
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            if (
                destination_path.exists()
                and not overwrite
                and not destination_path.is_dir()
                and not is_likely_directory
            ):
                print(
                    f"Error: Destination file already exists: {destination_path}. "
                    "Use --force to overwrite."
                )
                return 1
            local_target = str(destination_path)

        remote_source = self._resolve_remote_path(source)

        scp_cmd: List[str] = [
            "sshpass",
            "-p",
            self.config["password"],
            "scp",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-P",
            str(self.config["port"]),
        ]

        # Add recursive flag if likely a directory
        if is_likely_directory:
            scp_cmd.append("-r")
            print(f"Downloading directory Colab:{source} to {destination_input}...")
        else:
            print(f"Downloading file Colab:{source} to {destination_input}...")

        scp_cmd.append(f"{self._ssh_target()}:{remote_source}")
        scp_cmd.append(local_target)

        # Execute download
        result = subprocess.run(scp_cmd, capture_output=True, text=True)

        if result.returncode == 0:
            if is_likely_directory:
                print(f"[OK] Directory downloaded successfully to: {local_target}")
            else:
                print(f"[OK] File downloaded successfully to: {local_target}")
            return 0
        else:
            # If it failed and we didn't use recursive, suggest trying with recursive
            if (
                not is_likely_directory
                and "not a regular file" in result.stderr.lower()
            ):
                print(f"Note: '{source}' appears to be a directory.")
                print(f"Retrying with recursive mode...")
                return self.download(
                    source,
                    destination_input,
                    recursive=True,
                    overwrite=overwrite,
                )
            print(f"[ERROR] Download failed: {result.stderr.strip()}")
            return 1

    def sync(
        self,
        directory: Optional[str] = None,
        exclude: Optional[List[str]] = None,
        dry_run: bool = False,
        progress: bool = False,
    ) -> int:
        """Sync current directory (or specified directory) to Colab.

        Args:
            directory: Directory to sync (default: current directory)
            exclude: List of exclusion patterns (e.g., ['*.pyc', '__pycache__'])
            dry_run: Show transfer plan without copying files when True.
            progress: Emit rsync transfer progress when True.
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

        excludes = exclude or self.default_excludes

        remote_dir = self._map_local_to_remote(source_dir)
        self._ensure_remote_directory(remote_dir)

        print(f"Syncing {source_dir} to Colab:{remote_dir}...")

        source = f"{source_dir}/"
        destination = f"{self._ssh_target()}:{remote_dir}/"

        result = self._run_rsync(
            source,
            destination,
            excludes=excludes,
            delete=True,
            operation="upload",
            dry_run=dry_run,
            progress=progress,
        )

        if result == 0:
            print(f"Sync complete: {remote_dir}")
        return result

    def disconnect(self):
        """Disconnect from Colab runtime."""
        self._load_config()

        if self.auto_sync and self.config:
            self._sync_remote_to_local()

        # Kill port forwards
        for process in self.port_forwards:
            process.kill()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.terminate()
        self.port_forwards.clear()

        # Unmount SSHFS
        self._unmount_sshfs()

        print("Disconnected from Colab runtime.")

    def _load_config(self):
        """Load configuration from file."""
        if os.path.exists(self.config_file):
            with open(self.config_file, "r") as f:
                self.config = json.load(f)
            if self.config:
                self.config.setdefault("username", "root")
                remote_default = self.config.get("workspace", "/content")
                self.config.setdefault("remote_workspace", remote_default)
                self.auto_sync = self.config.get("auto_sync", self.auto_sync)

    def _save_config(self):
        """Save configuration to file."""
        with open(self.config_file, "w") as f:
            json.dump(self.config, f, indent=2)
        print(f"Configuration saved to: {self.config_file}")

    def _setup_ssh_config(self):
        """Create SSH config for easy connection."""
        ssh_config_content = f"""
Host colablink
    HostName {self.config['host']}
    Port {self.config['port']}
    User {self.config.get('username', 'root')}
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
    ServerAliveInterval 60
    ServerAliveCountMax 3
"""

        with open(self.ssh_config_file, "w") as f:
            f.write(ssh_config_content)

        # Also add to user's SSH config if not already there
        user_ssh_config = os.path.expanduser("~/.ssh/config")
        if os.path.exists(user_ssh_config):
            with open(user_ssh_config, "r") as f:
                if "Host colablink" not in f.read():
                    with open(user_ssh_config, "a") as f_append:
                        f_append.write(ssh_config_content)
        else:
            os.makedirs(os.path.dirname(user_ssh_config), exist_ok=True)
            with open(user_ssh_config, "w") as f:
                f.write(ssh_config_content)

    def _test_connection(self, verbose: bool = True) -> bool:
        """Test SSH connection to Colab."""
        ssh_cmd = self._build_ssh_command()
        result = subprocess.run(
            ssh_cmd + ["bash", "-lc", "echo connection_test"],
            capture_output=True,
            text=True,
        )

        success = result.returncode == 0 and "connection_test" in result.stdout

        if verbose:
            if success:
                print("   Connection test passed")
            else:
                print("   Connection test failed")

        return success

    def _setup_local_mount(self):
        """Setup local mount - mount Colab's /content directory locally using SSHFS."""
        # Check if sshfs is installed
        result = subprocess.run(["which", "sshfs"], capture_output=True, text=True)

        if result.returncode != 0:
            print("\n   Warning: sshfs not installed. Bidirectional sync unavailable.")
            print(
                "   Install with: sudo apt-get install sshfs  (or brew install macfuse sshfs on macOS)"
            )
            print("   You can still upload files manually with 'colablink upload'")
            return False

        # Create local mount point
        os.makedirs(self.local_mount_point, exist_ok=True)

        # Check if already mounted
        result = subprocess.run(
            ["mountpoint", "-q", self.local_mount_point], capture_output=True
        )

        if result.returncode == 0:
            print(f"   Colab directory already mounted at: {self.local_mount_point}")
            return True

        remote_path = self.remote_workspace or "/content"
        print(
            f"   Mounting Colab workspace ({remote_path}) to: {self.local_mount_point}"
        )

        mount_cmd = [
            "sshfs",
            f"{self._ssh_target()}:{remote_path}",
            self.local_mount_point,
            "-p",
            str(self.config["port"]),
            "-o",
            "password_stdin",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-o",
            "reconnect",
            "-o",
            "ServerAliveInterval=15",
            "-o",
            "ServerAliveCountMax=3",
        ]

        try:
            process = subprocess.Popen(
                mount_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            # Send password
            stdout, stderr = process.communicate(
                input=f"{self.config['password']}\n", timeout=10
            )

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
            ["mountpoint", "-q", self.local_mount_point], capture_output=True
        )
        return result.returncode == 0

    def _check_dependencies(self):
        """Verify system dependencies, attempting auto-install when possible."""
        import platform

        system = platform.system()

        sshpass_missing = shutil.which("sshpass") is None
        sshfs_available = shutil.which("sshfs") is not None
        rsync_available = shutil.which("rsync") is not None
        self.has_rsync = rsync_available

        packages_needed: List[str] = []
        if sshpass_missing:
            packages_needed.append("sshpass")
        if not sshfs_available:
            packages_needed.append("sshfs")
        if not rsync_available:
            packages_needed.append("rsync")

        install_attempted = False
        if packages_needed:
            install_attempted = self._auto_install_dependencies(packages_needed)
            if install_attempted:
                sshpass_missing = shutil.which("sshpass") is None
                sshfs_available = shutil.which("sshfs") is not None
                rsync_available = shutil.which("rsync") is not None
                self.has_rsync = rsync_available

        if sshpass_missing or not sshfs_available or not rsync_available:
            print("\n" + "=" * 70)
            header = (
                "WARNING: MISSING REQUIRED DEPENDENCIES"
                if sshpass_missing
                else "NOTICE: OPTIONAL DEPENDENCIES UNAVAILABLE"
            )
            print(header)
            print("=" * 70)

        if sshpass_missing:
            print("\nColabLink requires the following system packages:\n")
            print("  - sshpass (required for SSH authentication)")
        elif packages_needed:
            print("\nSome optional packages are still missing after auto-install.")

        if sshpass_missing or not sshfs_available or not rsync_available:
            if system == "Linux":
                print("\nInstall with:")
                print("  sudo apt-get update")
                if sshpass_missing:
                    print("  sudo apt-get install sshpass")
                if not sshfs_available:
                    print("  sudo apt-get install sshfs  # Optional: for auto-sync")
                if not rsync_available:
                    print("  sudo apt-get install rsync")
            elif system == "Darwin":
                print("\nInstall with Homebrew:")
                if sshpass_missing:
                    print("  brew install hudochenkov/sshpass/sshpass")
                if not sshfs_available:
                    print("  brew install macfuse sshfs  # Optional: for auto-sync")
                if not rsync_available:
                    print("  brew install rsync")
            else:
                print(
                    "\nAutomatic installation is not supported on this platform. "
                    "Please install the packages manually."
                )

            print("\n" + "=" * 70)

        if sshpass_missing:
            print(
                "\nError: Cannot proceed without sshpass. Please install and try again."
            )
            print("=" * 70)
            sys.exit(1)

        if not sshfs_available:
            print(
                "\nNote: sshfs not found. Automatic bidirectional sync will be unavailable."
            )
            print(
                "      Install sshfs for automatic file sync, or use manual upload/download commands."
            )

        if not rsync_available:
            if self.auto_sync:
                print(
                    "\nNote: rsync not found. Disabling automatic sync and falling back to manual commands."
                )
                self.auto_sync = False
            else:
                print(
                    "\nNote: rsync not found. Automatic sync features will be disabled until installed."
                )

    def _auto_install_dependencies(self, packages: List[str]) -> bool:
        """Attempt to install dependencies automatically."""
        import platform

        packages = sorted(set(packages))
        if not packages:
            return False

        system = platform.system()
        print("\nAttempting to install missing dependencies automatically...")

        if system == "Linux" and shutil.which("apt-get"):
            env = os.environ.copy()
            env.setdefault("DEBIAN_FRONTEND", "noninteractive")

            cmd_prefix: List[str] = []
            if hasattr(os, "geteuid") and os.geteuid() != 0:
                sudo_path = shutil.which("sudo")
                if not sudo_path:
                    print("sudo not found; cannot auto-install dependencies.")
                    return False
                cmd_prefix = [sudo_path, "-n"]

            update_cmd = cmd_prefix + ["apt-get", "update", "-qq"]
            subprocess.run(
                update_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )

            install_cmd = cmd_prefix + ["apt-get", "install", "-y"] + packages
            result = subprocess.run(
                install_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )

            if result.returncode != 0:
                print(f"[auto-install] apt-get install failed: {result.stderr.strip()}")
                return False

            return True

        if system == "Darwin" and shutil.which("brew"):
            commands: List[List[str]] = []
            for pkg in packages:
                if pkg == "sshpass":
                    commands.append(["brew", "install", "hudochenkov/sshpass/sshpass"])
                elif pkg == "sshfs":
                    commands.append(["brew", "install", "macfuse"])
                    commands.append(["brew", "install", "sshfs"])
                else:
                    commands.append(["brew", "install", pkg])

            seen = set()
            for cmd in commands:
                key = tuple(cmd)
                if key in seen:
                    continue
                seen.add(key)
                result = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                if result.returncode != 0:
                    print(
                        f"[auto-install] {' '.join(cmd)} failed: {result.stderr.strip()}"
                    )
                    return False
            return True

        print("Automatic installation is not supported on this platform.")
        return False

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
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                # Try umount (macOS/BSD)
                subprocess.run(
                    ["umount", self.local_mount_point],
                    capture_output=True,
                    text=True,
                )

    def _clean_port_forwards(self):
        """Remove finished port forward processes from tracking."""
        alive: List[subprocess.Popen] = []
        for process in self.port_forwards:
            if process.poll() is None:
                alive.append(process)
        self.port_forwards = alive

    def _map_local_to_remote(self, local_path: str) -> str:
        """Translate a local path to its remote workspace counterpart."""
        remote_base = PurePosixPath(
            self.remote_workspace or self.config.get("workspace", "/content")
        )
        project_root = Path(self.config.get("project_root", os.getcwd())).resolve()
        local_path_obj = Path(local_path).resolve()

        try:
            relative = local_path_obj.relative_to(project_root)
        except ValueError:
            return str(remote_base)

        if relative == Path("."):
            return str(remote_base)

        remote_path = remote_base.joinpath(PurePosixPath(*relative.parts))
        return str(remote_path)

    def _resolve_remote_path(self, remote_path: str) -> str:
        """Return an absolute remote path inside the workspace."""
        if not remote_path:
            return str(PurePosixPath(self.remote_workspace or "/content"))
        if remote_path.startswith("/"):
            return remote_path
        base = PurePosixPath(self.remote_workspace or "/content")
        return str(base.joinpath(PurePosixPath(remote_path)))

    def _build_ssh_command(
        self,
        interactive: bool = False,
        port_forward: Optional[str] = None,
        force_tty: bool = False,
        extra_args: Optional[List[str]] = None,
    ) -> List[str]:
        """Build SSH command with proper options."""
        cmd_parts: List[str] = [
            "sshpass",
            "-p",
            self.config["password"],
            "ssh",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-o",
            "ServerAliveInterval=60",
            "-o",
            "ServerAliveCountMax=3",
            "-p",
            str(self.config["port"]),
        ]

        if port_forward:
            cmd_parts.extend(["-L", port_forward])

        if interactive:
            cmd_parts.append("-t")
        elif force_tty:
            cmd_parts.append("-tt")

        if extra_args:
            cmd_parts.extend(extra_args)

        cmd_parts.append(self._ssh_target())
        return cmd_parts

    def _ssh_target(self) -> str:
        """Return user@host string for SSH commands."""
        username = self.config.get("username", "root")
        return f"{username}@{self.config['host']}"

    def _join_command_tokens(self, tokens: List[str]) -> str:
        """Join command tokens with shlex, supporting older Python versions."""
        if hasattr(shlex, "join"):
            return shlex.join(tokens)  # type: ignore[attr-defined]
        return " ".join(shlex.quote(token) for token in tokens)

    def _normalize_command(
        self, command: Union[str, Sequence[Union[str, Path]]]
    ) -> List[str]:
        """Return normalized token list for a command."""
        if isinstance(command, Sequence) and not isinstance(command, str):
            tokens = [str(token) for token in command]
        else:
            tokens = shlex.split(command)

        if tokens:
            first = tokens[0]
            if first.startswith("python") and "-u" not in tokens[1:]:
                tokens.insert(1, "-u")
        return tokens

    def _ensure_remote_workspace(self):
        """Make sure the remote workspace directory exists."""
        self.remote_workspace = self.config.get(
            "remote_workspace", self.config.get("workspace", "/content")
        )
        remote_path = shlex.quote(self.remote_workspace)
        ssh_cmd = self._build_ssh_command()
        subprocess.run(
            ssh_cmd + ["bash", "-lc", f"mkdir -p {remote_path}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )

    def _ensure_remote_directory(self, remote_path: str):
        """Create a remote directory if it does not exist."""
        if not remote_path:
            return
        ssh_cmd = self._build_ssh_command()
        subprocess.run(
            ssh_cmd + ["bash", "-lc", f"mkdir -p {shlex.quote(remote_path)}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )

    def _run_rsync(
        self,
        source: str,
        destination: str,
        excludes: Optional[List[str]] = None,
        delete: bool = False,
        operation: str = "transfer",
        dry_run: bool = False,
        progress: bool = False,
    ) -> int:
        """Execute rsync with shared SSH configuration."""
        if not self.has_rsync:
            return 1

        excludes = excludes or []
        ssh_command = [
            "ssh",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-o",
            "ServerAliveInterval=60",
            "-o",
            "ServerAliveCountMax=3",
            "-p",
            str(self.config["port"]),
        ]

        rsync_cmd: List[str] = [
            "sshpass",
            "-p",
            self.config["password"],
            "rsync",
            "-az",
        ]

        if delete:
            rsync_cmd.append("--delete")

        for pattern in excludes:
            rsync_cmd.extend(["--exclude", pattern])

        if dry_run:
            rsync_cmd.append("--dry-run")

        if progress:
            rsync_cmd.append("--info=progress2")

        rsync_cmd.extend(
            ["-e", self._join_command_tokens(ssh_command), source, destination]
        )

        stdout_target = None if progress else subprocess.PIPE
        result = subprocess.run(
            rsync_cmd,
            text=True,
            stdout=stdout_target,
            stderr=subprocess.PIPE,
        )

        if result.returncode != 0:
            print(f"[WARNING] rsync {operation} failed: {result.stderr.strip()}")
        elif dry_run and not progress and result.stdout:
            print(result.stdout.strip())

        return result.returncode

    def _sync_local_to_remote(self, initial: bool = False) -> int:
        """Synchronize local project to remote workspace."""
        if not self.auto_sync or not self.has_rsync:
            return 1

        project_root = Path(self.config.get("project_root", os.getcwd())).resolve()
        remote_path = self.remote_workspace or "/content"
        self._ensure_remote_directory(remote_path)

        if initial:
            print("   Uploading local project state to Colab...")

        source = f"{str(project_root)}/"
        destination = f"{self._ssh_target()}:{remote_path}/"

        return self._run_rsync(
            source,
            destination,
            excludes=self.default_excludes,
            delete=True,
            operation="upload",
        )

    def _sync_remote_to_local(self, initial: bool = False) -> int:
        """Pull remote workspace back to the local project directory."""
        if not self.auto_sync or not self.has_rsync:
            return 1

        project_root = Path(self.config.get("project_root", os.getcwd())).resolve()
        remote_path = self.remote_workspace or "/content"

        if initial:
            print("   Reconciling remote workspace back to local files...")

        source = f"{self._ssh_target()}:{remote_path}/"
        destination = f"{str(project_root)}/"

        return self._run_rsync(
            source,
            destination,
            excludes=self.default_excludes,
            delete=False,
            operation="download",
        )

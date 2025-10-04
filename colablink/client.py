"""
LocalClient - Runs on local machine to connect to Colab runtime.

This module handles:
- Connection to Colab via SSH
- Bidirectional file synchronization with proper isolation
- Command execution with real-time output streaming
- Port forwarding
"""

import os
import sys
import subprocess
import json
import shlex
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Sequence, Union

try:
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer
except ImportError:
    FileSystemEventHandler = None
    Observer = None


class LocalClient:
    """Client for connecting local machine to Colab runtime."""

    def __init__(
        self,
        profile: str = "default",
        config_file: Optional[str] = None,
        auto_sync: bool = True,
        source_sync_debounce: float = 0.1,
        remote_sync_interval: float = 0.5,
    ):
        """
        Initialize local client.

        Args:
            profile: ColabLink profile name
            config_file: Path to config file (default: ~/.colablink/profiles/{profile}.json)
            auto_sync: Enable automatic bidirectional file sync
            source_sync_debounce: Debounce time for local file changes (seconds)
            remote_sync_interval: Interval for remote-to-local sync when sshfs unavailable (seconds)
        """
        self.profile = profile
        
        # Config file setup
        if config_file:
            self.config_file = os.path.abspath(config_file)
            self.config_dir = os.path.dirname(self.config_file)
        else:
            profiles_root = os.path.expanduser("~/.colablink/profiles")
            os.makedirs(profiles_root, exist_ok=True)
            self.config_dir = profiles_root
            self.config_file = os.path.join(profiles_root, f"{self.profile}.json")

        # SSH setup
        ssh_config_name = f"{self.profile}_ssh_config"
        self.ssh_config_file = os.path.join(self.config_dir, ssh_config_name)
        self.ssh_alias = (
            "colablink" if self.profile == "default" else f"colablink-{self.profile}"
        )

        # Connection state
        self.config: Dict = {}
        self.project_root: Optional[Path] = None
        self.local_output_dir: Optional[Path] = None
        self.remote_root = "/content"  # Base path on Colab
        self.remote_output_dir = ""    # Full path to connection directory on Colab
        self.connection_id = ""        # Connection identifier

        # Process management
        self.port_forwards: List[subprocess.Popen] = []
        
        # Sync configuration
        self.auto_sync = auto_sync
        self.source_sync_debounce = source_sync_debounce
        self.remote_sync_interval = remote_sync_interval
        
        # System capabilities
        self.has_rsync = shutil.which("rsync") is not None
        self.has_sshfs = shutil.which("sshfs") is not None
        
        # Sync exclusions
        self.default_excludes = [
            "__pycache__", "*.pyc", ".git", ".gitignore", ".venv", "venv", "env",
            "node_modules", "*.egg-info", "dist", "build", ".colablink", "connection_*",
            "colab-workspace", "my-colab-files"  # Exclude broken legacy directories
        ]
        
        # Background sync threads
        self._source_observer: Optional[Observer] = None
        self._source_event: Optional[threading.Event] = None
        self._source_stop: Optional[threading.Event] = None
        self._source_thread: Optional[threading.Thread] = None
        self._remote_sync_thread: Optional[threading.Thread] = None
        self._remote_sync_stop: Optional[threading.Event] = None

    def initialize(
        self,
        connection_info: Dict,
        remote_dir: str,
        local_dir: str,
        remote_root: str = None,
    ):
        """
        Initialize connection to Colab runtime.
        
        Args:
            connection_info: Connection details from Colab runtime
            remote_dir: Directory name on Colab for this connection's outputs
            local_dir: Local directory name for outputs
            remote_root: Base directory on Colab (uses connection_info if None)
        """
        print("Initializing connection to Colab runtime...")

        self._check_dependencies()

        # Setup paths
        self.project_root = Path(os.getcwd()).resolve()
        self.local_output_dir = self.project_root / local_dir
        
        # Get remote root from parameter or connection info
        self.remote_root = remote_root or connection_info.get("remote_root", "/content")
        self.remote_output_dir = f"{self.remote_root.rstrip('/')}/{remote_dir}"
        self.connection_id = remote_dir

        # Create local output directory
        self.local_output_dir.mkdir(exist_ok=True)
        
        # Build config
        self.config = {
            **connection_info,
            "project_root": str(self.project_root),
            "local_output_dir": str(self.local_output_dir),
            "remote_root": self.remote_root,
            "remote_output_dir": self.remote_output_dir,
            "connection_id": self.connection_id,
            "auto_sync": self.auto_sync,
            "source_sync_debounce": self.source_sync_debounce,
            "remote_sync_interval": self.remote_sync_interval,
        }

        # Setup connection
        os.makedirs(self.config_dir, exist_ok=True)
        self._save_config()
        self._setup_ssh_config()
        
        # Test connection FIRST before any file operations
        print("\nTesting connection...")
        if not self._test_connection():
            print("Connection failed. Please check the connection details.")
            return False

        print("Connection successful!")
        
        # Create connection directory on Colab
        print(f"   Creating connection directory on Colab: {self.remote_output_dir}")
        self._ensure_remote_directory(self.remote_output_dir)

        # Initial sync of source files
        self._sync_local_source_to_remote(initial=True)

        # Setup file sync
        self._setup_connection_mount()

        if self.auto_sync:
            print("\nSetting up automatic bidirectional file sync...")
            print("\n" + "=" * 70)
            print("READY TO USE!")
            print("=" * 70)
            if self.has_sshfs and self._verify_mount():
                print(f"\n  [OK] Remote outputs mounted at: {self.local_output_dir}")
            elif not self.has_sshfs:
                print("\n  [INFO] sshfs unavailable; using rsync fallback for remote outputs")
            print(f"\nRemote root: {self.remote_root}")
            print(f"Connection directory: {self.remote_output_dir}")
            print(f"Local outputs: {self.local_output_dir}")
            print(f"\nRun 'colablink watch --profile {self.profile}' in another terminal "
                  "to keep local ↔ remote sync active while you work.")
        else:
            print("\n" + "=" * 70)
            print("READY TO USE! (Manual sync mode)")
            print("=" * 70)

        print("\nYou can now execute commands on Colab runtime:")
        print("  colablink exec python train.py")
        print("  colablink exec nvidia-smi")
        print("\nOr start a shell:")
        print("  colablink shell")
        print("\nOr use VS Code Remote-SSH:")
        print(f"  Host: {self.ssh_alias}")
        print("=" * 70)

        return True

    def execute(
        self,
        command: Union[str, Sequence[Union[str, Path]]],
        stream_output: bool = True,
        cwd: Optional[Union[str, Path]] = None,
    ) -> int:
        """Execute a command on the Colab runtime."""
        self._load_config()

        if not self.config:
            print("Error: Not connected. Run 'colablink init' first.")
            return 1

        # Normalize command
        if isinstance(command, (list, tuple)):
            cmd_tokens = [str(token) for token in command]
        else:
            cmd_tokens = shlex.split(str(command))

        # Add -u flag for python commands
        if cmd_tokens and cmd_tokens[0].startswith("python") and "-u" not in cmd_tokens:
            cmd_tokens.insert(1, "-u")

        command_str = " ".join(shlex.quote(token) for token in cmd_tokens)

        # Determine working directory
        if cwd is None:
            remote_cwd = self.remote_output_dir or self.remote_root
        else:
            # Map local path to remote
            local_cwd = Path(cwd).resolve()
            if self.local_output_dir and self._is_path_inside(local_cwd, self.local_output_dir):
                # Path is inside connection directory
                rel_path = local_cwd.relative_to(self.local_output_dir)
                remote_cwd = f"{self.remote_output_dir}/{rel_path}"
            else:
                # Path is in project root
                rel_path = local_cwd.relative_to(self.project_root)
                remote_cwd = f"{self.remote_root}/{rel_path}"

        # Sync local changes before execution
        self._sync_local_source_to_remote()

        # Ensure remote directory exists
        self._ensure_remote_directory(remote_cwd)

        # Build SSH command
        ssh_cmd = self._build_ssh_command()
        env_setup = (
            "export PYTHONUNBUFFERED=1 && "
            "if [ -d /usr/lib64-nvidia ]; then "
            "export LD_LIBRARY_PATH=/usr/lib64-nvidia:/usr/local/cuda/lib64:$LD_LIBRARY_PATH; "
            "export PATH=/usr/local/cuda/bin:$PATH; "
            "fi"
        )
        remote_script = f"{env_setup} && cd {shlex.quote(remote_cwd)} && {command_str}"
        full_cmd = ssh_cmd + ["bash", "-lc", remote_script]

        # Execute command
        if stream_output:
            process = subprocess.Popen(
                full_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1
            )
            try:
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

        # Sync remote changes after execution
        if not self.has_sshfs:
            self._sync_remote_outputs_to_local()

        return returncode

    def shell(self):
        """Start an interactive SSH shell to Colab runtime."""
        self._load_config()

        if not self.config:
            print("Error: Not connected. Run 'colablink init' first.")
            return 1

        print("Starting interactive shell on Colab runtime...")
        print("Type 'exit' to return to local shell.\n")

        # Start interactive SSH session
        ssh_cmd = self._build_ssh_command(force_tty=True)
        remote_shell_cmd = [
            "bash", "-lc", f"cd {shlex.quote(self.remote_output_dir)} && exec bash -l"
        ]
        subprocess.call(ssh_cmd + remote_shell_cmd)

    def watch(self, interval: float = None) -> int:
        """Run the sync agent until interrupted."""
        if interval is not None:
            self.remote_sync_interval = interval
            
        self._load_config()

        if not self.config:
            print("Error: Not connected. Run 'colablink init' first.")
            return 1

        self._setup_connection_mount()

        if not self.has_sshfs:
            self._start_remote_sync_loop()
            self._sync_remote_outputs_to_local()

        self._start_source_watch()

        print("Sync agent running. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping sync agent...")
        finally:
            self._stop_source_watch()
            self._stop_remote_sync_loop()

        return 0

    def forward_port(self, remote_port: int, local_port: Optional[int] = None):
        """Forward a port from Colab to local machine."""
        self._load_config()

        if not self.config:
            print("Error: Not connected. Run 'colablink init' first.")
            return

        local_port = local_port or remote_port
        print(f"Forwarding port {remote_port} to localhost:{local_port}")

        self._clean_port_forwards()

        ssh_cmd = self._build_ssh_command(port_forward=f"{local_port}:localhost:{remote_port}")
        ssh_cmd.extend(["-N"])  # No command execution

        process = subprocess.Popen(ssh_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
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
                self._build_ssh_command() + [
                    "bash", "-lc",
                    "nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader"
                ],
                capture_output=True, text=True
            )

            if result.returncode == 0 and result.stdout.strip():
                print(f"\nGPU: {result.stdout.strip()}")
        else:
            print("  Status: Disconnected")

    def disconnect(self):
        """Disconnect from Colab runtime."""
        self._load_config()

        self._stop_source_watch()
        self._stop_remote_sync_loop()

        if self.config:
            self._sync_remote_outputs_to_local()

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

    def upload(
        self, 
        source: str, 
        destination: Optional[str] = None, 
        recursive: bool = False
    ) -> int:
        """Upload files or directories to Colab."""
        self._load_config()

        if not self.config:
            print("Error: Not connected. Run 'colablink init' first.")
            return 1

        # Resolve source path
        source_path = Path(source).resolve()
        if not source_path.exists():
            print(f"Error: Source path does not exist: {source}")
            return 1

        is_directory = source_path.is_dir()

        # Determine destination
        if destination is None:
            if is_directory:
                target_path = self.remote_root
            else:
                target_path = f"{self.remote_root}/{source_path.name}"
        else:
            # Resolve destination path
            if destination.startswith("/"):
                target_path = destination
            else:
                target_path = f"{self.remote_root}/{destination}"

        # Ensure parent directory exists
        if destination and destination.endswith("/"):
            self._ensure_remote_directory(target_path)
        else:
            parent_path = str(Path(target_path).parent)
            self._ensure_remote_directory(parent_path)

        # Build scp command
        scp_cmd = [
            "sshpass", "-p", self.config["password"],
            "scp", "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-P", str(self.config["port"])
        ]

        if recursive or is_directory:
            scp_cmd.append("-r")

        scp_cmd.extend([str(source_path), f"{self._ssh_target()}:{target_path}"])

        print(f"Uploading {'directory' if is_directory else 'file'} {source} to Colab:{target_path}...")

        result = subprocess.run(scp_cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"[OK] {'Directory' if is_directory else 'File'} uploaded successfully: {target_path}")
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
        """Download files or directories from Colab to local machine."""
        self._load_config()

        if not self.config:
            print("Error: Not connected. Run 'colablink init' first.")
            return 1

        # Resolve remote source path
        if source.startswith("/"):
            remote_source = source
        else:
            remote_source = f"{self.remote_root}/{source}"

        # Auto-detect if source is likely a directory
        is_likely_directory = (
            source.endswith("/") or "." not in Path(source).name or recursive
        )

        # Determine local destination
        if destination is None:
            destination = os.getcwd()

        dest_path = Path(destination).expanduser().resolve(strict=False)

        if is_likely_directory or destination.endswith("/"):
            dest_path.mkdir(parents=True, exist_ok=True)
            local_target = str(dest_path)
        else:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            if dest_path.exists() and not overwrite and not dest_path.is_dir():
                print(f"Error: Destination file already exists: {dest_path}. Use --force to overwrite.")
                return 1
            local_target = str(dest_path)

        # Build scp command
        scp_cmd = [
            "sshpass", "-p", self.config["password"],
            "scp", "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-P", str(self.config["port"])
        ]

        if is_likely_directory:
            scp_cmd.append("-r")
            print(f"Downloading directory Colab:{source} to {destination}...")
        else:
            print(f"Downloading file Colab:{source} to {destination}...")

        scp_cmd.extend([f"{self._ssh_target()}:{remote_source}", local_target])

        result = subprocess.run(scp_cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"[OK] {'Directory' if is_likely_directory else 'File'} downloaded successfully to: {local_target}")
            return 0
        else:
            # If failed and we didn't use recursive, suggest trying with recursive
            if not is_likely_directory and "not a regular file" in result.stderr.lower():
                print(f"Note: '{source}' appears to be a directory.")
                print("Retrying with recursive mode...")
                return self.download(source, destination, recursive=True, overwrite=overwrite)
            
            print(f"[ERROR] Download failed: {result.stderr.strip()}")
            return 1

    def sync(
        self,
        directory: Optional[str] = None,
        exclude: Optional[List[str]] = None,
        dry_run: bool = False,
        progress: bool = False,
    ) -> int:
        """Sync current directory (or specified directory) to Colab."""
        self._load_config()

        if not self.config:
            print("Error: Not connected. Run 'colablink init' first.")
            return 1

        # Determine source directory
        if directory is None:
            directory = os.getcwd()

        source_dir = Path(directory).resolve()
        if not source_dir.is_dir():
            print(f"Error: Not a directory: {directory}")
            return 1

        excludes = exclude or self.default_excludes

        print(f"Syncing {source_dir} to Colab:{self.remote_root}...")

        source = f"{source_dir}/"
        destination = f"{self._ssh_target()}:{self.remote_root}/"

        return self._run_rsync(
            source, destination, excludes=excludes,
            delete=True, operation="upload",
            dry_run=dry_run, progress=progress
        )

    # ========================================================================================
    # PRIVATE METHODS
    # ========================================================================================

    def _load_config(self):
        """Load configuration from file."""
        if os.path.exists(self.config_file):
            with open(self.config_file, "r") as f:
                self.config = json.load(f)
            
            if self.config:
                self.config.setdefault("username", "root")
                self.project_root = Path(self.config.get("project_root", os.getcwd())).resolve()
                self.local_output_dir = Path(self.config.get("local_output_dir", "./connection"))
                self.remote_root = self.config.get("remote_root", "/content")
                self.remote_output_dir = self.config.get("remote_output_dir", f"{self.remote_root}/connection")
                self.connection_id = self.config.get("connection_id", "connection")
                self.auto_sync = self.config.get("auto_sync", self.auto_sync)
                self.source_sync_debounce = self.config.get("source_sync_debounce", self.source_sync_debounce)
                self.remote_sync_interval = self.config.get("remote_sync_interval", self.remote_sync_interval)

    def _save_config(self):
        """Save configuration to file."""
        with open(self.config_file, "w") as f:
            json.dump(self.config, f, indent=2)
        print(f"Configuration saved to: {self.config_file}")

    def _setup_ssh_config(self):
        """Create SSH config for easy connection."""
        ssh_config_content = f"""
Host {self.ssh_alias}
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
                if f"Host {self.ssh_alias}" not in f.read():
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
            ssh_cmd + ["echo", "connection_test"],
            capture_output=True, text=True
        )

        success = result.returncode == 0 and "connection_test" in result.stdout

        if verbose:
            if success:
                print("   Connection test passed")
            else:
                print("   Connection test failed")

        return success

    def _setup_connection_mount(self):
        """Setup SSHFS mount for connection directory."""
        if not self.remote_output_dir or not self.local_output_dir:
            return

        if self.has_sshfs:
            mount_cmd = [
                "sshfs",
                f"{self._ssh_target()}:{self.remote_output_dir}",
                str(self.local_output_dir),
                "-p", str(self.config["port"]),
                "-o", "password_stdin",
                "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null",
                "-o", "reconnect",
                "-o", "ServerAliveInterval=15",
                "-o", "ServerAliveCountMax=3",
            ]

            try:
                process = subprocess.Popen(
                    mount_cmd, stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                )
                stdout, stderr = process.communicate(
                    input=f"{self.config['password']}\n", timeout=10
                )
                if process.returncode == 0 or self._verify_mount():
                    self.has_sshfs = True
                else:
                    print(f"   Warning: Mount failed: {stderr.strip()}")
                    self.has_sshfs = False
            except (subprocess.TimeoutExpired, Exception) as exc:
                print(f"   Warning: Mount failed: {exc}")
                self.has_sshfs = False

        if not self.has_sshfs:
            self.local_output_dir.mkdir(exist_ok=True)

    def _verify_mount(self):
        """Verify that the mount point is actually mounted."""
        result = subprocess.run(
            ["mountpoint", "-q", str(self.local_output_dir)], capture_output=True
        )
        return result.returncode == 0

    def _unmount_sshfs(self):
        """Unmount SSHFS if mounted."""
        if not self.local_output_dir or not self.local_output_dir.exists():
            return

        if self._verify_mount():
            print(f"Unmounting {self.local_output_dir}...")
            # Try fusermount first (Linux)
            result = subprocess.run(
                ["fusermount", "-u", str(self.local_output_dir)],
                capture_output=True, text=True
            )
            if result.returncode != 0:
                # Try umount (macOS/BSD)
                subprocess.run(
                    ["umount", str(self.local_output_dir)],
                    capture_output=True, text=True
                )

    def _start_source_watch(self):
        """Start watching for local source file changes."""
        if Observer is None:
            print("Install 'watchdog' to enable automatic source sync (pip install watchdog).")
            return

        if not self.project_root:
            return

        self._source_event = threading.Event()
        self._source_stop = threading.Event()

        class SourceHandler(FileSystemEventHandler):
            def __init__(self, client):
                super().__init__()
                self.client = client

            def on_any_event(self, event):
                if getattr(event, "is_directory", False) and event.event_type == "modified":
                    return
                if self.client._should_ignore_source_path(event.src_path):
                    return
                if self.client._source_event:
                    self.client._source_event.set()

        handler = SourceHandler(self)
        self._source_observer = Observer()
        self._source_observer.schedule(handler, str(self.project_root), recursive=True)
        self._source_observer.start()

        self._source_thread = threading.Thread(
            target=self._source_sync_worker, name="colablink-source-sync", daemon=True
        )
        self._source_thread.start()

    def _source_sync_worker(self):
        """Worker thread for source file sync."""
        while not self._source_stop.is_set():
            triggered = self._source_event.wait(self.source_sync_debounce)
            if triggered:
                self._source_event.clear()
                self._sync_local_source_to_remote()

    def _stop_source_watch(self):
        """Stop watching for local source file changes."""
        if self._source_stop:
            self._source_stop.set()
        if self._source_observer:
            self._source_observer.stop()
            self._source_observer.join(timeout=2)
        if self._source_thread and self._source_thread.is_alive():
            self._source_thread.join(timeout=2)
        self._source_event = None
        self._source_stop = None
        self._source_thread = None
        self._source_observer = None

    def _start_remote_sync_loop(self):
        """Start polling for remote output changes."""
        if self.has_sshfs or not self.remote_output_dir:
            return
        self._remote_sync_stop = threading.Event()
        self._remote_sync_thread = threading.Thread(
            target=self._remote_sync_worker, name="colablink-remote-sync", daemon=True
        )
        self._remote_sync_thread.start()

    def _remote_sync_worker(self):
        """Worker thread for remote output sync."""
        while not self._remote_sync_stop.is_set():
            self._sync_remote_outputs_to_local()
            self._remote_sync_stop.wait(self.remote_sync_interval)

    def _stop_remote_sync_loop(self):
        """Stop polling for remote output changes."""
        if self._remote_sync_stop:
            self._remote_sync_stop.set()
        if self._remote_sync_thread and self._remote_sync_thread.is_alive():
            self._remote_sync_thread.join(timeout=2)
        self._remote_sync_thread = None
        self._remote_sync_stop = None

    def _should_ignore_source_path(self, path: Union[str, Path]) -> bool:
        """Check if a source path should be ignored during sync."""
        if not self.project_root:
            return False
        try:
            p = Path(path).resolve()
        except FileNotFoundError:
            return True
        
        # Ignore if path is inside connection output directory
        if self.local_output_dir and self._is_path_inside(p, self.local_output_dir):
            return True
        
        # Ignore if path matches exclusion patterns
        rel_path = p.relative_to(self.project_root)
        path_str = str(rel_path)
        for pattern in self.default_excludes:
            if pattern.startswith("*"):
                if path_str.endswith(pattern[1:]):
                    return True
            elif pattern in path_str:
                return True
        
        return False

    def _sync_local_source_to_remote(self, initial: bool = False) -> int:
        """
        Sync local source files to remote root.
        ONE-WAY: local source → remote root (excludes connection directories)
        """
        if not self.remote_root:
            return 1

        excludes = list(self.default_excludes)
        
        # Exclude connection directories
        if self.local_output_dir:
            excludes.append(self.local_output_dir.name)
        
        # Scan for any connection_* directories and exclude them
        try:
            for item in self.project_root.iterdir():
                if item.is_dir() and item.name.startswith("connection_"):
                    if item.name not in excludes:
                        excludes.append(item.name)
        except (OSError, IOError) as e:
            # Handle broken symlinks or I/O errors gracefully
            print(f"   Warning: Could not scan project directory: {e}")
            print("   Continuing with sync using default exclusions...")
        except Exception:
            pass

        if initial:
            print("   Uploading local source files to Colab...")

        source = f"{str(self.project_root)}/"
        destination = f"{self._ssh_target()}:{self.remote_root}/"

        return self._run_rsync(
            source, destination, excludes=excludes,
            delete=False, operation="upload"
        )

    def _sync_remote_outputs_to_local(self, delete: bool = False) -> int:
        """
        Sync remote connection directory to local connection directory.
        ONE-WAY: remote connection dir → local connection dir
        """
        if not self.remote_output_dir or not self.local_output_dir:
            return 1

        source = f"{self._ssh_target()}:{self.remote_output_dir}/"
        self.local_output_dir.mkdir(parents=True, exist_ok=True)
        destination = f"{str(self.local_output_dir)}/"

        return self._run_rsync(
            source, destination, excludes=[],
            delete=delete, operation="download"
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
        """Execute rsync with SSH configuration."""
        excludes = excludes or []
        ssh_command = [
            "ssh", "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ServerAliveInterval=60",
            "-o", "ServerAliveCountMax=3",
            "-p", str(self.config["port"])
        ]

        rsync_cmd = [
            "sshpass", "-p", self.config["password"],
            "rsync", "-az", "--safe-links"  # Skip broken symlinks
        ]

        if delete:
            rsync_cmd.append("--delete")

        for pattern in excludes:
            rsync_cmd.extend(["--exclude", pattern])

        if dry_run:
            rsync_cmd.append("--dry-run")

        if progress:
            rsync_cmd.append("--info=progress2")

        rsync_cmd.extend(["-e", " ".join(shlex.quote(arg) for arg in ssh_command)])
        rsync_cmd.extend([source, destination])

        result = subprocess.run(
            rsync_cmd, text=True,
            stdout=None if progress else subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        if result.returncode != 0:
            print(f"[WARNING] rsync {operation} failed: {result.stderr.strip()}")
        elif dry_run and not progress and result.stdout:
            print(result.stdout.strip())

        return result.returncode

    def _build_ssh_command(
        self,
        interactive: bool = False,
        port_forward: Optional[str] = None,
        force_tty: bool = False,
    ) -> List[str]:
        """Build SSH command with proper options."""
        cmd = [
            "sshpass", "-p", self.config["password"],
            "ssh",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ServerAliveInterval=60",
            "-o", "ServerAliveCountMax=3",
            "-p", str(self.config["port"])
        ]

        if port_forward:
            cmd.extend(["-L", port_forward])

        if interactive:
            cmd.append("-t")
        elif force_tty:
            cmd.append("-tt")

        cmd.append(self._ssh_target())
        return cmd

    def _ssh_target(self) -> str:
        """Return user@host string for SSH commands."""
        username = self.config.get("username", "root")
        return f"{username}@{self.config['host']}"

    def _ensure_remote_directory(self, remote_path: str):
        """Create a remote directory if it does not exist."""
        if not remote_path or not remote_path.strip():
            print(f"   Warning: Empty remote path provided to _ensure_remote_directory")
            return
        
        ssh_cmd = self._build_ssh_command()
        # Use proper shell quoting for the mkdir command
        mkdir_cmd = f'mkdir -p {shlex.quote(remote_path)}'
        
        result = subprocess.run(
            ssh_cmd + ["bash", "-lc", mkdir_cmd],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"   Warning: Could not create remote directory {remote_path}")
            print(f"   Error: {result.stderr.strip()}")
            if "Permission denied" in result.stderr:
                print("   This may indicate the Colab runtime needs to be restarted with updated permissions.")
                print("   Please rerun the ColabRuntime.setup() cell in your Colab notebook.")

    def _clean_port_forwards(self):
        """Remove finished port forward processes."""
        alive = []
        for process in self.port_forwards:
            if process.poll() is None:
                alive.append(process)
        self.port_forwards = alive

    def _check_dependencies(self):
        """Check and attempt to install system dependencies."""
        # Initial check
        sshpass_missing = shutil.which("sshpass") is None
        rsync_missing = shutil.which("rsync") is None
        sshfs_missing = shutil.which("sshfs") is None
        
        # Try auto-install if anything is missing
        packages_needed = []
        if sshpass_missing:
            packages_needed.append("sshpass")
        if rsync_missing:
            packages_needed.append("rsync")
        if sshfs_missing:
            packages_needed.append("sshfs")
        
        if packages_needed:
            print(f"\nAttempting to install missing dependencies: {', '.join(packages_needed)}")
            if self._auto_install_dependencies(packages_needed):
                print("Auto-installation completed. Rechecking dependencies...")
                # Recheck after installation
                sshpass_missing = shutil.which("sshpass") is None
                rsync_missing = shutil.which("rsync") is None
                sshfs_missing = shutil.which("sshfs") is None
            else:
                print("Auto-installation failed or not supported on this platform.")

        # Update capabilities
        self.has_rsync = not rsync_missing
        self.has_sshfs = not sshfs_missing

        # Handle critical missing dependencies
        if sshpass_missing:
            print("\nError: sshpass is required but not found.")
            print("Please install manually:")
            print("  Linux: sudo apt-get install sshpass")
            print("  macOS: brew install hudochenkov/sshpass/sshpass")
            sys.exit(1)

        if rsync_missing:
            print("\nError: rsync is required for file synchronization but not found.")
            print("Please install manually:")
            print("  Linux: sudo apt-get install rsync")
            print("  macOS: brew install rsync")
            sys.exit(1)

        # Handle optional dependencies
        if sshfs_missing:
            print("\nNote: sshfs not found. Using rsync fallback for remote outputs.")

    def _auto_install_dependencies(self, packages: List[str]) -> bool:
        """Attempt to install dependencies automatically."""
        import platform
        
        if not packages:
            return True
            
        system = platform.system()
        
        if system == "Linux" and shutil.which("apt-get"):
            return self._install_with_apt(packages)
        elif system == "Darwin" and shutil.which("brew"):
            return self._install_with_brew(packages)
        else:
            print(f"Automatic installation not supported on {system}")
            return False
    
    def _install_with_apt(self, packages: List[str]) -> bool:
        """Install packages using apt-get."""
        try:
            # Check if we need sudo
            cmd_prefix = []
            if hasattr(os, "geteuid") and os.geteuid() != 0:
                if not shutil.which("sudo"):
                    print("sudo not found; cannot auto-install dependencies.")
                    return False
                cmd_prefix = ["sudo", "-n"]  # Non-interactive sudo
            
            # Set non-interactive environment
            env = os.environ.copy()
            env["DEBIAN_FRONTEND"] = "noninteractive"
            
            # Update package list
            print("  Updating package list...")
            result = subprocess.run(
                cmd_prefix + ["apt-get", "update", "-qq"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, env=env
            )
            if result.returncode != 0:
                print(f"  Warning: apt-get update failed: {result.stderr[:200]}")
            
            # Install packages
            print(f"  Installing {', '.join(packages)}...")
            result = subprocess.run(
                cmd_prefix + ["apt-get", "install", "-y", "--fix-missing"] + packages,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, env=env
            )
            
            if result.returncode == 0:
                print("  Installation successful!")
                return True
            else:
                print(f"  Installation failed: {result.stderr[:200]}")
                return False
                
        except Exception as e:
            print(f"  Installation error: {e}")
            return False
    
    def _install_with_brew(self, packages: List[str]) -> bool:
        """Install packages using Homebrew."""
        try:
            for package in packages:
                print(f"  Installing {package}...")
                
                # Special handling for sshpass on macOS
                if package == "sshpass":
                    cmd = ["brew", "install", "hudochenkov/sshpass/sshpass"]
                elif package == "sshfs":
                    # Install macfuse first, then sshfs
                    result = subprocess.run(
                        ["brew", "install", "macfuse"],
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                    )
                    if result.returncode != 0:
                        print(f"  Warning: macfuse install failed: {result.stderr[:200]}")
                    
                    cmd = ["brew", "install", "sshfs"]
                else:
                    cmd = ["brew", "install", package]
                
                result = subprocess.run(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                )
                
                if result.returncode != 0:
                    print(f"  Failed to install {package}: {result.stderr[:200]}")
                    return False
            
            print("  Installation successful!")
            return True
            
        except Exception as e:
            print(f"  Installation error: {e}")
            return False

    @staticmethod
    def _is_path_inside(path: Path, parent: Path) -> bool:
        """Check if path is inside parent directory."""
        try:
            path.relative_to(parent.resolve())
            return True
        except ValueError:
            return False
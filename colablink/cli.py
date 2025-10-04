"""
CLI tool for ColabLink.

Provides command-line interface for connecting to and executing commands on Colab.
"""

import sys
import json
import argparse
from pathlib import Path
from .client import LocalClient
from . import __version__


class ColablinkArgumentParser(argparse.ArgumentParser):
    """Custom parser that prints full help on errors."""

    def error(self, message):
        self.print_usage(sys.stderr)
        self.print_help(sys.stderr)
        self.exit(2, f"\nError: {message}\n")


def main():
    """Main CLI entry point."""
    parser = ColablinkArgumentParser(
        description="ColabLink - Execute code on Google Colab runtime from your local terminal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Initialize connection to Colab with auto-generated directory
  colablink init '{"host": "0.tcp.ngrok.io", "port": "12345", "password": "xxx"}'
  
  # Initialize with custom directories
  colablink init '{...}' --remote-dir training --local-dir train_outputs
  
  # Execute commands on Colab runtime
  colablink exec python train.py
  colablink exec nvidia-smi
  
  # Manual file transfer
  colablink upload train.py              # Upload file to Colab
  colablink upload data/                 # Upload directory
  colablink download /content/model.pt   # Download file from Colab
  colablink sync                         # Sync entire directory
  
  # Interactive shell and port forwarding
  colablink shell                        # Interactive shell on Colab
  colablink forward 8888                 # Forward Jupyter port
  
  # Check status
  colablink status
""",
    )

    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"colablink {__version__}",
        help="Show the installed ColabLink version and exit.",
    )
    parser.add_argument(
        "--profile",
        default="default",
        help="Name of the ColabLink profile to use (default: default).",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Init command
    init_parser = subparsers.add_parser(
        "init",
        help="Initialize connection to Colab runtime",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Initialize connection to Colab runtime and set up automatic file sync.",
        epilog="""Examples:
        colablink init '{"host": "...", "port": "...", "username": "...", "password": "...", "remote_root": "/content"}'
        colablink init '{...}' --remote-dir training --local-dir train_outputs --profile exp1
        colablink init '{...}' --remote-root /content/workspace --remote-dir experiment1
        """,
    )
    init_parser.add_argument(
        "config", help="Connection config JSON string from Colab output"
    )
    init_parser.add_argument(
        "--remote-dir",
        help="Directory name on Colab for this connection's outputs (default: connection_<random>)",
    )
    init_parser.add_argument(
        "--local-dir",
        help="Local directory name for outputs (default: same as --remote-dir)",
    )
    init_parser.add_argument(
        "--remote-root",
        help="Base directory on Colab (default: from connection info or /content)",
    )

    # Exec command
    exec_parser = subparsers.add_parser(
        "exec",
        help="Execute command on Colab runtime",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Execute any command on Google Colab runtime with real-time output streaming.",
        epilog="""Common commands:
        colablink exec python train.py          # Run Python script
        colablink exec nvidia-smi                # Check GPU
        colablink exec pip install torch         # Install packages
        colablink exec "ls -la /content"         # Shell commands
        
        Output streams in real-time to your terminal.""",
    )
    exec_parser.add_argument(
        "cmd", nargs=argparse.REMAINDER, help="Command to execute on Colab"
    )

    # Shell command
    subparsers.add_parser(
        "shell",
        help="Start interactive shell on Colab",
        description='Start an interactive bash shell on Colab with full access. Type "exit" to return to local shell.',
    )

    # Status command
    subparsers.add_parser(
        "status",
        help="Check connection status and GPU info",
        description="Display connection status, host/port details, and GPU information.",
    )

    # Forward command
    p = subparsers.add_parser(
        "forward",
        help="Forward port from Colab to local",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Forward a port from Colab to your local machine.",
        epilog="""Examples:
        colablink forward 8888                   # Jupyter
        colablink forward 6006                   # TensorBoard
        colablink forward 5000 --local-port 3000 # Custom mapping""",
    )
    p.add_argument("port", type=int, help="Remote port on Colab to forward")
    p.add_argument(
        "--local-port", type=int, help="Local port (default: same as remote)"
    )

    # Upload command
    p = subparsers.add_parser(
        "upload",
        help="Upload file or directory to Colab",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Upload files or directories to Colab.\nDirectories are automatically uploaded recursively.",
        epilog="""Examples:
        colablink upload train.py                # Single file
        colablink upload data/                   # Directory (auto-recursive)
        colablink upload model.py -d /content/models/""",
    )
    p.add_argument("source", help="Local file or directory to upload")
    p.add_argument(
        "--destination", "-d", help="Remote destination path (default: /content/)"
    )
    p.add_argument(
        "--recursive", "-r", action="store_true", help="Force recursive mode"
    )

    # Download command
    p = subparsers.add_parser(
        "download",
        help="Download file or directory from Colab",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Download files or directories from Colab.\nDirectories are automatically detected.",
        epilog="""Examples:
        colablink download /content/model.pt               # Single file
        colablink download /content/output/                # Directory
        colablink download /content/data/ -d ./local_data/""",
    )
    p.add_argument("source", help="Remote file or directory path on Colab")
    p.add_argument(
        "--destination",
        "-d",
        help="Local destination path (default: current directory)",
    )
    p.add_argument(
        "--recursive", "-r", action="store_true", help="Force recursive mode"
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing local files when downloading",
    )

    # Watch command
    p = subparsers.add_parser(
        "watch",
        help="Run background sync agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Keep local source ↔ remote outputs synchronized until interrupted.",
        epilog="""Example:
        colablink watch --profile connection_x
        colablink watch --interval 0.5 --debounce 0.1  # Near-instant sync
        """,
    )
    p.add_argument(
        "--interval",
        type=float,
        default=0.5,
        help="Seconds between remote→local sync passes when sshfs is unavailable (default: 0.5).",
    )
    p.add_argument(
        "--debounce",
        type=float,
        default=0.1,
        help="Debounce time for local file changes before syncing (default: 0.1).",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing local files when downloading",
    )

    # Sync command
    p = subparsers.add_parser(
        "sync",
        help="Sync current directory to Colab",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Sync entire directory to Colab.\nAutomatically excludes: .git, __pycache__, venv, node_modules, *.pyc",
        epilog="""Examples:
        colablink sync                  # Sync current directory
        colablink sync -d /path/to/project""",
    )
    p.add_argument(
        "--directory", "-d", help="Directory to sync (default: current directory)"
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without copying files",
    )
    p.add_argument(
        "--progress",
        action="store_true",
        help="Display transfer progress during sync",
    )

    # Disconnect command
    subparsers.add_parser(
        "disconnect",
        help="Disconnect from Colab",
        description="Disconnect from Colab runtime, unmount file systems, and clean up resources.",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    profile = getattr(args, "profile", "default")

    if args.command == "init":
        try:
            config = json.loads(args.config)
        except json.JSONDecodeError:
            print("Error: Invalid JSON config string")
            return 1

        # Generate remote directory name if not provided
        if not args.remote_dir:
            import random
            import string
            random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
            remote_dir_name = f"connection_{random_suffix}"
        else:
            remote_dir_name = args.remote_dir
            
        # Set local directory name (defaults to remote directory name)
        local_dir_name = args.local_dir or remote_dir_name
        
        client = LocalClient(profile=profile)
        ok = client.initialize(
            config,
            remote_dir=remote_dir_name,
            local_dir=local_dir_name,
            remote_root=args.remote_root,
        )
        return 0 if ok else 1

    # Create client with sync intervals if specified
    if args.command == "watch":
        client = LocalClient(
            profile=profile,
            source_sync_debounce=getattr(args, "debounce", 0.1),
            remote_sync_interval=getattr(args, "interval", 0.5),
        )
        return client.watch(interval=args.interval)
    
    client = LocalClient(profile=profile)

    if args.command == "exec":
        if not args.cmd:
            print("Error: No command specified")
            return 1

        return client.execute(args.cmd)

    if args.command == "shell":
        return client.shell()

    if args.command == "status":
        client.status()
        return 0

    if args.command == "forward":
        client.forward_port(args.port, args.local_port)
        print("Press Ctrl+C to stop port forwarding...")
        try:
            import time

            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping port forwarding...")
        return 0

    if args.command == "upload":
        return client.upload(
            args.source, destination=args.destination, recursive=args.recursive
        )

    if args.command == "download":
        return client.download(
            args.source,
            destination=args.destination,
            recursive=args.recursive,
            overwrite=args.force,
        )

    if args.command == "sync":
        return client.sync(
            directory=args.directory,
            dry_run=args.dry_run,
            progress=args.progress,
        )

    if args.command == "disconnect":
        client.disconnect()
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())

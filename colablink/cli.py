"""
CLI tool for ColabLink.

Provides command-line interface for connecting to and executing commands on Colab.
"""

import sys
import json
import argparse
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
        description="ColabLink - Execute code on Google Colab GPU from your local terminal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Initialize connection to Colab (automatic sync enabled by default)
  colablink init '{"host": "0.tcp.ngrok.io", "port": "12345", "password": "xxx"}'
  
  # Custom mount directory
  colablink init '{"host": "...", "port": "...", "password": "..."}' --mount-dir /custom/path
  
  # Files generated on Colab appear in ./colab-workspace/ automatically
  
  # Manual file management (if auto-sync disabled):
  colablink upload train.py              # Push file to Colab
  colablink upload data/                 # Push directory (auto-recursive)
  colablink download /content/model.pt   # Pull file from Colab
  colablink download /content/output/    # Pull directory (auto-detected)
  colablink sync                         # Push entire directory
  
  # Execute a Python script on Colab GPU
  colablink exec python train.py
  
  # Check GPU status
  colablink exec nvidia-smi
  
  # Start interactive shell
  colablink shell
  
  # Check connection status
  colablink status
  
  # Forward Jupyter port to local
  colablink forward 8888
""",
    )

    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"colablink {__version__}",
        help="Show the installed ColabLink version and exit.",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Init command
    init_parser = subparsers.add_parser(
        "init",
        help="Initialize connection to Colab runtime",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Initialize connection to Colab runtime and set up automatic file sync.",
        epilog='Example: colablink init \'{"host": "0.tcp.ngrok.io", "port": "12345", "password": "mypass"}\'',
    )
    init_parser.add_argument(
        "config", help="Connection config JSON string from Colab output"
    )
    init_parser.add_argument(
        "--mount-dir",
        default="./colab-workspace",
        help="Local directory to mount Colab files (default: ./colab-workspace)",
    )

    # Exec command
    exec_parser = subparsers.add_parser(
        "exec",
        help="Execute command on Colab GPU",
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
        description='Start an interactive bash shell on Colab with full GPU access. Type "exit" to return to local shell.',
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

    # Create client with mount directory (for init command)
    local_mount_dir = None
    if args.command == "init":
        local_mount_dir = args.mount_dir

    client = LocalClient(local_mount_dir=local_mount_dir)

    if args.command == "init":
        try:
            config = json.loads(args.config)
            client.initialize(config)
        except json.JSONDecodeError:
            print("Error: Invalid JSON config string")
            return 1

    elif args.command == "exec":
        if not args.cmd:
            print("Error: No command specified")
            return 1

        return client.execute(args.cmd)

    elif args.command == "shell":
        return client.shell()

    elif args.command == "status":
        client.status()

    elif args.command == "forward":
        client.forward_port(args.port, args.local_port)
        print("Press Ctrl+C to stop port forwarding...")
        try:
            import time

            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping port forwarding...")

    elif args.command == "upload":
        return client.upload(
            args.source, destination=args.destination, recursive=args.recursive
        )

    elif args.command == "download":
        return client.download(
            args.source,
            destination=args.destination,
            recursive=args.recursive,
            overwrite=args.force,
        )

    elif args.command == "sync":
        return client.sync(
            directory=args.directory,
            dry_run=args.dry_run,
            progress=args.progress,
        )

    elif args.command == "disconnect":
        client.disconnect()

    return 0


if __name__ == "__main__":
    sys.exit(main())

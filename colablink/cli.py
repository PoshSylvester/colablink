"""
CLI tool for ColabLink.

Provides command-line interface for connecting to and executing commands on Colab.
"""

import sys
import json
import argparse
from .client import LocalClient


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="ColabLink - Execute code on Google Colab GPU from your local terminal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Initialize connection to Colab
  colablink init '{"host": "0.tcp.ngrok.io", "port": "12345", "password": "xxx", "mount_point": "/mnt/local"}'
  
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
"""
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Init command
    init_parser = subparsers.add_parser('init', help='Initialize connection to Colab runtime')
    init_parser.add_argument('config', help='Connection config JSON string')
    
    # Exec command
    exec_parser = subparsers.add_parser('exec', help='Execute command on Colab')
    exec_parser.add_argument('cmd', nargs=argparse.REMAINDER, help='Command to execute')
    
    # Shell command
    shell_parser = subparsers.add_parser('shell', help='Start interactive shell on Colab')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Check connection status')
    
    # Forward command
    forward_parser = subparsers.add_parser('forward', help='Forward port from Colab to local')
    forward_parser.add_argument('port', type=int, help='Port to forward')
    forward_parser.add_argument('--local-port', type=int, help='Local port (default: same as remote)')
    
    # Disconnect command
    disconnect_parser = subparsers.add_parser('disconnect', help='Disconnect from Colab')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 0
    
    client = LocalClient()
    
    if args.command == 'init':
        try:
            config = json.loads(args.config)
            client.initialize(config)
        except json.JSONDecodeError:
            print("Error: Invalid JSON config string")
            return 1
    
    elif args.command == 'exec':
        if not args.cmd:
            print("Error: No command specified")
            return 1
        
        command = ' '.join(args.cmd)
        return client.execute(command)
    
    elif args.command == 'shell':
        return client.shell()
    
    elif args.command == 'status':
        client.status()
    
    elif args.command == 'forward':
        client.forward_port(args.port, args.local_port)
        print("Press Ctrl+C to stop port forwarding...")
        try:
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping port forwarding...")
    
    elif args.command == 'disconnect':
        client.disconnect()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())


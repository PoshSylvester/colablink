"""
Display GPU information.

Run with: colablink exec python examples/gpu_info.py
"""

import sys


def check_gpu():
    """Check and display GPU information."""
    print("Checking GPU availability...\n")
    
    try:
        import torch
        
        print("PyTorch GPU Information:")
        print("-" * 60)
        print(f"CUDA Available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"Number of GPUs: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                print(f"\nGPU {i}:")
                print(f"  Name: {torch.cuda.get_device_name(i)}")
                props = torch.cuda.get_device_properties(i)
                print(f"  Total Memory: {props.total_memory / 1e9:.2f} GB")
                print(f"  Compute Capability: {props.major}.{props.minor}")
                
                if torch.cuda.is_available():
                    print(f"  Memory Allocated: {torch.cuda.memory_allocated(i) / 1e9:.2f} GB")
                    print(f"  Memory Reserved: {torch.cuda.memory_reserved(i) / 1e9:.2f} GB")
        else:
            print("\nNo GPU detected. Running on CPU.")
            
    except ImportError:
        print("PyTorch not installed.")
        print("Install with: colablink exec pip install torch")
        return
    
    print("\n" + "=" * 60)
    
    # Try to get nvidia-smi output
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("\nnvidia-smi output:")
            print("-" * 60)
            print(result.stdout)
    except Exception as e:
        print(f"\nCould not run nvidia-smi: {e}")


if __name__ == "__main__":
    check_gpu()


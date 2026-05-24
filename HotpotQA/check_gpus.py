#!/usr/bin/env python3
"""
Script to check available GPUs and their status
"""
import torch
import subprocess
import sys

def check_gpus():
    print("=== GPU Information ===")
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"CUDA is available: {torch.cuda.is_available()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
            print(f"  Memory allocated: {torch.cuda.memory_allocated(i) / 1024**3:.1f} GB")
            print(f"  Memory cached: {torch.cuda.memory_reserved(i) / 1024**3:.1f} GB")
    else:
        print("CUDA is not available")
    
    print("\n=== nvidia-smi output ===")
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        print(result.stdout)
    except FileNotFoundError:
        print("nvidia-smi not found - NVIDIA drivers may not be installed")

if __name__ == "__main__":
    check_gpus()

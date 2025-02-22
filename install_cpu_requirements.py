#!/usr/bin/env python
"""
===============================================================================
   CPU REQUIREMENTS INSTALLER FOR THE ULTIMATE HYBRID CHESS AI TRAINING ENGINE
===============================================================================
Title: CPU Requirements Installer for the Ultimate Hybrid Chess AI Training Engine

About:
    This Python script installs all required packages to run the Ultimate Hybrid
    Chess AI Training Engine on a CPU-only system. It installs the CPU-enabled versions
    of PyTorch (torch, torchvision, torchaudio) using the appropriate PyTorch wheels,
    as well as other essential libraries such as python-chess, numpy, and matplotlib.

    Note: For systems with CUDA-enabled GPUs, a different installation procedure is
    recommended to install the GPU-enabled version of PyTorch.

Usage:
    Simply run this script with Python. It will automatically install the necessary
    packages using pip.
===============================================================================
"""

import subprocess
import sys

def install_packages():
    """
    Installs the required packages for a CPU-only environment.
    
    The script installs:
      - PyTorch (CPU version), torchvision, and torchaudio using the CPU wheels.
      - python-chess, numpy, and matplotlib.
    """
    try:
        print("Installing PyTorch (CPU version), torchvision, and torchaudio...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "torch", "torchvision", "torchaudio",
            "--extra-index-url", "https://download.pytorch.org/whl/cpu"
        ])
    except subprocess.CalledProcessError as e:
        print("Failed to install PyTorch CPU version packages:", e)
    
    try:
        print("Installing python-chess, numpy, and matplotlib...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "python-chess", "numpy", "matplotlib"
        ])
    except subprocess.CalledProcessError as e:
        print("Failed to install one or more required packages:", e)

def main():
    print("Starting installation of required packages for CPU environment...")
    install_packages()
    print("Installation complete.")

if __name__ == "__main__":
    main()

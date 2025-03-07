"""
Setup script for CPU-only mode.
This script installs the required dependencies for CPU-only mode.
"""

import subprocess
import sys
import os

def run_command(command):
    """Run a command and print the output"""
    print(f"Running: {command}")
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    return process.returncode

def install_dependencies():
    """Install the required dependencies for CPU-only mode"""
    print("=== Installing Dependencies for CPU-Only Mode ===")
    
    # Basic dependencies
    basic_deps = [
        "fastapi",
        "uvicorn",
        "python-multipart",
        "PyPDF2",
        "python-docx",
        "ftfy",
        "requests"
    ]
    
    # NLP dependencies
    nlp_deps = [
        "transformers",
        "datasets",
        "torch --index-url https://download.pytorch.org/whl/cpu",  # CPU-only PyTorch
        "accelerate"
    ]
    
    # RAG dependencies
    rag_deps = [
        "langchain",
        "sentence-transformers",
        "faiss-cpu"
    ]
    
    # Install basic dependencies
    print("\n1. Installing basic dependencies...")
    if run_command(f"{sys.executable} -m pip install {' '.join(basic_deps)}") != 0:
        print("Error installing basic dependencies")
        return False
    
    # Install NLP dependencies
    print("\n2. Installing NLP dependencies...")
    if run_command(f"{sys.executable} -m pip install {' '.join(nlp_deps)}") != 0:
        print("Error installing NLP dependencies")
        return False
    
    # Install RAG dependencies
    print("\n3. Installing RAG dependencies...")
    if run_command(f"{sys.executable} -m pip install {' '.join(rag_deps)}") != 0:
        print("Error installing RAG dependencies")
        return False
    
    print("\n=== Installation Complete ===")
    print("You can now run the system in CPU-only mode.")
    print("To test the system, run: python test_api.py")
    print("To run the full system, run: python app.py")
    return True

if __name__ == "__main__":
    success = install_dependencies()
    sys.exit(0 if success else 1) 
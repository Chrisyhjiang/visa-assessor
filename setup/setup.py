"""
Setup script for the O-1A Visa Qualification Assessment System.
This script installs the required dependencies.
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
    """Install the required dependencies"""
    print("=== Installing Dependencies for O-1A Visa Qualification Assessment System ===")
    
    # Install from requirements.txt
    print("\nInstalling dependencies from requirements.txt...")
    if run_command(f"{sys.executable} -m pip install -r requirements.txt") != 0:
        print("Error installing dependencies from requirements.txt")
        return False
    
    print("\n=== Installation Complete ===")
    print("You can now run the system.")
    print("To test the system, run: python test_api.py")
    print("To run the full system, run: python app.py")
    return True

if __name__ == "__main__":
    success = install_dependencies()
    sys.exit(0 if success else 1) 
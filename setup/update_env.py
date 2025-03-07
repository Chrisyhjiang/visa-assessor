"""
Script to update the environment with the latest dependencies.
This will ensure you have the correct versions of all libraries.
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

def update_environment():
    """Update the environment with the latest dependencies"""
    print("=== Updating Environment ===")
    
    # Upgrade pip first
    print("\nUpgrading pip...")
    if run_command(f"{sys.executable} -m pip install --upgrade pip") != 0:
        print("Error upgrading pip")
        return False
    
    # Install/upgrade dependencies from requirements.txt
    print("\nInstalling/upgrading dependencies from requirements.txt...")
    if run_command(f"{sys.executable} -m pip install -r requirements.txt --upgrade") != 0:
        print("Error installing dependencies")
        return False
    
    # Verify transformers version
    print("\nVerifying transformers version...")
    try:
        import transformers
        print(f"Transformers version: {transformers.__version__}")
        if transformers.__version__ < "4.37.0":
            print("Warning: Transformers version is still below 4.37.0")
            print("Trying to force upgrade transformers...")
            if run_command(f"{sys.executable} -m pip install 'transformers>=4.37.0' --force-reinstall") != 0:
                print("Error upgrading transformers")
                return False
    except ImportError:
        print("Error: Transformers not installed")
        return False
    
    print("\n=== Environment Update Complete ===")
    print("You can now run the API with: python app.py")
    return True

if __name__ == "__main__":
    success = update_environment()
    sys.exit(0 if success else 1) 
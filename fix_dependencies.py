import subprocess
import sys
import importlib

def install_package(package_name):
    print(f"Installing {package_name}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--force-reinstall", package_name])
    print(f"{package_name} installed successfully!")

def verify_import(package_name):
    try:
        importlib.import_module(package_name)
        print(f"✅ {package_name} imported successfully!")
        return True
    except ImportError as e:
        print(f"❌ Failed to import {package_name}: {str(e)}")
        return False

if __name__ == "__main__":
    # List of packages to check and install
    packages = ["sentence_transformers", "transformers", "torch"]
    
    for package in packages:
        if not verify_import(package):
            install_package(package)
            verify_import(package)
    
    # Verify HuggingFaceBgeEmbeddings can be imported
    try:
        from langchain.embeddings import HuggingFaceBgeEmbeddings
        print("✅ HuggingFaceBgeEmbeddings imported successfully!")
    except ImportError as e:
        print(f"❌ Failed to import HuggingFaceBgeEmbeddings: {str(e)}")
        install_package("langchain-community")
        try:
            from langchain.embeddings import HuggingFaceBgeEmbeddings
            print("✅ HuggingFaceBgeEmbeddings imported successfully after reinstall!")
        except ImportError as e:
            print(f"❌ Still failed to import HuggingFaceBgeEmbeddings: {str(e)}")
    
    print("\nAll dependencies checked and fixed if needed!") 
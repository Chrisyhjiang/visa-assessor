"""
Script to download the Qwen model files in advance.
This will cache the model locally so future startups are faster.
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def download_model(model_name="Qwen/Qwen1.5-1.8B"):
    """Download and cache the model files"""
    print(f"Downloading model: {model_name}")
    print("This may take several minutes depending on your internet connection...")
    
    # Download tokenizer
    print("\nDownloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Download model
    print("\nDownloading model weights...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    
    # Get cache location
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    
    print("\nDownload complete!")
    print(f"Model files are cached at: {cache_dir}")
    print("Future startups will be faster as the model is now cached locally.")

if __name__ == "__main__":
    # You can change the model name here if you want to use a different model
    download_model("Qwen/Qwen1.5-1.8B")
    
    # Uncomment below if you want to also download the smaller model as a backup
    # download_model("Qwen/Qwen1.5-0.5B") 
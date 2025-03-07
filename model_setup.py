# Set environment variables before any imports
import os
import platform

# Configure GPU acceleration based on platform
if platform.system() == "Darwin":  # macOS
    # Force CPU fallback for MPS issues
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    # Set watermark ratio to 0.0 to avoid the invalid ratio error
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import torch
import logging
import functools

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import required libraries
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Try to import bitsandbytes for quantization
try:
    import bitsandbytes as bnb
    QUANTIZATION_AVAILABLE = True
    logger.info("bitsandbytes is available for quantization")
except ImportError:
    QUANTIZATION_AVAILABLE = False
    logger.warning("bitsandbytes not available. Quantization will be disabled.")

# Define cache directories
LOCAL_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_cache")
HF_CACHE_DIR = os.path.expanduser("~/.cache/huggingface/hub")

# Create local cache directory if it doesn't exist
os.makedirs(LOCAL_CACHE_DIR, exist_ok=True)

# Simple LRU cache for model responses
RESPONSE_CACHE_SIZE = 100
response_cache = {}

# Check for vLLM availability
try:
    import vllm
    VLLM_AVAILABLE = True
    logger.info("vLLM is available for faster inference")
except ImportError:
    VLLM_AVAILABLE = False
    logger.warning("vLLM not available. Using HuggingFace for inference.")

# Detect system capabilities
IS_MAC = platform.system() == "Darwin"
HAS_CUDA = torch.cuda.is_available()

# Safely check for MPS availability
try:
    # Correct way to check for MPS availability
    HAS_MPS = IS_MAC and torch.backends.mps.is_available()
    if HAS_MPS:
        logger.info("MPS (Metal Performance Shaders) is available")
    else:
        logger.warning("MPS is not available on this system")
except Exception as e:
    logger.warning(f"Error checking MPS availability: {str(e)}")
    HAS_MPS = False

# Log available devices
if HAS_CUDA:
    logger.info(f"CUDA is available with {torch.cuda.device_count()} device(s)")
    logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
elif HAS_MPS:
    logger.info("MPS (Metal Performance Shaders) is available")
else:
    logger.warning("No GPU acceleration available. Using CPU only.")

def initialize_model(model_name="Qwen/Qwen1.5-1.8B"):
    """
    Initialize the model for inference
    
    Args:
        model_name: Name or path of the model to initialize
    """
    global VLLM_AVAILABLE, QUANTIZATION_AVAILABLE
    
    # Try to use vLLM first if available (fastest option)
    if VLLM_AVAILABLE and HAS_CUDA:
        try:
            logger.info(f"Initializing model {model_name} with vLLM")
            from vllm import LLM
            
            # Initialize vLLM model
            llm = LLM(
                model=model_name,
                trust_remote_code=True,
                tensor_parallel_size=1  # Use all available GPUs
            )
            
            # Create a wrapper function to match the HF pipeline interface
            def vllm_generate(prompt, **kwargs):
                from vllm import SamplingParams
                
                # Convert HF parameters to vLLM parameters
                temperature = kwargs.get("temperature", 0.1)
                top_p = kwargs.get("top_p", 0.9)
                max_tokens = kwargs.get("max_new_tokens", 2048)
                
                # Create sampling parameters
                sampling_params = SamplingParams(
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens
                )
                
                # Generate text
                outputs = llm.generate([prompt], sampling_params)
                
                # Format output to match HF pipeline
                return [{"generated_text": prompt + outputs[0].outputs[0].text}]
            
            return vllm_generate, "vllm"
        except Exception as e:
            logger.error(f"Error initializing vLLM: {str(e)}")
            logger.warning("Falling back to HuggingFace")
    
    # If vLLM is not available or failed, use HuggingFace
    logger.info(f"Initializing model {model_name} with HuggingFace")
    logger.info(f"Checking HF cache directory: {HF_CACHE_DIR}")
    logger.info(f"Checking local cache directory: {LOCAL_CACHE_DIR}")
    
    # Determine the best device to use
    if HAS_CUDA:
        device = "cuda"
        logger.info("Using CUDA for model inference")
    elif HAS_MPS:
        device = "mps"
        logger.info("Using MPS for model inference on Apple Silicon")
    else:
        device = "cpu"
        logger.info("Using CPU for model inference")
    
    # Check if model exists in HF cache
    model_folder_name = model_name.replace("/", "--")
    model_exists_in_hf_cache = os.path.exists(os.path.join(HF_CACHE_DIR, f"models--{model_folder_name}"))
    if model_exists_in_hf_cache:
        logger.info("Model found in Hugging Face cache")
    else:
        logger.info("Model not found in Hugging Face cache")
    
    try:
        # For Qwen1.5 models, we need to use trust_remote_code=True
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True,
            local_files_only=model_exists_in_hf_cache  # Only use local files if they exist
        )
        
        # Set up model loading parameters based on device
        model_kwargs = {
            "trust_remote_code": True,
            "local_files_only": model_exists_in_hf_cache,  # Only use local files if they exist
        }
        
        # Set appropriate dtype and device map based on device
        if device == "cuda":
            model_kwargs["torch_dtype"] = torch.float16
            model_kwargs["device_map"] = "auto"
        elif device == "mps":
            model_kwargs["torch_dtype"] = torch.float16
            # For MPS, we load on CPU first then move to MPS
            model_kwargs["device_map"] = "cpu"
        else:
            model_kwargs["torch_dtype"] = torch.float32
            model_kwargs["device_map"] = "cpu"
        
        # Load the model
        logger.info(f"Loading model on {device}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        # If using MPS, move model to MPS device after loading
        if device == "mps":
            logger.info("Moving model to MPS device...")
            model = model.to("mps")
        
        # Create pipeline with appropriate device
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map=device if device != "mps" else "cpu"  # For MPS, we already moved the model
        )
        
        # For MPS, we need to ensure the pipeline uses the MPS device
        if device == "mps":
            generator.device = torch.device("mps")
            
        return generator, "hf"
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        logger.warning("Falling back to smaller model")
        
        # Fallback to a smaller model
        try:
            # Try a smaller model that should be faster
            fallback_model = "Qwen/Qwen1.5-0.5B"  # Much smaller model
            logger.info(f"Trying fallback model: {fallback_model}")
            
            # Try loading with the best available device
            logger.info(f"Using {device} for fallback model")
            tokenizer = AutoTokenizer.from_pretrained(fallback_model, trust_remote_code=True)
            
            if device == "mps":
                # For MPS, load on CPU first then move to MPS
                model = AutoModelForCausalLM.from_pretrained(
                    fallback_model,
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    device_map="cpu"
                ).to("mps")
                
                generator = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    device_map="cpu"  # We already moved the model to MPS
                )
                generator.device = torch.device("mps")
            else:
                # For other devices
                model = AutoModelForCausalLM.from_pretrained(
                    fallback_model,
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    device_map=device
                )
                
                generator = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    device_map=device
                )
            
            return generator, "hf"
        except Exception as e:
            logger.error(f"Error loading fallback model: {str(e)}")
            
            # Try an even more basic model as a last resort
            try:
                logger.info("Trying last resort model: gpt2")
                
                # Try loading with the best available device
                logger.info(f"Using {device} for last resort model")
                tokenizer = AutoTokenizer.from_pretrained("gpt2")
                
                if device == "mps":
                    # For MPS, load on CPU first then move to MPS
                    model = AutoModelForCausalLM.from_pretrained(
                        "gpt2",
                        device_map="cpu"
                    ).to("mps")
                    
                    generator = pipeline(
                        "text-generation",
                        model=model,
                        tokenizer=tokenizer,
                        device_map="cpu"  # We already moved the model to MPS
                    )
                    generator.device = torch.device("mps")
                else:
                    # For other devices
                    model = AutoModelForCausalLM.from_pretrained(
                        "gpt2",
                        device_map=device
                    )
                    
                    generator = pipeline(
                        "text-generation",
                        model=model,
                        tokenizer=tokenizer,
                        device_map=device
                    )
                
                return generator, "hf"
            except Exception as e:
                logger.error(f"Error loading last resort model: {str(e)}")
                raise RuntimeError("Failed to load any model. Please check your installation.")

def get_sampling_params():
    """Define sampling parameters for generation"""
    return {
        "temperature": 0.1,
        "top_p": 0.9,
        "max_new_tokens": 2048,
        "do_sample": True
    }

def safe_tensor(data, dtype=None):
    """
    Create a tensor with a safe data type for compatibility
    
    Args:
        data: The data to convert to a tensor
        dtype: The data type to use (default: None, will use appropriate dtype based on device)
    
    Returns:
        A tensor with the specified data type
    """
    # Determine the best device to use
    if torch.cuda.is_available():
        device = "cuda"
        # Default to float16 on CUDA for better performance
        if dtype is None:
            dtype = torch.float16
    elif hasattr(torch, 'mps') and torch.mps.is_available():
        device = "mps"
        # Default to float16 on MPS for better performance
        if dtype is None:
            dtype = torch.float16
    else:
        device = "cpu"
        # Default to float32 on CPU
        if dtype is None:
            dtype = torch.float32
    
    # If dtype is int64, convert to int32 to avoid issues on some devices
    if dtype == torch.int64:
        dtype = torch.int32
        logger.info("Converting int64 tensor to int32 for compatibility")
    
    # Create tensor on the appropriate device
    try:
        return torch.tensor(data, dtype=dtype, device=device)
    except Exception as e:
        logger.warning(f"Error creating tensor on {device}: {str(e)}")
        logger.warning("Falling back to CPU tensor")
        return torch.tensor(data, dtype=torch.float32, device="cpu")

# Simple LRU cache implementation for generate_text
def lru_cache_with_size_limit(maxsize=100):
    """Simple LRU cache decorator with size limit"""
    def decorator(func):
        cache = {}
        order = []
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create a key from the arguments
            key = str(args) + str(kwargs)
            
            # Check if result is in cache
            if key in cache:
                # Move to the end of the order list (most recently used)
                order.remove(key)
                order.append(key)
                logger.debug("Cache hit for generate_text")
                return cache[key]
            
            # Call the function
            result = func(*args, **kwargs)
            
            # Add to cache
            cache[key] = result
            order.append(key)
            
            # Remove oldest entries if cache is too large
            while len(order) > maxsize:
                oldest_key = order.pop(0)
                del cache[oldest_key]
            
            return result
        
        return wrapper
    
    return decorator

@lru_cache_with_size_limit(maxsize=RESPONSE_CACHE_SIZE)
def generate_text(model, prompt, model_type="hf"):
    """
    Generate text using the model
    
    Args:
        model: The model to use for generation
        prompt: The prompt to generate from
        model_type: The type of model ("vllm" or "hf")
    
    Returns:
        The generated text
    """
    # Start timing the generation
    import time
    start_time = time.time()
    
    # For vLLM, the model is actually a function
    if model_type == "vllm":
        # vLLM already has optimized generation
        outputs = model(prompt, **get_sampling_params())
    else:
        # HuggingFace generation
        sampling_params = get_sampling_params()
        
        # Optimize generation by setting appropriate batch size
        if "batch_size" not in sampling_params:
            if hasattr(model, "device") and model.device.type == "cpu":
                # Use smaller batch size on CPU for better performance
                sampling_params["batch_size"] = 1
            else:
                # Use larger batch size on GPU for better performance
                sampling_params["batch_size"] = 4
        
        # Use different generation strategies based on device
        if hasattr(model, "model") and hasattr(model.model, "device"):
            device_type = model.model.device.type
            if device_type == "cuda" or device_type == "mps":
                # For CUDA/MPS, we can use more aggressive settings
                sampling_params["use_cache"] = True
        
        outputs = model(prompt, **sampling_params)
    
    # Log generation time
    end_time = time.time()
    generation_time = end_time - start_time
    logger.info(f"Text generation took {generation_time:.2f} seconds")
    
    # Return just the generated text (without the prompt)
    return outputs[0]["generated_text"][len(prompt):] 
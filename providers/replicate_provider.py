import os
import base64
import replicate
import requests
import io

# Initialize Replicate client
replicate_client = replicate.Client(api_token=os.getenv("REPLICATE_API_TOKEN"))

# Model configurations
AVAILABLE_MODELS = {
    "Imagen 4": {
        "model": "google/imagen-4",
        "input_params": lambda prompt, width, height: {
            "prompt": prompt, 
            "width": width, 
            "height": height
        },
        "edit_params": lambda prompt, width, height, image_file: {
            "prompt": prompt,
            "input_image": image_file,
            "width": width, 
            "height": height
        }
    },
    "Flux Kontext Pro": {
        "model": "black-forest-labs/flux-kontext-pro",
        "input_params": lambda prompt, width, height: {
            "prompt": prompt, 
            "width": width, 
            "height": height
        },
        "edit_params": lambda prompt, width, height, image_file: {
            "prompt": prompt,
            "input_image": image_file,
            "aspect_ratio": "match_input_image",
            "output_format": "png",
            "safety_tolerance": 2,
            "prompt_upsampling": False
        }
    }
}

def generate_image(prompt: str, model_name: str = "Flux Kontext Pro", width=1024, height=1024) -> bytes:
    """
    Generate an image using the specified model.
    
    Args:
        prompt: Text prompt for image generation
        model_name: Name of the model to use (must be in AVAILABLE_MODELS)
        width: Image width in pixels
        height: Image height in pixels
    
    Returns:
        Image content as bytes
    """
    if not os.getenv("REPLICATE_API_TOKEN"):
        raise RuntimeError("REPLICATE_API_TOKEN not set")
    
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Model '{model_name}' not available. Choose from: {list(AVAILABLE_MODELS.keys())}")
    
    model_config = AVAILABLE_MODELS[model_name]
    input_params = model_config["input_params"](prompt, width, height)
    
    try:
        # Run the prediction using the replicate library
        output = replicate_client.run(
            model_config["model"],
            input=input_params
        )
        
        # Handle different output formats
        if hasattr(output, 'read'):
            # New Replicate API format
            return output.read()
        elif hasattr(output, 'url'):
            # Get URL and download
            response = requests.get(output.url(), timeout=120)
            response.raise_for_status()
            return response.content
        elif isinstance(output, list):
            # List of URLs
            image_url = output[0]
            response = requests.get(image_url, timeout=120)
            response.raise_for_status()
            return response.content
        else:
            # Single URL
            response = requests.get(output, timeout=120)
            response.raise_for_status()
            return response.content
        
    except Exception as e:
        raise RuntimeError(f"Replicate {model_name} failed: {str(e)}")

def edit_image_replicate(image_bytes: bytes, prompt: str, model_name: str = "Flux Kontext Pro", width=1024, height=1024) -> bytes:
    """
    Edit an image using the specified Replicate model's img2img capabilities.
    
    Args:
        image_bytes: Input image as bytes
        prompt: Text prompt for image editing
        model_name: Name of the model to use for editing (must be in AVAILABLE_MODELS)
        width: Output image width in pixels
        height: Output image height in pixels
    
    Returns:
        Edited image content as bytes
    """
    if not os.getenv("REPLICATE_API_TOKEN"):
        raise RuntimeError("REPLICATE_API_TOKEN not set")
    
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Model '{model_name}' not available. Choose from: {list(AVAILABLE_MODELS.keys())}")
    
    try:
        model_config = AVAILABLE_MODELS[model_name]
        
        # Check if the model supports editing
        if "edit_params" not in model_config:
            raise ValueError(f"Model '{model_name}' does not support image editing")
        
        # Create a file-like object from the image bytes
        image_file = io.BytesIO(image_bytes)
        
        input_params = model_config["edit_params"](prompt, width, height, image_file)
        
        # Use the selected model for editing
        output = replicate_client.run(
            model_config["model"],
            input=input_params
        )
        
        # Handle different output formats based on the new Replicate API
        if hasattr(output, 'read'):
            # New Replicate API format - can read directly
            return output.read()
        elif hasattr(output, 'url'):
            # Has URL method
            response = requests.get(output.url(), timeout=120)
            response.raise_for_status()
            return response.content
        elif isinstance(output, list):
            # List of URLs
            image_url = output[0]
            response = requests.get(image_url, timeout=120)
            response.raise_for_status()
            return response.content
        else:
            # Single URL string
            response = requests.get(output, timeout=120)
            response.raise_for_status()
            return response.content
        
    except Exception as e:
        raise RuntimeError(f"Replicate {model_name} image editing failed: {str(e)}")

# Keep the old function for backward compatibility
def generate_image_sdxl(prompt: str, width=1024, height=1024) -> bytes:
    """Legacy function - use generate_image instead"""
    return generate_image(prompt, "Flux Kontext Pro", width, height)

def get_available_models():
    """Return list of available model names for UI selection"""
    return list(AVAILABLE_MODELS.keys())

def get_available_edit_models():
    """Return list of available model names that support image editing"""
    return [name for name, config in AVAILABLE_MODELS.items() if "edit_params" in config]
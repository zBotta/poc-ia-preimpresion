import io
import os
import base64
import replicate
from PIL import Image

# Initialize Replicate client
replicate_client = replicate.Client(api_token=os.getenv("REPLICATE_API_TOKEN"))

# Available upscaling models
UPSCALE_MODELS = {
    "Real-ESRGAN": {
        "model": "nightmareai/real-esrgan:42fed1c4974146d4d2414e2be2c5277c7fcf05fcc972b6f011297b040ecc4694",
        "input_params": lambda image_data, scale: {
            "image": f"data:image/png;base64,{base64.b64encode(image_data).decode()}",
            "scale": scale
        }
    },
    "Topaz Labs": {
        "model": "topazlabs/image-upscale",
        "input_params": lambda image_data, scale: {
            "image": f"data:image/png;base64,{base64.b64encode(image_data).decode()}",
            "scale": scale
        }
    }
}

def upscale_lanczos(image_bytes: bytes, scale: float = 2.0) -> bytes:
    """
    Upscale image using Lanczos algorithm (local processing).
    
    Args:
        image_bytes: Image data as bytes
        scale: Scale factor
    
    Returns:
        Upscaled image as bytes
    """
    img = Image.open(io.BytesIO(image_bytes))
    w, h = img.size
    new_w, new_h = int(w * scale), int(h * scale)
    upscaled = img.resize((new_w, new_h), Image.LANCZOS)
    
    output = io.BytesIO()
    upscaled.save(output, format="PNG")
    return output.getvalue()

def upscale_replicate(image_bytes: bytes, model_name: str = "Real-ESRGAN", scale: int = 2) -> bytes:
    """
    Upscale image using Replicate models.
    
    Args:
        image_bytes: Image data as bytes
        model_name: Name of the upscaling model to use
        scale: Scale factor (must be integer)
    
    Returns:
        Upscaled image as bytes
    """
    if not os.getenv("REPLICATE_API_TOKEN"):
        raise RuntimeError("REPLICATE_API_TOKEN not set")
    
    if model_name not in UPSCALE_MODELS:
        raise ValueError(f"Model '{model_name}' not available. Choose from: {list(UPSCALE_MODELS.keys())}")
    
    model_config = UPSCALE_MODELS[model_name]
    input_params = model_config["input_params"](image_bytes, scale)
    
    try:
        # Run the upscaling prediction
        output = replicate_client.run(
            model_config["model"],
            input=input_params
        )
        
        # Handle different output formats
        if isinstance(output, list):
            image_url = output[0]
        else:
            image_url = output
        
        # Download the upscaled image
        import requests
        response = requests.get(image_url, timeout=120)
        response.raise_for_status()
        return response.content
        
    except Exception as e:
        raise RuntimeError(f"Replicate {model_name} upscaling failed: {str(e)}")

# Keep legacy function for backward compatibility
def upscale_realesrgan_replicate(image_bytes: bytes, scale: int = 2) -> bytes:
    """Legacy function - use upscale_replicate instead"""
    return upscale_replicate(image_bytes, "Real-ESRGAN", scale)

def get_upscale_models():
    """Return list of available upscaling model names for UI selection"""
    return list(UPSCALE_MODELS.keys())
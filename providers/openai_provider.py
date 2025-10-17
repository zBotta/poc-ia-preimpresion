import os, base64, requests

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def generate_image_openai(prompt: str, size: str="1024x1024") -> bytes:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set")
    url = "https://api.openai.com/v1/images/generations"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    payload = {"prompt": prompt, "size": size, "response_format": "b64_json"}
    r = requests.post(url, json=payload, headers=headers, timeout=120)
    r.raise_for_status()
    b64 = r.json()["data"][0]["b64_json"]
    return base64.b64decode(b64)

def edit_image_openai(image_bytes: bytes, prompt: str, size: str="1024x1024") -> bytes:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set")
    url = "https://api.openai.com/v1/images/edits"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    files = {
        "image": ("image.png", image_bytes, "image/png"),
        "prompt": (None, prompt),
        "size": (None, size),
        "response_format": (None, "b64_json"),
    }
    r = requests.post(url, files=files, headers=headers, timeout=180)
    r.raise_for_status()
    b64 = r.json()["data"][0]["b64_json"]
    return base64.b64decode(b64)

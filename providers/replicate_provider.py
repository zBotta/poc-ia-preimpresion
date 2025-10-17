import os, time, requests

REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

def _start_prediction(model: str, version: str, inp: dict) -> str:
    if not REPLICATE_API_TOKEN:
        raise RuntimeError("REPLICATE_API_TOKEN not set")
    url = "https://api.replicate.com/v1/predictions"
    headers = {"Authorization": f"Token {REPLICATE_API_TOKEN}", "Content-Type": "application/json"}
    payload = {"version": version, "input": inp}
    r = requests.post(url, json=payload, headers=headers, timeout=60)
    r.raise_for_status()
    return r.json()["id"]

def _poll(pred_id: str, timeout_s: int=300):
    url = f"https://api.replicate.com/v1/predictions/{pred_id}"
    headers = {"Authorization": f"Token {REPLICATE_API_TOKEN}"}
    t0 = time.time()
    while True:
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        data = r.json()
        status = data["status"]
        if status in ("succeeded","failed","canceled"):
            return data
        if time.time() - t0 > timeout_s:
            raise TimeoutError("Replicate prediction timed out")
        time.sleep(2.0)

_SDXL_MODEL = "stability-ai/sdxl"
_SDXL_VERSION = "8f8d5e7f2a58c12a1e5b1e29a51b9db6d735812c1de9e8b456e17b6f8c3d5f39"  # example, may change

def generate_image_sdxl(prompt: str, width=1024, height=1024) -> bytes:
    pred_id = _start_prediction(_SDXL_MODEL, _SDXL_VERSION, {"prompt": prompt, "width": width, "height": height})
    data = _poll(pred_id)
    if data["status"] != "succeeded":
        raise RuntimeError(f"Replicate SDXL failed: {data}")
    out_url = data["output"][0]
    img = requests.get(out_url, timeout=120)
    img.raise_for_status()
    return img.content

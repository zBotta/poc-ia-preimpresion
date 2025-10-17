import io, requests, os, time
from PIL import Image

REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

def upscale_lanczos(image_bytes: bytes, scale: float=2.0) -> bytes:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    new_size = (int(w*scale), int(h*scale))
    up = img.resize(new_size, Image.LANCZOS)
    out = io.BytesIO()
    up.save(out, format="PNG", optimize=True)
    return out.getvalue()

_REALESRGAN_VERSION = "9936c2332f4869889dae465d174a264a3ae59a1ce5f9b89e0ee1d3f3f94a515a"  # example

def upscale_realesrgan_replicate(image_bytes: bytes, scale:int=2) -> bytes:
    if not REPLICATE_API_TOKEN:
        raise RuntimeError("REPLICATE_API_TOKEN not set")
    upload_url = "https://api.replicate.com/v1/uploads"
    headers = {"Authorization": f"Token {REPLICATE_API_TOKEN}"}
    r = requests.post(upload_url, headers=headers, timeout=30)
    r.raise_for_status()
    upload = r.json()
    put_url = upload["upload_url"]
    r2 = requests.put(put_url, data=image_bytes, headers={"Content-Type":"image/png"}, timeout=120)
    r2.raise_for_status()
    serving_url = upload["serving_url"]
    pred_req = {"version": _REALESRGAN_VERSION, "input": {"image": serving_url, "scale": scale}}
    r3 = requests.post("https://api.replicate.com/v1/predictions", json=pred_req, headers={"Authorization": f"Token {REPLICATE_API_TOKEN}", "Content-Type":"application/json"}, timeout=60)
    r3.raise_for_status()
    pred_id = r3.json()["id"]
    t0 = time.time()
    while True:
        rp = requests.get(f"https://api.replicate.com/v1/predictions/{pred_id}", headers={"Authorization": f"Token {REPLICATE_API_TOKEN}"}, timeout=30)
        rp.raise_for_status()
        data = rp.json()
        if data["status"] == "succeeded":
            out_url = data["output"]
            if isinstance(out_url, list):
                out_url = out_url[0]
            img = requests.get(out_url, timeout=120)
            img.raise_for_status()
            return img.content
        if data["status"] in ("failed","canceled"):
            raise RuntimeError(f"Real-ESRGAN failed: {data}")
        if time.time() - t0 > 420:
            raise TimeoutError("Real-ESRGAN timeout")
        time.sleep(2.0)

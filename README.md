# PoC IA Preimpresión (Streamlit, Python)

Este PoC cubre:
1) Generación de imágenes desde prompt o foto.
2) Upscaling / cambio de tamaño (Lanczos local y Real-ESRGAN en Replicate).
3) Chequeo de PPI efectivo en imágenes y PDFs.

## Arranque rápido
- `pip install -r requirements.txt`
- `streamlit run app.py`

## Variables de entorno
- `OPENAI_API_KEY` (para generación)
- `REPLICATE_API_TOKEN` (para SDXL y Real-ESRGAN en Replicate)

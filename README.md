# AI Prepress PoC (Streamlit, Python)

This PoC covers:
1) Image generation from a prompt or photo.
2) Upscaling/resizing (local Lanczos and Real-ESRGAN in Replicate).
3) Effective PPI checking on images and PDFs.

## Quick Start
- `pip install -r requirements.txt`
- `streamlit run app.py`

## Environment Variables
A `.env` file must be included in the root folder with the following variables:
- `OPENAI_API_KEY` (for generation)
- `REPLICATE_API_TOKEN` (for SDXL and Real-ESRGAN in Replicate)
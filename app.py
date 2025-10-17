import os, io
import streamlit as st
from dotenv import load_dotenv
from PIL import Image

from providers.openai_provider import generate_image_openai, edit_image_openai
from providers.replicate_provider import generate_image_sdxl
from services.upscale import upscale_lanczos, upscale_realesrgan_replicate
from utils.ppi import ppi_from_image_bytes, min_effective_ppi_in_pdf

load_dotenv()
st.set_page_config(page_title="PoC IA Preimpresión", page_icon="🖨️", layout="wide")
st.title("🖨️ PoC IA para Preimpresión (B2C)")

with st.sidebar:
    st.markdown("**Proveedor IA (generación):**")
    prov = st.selectbox("Modelo", ["OpenAI Images", "Replicate SDXL"], index=0)
    st.markdown("---")
    st.markdown("**Upscaling:**")
    up_mode = st.selectbox("Método", ["Lanczos (local)", "Real-ESRGAN (Replicate)"], index=0)
    st.markdown("---")
    st.info("Configura tus claves en variables de entorno: OPENAI_API_KEY, REPLICATE_API_TOKEN")

tabs = st.tabs(["1) Generar", "2) Upscale / Resize", "3) Chequeo PPI"])

with tabs[0]:
    st.subheader("Generación de imágenes (prompt o a partir de foto)")
    mode = st.radio("Modo", ["Prompt → Imagen", "Imagen → Edit (prompt)"], horizontal=True)
    if mode == "Prompt → Imagen":
        prompt = st.text_area("Prompt", placeholder="Una ilustración de un faro en la costa al atardecer…")
        size = st.selectbox("Tamaño", ["1024x1024","1024x1536","1536x1024","2048x2048"], index=0)
        if st.button("Generar", type="primary"):
            try:
                if prov == "OpenAI Images":
                    img_bytes = generate_image_openai(prompt, size=size)
                else:
                    w,h = [int(x) for x in size.split("x")]
                    img_bytes = generate_image_sdxl(prompt, width=w, height=h)
                st.image(img_bytes, caption="Resultado", use_column_width=True)
                st.download_button("Descargar PNG", data=img_bytes, file_name="generated.png", mime="image/png")
            except Exception as e:
                st.error(f"Error generando imagen: {e}")
    else:
        upl = st.file_uploader("Sube una imagen (PNG/JPG)", type=["png","jpg","jpeg"])
        prompt_edit = st.text_input("Prompt de edición", value="Mejora colores y contraste")
        size2 = st.selectbox("Tamaño", ["1024x1024","1024x1536","1536x1024"], index=0)
        if st.button("Editar con IA"):
            if upl is None:
                st.warning("Sube una imagen primero.")
            else:
                try:
                    if prov == "OpenAI Images":
                        img_bytes = edit_image_openai(upl.read(), prompt_edit, size=size2)
                        st.image(img_bytes, caption="Editado", use_column_width=True)
                        st.download_button("Descargar PNG", data=img_bytes, file_name="edited.png", mime="image/png")
                    else:
                        st.warning("La edición image-to-image está implementada solo para OpenAI en este PoC.")
                except Exception as e:
                    st.error(f"Error en edición: {e}")

with tabs[1]:
    st.subheader("Upscaling y cambio de tamaño")
    up_img = st.file_uploader("Sube una imagen (PNG/JPG)", type=["png","jpg","jpeg"], key="upimg")
    colA, colB = st.columns(2)
    with colA:
        scale = st.slider("Factor de escalado (x)", 1.0, 4.0, 2.0, 0.5)
        target_w = st.number_input("Ancho objetivo (px, opcional)", min_value=0, value=0, step=10)
        target_h = st.number_input("Alto objetivo (px, opcional)", min_value=0, value=0, step=10)
    with colB:
        st.info("Si conoces el tamaño físico final, calcula píxeles: mm / 25,4 × ppp")
    if st.button("Aplicar Upscale / Resize", type="primary"):
        if up_img is None:
            st.warning("Sube una imagen primero.")
        else:
            raw = up_img.read()
            try:
                if target_w>0 and target_h>0:
                    im = Image.open(io.BytesIO(raw)).convert("RGB")
                    im2 = im.resize((int(target_w), int(target_h)), Image.LANCZOS)
                    out = io.BytesIO(); im2.save(out, format="PNG")
                    result = out.getvalue()
                else:
                    if up_mode == "Lanczos (local)":
                        result = upscale_lanczos(raw, scale=scale)
                    else:
                        result = upscale_realesrgan_replicate(raw, scale=int(scale))
                st.image(result, use_column_width=True)
                st.download_button("Descargar PNG", data=result, file_name="upscaled.png", mime="image/png")
            except Exception as e:
                st.error(f"Error en upscaling: {e}")

with tabs[2]:
    st.subheader("Chequeo PPI (imagen o PDF)")
    mode_ppi = st.radio("Tipo de archivo", ["Imagen", "PDF"], horizontal=True)
    if mode_ppi == "Imagen":
        file = st.file_uploader("Sube imagen", type=["png","jpg","jpeg"], key="ppi_img")
        c1, c2 = st.columns(2)
        with c1:
            tw = st.number_input("Ancho objetivo (mm)", min_value=1.0, value=210.0, step=1.0)
        with c2:
            th = st.number_input("Alto objetivo (mm)", min_value=1.0, value=297.0, step=1.0)
        if st.button("Calcular PPI requerido", key="btn_ppi_img"):
            if file is None:
                st.warning("Sube una imagen.")
            else:
                info = ppi_from_image_bytes(file.read(), (tw, th))
                st.json(info)
    else:
        file = st.file_uploader("Sube PDF", type=["pdf"], key="ppi_pdf")
        if st.button("Analizar PDF", key="btn_ppi_pdf"):
            if file is None:
                st.warning("Sube un PDF.")
            else:
                data = min_effective_ppi_in_pdf(file.read())
                st.write("Resultados por página:")
                st.json(data)

st.caption("© PoC IA Preimpresión — Streamlit + Python. Amplía a Firefly/Bedrock, añade preflight y CMYK en tu pipeline.")

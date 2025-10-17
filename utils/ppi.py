import io
from typing import Tuple, List, Dict, Any
from PIL import Image
import fitz  # PyMuPDF

def ppi_from_image_bytes(data: bytes, target_mm: Tuple[float,float]) -> Dict[str, Any]:
    img = Image.open(io.BytesIO(data))
    px_w, px_h = img.size
    tw_mm, th_mm = target_mm
    tw_in, th_in = tw_mm / 25.4, th_mm / 25.4
    req_ppi_w = px_w / tw_in if tw_in > 0 else None
    req_ppi_h = px_h / th_in if th_in > 0 else None
    return {
        "pixels": (px_w, px_h),
        "target_mm": target_mm,
        "required_ppi": (round(req_ppi_w,2) if req_ppi_w else None, round(req_ppi_h,2) if req_ppi_h else None),
    }

def min_effective_ppi_in_pdf(pdf_bytes: bytes) -> List[Dict[str, Any]]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    results = []
    for pno, page in enumerate(doc):
        page_min = None
        page_images = []
        images = page.get_images(full=True)
        for (xref, smask, w, h, bpc, cs, name, filter, d, enc, interp) in images:
            rect = page.get_image_bbox(xref)  # display bbox in points
            w_in, h_in = rect.width/72.0, rect.height/72.0
            if w_in <= 0 or h_in <= 0:
                continue
            ppi_x = w / w_in
            ppi_y = h / h_in
            eff = min(ppi_x, ppi_y)
            entry = {"xref": xref, "display_in": (round(w_in,2), round(h_in,2)), "pixels": (w, h),
                     "ppi_x": round(ppi_x,1), "ppi_y": round(ppi_y,1), "min_ppi": round(eff,1)}
            page_images.append(entry)
            page_min = entry if (page_min is None or eff < page_min["min_ppi"]) else page_min
        results.append({"page": pno+1, "min_ppi": (page_min["min_ppi"] if page_min else None), "images": page_images})
    return results

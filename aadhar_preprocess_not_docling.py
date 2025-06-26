import os
import re
import csv
import pickle
import io
import logging
import subprocess

from PIL import Image
from dotenv import load_dotenv

# === Optional docling OCR ===
# try:
#     from docling.document_converter import DocumentConverter
#     from docling_core.types.io import DocumentStream
#     use_docling = True
# except ImportError:
#     use_docling = False
#     import pytesseract
use_docling = False

# === Setup ===
load_dotenv()
logging.basicConfig(level=logging.INFO)

input_folder = "Users/yrevash/Qoneqt/Test_model/pipeline_output/session_20250620_093304/3_cropped_entities"
output_csv = "./aadhaar_data_local.csv"
output_pkl = "./aadhaar_data_local.pkl"
os.makedirs(input_folder, exist_ok=True)

# === Load OCR backend ===
if use_docling:
    converter = DocumentConverter()
    logging.info("✅ Using docling for OCR")
else:
    logging.info("⚠️ Using Tesseract as fallback")

def upscale_image(pil_img: Image.Image, hint: str) -> Image.Image:
    inp = f"/tmp/{hint}_in.png"
    out_dir = "/tmp"
    pil_img.save(inp)
    try:
        subprocess.run([
            "upscayl", "--input", inp, "--output", out_dir,
            "--scale", "2", "--mode", "real-esrgan"
        ], check=True)
        for f in os.listdir(out_dir):
            if f.startswith(hint) and f.endswith(".png"):
                return Image.open(os.path.join(out_dir, f))
    except Exception as e:
        logging.warning(f"Upscaling failed: {e}")
    return pil_img

def extract_text(img: Image.Image) -> str:
    max_side = 1200
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / float(max(w, h))
        new_size = (int(w * scale), int(h * scale))
        img = img.resize(new_size, Image.LANCZOS)

    if use_docling:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        doc_stream = DocumentStream(name="input.png", stream=buf)
        result = converter.convert(doc_stream)
        return result.document.export_to_markdown()
    else:
        return pytesseract.image_to_string(img)

# === Extraction logic reused ===
def extract_info(front_text: str, back_text: str = None):
    # Same as your extract_info function (copy/paste from your code)
    # ... skipped for brevity ...
    # Use the same function you already have

    # Just for this demo (replace with your real extract_info logic)
    return {
        "Name": "Unknown",
        "Gender": "Unknown",
        "Aadhaar Number": "000000000000",
        "VID": None,
        "Address": None,
        "Pincode": None,
    }

def save_data(info: dict):
    try:
        if os.path.exists(output_pkl):
            with open(output_pkl, 'rb') as f:
                all_data = pickle.load(f)
        else:
            all_data = []

        existing = [x["Aadhaar Number"] for x in all_data if "Aadhaar Number" in x]
        if info["Aadhaar Number"] in existing:
            logging.info(f"⚠️ Duplicate: {info['Aadhaar Number']}")
            return False

        all_data.append(info)
        with open(output_pkl, 'wb') as f:
            pickle.dump(all_data, f)

        write_header = not os.path.exists(output_csv)
        with open(output_csv, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=info.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(info)

        logging.info(f"✅ Saved: {info['Aadhaar Number']}")
        return True
    except Exception as e:
        logging.error(f"❌ Save failed: {e}")
        return False

def run_batch():
    image_files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    logging.info(f"📁 Found {len(image_files)} images")

    for i in range(0, len(image_files), 2):
        front_path = os.path.join(input_folder, image_files[i])
        back_path = os.path.join(input_folder, image_files[i + 1]) if i + 1 < len(image_files) else None

        front_img = Image.open(front_path)
        back_img = Image.open(back_path) if back_path else None

        logging.info(f"🔍 Processing: {image_files[i]}")

        front_img = upscale_image(front_img, "front")
        front_txt = extract_text(front_img)

        back_txt = None
        if back_img:
            back_img = upscale_image(back_img, "back")
            back_txt = extract_text(back_img)

        info = extract_info(front_txt, back_txt)
        save_data(info)

if __name__ == "__main__":
    run_batch()

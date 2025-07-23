# final_api_with_statuses.py

import asyncio
import hashlib
import json
import logging
import os
import pickle
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiofiles
import aiohttp
import cv2
import numpy as np
import pandas as pd
import pytesseract
import torch
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, Field, HttpUrl
from ultralytics import YOLO

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComprehensiveAadhaarPipeline:
    def __init__(self, model1_path, model2_path, other_lang_code='hin+tel+ben'):
        self.model1_path = model1_path
        self.model2_path = model2_path
        
        if not Path(self.model1_path).exists():
            raise FileNotFoundError(f"Model1 not found at {self.model1_path}")
        if not Path(self.model2_path).exists():
            raise FileNotFoundError(f"Model2 not found at {self.model2_path}")

        self.model1 = YOLO(self.model1_path)
        self.model2 = YOLO(self.model2_path)
        
        self.other_lang_code = other_lang_code
        self._check_tesseract()

        self.card_classes = {i: name for i, name in self.model1.names.items()}
        self.entity_classes = {
            0: 'aadharNumber', 1: 'address', 2: 'address_other_lang', 3: 'city',
            4: 'dob', 5: 'gender', 6: 'gender_other_lang', 7: 'mobile_no',
            8: 'name', 9: 'name_otherlang', 10: 'pincode', 11: 'state'
        }

    def _check_tesseract(self):
        try:
            pytesseract.get_tesseract_version()
        except pytesseract.TesseractNotFoundError:
            raise RuntimeError("Tesseract not found. Please install it and ensure it's in your PATH.")

    def detect_and_crop_cards(self, image_paths, confidence_threshold: float):
        cropped_card_data = {'front': [], 'back': []}
        for image_path in image_paths:
            img = cv2.imread(str(image_path))
            if img is None: continue
            results = self.model1(img)
            for box in results[0].boxes:
                if float(box.conf[0]) < confidence_threshold: continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_name = self.card_classes.get(int(box.cls[0]), "unknown")
                if class_name in ['aadhar_front', 'aadhar_back']:
                    crop = img[y1:y2, x1:x2]
                    cropped_card_data[class_name.replace('aadhar_', '')].append(crop)
        return cropped_card_data

    def correct_card_orientation(self, cropped_card_data: Dict[str, List[np.ndarray]], osd_confidence: float = 1.0) -> Dict[str, List[np.ndarray]]:
        corrected_card_data = {'front': [], 'back': []}
        for card_type, images in cropped_card_data.items():
            for img in images:
                try:
                    h, w = img.shape[:2]
                    if h > w:
                        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                    
                    osd = pytesseract.image_to_osd(img, config='--psm 0', output_type=pytesseract.Output.DICT)
                    rotation_angle = osd.get('rotate', 0)
                    confidence = osd.get('orientation_conf', 0.0)

                    corrected_img = img
                    if rotation_angle != 0 and confidence >= osd_confidence:
                        if rotation_angle == 180:
                            corrected_img = cv2.rotate(img, cv2.ROTATE_180)
                        elif rotation_angle == 90:
                            corrected_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                        elif rotation_angle == 270:
                             corrected_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    
                    corrected_card_data[card_type].append(corrected_img)
                except Exception as e:
                    logger.error(f"Error during orientation correction: {e}. Using pre-corrected image.")
                    corrected_card_data[card_type].append(img)
        return corrected_card_data
    
    def process_images(self, image_paths: List[str], confidence_threshold: float):
        try:
            organized_results = {'front': {'entities': {}}, 'back': {'entities': {}}}
            
            cropped_data = self.detect_and_crop_cards(image_paths, confidence_threshold)
            if not cropped_data.get('front') and not cropped_data.get('back'):
                return {'error': 'No Aadhaar cards detected.'}

            corrected_data = self.correct_card_orientation(cropped_data)

            for card_type, images in corrected_data.items():
                for card_image in images:
                    results = self.model2(card_image)
                    for box in results[0].boxes:
                        if float(box.conf[0]) < confidence_threshold: continue
                        
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        entity_class = self.entity_classes.get(int(box.cls[0]), "unknown")
                        
                        entity_crop = card_image[y1:y2, x1:x2]
                        if entity_crop.size == 0: continue

                        try:
                            pil_img = Image.fromarray(cv2.cvtColor(entity_crop, cv2.COLOR_BGR2GRAY))
                            lang_to_use = self.other_lang_code if entity_class.endswith('_other_lang') else 'eng'
                            text = pytesseract.image_to_string(pil_img, lang=lang_to_use, config='--psm 6').strip()
                            
                            if text:
                                if entity_class not in organized_results[card_type]['entities']:
                                    organized_results[card_type]['entities'][entity_class] = []
                                
                                organized_results[card_type]['entities'][entity_class].append({
                                    'confidence': float(box.conf[0]),
                                    'extracted_text': ' '.join(text.split())
                                })
                        except Exception as ocr_error:
                            logger.error(f"OCR failed for entity {entity_class}: {ocr_error}")

            return {'organized_results': organized_results}
        except Exception as e:
            logger.error(f"Unhandled error in pipeline: {e}\n{traceback.format_exc()}")
            return {'error': str(e)}

app = FastAPI(title="Aadhaar Processing API", version="3.1.0-with-statuses")

class Config:
    BASE_DIR = Path(__file__).parent
    MODEL1_PATH = BASE_DIR / os.environ.get("MODEL1_PATH", "models/best.pt")
    MODEL2_PATH = BASE_DIR / os.environ.get("MODEL2_PATH", "models/best4.pt")
    DOWNLOAD_DIR = BASE_DIR / "downloads"
    SUMMARY_DATA_DIR = Path(os.environ.get("SUMMARY_DATA_DIR", "/Users/hqpl/Desktop/aadhar_testing/AadharAuth/data"))

config = Config()
pipeline: Optional[ComprehensiveAadhaarPipeline] = None

class AadhaarProcessRequest(BaseModel):
    user_id: str = "default_user"
    front_url: HttpUrl
    back_url: HttpUrl
    confidence_threshold: float = Field(0.15, ge=0.0, le=1.0)

@app.on_event("startup")
async def startup_event():
    global pipeline
    try:
        config.DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
        config.SUMMARY_DATA_DIR.mkdir(parents=True, exist_ok=True)
        pipeline = ComprehensiveAadhaarPipeline(
            model1_path=str(config.MODEL1_PATH),
            model2_path=str(config.MODEL2_PATH)
        )
    except Exception as e:
        logger.critical(f"Pipeline initialization failed: {e}", exc_info=True)
        sys.exit(1)

async def download_image(session: aiohttp.ClientSession, url: str, filepath: Path) -> bool:
    try:
        async with session.get(str(url), timeout=30) as response:
            response.raise_for_status()
            async with aiofiles.open(filepath, 'wb') as f: await f.write(await response.read())
            return True
    except Exception:
        return False

def extract_main_fields(organized_results: Dict[str, Any]) -> Dict[str, Any]:
    fields = ['aadharNumber', 'dob', 'gender', 'name', 'address', 'pincode', 'state']
    data = {key: "" for key in fields}
    for side in ['front', 'back']:
        if side in organized_results:
            for field in fields:
                if field in organized_results[side]['entities'] and organized_results[side]['entities'][field]:
                    text = organized_results[side]['entities'][field][0].get('extracted_text', '')
                    if text: data[field] = text
    return data

@app.post("/verify_aadhar", response_class=JSONResponse)
async def verify_aadhaar_sync(request: AadhaarProcessRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Service Unavailable: Pipeline not initialized")
    
    task_id = hashlib.md5(f"{request.user_id}_{datetime.now().timestamp()}".encode()).hexdigest()
    user_download_dir = config.DOWNLOAD_DIR / request.user_id / task_id
    user_download_dir.mkdir(parents=True, exist_ok=True)

    try:
        front_path = user_download_dir / "front.jpg"
        back_path = user_download_dir / "back.jpg"
        async with aiohttp.ClientSession() as session:
            downloads = await asyncio.gather(
                download_image(session, str(request.front_url), front_path),
                download_image(session, str(request.back_url), back_path)
            )
            if not all(downloads):
                raise HTTPException(status_code=400, detail="Failed to download one or both Aadhaar images")

        result = pipeline.process_images([str(front_path), str(back_path)], request.confidence_threshold)
        
        # --- MODIFIED ERROR HANDLING ---
        if 'error' in result:
            if result['error'] == 'No Aadhaar cards detected.':
                return JSONResponse(
                    status_code=404, 
                    content={"status": "No Aadhaar Card Detected", "data": None}
                )
            # For all other unexpected errors, raise a 500
            raise HTTPException(status_code=500, detail=f"Internal Server Error: {result['error']}")

        organized = result.get('organized_results', {})
        main_data = extract_main_fields(organized)
        main_data['User ID'] = request.user_id

        pkl_path = config.SUMMARY_DATA_DIR / "summary.pkl"
        csv_path = config.SUMMARY_DATA_DIR / "summary.csv"
        
        aadhar_number = main_data.get('aadharNumber', '').replace(' ', '')
        if pkl_path.exists() and aadhar_number:
            try:
                with open(pkl_path, 'rb') as pf:
                    all_data = pickle.load(pf)
                for entry in all_data:
                    if entry.get('aadharNumber', '').replace(' ', '') == aadhar_number:
                        return JSONResponse(
                            status_code=200,
                            content={"status": "Aadhar Data Already Exists", "data": {"name": entry.get("name", ""), "aadharNumber": entry.get("aadharNumber", ""),"matched_user_id": entry.get("User ID", "")}}
                        )
            except Exception as e:
                logger.error(f"Failed to check PKL for duplicates: {e}")

        try:
            all_data = []
            if pkl_path.exists():
                with open(pkl_path, 'rb') as pf: all_data = pickle.load(pf)
            all_data.append(main_data)
            with open(pkl_path, 'wb') as pf: pickle.dump(all_data, pf)
        except Exception as e: logger.error(f"Failed to update PKL file: {e}")
        
        try:
            pd.DataFrame([main_data]).to_csv(csv_path, mode='a', header=not csv_path.exists(), index=False)
        except Exception as e: logger.error(f"Failed to update CSV file: {e}")
        
        for k, v in main_data.items():
            main_data[k] = "" if v is None else str(v)

        return JSONResponse(content={"status": "saved", "data": main_data})

    finally:
        if user_download_dir.exists():
            shutil.rmtree(user_download_dir, ignore_errors=True)

if __name__ == "__main__":
    uvicorn.run("final_api_with_statuses:app", host="0.0.0.0", port=8200, reload=True)
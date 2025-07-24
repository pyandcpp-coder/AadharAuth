import asyncio
import hashlib
import json
import logging
import math # Added for rotation calculation
import os
import pickle
import shutil
import sys
import tempfile
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import aiofiles
import aiohttp
import cv2
import numpy as np
import pandas as pd  # Added for CSV writing
import pytesseract
import torch
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, Field, HttpUrl
from ultralytics import YOLO

# Load environment variables from .env file
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- Core Pipeline Logic (Refactored for Fully In-Memory Processing) ---

class ComprehensiveAadhaarPipeline:
    ### MODIFIED: Removed output_base_dir from constructor ###
    def __init__(self, model1_path, model2_path, confidence_threshold=0.10, other_lang_code='hin+tel+ben'):
        ### NEW: Dynamic device selection (CUDA/CPU) ###
        if torch.cuda.is_available():
            self.device = "cuda"
            logger.info(f"✅ CUDA is available. Models will run on GPU ({torch.cuda.get_device_name(0)}).")
        else:
            self.device = "cpu"
            logger.info("⚠️ CUDA not available. Models will fall back to CPU.")
        ### END NEW ###

        self.model1_path = model1_path
        self.model2_path = model2_path
        
        logger.info("🔁 Checking for YOLO models on filesystem...")
        if not Path(self.model1_path).exists():
            logger.critical(f"❌ Model1 not found at {self.model1_path}. Aborting startup.")
            raise FileNotFoundError(f"Model1 not found at {self.model1_path}")
        if not Path(self.model2_path).exists():
            logger.critical(f"❌ Model2 not found at {self.model2_path}. Aborting startup.")
            raise FileNotFoundError(f"Model2 not found at {self.model2_path}")

        logger.info("✅ Loading models directly from filesystem...")
        self.model1 = YOLO(self.model1_path)
        self.model2 = YOLO(self.model2_path)
        
        self.default_confidence_threshold = confidence_threshold
        self.other_lang_code = other_lang_code

        self._check_tesseract()

        logger.info("✅ YOLOv8 models loaded successfully from filesystem.")
        logger.info(f"Model1 classes: {self.model1.names}")
        logger.info(f"Model2 classes: {self.model2.names}")

        self.card_classes = {i: name for i, name in self.model1.names.items()}

        self.entity_classes = {
            0: 'aadharNumber', 1: 'address', 2: 'address_other_lang', 3: 'city',
            4: 'dob', 5: 'gender', 6: 'gender_other_lang', 7: 'mobile_no',
            8: 'name', 9: 'name_otherlang', 10: 'pincode', 11: 'state'
        }
        logger.info(f"✅ Pipeline initialized to use '{self.other_lang_code}' for other language fields.")

    def _check_tesseract(self):
        try:
            pytesseract.get_tesseract_version()
        except pytesseract.TesseractNotFoundError:
            logger.critical("❌ Tesseract executable not found. Please install Tesseract OCR and ensure it's in your PATH.")
            raise RuntimeError("Tesseract not found")
        except Exception as e:
             logger.critical(f"An error occurred while checking Tesseract: {e}")
             raise RuntimeError(f"Error checking Tesseract: {e}")

    def detect_and_crop_cards(self, image_arrays: List[np.ndarray], confidence_threshold: float) -> Dict[str, List[np.ndarray]]:
        """Step 1: Detect Aadhaar front and back cards and crop them from in-memory images."""
        logger.info(f"\n🔍 Step 1: Detecting Aadhaar cards in {len(image_arrays)} images (Threshold: {confidence_threshold})")
        cropped_cards = {'front': [], 'back': []}
        for i, img in enumerate(image_arrays):
            logger.info(f"  Processing image index: {i}")
            results = self.model1(img, device=self.device)
            detected = False
            for box in results[0].boxes:
                if float(box.conf[0]) < confidence_threshold: continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_name = self.card_classes.get(int(box.cls[0]), "unknown")
                if class_name == 'print_aadhar':
                    raise ValueError("Print Aadhaar detected - processing stopped for security reasons")
                if class_name not in ['aadhar_front', 'aadhar_back']: continue
                detected = True
                crop = img[y1:y2, x1:x2]
                cropped_cards[class_name.replace('aadhar_', '')].append(crop)
                logger.info(f"    ✅ Cropped {class_name} in-memory.")
            if not detected:
                logger.warning(f"    ⚠️ No Aadhaar card detected in image index {i}.")
        return cropped_cards

    def _get_rotated_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotates an image by a given angle, expanding the canvas to avoid cropping."""
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
        return cv2.warpAffine(image, M, (nW, nH))

    def correct_card_orientation_opencv(self, cropped_cards: Dict[str, List[np.ndarray]], osd_confidence: float = 1.0) -> Dict[str, List[np.ndarray]]:
        """Step 1.5: Corrects card orientation using in-memory numpy arrays."""
        logger.info("\n🔄 Step 1.5: Correcting card orientation using OpenCV contours and Tesseract OSD")
        corrected_card_arrays = {'front': [], 'back': []}
        all_cards_to_process = []
        all_cards_to_process.extend([(img, 'front') for img in cropped_cards.get('front', [])])
        all_cards_to_process.extend([(img, 'back') for img in cropped_cards.get('back', [])])

        if not all_cards_to_process:
            logger.warning("  No cards to correct orientation for.")
            return corrected_card_arrays

        for i, (img, card_type) in enumerate(all_cards_to_process):
            try:
                if img is None:
                    logger.warning(f"  ⚠️ Null image array for {card_type} card index {i}, skipping.")
                    continue
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (5, 5), 0)
                edged = cv2.Canny(gray, 50, 150)
                contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                rotated_img = img
                if contours:
                    main_contour = max(contours, key=cv2.contourArea)
                    rect = cv2.minAreaRect(main_contour)
                    angle = rect[-1]
                    box_w, box_h = rect[1]
                    if box_w < box_h: angle += 90
                    logger.info(f"  Processing {card_type} card index {i}: Detected OpenCV rotation angle: {angle:.2f}°")
                    rotated_img = self._get_rotated_image(img, angle)
                final_corrected_img = rotated_img
                osd = pytesseract.image_to_osd(rotated_img, config='--psm 0', output_type=pytesseract.Output.DICT)
                rotation_angle = osd.get('rotate', 0)
                confidence = osd.get('orientation_conf', 0.0)
                if rotation_angle == 180 and confidence >= osd_confidence:
                    logger.info(f"    -> Tesseract detected 180° flip (conf: {confidence:.2f}). Applying final correction.")
                    final_corrected_img = cv2.rotate(rotated_img, cv2.ROTATE_180)
                corrected_card_arrays[card_type].append(final_corrected_img)
                logger.info(f"    ✅ Corrected orientation for {card_type} card index {i} in-memory.")
            except Exception as e:
                logger.error(f"  ❌ Error during orientation correction for {card_type} card index {i}: {e}. Using original image as fallback.", exc_info=True)
                corrected_card_arrays[card_type].append(img)
        return corrected_card_arrays
    
    def detect_entities_in_card(self, card_array: np.ndarray, confidence_threshold: float) -> List[Dict]:
        """Step 2: Detect entities in a single orientation-corrected card array."""
        logger.info(f"  Detecting entities in a single card (Threshold: {confidence_threshold})")
        results = self.model2(card_array, device=self.device)
        card_detections = []
        for box in results[0].boxes:
            if float(box.conf[0]) < confidence_threshold: continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_name = self.entity_classes.get(int(box.cls[0]), "unknown")
            card_detections.append({'bbox': (x1, y1, x2, y2), 'class_name': class_name, 'confidence': float(box.conf[0])})
        logger.info(f"    ✅ Detected {len(card_detections)} entities.")
        return card_detections

    def _correct_entity_orientation_and_preprocess(self, image_array: np.ndarray, class_name: str, osd_confidence_threshold: float = 1.0) -> Optional[Image.Image]:
        """Takes a numpy array of a cropped entity, corrects orientation, and returns a preprocessed PIL Image."""
        try:
            img = image_array
            if img is None or img.size == 0:
                 logger.warning(f"    ⚠️ Received an empty image array for entity '{class_name}', skipping.")
                 return None
            h, w = img.shape[:2]
            if h < 100:
                scale_factor = 100 / h
                new_w, new_h = int(w * scale_factor), int(h * scale_factor)
                img_for_osd = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            else:
                img_for_osd = img
            rotation = 0
            try:
                osd = pytesseract.image_to_osd(img_for_osd, output_type=pytesseract.Output.DICT)
                if osd['orientation_conf'] > osd_confidence_threshold: rotation = osd['rotate']
            except pytesseract.TesseractError as e:
                logger.warning(f"    ⚠️ OSD failed for {class_name} (likely too small). Assuming 0° rotation. Details: {e}")
            corrected_img = img
            if rotation != 0:
                logger.info(f"    🔄 Fine-tuning entity {class_name} orientation by {rotation}°")
                if rotation == 90: corrected_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif rotation == 180: corrected_img = cv2.rotate(img, cv2.ROTATE_180)
                elif rotation == 270: corrected_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            h_corr, w_corr = corrected_img.shape[:2]
            if h_corr > w_corr and 'address' not in class_name:
                logger.info(f"    ↪️ Rotating vertical entity '{class_name}' to horizontal format")
                corrected_img = cv2.rotate(corrected_img, cv2.ROTATE_90_CLOCKWISE)
            pil_img = Image.fromarray(cv2.cvtColor(corrected_img, cv2.COLOR_BGR2GRAY))
            return pil_img
        except Exception as e:
            logger.error(f"    ❌ Unhandled error during entity orientation/preprocessing for '{class_name}': {e}")
            return None

    ### REMOVED: save_results_to_json method is no longer needed ###
            
    def process_images(self, image_arrays: List[np.ndarray], user_id: str, task_id: str, confidence_threshold: float, verbose=True):
        """Main pipeline function to process multiple in-memory images."""
        try:
            ### REMOVED: Session directory creation ###
            if verbose: logger.info(f"🚀 Starting Fully In-Memory Pipeline for task {task_id}")

            # Step 1
            cropped_cards = self.detect_and_crop_cards(image_arrays, confidence_threshold)
            if not cropped_cards.get('front') and not cropped_cards.get('back'):
                return {'error': 'No Aadhaar cards detected.', 'step': 'card_detection', 'front_detected': False, 'back_detected': False}

            # Step 1.5
            corrected_cards = self.correct_card_orientation_opencv(cropped_cards)

            # Initialize results structure
            organized_results = {
                'front': {}, 'back': {},
                'metadata': {
                    'processing_timestamp': datetime.now().isoformat(),
                    ### REMOVED: session_directory key ###
                    'confidence_threshold_used': confidence_threshold
                }
            }

            # Process both front and back cards
            for card_type in ['front', 'back']:
                for i, card_array in enumerate(corrected_cards.get(card_type, [])):
                    card_key = f"{card_type}_card_{i}"
                    organized_results[card_type][card_key] = {'entities': {}}
                    
                    logger.info(f"\n📝 Processing {card_type} card index {i}")
                    # Step 2: Detect entities
                    detections = self.detect_entities_in_card(card_array, confidence_threshold)
                    
                    # Steps 3, 4, 5: Crop, Preprocess, OCR, and Organize
                    for detection in detections:
                        class_name = detection['class_name']
                        logger.info(f"  Processing entity: {class_name}")
                        
                        x1, y1, x2, y2 = detection['bbox']
                        entity_crop_array = card_array[y1:y2, x1:x2]
                        
                        processed_pil_img = self._correct_entity_orientation_and_preprocess(entity_crop_array, class_name)
                        
                        extracted_text = ""
                        if processed_pil_img:
                            try:
                                lang_to_use = self.other_lang_code if class_name and class_name.endswith('_other_lang') else 'eng'
                                text = pytesseract.image_to_string(processed_pil_img, lang=lang_to_use, config='--psm 6')
                                extracted_text = ' '.join(text.split()).strip()
                            except Exception as e:
                                logger.error(f"    ❌ OCR failed for {class_name}: {e}")
                        
                        if class_name not in organized_results[card_type][card_key]['entities']:
                            organized_results[card_type][card_key]['entities'][class_name] = []
                        
                        organized_results[card_type][card_key]['entities'][class_name].append({
                            'confidence': detection['confidence'], 'bbox': detection['bbox'],
                            'extracted_text': extracted_text
                        })

            ### REMOVED: Call to save_results_to_json ###

            if verbose: logger.info("🎉 Pipeline processing completed.")
            
            final_result = {'organized_results': organized_results}
            final_result['front_detected'] = bool(organized_results['front'])
            final_result['back_detected'] = bool(organized_results['back'])
            return final_result

        except ValueError as ve: 
            logger.error(f"🚫 SECURITY ERROR in pipeline: {ve}")
            return {'error': str(ve), 'security_flagged': True, 'step': 'card_detection'}
        except Exception as e:
            logger.error(f"❌ Unhandled error in pipeline: {e}\n{traceback.format_exc()}")
            return {'error': str(e), 'traceback': traceback.format_exc(), 'step': 'unknown'}


# --- FastAPI Application Setup ---

app = FastAPI(
    title="Aadhaar Processing API",
    description="API for synchronously processing Aadhaar cards using YOLO and multi-language OCR.",
    version="2.4.0-fully-in-memory"
)

class Config:
    BASE_DIR = Path(__file__).parent
    MODEL1_PATH = BASE_DIR / os.environ.get("MODEL1_PATH", "models/best_updated.pt")
    MODEL2_PATH = BASE_DIR / os.environ.get("MODEL2_PATH", "models/best.pt")
    ### REMOVED: OUTPUT_DIR is no longer used. ###
    SUMMARY_DATA_DIR = Path(os.environ.get("SUMMARY_DATA_DIR", "/Users/hqpl/Desktop/aadhar_testing/AadharAuth/data"))
    DEFAULT_CONFIDENCE_THRESHOLD = float(os.environ.get("DEFAULT_CONFIDENCE_THRESHOLD", "0.4"))

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
        ### MODIFIED: Removed creation of unused directories ###
        config.SUMMARY_DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        ### MODIFIED: Removed output_base_dir argument ###
        pipeline = ComprehensiveAadhaarPipeline(
            model1_path=str(config.MODEL1_PATH),
            model2_path=str(config.MODEL2_PATH),
            confidence_threshold=config.DEFAULT_CONFIDENCE_THRESHOLD,
            other_lang_code='hin+tel+ben'
        )
        logger.info("✅ Pipeline models loaded successfully.")
    except Exception as e:
        logger.critical(f"❌ Pipeline initialization failed: {e}", exc_info=True)
        sys.exit(1)

async def download_image(session: aiohttp.ClientSession, url: str) -> Optional[bytes]:
    """Downloads image into memory as bytes, not to a file."""
    try:
        async with session.get(str(url), timeout=30) as response:
            response.raise_for_status()
            return await response.read()
    except Exception as e:
        logger.error(f"Download failed for {url}: {e}")
        return None

def extract_main_fields(organized_results: Dict[str, Any]) -> Dict[str, Any]:
    fields = ['aadharNumber', 'dob', 'gender', 'name', 'address', 'pincode', 'state']
    data = {key: "" for key in fields}
    for side in ['front', 'back']:
        for card in organized_results.get(side, {}).values():
            for field in fields:
                if field in card['entities'] and card['entities'][field]:
                    all_texts = [item.get('extracted_text', '') for item in card['entities'][field]]
                    first_valid_text = next((text for text in all_texts if text), '')
                    if first_valid_text:
                        data[field] = first_valid_text
    return data

@app.post("/verify_aadhar", response_class=JSONResponse, tags=["Aadhaar Processing"])
async def verify_aadhaar_sync(request: AadhaarProcessRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Service Unavailable: Pipeline not initialized")

    task_id = hashlib.md5(f"{request.user_id}_{datetime.now().timestamp()}".encode()).hexdigest()
    
    async with aiohttp.ClientSession() as session:
        front_bytes, back_bytes = await asyncio.gather(
            download_image(session, str(request.front_url)),
            download_image(session, str(request.back_url))
        )
        if not all([front_bytes, back_bytes]):
            raise HTTPException(status_code=400, detail="Failed to download one or both Aadhaar images")

    try:
        front_array = cv2.imdecode(np.frombuffer(front_bytes, np.uint8), cv2.IMREAD_COLOR)
        back_array = cv2.imdecode(np.frombuffer(back_bytes, np.uint8), cv2.IMREAD_COLOR)
        if front_array is None or back_array is None:
            raise ValueError("Could not decode one or both images. They might be corrupt or in an unsupported format.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image decoding failed: {e}")

    result = pipeline.process_images([front_array, back_array], request.user_id, task_id, request.confidence_threshold)
    
    if 'error' in result:
        raise HTTPException(status_code=500, detail=result['error'])

    organized = result.get('organized_results', {})
    main_data = extract_main_fields(organized)
    main_data['User ID'] = request.user_id

    if not result.get('front_detected'):
        logger.info(f"❗ Front image for task {task_id} did not contain a detectable card.")
    if not result.get('back_detected'):
        logger.info(f"❗ Back image for task {task_id} did not contain a detectable card.")

    pkl_path = config.SUMMARY_DATA_DIR / "summary.pkl"
    csv_path = config.SUMMARY_DATA_DIR / "summary.csv"
    json_path = config.SUMMARY_DATA_DIR / "summary.json"
    
    aadhar_number = main_data.get('aadharNumber', '').replace(' ', '')
    if pkl_path.exists() and aadhar_number:
        try:
            with open(pkl_path, 'rb') as pf:
                all_data = pickle.load(pf)
            for entry in all_data:
                if entry.get('aadharNumber', '').replace(' ', '') == aadhar_number:
                    minimal_data = {
                        "name": entry.get("name", ""), "aadharNumber": entry.get("aadharNumber", ""),
                        "matched_user_id": entry.get("User ID", "")
                    }
                    logger.warning(f"Duplicate Aadhaar {aadhar_number} found for user {entry.get('User ID')}.")
                    return JSONResponse(content={"status": "Aadhar Data Already Exists", "data": minimal_data})
        except Exception as e:
            logger.error(f"Failed to check PKL for duplicates: {e}")

    with open(json_path, 'w', encoding='utf-8') as jf:
        json.dump(main_data, jf, indent=4, ensure_ascii=False)

    try:
        all_data = []
        if pkl_path.exists():
            with open(pkl_path, 'rb') as pf:
                all_data = pickle.load(pf)
        all_data.append(main_data)
        with open(pkl_path, 'wb') as pf:
            pickle.dump(all_data, pf)
    except Exception as e:
        logger.error(f"Failed to update PKL file: {e}")

    try:
        df = pd.DataFrame([main_data])
        df.to_csv(csv_path, mode='a', header=not csv_path.exists(), index=False)
    except Exception as e:
        logger.error(f"Failed to update CSV file: {e}")
    
    for k, v in main_data.items():
        main_data[k] = "" if v is None else str(v)

    return JSONResponse(content={"status": "saved", "data": main_data})

@app.get("/health", tags=["Monitoring"])
async def health_check():
    """
    Checks the health of the service, including pipeline initialization and device availability.
    """
    if pipeline and hasattr(pipeline, 'device'):
        return JSONResponse(
            status_code=200,
            content={
                "status": "ok",
                "pipeline_status": "initialized",
                "inference_device": pipeline.device,
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
            }
        )
    else:
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "pipeline_status": "not_initialized",
                "detail": "The main processing pipeline is not available.",
            },
        )

if __name__ == "__main__":
    uvicorn.run("gpupipeline:app", host="0.0.0.0", port=8200, reload=True)
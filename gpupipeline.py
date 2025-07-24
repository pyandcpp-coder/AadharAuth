# merged_pipeline_api.py

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


# --- Core Pipeline Logic (from comprehensive_pipeline_api_redis.py, with Redis removed) ---

class ComprehensiveAadhaarPipeline:
    def __init__(self, model1_path, model2_path, output_base_dir="pipeline_output", confidence_threshold=0.10, other_lang_code='hin+tel+ben'):
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
        # The device will be passed during inference calls, not at initialization.
        self.model1 = YOLO(self.model1_path)
        self.model2 = YOLO(self.model2_path)
        
        self.output_base_dir = Path(output_base_dir)
        self.default_confidence_threshold = confidence_threshold
        self.other_lang_code = other_lang_code

        self._check_tesseract()

        logger.info("✅ YOLOv8 models loaded successfully from filesystem.")
        logger.info(f"Model1 classes: {self.model1.names}")
        logger.info(f"Model2 classes: {self.model2.names}")

        self.card_classes = {i: name for i, name in self.model1.names.items()}
        self.session_dir = None

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

    def setup_session_directories(self, user_id: str, task_id: str):
        """Create the required directory structure for a specific processing session"""
        self.session_dir = self.output_base_dir / user_id / task_id
        logger.info(f"Creating session directory: {self.session_dir}")
        self.front_back_dir = self.session_dir / "1_front_back_cards"
        self.corrected_cards_dir = self.session_dir / "1a_corrected_cards"
        self.detected_entities_dir = self.session_dir / "2_detected_entities"
        self.cropped_entities_dir = self.session_dir / "3_cropped_entities"
        self.corrected_entities_dir = self.session_dir / "3a_corrected_entities"
        self.preprocessed_entities_dir = self.session_dir / "4_preprocessed_entities"
        self.image_not_scan_dir = self.session_dir / "ImageNotScan"
        for directory in [
            self.front_back_dir, 
            self.corrected_cards_dir,
            self.detected_entities_dir, 
            self.cropped_entities_dir, 
            self.corrected_entities_dir,
            self.preprocessed_entities_dir,
            self.image_not_scan_dir
        ]:
            directory.mkdir(parents=True, exist_ok=True)

    def detect_and_crop_cards(self, image_paths, confidence_threshold: float):
        """Step 1: Detect Aadhaar front and back cards and crop them"""
        logger.info(f"\n🔍 Step 1: Detecting Aadhaar cards in {len(image_paths)} images (Threshold: {confidence_threshold})")
        cropped_cards = {'front': [], 'back': []}
        for image_path in image_paths:
            logger.info(f"  Processing: {Path(image_path).name}")
            ### MODIFIED: Pass the selected device to the model for inference ###
            results = self.model1(str(image_path), device=self.device)
            img = cv2.imread(str(image_path))
            input_filename = Path(image_path).stem
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
                crop_filename = self.front_back_dir / f"{input_filename}_{class_name}_conf{int(float(box.conf[0])*100)}.jpg"
                cv2.imwrite(str(crop_filename), crop)
                cropped_cards[class_name.replace('aadhar_', '')].append(crop_filename)
                logger.info(f"    ✅ Saved {class_name}: {crop_filename.name}")
            if not detected:
                not_scan_path = self.image_not_scan_dir / Path(image_path).name
                shutil.copy(image_path, not_scan_path)
                logger.warning(f"    ⚠️ No Aadhaar card detected in {Path(image_path).name}. Saved to ImageNotScan.")
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

    def correct_card_orientation_opencv(self, cropped_cards: Dict[str, List[Path]], osd_confidence: float = 1.0) -> Dict[str, List[Path]]:
        """
        Step 1.5: Corrects card orientation using a hybrid OpenCV and Tesseract approach.
        """
        logger.info("\n🔄 Step 1.5: Correcting card orientation using OpenCV contours and Tesseract OSD")
        corrected_card_paths = {'front': [], 'back': []}
        all_cards = cropped_cards.get('front', []) + cropped_cards.get('back', [])

        if not all_cards:
            logger.warning("  No cards to correct orientation for.")
            return corrected_card_paths

        for card_path in all_cards:
            try:
                img = cv2.imread(str(card_path))
                if img is None:
                    logger.warning(f"  ⚠️ Could not read card image {card_path.name}, skipping.")
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
                    if box_w < box_h:
                        angle += 90
                    
                    logger.info(f"  Processing {card_path.name}: Detected OpenCV rotation angle: {angle:.2f}°")
                    rotated_img = self._get_rotated_image(img, angle)

                final_corrected_img = rotated_img
                osd = pytesseract.image_to_osd(rotated_img, config='--psm 0', output_type=pytesseract.Output.DICT)
                rotation_angle = osd.get('rotate', 0)
                confidence = osd.get('orientation_conf', 0.0)

                if rotation_angle == 180 and confidence >= osd_confidence:
                    logger.info(f"    -> Tesseract detected 180° flip (conf: {confidence:.2f}). Applying final correction.")
                    final_corrected_img = cv2.rotate(rotated_img, cv2.ROTATE_180)
                
                card_type = 'front' if 'aadhar_front' in card_path.stem else 'back'
                corrected_filename = self.corrected_cards_dir / f"{card_path.stem}_corrected.jpg"
                cv2.imwrite(str(corrected_filename), final_corrected_img)
                corrected_card_paths[card_type].append(corrected_filename)
                logger.info(f"    ✅ Saved orientation-corrected card: {corrected_filename.name}")

            except Exception as e:
                logger.error(f"  ❌ Error during orientation correction for {card_path.name}: {e}. Using original image as fallback.", exc_info=True)
                card_type = 'front' if 'aadhar_front' in card_path.stem else 'back'
                corrected_filename = self.corrected_cards_dir / f"{card_path.stem}_fallback.jpg"
                shutil.copy(str(card_path), str(corrected_filename))
                corrected_card_paths[card_type].append(corrected_filename)
        return corrected_card_paths

    def detect_entities_in_cards(self, orientation_corrected_cards, confidence_threshold: float):
        """Step 2: Detect entities in orientation-corrected cards."""
        all_card_paths = orientation_corrected_cards.get('front', []) + orientation_corrected_cards.get('back', [])
        logger.info(f"\n🔍 Step 2: Detecting entities in {len(all_card_paths)} orientation-corrected cards (Threshold: {confidence_threshold})")
        all_detections = {}

        for card_path in all_card_paths:
            logger.info(f"  Processing: {card_path.name}")
            ### MODIFIED: Pass the selected device to the model for inference ###
            results = self.model2(str(card_path), device=self.device)
            img = cv2.imread(str(card_path))
            img_with_boxes = img.copy()
            card_detections = []
            
            for box in results[0].boxes:
                if float(box.conf[0]) < confidence_threshold: continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_name = self.entity_classes.get(int(box.cls[0]), "unknown")
                cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
                card_detections.append({'bbox': (x1, y1, x2, y2), 'class_name': class_name, 'confidence': float(box.conf[0])})
                
            detection_filename = self.detected_entities_dir / f"{card_path.stem}_with_entities.jpg"
            cv2.imwrite(str(detection_filename), img_with_boxes)
            logger.info(f"    ✅ Detected {len(card_detections)} entities, saved: {detection_filename.name}")
            all_detections[card_path] = card_detections
        return all_detections

    def crop_entities(self, all_detections):
        """Step 3: Crop individual entities and enrich the detection dictionary"""
        logger.info(f"\n✂️  Step 3: Cropping individual entities")
        for card_path, detections in all_detections.items():
            img = cv2.imread(str(card_path))
            for i, detection in enumerate(detections):
                x1, y1, x2, y2 = detection['bbox']
                crop = img[y1:y2, x1:x2]
                entity_filename = self.cropped_entities_dir / f"{card_path.stem}_{detection['class_name']}_{i}.jpg"
                cv2.imwrite(str(entity_filename), crop)
                detection['cropped_filename'] = str(entity_filename) 
                logger.info(f"    ✅ Saved entity: {entity_filename.name}")
        return all_detections 

    def _correct_entity_orientation_and_preprocess(self, image_path: Path, osd_confidence_threshold: float = 1.0) -> Optional[Image.Image]:
        """Takes a path to a single cropped entity, attempts to correct its orientation, and returns a preprocessed PIL Image."""
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                logger.warning(f"    ⚠️ Could not read entity image {image_path.name}, skipping.")
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
                if osd['orientation_conf'] > osd_confidence_threshold:
                    rotation = osd['rotate']
            except pytesseract.TesseractError as e:
                logger.warning(f"    ⚠️ OSD failed for {image_path.name} (likely too small). Assuming 0° rotation. Details: {e}")
                rotation = 0
            
            corrected_img = img
            if rotation != 0:
                logger.info(f"    🔄 Fine-tuning entity {image_path.name} orientation by {rotation}°")
                if rotation == 90: corrected_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif rotation == 180: corrected_img = cv2.rotate(img, cv2.ROTATE_180)
                elif rotation == 270: corrected_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

            h_corr, w_corr = corrected_img.shape[:2]
            if h_corr > w_corr and 'address' not in image_path.name:
                logger.info(f"    ↪️ Rotating vertical entity {image_path.name} to horizontal format")
                corrected_img = cv2.rotate(corrected_img, cv2.ROTATE_90_CLOCKWISE)

            corrected_path = self.corrected_entities_dir / f"{image_path.stem}_corrected.jpg"
            cv2.imwrite(str(corrected_path), corrected_img)
            pil_img = Image.fromarray(cv2.cvtColor(corrected_img, cv2.COLOR_BGR2GRAY))
            preprocessed_filename = self.preprocessed_entities_dir / (image_path.stem + "_preprocessed.png")
            pil_img.save(str(preprocessed_filename), format='PNG')
            return pil_img
        except Exception as e:
            logger.error(f"    ❌ Unhandled error during entity orientation/preprocessing for {image_path.name}: {e}")
            return None

    def perform_multi_language_ocr(self, all_detections: Dict[Path, List[Dict[str, Any]]]):
        """Step 4: Correct entity orientation and perform OCR on cropped entities."""
        logger.info(f"\n📝 Step 4: Correcting Entity Orientation & Performing Multi-Language OCR")
        ocr_results = {}
        for card_path, detections in all_detections.items():
            for detection in detections:
                entity_path_str = detection.get('cropped_filename')
                class_name = detection.get('class_name')
                if not entity_path_str: continue

                entity_path = Path(entity_path_str)
                logger.info(f"  Processing entity: {entity_path.name} (Class: {class_name})")
                lang_to_use = self.other_lang_code if class_name and class_name.endswith('_other_lang') else 'eng'
                processed_pil_img = self._correct_entity_orientation_and_preprocess(entity_path)

                if processed_pil_img:
                    try:
                        text = pytesseract.image_to_string(processed_pil_img, lang=lang_to_use, config='--psm 6')
                        extracted_text = ' '.join(text.split()).strip()
                        ocr_results[entity_path.name] = extracted_text
                    except Exception as e:
                        logger.error(f"    ❌ OCR failed for {entity_path.name}: {e}")
                        ocr_results[entity_path.name] = None
        return ocr_results

    def organize_results_by_card_type(self, corrected_cards, all_detections, ocr_results, confidence_threshold: float):
        logger.info("\n🗂️  Step 5: Organizing final results")
        organized_results = {
            'front': {}, 'back': {},
            'metadata': {
                'processing_timestamp': datetime.now().isoformat(),
                'session_directory': str(self.session_dir),
                'confidence_threshold_used': confidence_threshold
            }
        }
        for card_path, detections in all_detections.items():
            card_type = 'front' if card_path in corrected_cards.get('front', []) else 'back'
            card_key = card_path.stem
            organized_results[card_type][card_key] = {'entities': {}}
            for detection in detections:
                 entity_name = detection['class_name']
                 if entity_name not in organized_results[card_type][card_key]['entities']:
                      organized_results[card_type][card_key]['entities'][entity_name] = []
                 cropped_filename = detection.get('cropped_filename')
                 extracted_text = ocr_results.get(Path(cropped_filename).name if cropped_filename else None)
                 organized_results[card_type][card_key]['entities'][entity_name].append({
                     'confidence': detection['confidence'], 'bbox': detection['bbox'],
                     'cropped_filename': Path(cropped_filename).name if cropped_filename else None,
                     'extracted_text': extracted_text
                 })
        return organized_results

    def save_results_to_json(self, organized_results: Dict[str, Any]) -> Optional[Path]:
        if not self.session_dir: return None
        json_path = self.session_dir / "complete_aadhaar_results.json"
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(organized_results, f, indent=4, ensure_ascii=False)
            logger.info(f"✅ Successfully saved JSON results to {json_path}")
            return json_path
        except Exception as e:
            logger.error(f"❌ Failed to save JSON file: {e}")
            return None
            
    def process_images(self, image_paths, user_id: str, task_id: str, confidence_threshold: float, verbose=True):
        """Main pipeline function to process multiple images"""
        try:
            self.setup_session_directories(user_id, task_id)
            if verbose: logger.info(f"🚀 Starting Pipeline for task {task_id}")

            cropped_cards = self.detect_and_crop_cards(image_paths, confidence_threshold)
            if not cropped_cards.get('front') and not cropped_cards.get('back'):
                return {'error': 'No Aadhaar cards detected.', 'step': 'card_detection'}
            
            corrected_cards = self.correct_card_orientation_opencv(cropped_cards)
            
            all_detections = self.detect_entities_in_cards(corrected_cards, confidence_threshold)
            self.crop_entities(all_detections)
            ocr_results = self.perform_multi_language_ocr(all_detections)
            organized_results = self.organize_results_by_card_type(corrected_cards, all_detections, ocr_results, confidence_threshold)
            
            self.save_results_to_json(organized_results)

            if verbose: logger.info("🎉 Pipeline processing completed.")
            return {'organized_results': organized_results}

        except ValueError as ve: 
            logger.error(f"🚫 SECURITY ERROR in pipeline: {ve}")
            return {'error': str(ve), 'security_flagged': True, 'step': 'card_detection'}
        except Exception as e:
            logger.error(f"❌ Unhandled error in pipeline: {e}\n{traceback.format_exc()}")
            return {'error': str(e), 'traceback': traceback.format_exc(), 'step': 'unknown'}


# --- FastAPI Application Setup (from comprehensive_pipeline_api_redis_test.py) ---

app = FastAPI(
    title="Aadhaar Processing API",
    description="API for synchronously processing Aadhaar cards using YOLO and multi-language OCR.",
    ### MODIFIED ###
    version="2.2.0-health-cuda"
)

class Config:
    BASE_DIR = Path(__file__).parent
    MODEL1_PATH = BASE_DIR / os.environ.get("MODEL1_PATH", "models/best4.pt")
    MODEL2_PATH = BASE_DIR / os.environ.get("MODEL2_PATH", "models/best.pt")
    DOWNLOAD_DIR = BASE_DIR / Path(os.environ.get("DOWNLOAD_DIR", "downloads"))
    OUTPUT_DIR = BASE_DIR / Path(os.environ.get("OUTPUT_DIR", "pipeline_output"))
    
    IMAGE_NOT_SCAN_DIR = Path(os.environ.get("IMAGE_NOT_SCAN_DIR", "/Users/hqpl/Desktop/aadhar_testing/AadharAuth/ImageNotScan"))
    SUMMARY_DATA_DIR = Path(os.environ.get("SUMMARY_DATA_DIR", "/Users/hqpl/Desktop/aadhar_testing/AadharAuth/data"))

    DEFAULT_CONFIDENCE_THRESHOLD = float(os.environ.get("DEFAULT_CONFIDENCE_THRESHOLD", "0.25"))

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
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        config.IMAGE_NOT_SCAN_DIR.mkdir(parents=True, exist_ok=True)
        config.SUMMARY_DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        pipeline = ComprehensiveAadhaarPipeline(
            model1_path=str(config.MODEL1_PATH),
            model2_path=str(config.MODEL2_PATH),
            output_base_dir=str(config.OUTPUT_DIR),
            confidence_threshold=config.DEFAULT_CONFIDENCE_THRESHOLD,
            other_lang_code='hin+tel+ben'
        )
        logger.info("✅ Pipeline models loaded successfully.")
    except Exception as e:
        logger.critical(f"❌ Pipeline initialization failed: {e}", exc_info=True)
        sys.exit(1)

async def download_image(session: aiohttp.ClientSession, url: str, filepath: Path) -> bool:
    try:
        async with session.get(str(url), timeout=30) as response:
            response.raise_for_status()
            async with aiofiles.open(filepath, 'wb') as f:
                await f.write(await response.read())
            return True
    except Exception as e:
        logger.error(f"Download failed for {url}: {e}")
        return False

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
    user_download_dir = config.DOWNLOAD_DIR / request.user_id / task_id
    user_download_dir.mkdir(parents=True, exist_ok=True)
    front_path = user_download_dir / "front.jpg"
    back_path = user_download_dir / "back.jpg"

    async with aiohttp.ClientSession() as session:
        downloads = await asyncio.gather(
            download_image(session, str(request.front_url), front_path),
            download_image(session, str(request.back_url), back_path)
        )
        if not all(downloads):
            shutil.rmtree(user_download_dir, ignore_errors=True)
            raise HTTPException(status_code=400, detail="Failed to download one or both Aadhaar images")

    result = pipeline.process_images([str(front_path), str(back_path)], request.user_id, task_id, request.confidence_threshold)
    
    if 'error' in result:
        shutil.rmtree(user_download_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=result['error'])

    organized = result.get('organized_results', {})
    main_data = extract_main_fields(organized)
    main_data['User ID'] = request.user_id

    if not organized.get("front"):
        shutil.copy(str(front_path), config.IMAGE_NOT_SCAN_DIR / f"{task_id}_front.jpg")
        logger.info(f"❗ Front image for task {task_id} saved to ImageNotScan")
    if not organized.get("back"):
        shutil.copy(str(back_path), config.IMAGE_NOT_SCAN_DIR / f"{task_id}_back.jpg")
        logger.info(f"❗ Back image for task {task_id} saved to ImageNotScan")

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
                    shutil.rmtree(user_download_dir, ignore_errors=True)
                    minimal_data = {
                        "name": entry.get("name", ""),
                        "aadharNumber": entry.get("aadharNumber", ""),
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
    
    shutil.rmtree(user_download_dir, ignore_errors=True)

    for k, v in main_data.items():
        main_data[k] = "" if v is None else str(v)

    return JSONResponse(content={"status": "saved", "data": main_data})

### NEW: Health check endpoint ###
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
### END NEW ###

if __name__ == "__main__":
    uvicorn.run("gpupipeline:app", host="0.0.0.0", port=8200, reload=True)
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl, Field
import uvicorn
import asyncio
import aiohttp
import aiofiles
from pathlib import Path
import hashlib
import json
import logging
from datetime import datetime
import os
import shutil

from typing import Optional, Dict, Any, List
import traceback
import time
from urllib.parse import urlparse
from fastapi.staticfiles import StaticFiles

from ultralytics import YOLO
import cv2
import numpy as np
import pytesseract
from PIL import Image
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComprehensiveAadhaarPipeline:
    def __init__(self, model1_path, model2_path, output_base_dir="pipeline_output", confidence_threshold=0.10, other_lang_code='hin+tel+ben'):
        """
        Initialize the comprehensive Aadhaar processing pipeline

        Args:
            model1_path (str): Path to the first model (front/back detection)
            model2_path (str): Path to the second model (entity detection)
            output_base_dir (str): Base output directory for saving results
            confidence_threshold (float): Default confidence threshold
            other_lang_code (str): Tesseract language code for secondary language entities (e.g., 'hin' for Hindi).
        """
        self.model1_path = model1_path
        self.model2_path = model2_path
        self.output_base_dir = Path(output_base_dir)
        self.default_confidence_threshold = confidence_threshold
        self.other_lang_code = other_lang_code

        self._check_tesseract()

        logger.info("Loading YOLOv8 models...")
        if not Path(self.model1_path).exists():
            raise FileNotFoundError(f"Model file not found: {self.model1_path}")
        if not Path(self.model2_path).exists():
            raise FileNotFoundError(f"Model file not found: {self.model2_path}")

        self.model1 = YOLO(self.model1_path)
        self.model2 = YOLO(self.model2_path)
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
        """Check if Tesseract is available"""
        try:
            pytesseract.get_tesseract_version()
        except pytesseract.TesseractNotFoundError:
            logger.critical("❌ Tesseract executable not found. Please install Tesseract OCR and ensure it's in your PATH.")
            raise RuntimeError("Tesseract not found")
        except Exception as e:
             logger.critical(f"An error occurred while checking Tesseract: {e}")
             raise RuntimeError(f"Error checking Tesseract: {e}")

    # def setup_session_directories(self, user_id: str, task_id: str):
    #     """Create the required directory structure for a specific processing session"""
    #     self.session_dir = self.output_base_dir / user_id / task_id
    #     logger.info(f"Creating session directory: {self.session_dir}")
    #     self.front_back_dir = self.session_dir / "1_front_back_cards"
    #     self.detected_entities_dir = self.session_dir / "2_detected_entities"
    #     self.cropped_entities_dir = self.session_dir / "3_cropped_entities"
    #     self.preprocessed_entities_dir = self.session_dir / "4_preprocessed_entities"
    #     for directory in [self.front_back_dir, self.detected_entities_dir, self.cropped_entities_dir, self.preprocessed_entities_dir]:
    #         directory.mkdir(parents=True, exist_ok=True)
    
    def setup_session_directories(self, user_id: str, task_id: str):
        """Create the required directory structure for a specific processing session"""
        self.session_dir = self.output_base_dir / user_id / task_id
        logger.info(f"Creating session directory: {self.session_dir}")
        self.front_back_dir = self.session_dir / "1_front_back_cards"
        self.detected_entities_dir = self.session_dir / "2_detected_entities"
        self.cropped_entities_dir = self.session_dir / "3_cropped_entities"
        self.preprocessed_entities_dir = self.session_dir / "4_preprocessed_entities"
        self.image_not_scan_dir = self.session_dir / "ImageNotScan"  
        for directory in [
            self.front_back_dir, 
            self.detected_entities_dir, 
            self.cropped_entities_dir, 
            self.preprocessed_entities_dir,
            self.image_not_scan_dir
        ]:
            directory.mkdir(parents=True, exist_ok=True)


    # def detect_and_crop_cards(self, image_paths, confidence_threshold: float):
    #     """Step 1: Detect Aadhaar front and back cards and crop them"""
    #     logger.info(f"\n🔍 Step 1: Detecting Aadhaar cards in {len(image_paths)} images (Threshold: {confidence_threshold})")
    #     cropped_cards = {'front': [], 'back': []}
    #     for image_path in image_paths:
    #         logger.info(f"  Processing: {Path(image_path).name}")
    #         results = self.model1(str(image_path))
    #         img = cv2.imread(str(image_path))
    #         input_filename = Path(image_path).stem
    #         for box in results[0].boxes:
    #             if float(box.conf[0]) < confidence_threshold: continue
    #             x1, y1, x2, y2 = map(int, box.xyxy[0])
    #             class_name = self.card_classes.get(int(box.cls[0]), "unknown")
    #             if class_name == 'print_aadhar':
    #                 raise ValueError("Print Aadhaar detected - processing stopped for security reasons")
    #             if class_name not in ['aadhar_front', 'aadhar_back']: continue
    #             crop = img[y1:y2, x1:x2]
    #             crop_filename = self.front_back_dir / f"{input_filename}_{class_name}_conf{int(float(box.conf[0])*100)}.jpg"
    #             cv2.imwrite(str(crop_filename), crop)
    #             cropped_cards[class_name.replace('aadhar_', '')].append(crop_filename)
    #             logger.info(f"    ✅ Saved {class_name}: {crop_filename.name}")
    #     return cropped_cards
    
    def detect_and_crop_cards(self, image_paths, confidence_threshold: float):
        """Step 1: Detect Aadhaar front and back cards and crop them"""
        logger.info(f"\n🔍 Step 1: Detecting Aadhaar cards in {len(image_paths)} images (Threshold: {confidence_threshold})")
        cropped_cards = {'front': [], 'back': []}

        for image_path in image_paths:
            logger.info(f"  Processing: {Path(image_path).name}")
            results = self.model1(str(image_path))
            img = cv2.imread(str(image_path))
            input_filename = Path(image_path).stem

            detected = False  # flag to track detection
            for box in results[0].boxes:
                if float(box.conf[0]) < confidence_threshold:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_name = self.card_classes.get(int(box.cls[0]), "unknown")

                if class_name == 'print_aadhar':
                    raise ValueError("Print Aadhaar detected - processing stopped for security reasons")

                if class_name not in ['aadhar_front', 'aadhar_back']:
                    continue

                detected = True
                crop = img[y1:y2, x1:x2]
                crop_filename = self.front_back_dir / f"{input_filename}_{class_name}_conf{int(float(box.conf[0])*100)}.jpg"
                cv2.imwrite(str(crop_filename), crop)
                cropped_cards[class_name.replace('aadhar_', '')].append(crop_filename)
                logger.info(f"    ✅ Saved {class_name}: {crop_filename.name}")

            if not detected:
                # Move the image to ImageNotScan in the ROOT directory
                root_not_scan_dir = Path(__file__).parent / "ImageNotScan"
                root_not_scan_dir.mkdir(parents=True, exist_ok=True)
                not_scan_path = root_not_scan_dir / Path(image_path).name
                shutil.copy(image_path, not_scan_path)
                logger.warning(f"    ⚠️ No Aadhaar card detected in {Path(image_path).name}. Saved to root ImageNotScan.")

        return cropped_cards

    def detect_entities_in_cards(self, cropped_cards, confidence_threshold: float):
        """Step 2: Detect entities in cropped cards"""
        all_card_paths = cropped_cards.get('front', []) + cropped_cards.get('back', [])
        logger.info(f"\n🔍 Step 2: Detecting entities in {len(all_card_paths)} cards (Threshold: {confidence_threshold})")
        all_detections = {}
        for card_path in all_card_paths:
            logger.info(f"  Processing: {card_path.name}")
            results = self.model2(str(card_path))
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

    def preprocess_image_for_ocr(self, image_path):
        """Preprocess image for better OCR accuracy"""
        try:
            img = Image.open(image_path).convert('L')
            return img
        except Exception as e:
            logger.error(f"    ❌ Error preprocessing image {Path(image_path).name}: {e}")
            return None

    def perform_multi_language_ocr(self, all_detections: Dict[Path, List[Dict[str, Any]]]):
        """
        Step 4: Perform OCR on cropped entities using different languages based on class name.
        """
        logger.info(f"\n📝 Step 4: Performing Multi-Language OCR")
        ocr_results = {}
        entity_count = 0

        for card_path, detections in all_detections.items():
            for detection in detections:
                entity_count += 1
                entity_path_str = detection.get('cropped_filename')
                class_name = detection.get('class_name')

                if not entity_path_str:
                    logger.warning(f"    ⚠️ Skipping OCR for a detection on {card_path.name} as it was not cropped.")
                    continue

                entity_path = Path(entity_path_str)
                logger.info(f"  Processing: {entity_path.name} (Class: {class_name})")

                if class_name and class_name.endswith('_other_lang'):
                    lang_to_use = self.other_lang_code
                    logger.info(f"    -> Using '{lang_to_use}' OCR for class '{class_name}'")
                else:
                    lang_to_use = 'eng'

                processed_img = self.preprocess_image_for_ocr(entity_path)
                if processed_img:
                    try:

                        preprocessed_filename = self.preprocessed_entities_dir / (entity_path.stem + "_preprocessed.png")
                        processed_img.save(str(preprocessed_filename), format='PNG')
                        
                        text = pytesseract.image_to_string(processed_img, lang=lang_to_use, config='--psm 6')
                        extracted_text = ' '.join(text.split()).strip()
                        
                        display_snippet = (extracted_text[:50] + '...') if len(extracted_text) > 50 else extracted_text
                        logger.info(f"    ✅ Extracted: \"{display_snippet}\"")
                        ocr_results[entity_path.name] = extracted_text
                    except Exception as e:
                        logger.error(f"    ❌ OCR failed for {entity_path.name}: {e}")
                        ocr_results[entity_path.name] = None
                else:
                    logger.warning(f"    ⚠️ Skipping OCR due to preprocessing failure for {entity_path.name}")
                    ocr_results[entity_path.name] = None

        successful_ocr = sum(1 for text in ocr_results.values() if text is not None)
        logger.info(f"  📊 OCR completed: {successful_ocr}/{entity_count} successful")
        return ocr_results

    def organize_results_by_card_type(self, cropped_cards, all_detections, ocr_results, confidence_threshold: float):
        logger.info("\nOrganizing results...")
        organized_results = {
            'front': {}, 'back': {},
            'metadata': {
                'processing_timestamp': datetime.now().isoformat(),
                'session_directory': str(self.session_dir),
                'confidence_threshold_used': confidence_threshold
            }
        }
        for card_path, detections in all_detections.items():
            card_type = 'front' if card_path in cropped_cards.get('front', []) else 'back'
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
        logger.info(f"Saving final results to: {json_path}")
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

            all_detections = self.detect_entities_in_cards(cropped_cards, confidence_threshold)
            self.crop_entities(all_detections)

            ocr_results = self.perform_multi_language_ocr(all_detections)

            organized_results = self.organize_results_by_card_type(cropped_cards, all_detections, ocr_results, confidence_threshold)
            
            if verbose: logger.info("🎉 Pipeline processing completed.")
            return {'organized_results': organized_results}

        except ValueError as ve: 
            logger.error(f"🚫 SECURITY ERROR in pipeline: {ve}")
            return {'error': str(ve), 'security_flagged': True, 'step': 'card_detection'}
        except Exception as e:
            logger.error(f"❌ Unhandled error in pipeline: {e}\n{traceback.format_exc()}")
            return {'error': str(e), 'traceback': traceback.format_exc(), 'step': 'unknown'}

app = FastAPI(
    title="Aadhaar Processing API",
    description="API for processing Aadhaar cards using YOLO models and multi-language OCR",
    version="1.1.0"
)

class Config:
    BASE_DIR = Path(__file__).parent
    MODEL1_PATH = BASE_DIR / os.environ.get("MODEL1_PATH", "models/best.pt")
    MODEL2_PATH = BASE_DIR / os.environ.get("MODEL2_PATH", "models/best4.pt")
    DOWNLOAD_DIR = BASE_DIR / Path(os.environ.get("DOWNLOAD_DIR", "downloads"))
    OUTPUT_DIR = BASE_DIR / Path(os.environ.get("OUTPUT_DIR", "pipeline_output"))
    _max_file_size_raw = os.environ.get("MAX_FILE_SIZE", str(10 * 1024 * 1024))
    try:
        MAX_FILE_SIZE = int(''.join(filter(str.isdigit, _max_file_size_raw)))
    except Exception:
        MAX_FILE_SIZE = 10 * 1024 * 1024
    ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png'}
    _default_conf_raw = os.environ.get("DEFAULT_CONFIDENCE_THRESHOLD", "0.4")
    try:
        import re
        match = re.search(r"[0-9]*\.?[0-9]+", _default_conf_raw)
        DEFAULT_CONFIDENCE_THRESHOLD = float(match.group(0)) if match else 0.4
    except Exception:
        DEFAULT_CONFIDENCE_THRESHOLD = 0.4

config = Config()
pipeline: Optional[ComprehensiveAadhaarPipeline] = None
processing_tasks: Dict[str, Dict[str, Any]] = {}

class AadhaarProcessRequest(BaseModel):
    user_id: str = "default_user"
    front_url: HttpUrl
    back_url: HttpUrl
    confidence_threshold: float = Field(0.4, ge=0.0, le=1.0)

class ProcessingStatus(BaseModel):
    status: str
    message: str
    task_id: str
    user_id: str
    status_url: Optional[str] = None 
    session_dir: Optional[str] = None
    results: Optional[Dict[str, Any]] = None
    json_results_path: Optional[str] = None
    json_results_url: Optional[str] = None
    processing_time: Optional[float] = None
    error: Optional[str] = None
    security_flagged: Optional[bool] = None
    failed_step: Optional[str] = None

@app.on_event("startup")
async def startup_event():
    global pipeline
    try:
        config.DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("Attempting to load models...")
        pipeline = ComprehensiveAadhaarPipeline(
            model1_path=str(config.MODEL1_PATH),
            model2_path=str(config.MODEL2_PATH),
            output_base_dir=str(config.OUTPUT_DIR),
            confidence_threshold=config.DEFAULT_CONFIDENCE_THRESHOLD,
            other_lang_code='hin+tel+ben' 
        )
        logger.info("✅ Pipeline models loaded successfully.")
    except Exception as e:
        logger.critical(f"❌ Failed to initialize pipeline on startup: {e}. Exiting.", exc_info=True)
        sys.exit(1)

async def download_image(session: aiohttp.ClientSession, url: str, filepath: Path) -> bool:
    try:
        async with session.get(str(url), timeout=30) as response:
            response.raise_for_status()
            async with aiofiles.open(filepath, 'wb') as f: await f.write(await response.read())
            return True
    except Exception as e:
        logger.error(f"Error downloading {url}: {e}")
        return False

def generate_task_id(user_id: str, front_url: str, back_url: str) -> str:
    content = f"{user_id}_{front_url}_{back_url}_{datetime.now().timestamp()}"
    return hashlib.md5(content.encode()).hexdigest()

async def process_aadhaar_task(task_id: str, user_id: str, front_url: str, back_url: str, confidence_threshold: float):
    global processing_tasks, pipeline
    start_time = time.time()
    user_download_dir = config.DOWNLOAD_DIR / user_id / task_id
    try:
        processing_tasks[task_id]['message'] = 'Downloading images...'
        user_download_dir.mkdir(parents=True, exist_ok=True)
        front_path = user_download_dir / "front.jpg"
        back_path = user_download_dir / "back.jpg"

        async with aiohttp.ClientSession() as session:
            downloads = await asyncio.gather(
                download_image(session, str(front_url), front_path),
                download_image(session, str(back_url), back_path)
            )
            if not all(downloads):
                raise RuntimeError("Failed to download one or both images.")

        processing_tasks[task_id]['message'] = 'Processing Aadhaar cards...'
        image_paths = [str(front_path), str(back_path)]
        results = pipeline.process_images(image_paths, user_id, task_id, confidence_threshold)

        if 'error' in results:
            raise ValueError(f"Pipeline failed at step '{results.get('step')}': {results['error']}")

        final_json_path = pipeline.save_results_to_json(results['organized_results'])
        if not final_json_path:
            raise IOError("Failed to save final JSON results.")

        processing_time_sec = time.time() - start_time
        processing_tasks[task_id].update({
            'status': 'completed', 'message': 'Processing completed successfully',
            'results': results['organized_results'],
            'json_results_path': str(final_json_path),
            'json_results_url': f"/results/{user_id}/{task_id}/complete_aadhaar_results.json",
            'session_dir': str(pipeline.session_dir),
            'processing_time': processing_time_sec
        })
        logger.info(f"Task {task_id}: Completed successfully in {processing_time_sec:.2f}s.")
    except Exception as e:
        logger.error(f"Task {task_id}: Error during execution: {e}", exc_info=True)
        processing_tasks[task_id].update({
            'status': 'error', 'message': f'Task failed: {e}', 'error': str(e),
            'processing_time': time.time() - start_time
        })
    finally:
        if user_download_dir and user_download_dir.exists():
            shutil.rmtree(user_download_dir)


@app.post("/process", status_code=202, response_model=ProcessingStatus)
async def process_aadhaar(request: AadhaarProcessRequest, background_tasks: BackgroundTasks):
    """
    Submit Aadhaar image URLs for background processing.
    Returns a task ID and a status URL to poll for results.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Service Unavailable: AI pipeline not initialized.")
    
    task_id = generate_task_id(request.user_id, str(request.front_url), str(request.back_url))
    
    initial_status = ProcessingStatus(
        status='pending',
        message='Task received and queued for processing.',
        task_id=task_id,
        user_id=request.user_id,
        status_url=f"/status/{task_id}"  
    )
    
    processing_tasks[task_id] = initial_status.model_dump()

    background_tasks.add_task(
        process_aadhaar_task,
        task_id,
        request.user_id,
        str(request.front_url),
        str(request.back_url),
        request.confidence_threshold
    )
    logger.info(f"Task {task_id} queued for user {request.user_id}")
    
    return initial_status

@app.get("/status/{task_id}", response_model=ProcessingStatus)
async def get_processing_status(task_id: str):
    """Check the status of a processing task."""
    task_info = processing_tasks.get(task_id)
    if not task_info:
        raise HTTPException(status_code=404, detail=f"Task ID '{task_id}' not found.")
    

    if not task_info.get('status_url'):
        task_info['status_url'] = f"/status/{task_id}"
        
    return ProcessingStatus(**task_info)

try:
    app.mount("/results", StaticFiles(directory=config.OUTPUT_DIR), name="results")
    logger.info(f"Mounted static files from '{config.OUTPUT_DIR}' at '/results'")
except Exception as e:
    logger.warning(f"Could not mount static files directory '{config.OUTPUT_DIR}': {e}.")


if __name__ == "__main__":
    uvicorn.run("comprehensive_pipeline_api:app", host="0.0.0.0", port=8000, reload=True)
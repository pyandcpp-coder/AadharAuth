# import os
# import sys
# import io
# import json
# import traceback
# import cv2
# import numpy as np
# import requests
# from PIL import Image
# import pytesseract
# from ultralytics import YOLO
# from fastapi import FastAPI, HTTPException, status
# from fastapi.responses import JSONResponse
# from pydantic import BaseModel
# from pathlib import Path
# from datetime import datetime
# import shutil

# # --- Configuration ---
# # Set the path to the Tesseract executable if it's not in your system's PATH
# # Example for macOS:
# # pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract'
# # Example for Windows:
# # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# # Example for Linux (often in PATH, but if not):
# # pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract' # or where your install places it

# # Uncomment and set the path if needed based on your Tesseract installation
# # pytesseract.pytesseract.tesseract_cmd = r'YOUR_TESSERACT_PATH_HERE'

# # --- Basic Tesseract Check ---
# try:
#     pytesseract.get_tesseract_version()
#     print("✅ Tesseract is installed and accessible.")
# except pytesseract.TesseractNotFoundError:
#     print("❌ Tesseract executable not found.", file=sys.stderr)
#     print("Please install Tesseract OCR and ensure it's in your PATH or set pytesseract.pytesseract.tesseract_cmd.", file=sys.stderr)
#     # Depending on your needs, you might want to exit here if OCR is critical.
#     # sys.exit(1)
#     # For now, we'll allow the app to start but OCR functions will fail.

# # --- Global Variables ---
# # Define your model paths here
# # Replace these with the actual paths on your system
# MODEL1_PATH = "/Users/yrevash/Downloads/aadhar_models/aadhar_model_1/best-4.pt" # Card detection model
# MODEL2_PATH = "/Users/yrevash/Downloads/aadhar_models/aadhar_model_2/best-5.pt" # Entity detection model

# OUTPUT_BASE_DIR = Path("api_pipeline_output")
# CONFIDENCE_THRESHOLD = 0.4 # Minimum confidence for detections

# # Load YOLOv8 models globally when the app starts
# # This avoids reloading models on every request
# try:
#     print("Loading YOLOv8 models...")
#     MODEL1 = YOLO(MODEL1_PATH) # Aadhaar front/back detection
#     MODEL2 = YOLO(MODEL2_PATH) # Entity detection
#     print("Models loaded successfully!")
# except Exception as e:
#     print(f"❌ Error loading models: {e}", file=sys.stderr)
#     MODEL1 = None
#     MODEL2 = None


# # Define class names based on your actual model classes
# CARD_CLASSES = {0: 'aadhar_front', 1: 'aadhar_back', 2: 'print_aadhar'} # Model1 classes
# ENTITY_CLASSES = {
#     0: 'aadharNumber', 1: 'address', 2: 'address_other_lang', 3: 'city',
#     4: 'dob', 5: 'gender', 6: 'gender_other_lang', 7: 'mobile_no',
#     8: 'name', 9: 'name_otherlang', 10: 'pincode', 11: 'state'
# } # Model2 classes

# # --- FastAPI Setup ---
# app = FastAPI(
#     title="Aadhaar Authentication Pipeline API",
#     description="API to process Aadhaar front and back images, detect entities, and extract text via OCR."
# )

# # --- Pydantic Models ---
# class AadhaarProcessRequest(BaseModel):
#     user_id: str = "default_user" # Optional user identifier
#     front_url: str # URL to the front side image
#     back_url: str # URL to the back side image

# class ExtractedEntity(BaseModel):
#     class_name: str
#     text: str | None # Text will be None if OCR fails

# class AadhaarProcessingResponse(BaseModel):
#     session_id: str
#     status: str # "success", "failed", "print_aadhar_detected"
#     message: str
#     front_entities: list[ExtractedEntity] | None = None
#     back_entities: list[ExtractedEntity] | None = None
#     output_dir: str # Path to the session output directory

# # --- Helper Functions ---

# def download_image(url: str) -> Image.Image:
#     """Downloads an image from a URL and returns a PIL Image."""
#     try:
#         response = requests.get(url, stream=True, timeout=15)
#         response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
#         img = Image.open(io.BytesIO(response.content)).convert("RGB") # Ensure RGB format
#         return img
#     except requests.exceptions.RequestException as e:
#         print(f"Error downloading image from {url}: {e}", file=sys.stderr)
#         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Could not download image from URL: {url}. Error: {e}")
#     except Exception as e:
#         print(f"Error processing image after download from {url}: {e}", file=sys.stderr)
#         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error processing image from URL: {url}. Error: {e}")

# def pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
#     """Converts a PIL Image to a OpenCV image (numpy array)."""
#     # OpenCV uses BGR format, PIL uses RGB
#     cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
#     return cv2_img

# def cv2_to_pil(cv2_img: np.ndarray) -> Image.Image:
#     """Converts a OpenCV image (numpy array) to a PIL Image."""
#     # OpenCV uses BGR format, PIL uses RGB
#     pil_img = Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))
#     return pil_img

# def preprocess_image_for_ocr(pil_img: Image.Image) -> Image.Image:
#     """
#     Applies basic preprocessing steps to a PIL Image for OCR.

#     Args:
#         pil_img (PIL.Image.Image): The input PIL image.

#     Returns:
#         PIL.Image.Image: The preprocessed PIL Image object.
#     """
#     # 1. Convert to grayscale: Often helps OCR accuracy.
#     img = pil_img.convert('L')

#     # 2. Resize: Large images can be slow, small text might need upscaling.
#     max_side = 1200
#     w, h = img.size
#     if max(w, h) > max_side:
#         scale = max_side / float(max(w, h))
#         new_size = (int(w * scale), int(h * scale))
#         img = img.resize(new_size, Image.Resampling.LANCZOS) # LANCZOS is good for text

#     # 3. (Optional) Binarization: Convert to strict black and white.
#     # img = img.point(lambda x: 0 if x < 128 else 255, '1') # Simple threshold at 128

#     return img

# def run_ocr_on_image(pil_img: Image.Image) -> str | None:
#     """
#     Performs OCR on a preprocessed PIL image.

#     Args:
#         pil_img (PIL.Image.Image): The input PIL image (preferably preprocessed).

#     Returns:
#         str | None: The extracted text, or None if OCR fails.
#     """
#     try:
#         # Perform OCR using pytesseract
#         # config='--psm 6' suggests assuming a single uniform block of text
#         text = pytesseract.image_to_string(pil_img, lang='eng', config='--psm 6')

#         # Clean up extracted text (remove leading/trailing whitespace)
#         cleaned_text = text.strip()
#         return cleaned_text
#     except pytesseract.TesseractError as te:
#          print(f"  ❌ Tesseract Error during OCR: {te}", file=sys.stderr)
#          return None
#     except Exception as e:
#         print(f"  ❌ Unexpected error during OCR: {e}", file=sys.stderr)
#         # traceback.print_exc() # Uncomment for detailed exception stack
#         return None

# # --- Core Pipeline Logic ---

# def process_card_image(
#     pil_img: Image.Image,
#     card_side: str, # "front" or "back"
#     session_dir: Path
# ) -> list[ExtractedEntity]:
#     """
#     Processes a single card image (front or back): verifies card type,
#     detects entities, crops entities, runs OCR, and saves intermediate steps.

#     Args:
#         pil_img (PIL.Image.Image): The input card image.
#         card_side (str): The expected side of the card ("front" or "back").
#         session_dir (Path): The base directory for this processing session.

#     Returns:
#         list[ExtractedEntity]: A list of detected entities with extracted text.
#     """
#     if MODEL1 is None or MODEL2 is None:
#          raise RuntimeError("YOLO models failed to load.")

#     card_name = f"aadhar_{card_side}"
#     print(f"\n--- Processing {card_name} ---")

#     # --- Directories for this side ---
#     front_back_dir = session_dir / "1_front_back_cards"
#     detected_entities_dir = session_dir / "2_detected_entities"
#     cropped_entities_dir = session_dir / "3_cropped_entities"
#     preprocessed_entities_dir = session_dir / "4_preprocessed_entities"

#     # Ensure directories exist (should be created by the main endpoint but double check)
#     for d in [front_back_dir, detected_entities_dir, cropped_entities_dir, preprocessed_entities_dir]:
#         d.mkdir(parents=True, exist_ok=True)

#     # Convert PIL to CV2 for model inference and drawing
#     cv2_img_original = pil_to_cv2(pil_img)
#     # Use a unique filename for this card within the session
#     img_stem = f"{card_side}_input" # Will be named like "front_input_aadhar_front_confXX.jpg" later

#     # --- Step 1: Verify Card Type and Crop (using Model 1) ---
#     print(f"  🔍 Step 1: Verifying and cropping {card_side} card...")
#     model1_results = MODEL1(cv2_img_original, verbose=False, conf=CONFIDENCE_THRESHOLD)
#     model1_result = model1_results[0]
#     card_crop = None
#     card_detection_found = False

#     for i, box in enumerate(model1_result.boxes):
#         conf = float(box.conf[0])
#         cls_id = int(box.cls[0])
#         class_name = CARD_CLASSES.get(cls_id, f"class_{cls_id}")

#         # Check for Print Aadhaar immediately
#         if class_name == 'print_aadhar' and conf >= CONFIDENCE_THRESHOLD:
#             error_msg = f"❌ PRINT AADHAAR DETECTED on {card_side} side (confidence: {conf:.2f})"
#             print(error_msg)
#             raise ValueError("Print Aadhaar detected - processing stopped for security reasons") # Caught by API endpoint

#         # Find the detection for the expected card side
#         if class_name == card_name and conf >= CONFIDENCE_THRESHOLD:
#              x1, y1, x2, y2 = map(int, box.xyxy[0])
#              card_crop = cv2_img_original[y1:y2, x1:x2].copy() # Use copy to avoid view issues
#              card_detection_found = True
#              print(f"    ✅ Detected {card_name} (confidence: {conf:.2f})")

#              # Save the cropped card
#              cropped_card_filename = front_back_dir / f"{img_stem}_{card_name}_conf{int(conf*100)}.jpg"
#              cv2.imwrite(str(cropped_card_filename), card_crop)
#              print(f"    📸 Saved cropped card: {cropped_card_filename.name}")

#              # If multiple detections, just take the first one for the expected type (assuming one main card)
#              # A more robust system might handle multiple or ambiguous detections differently.
#              break

#     if not card_detection_found or card_crop is None:
#         error_msg = f"❌ No strong detection for {card_name} found in the provided image."
#         print(error_msg)
#         raise ValueError(error_msg) # Caught by API endpoint

#     # --- Step 2: Detect Entities in the Cropped Card (using Model 2) ---
#     print(f"  🔍 Step 2: Detecting entities in cropped {card_name}...")
#     model2_results = MODEL2(card_crop, verbose=False, conf=CONFIDENCE_THRESHOLD)
#     model2_result = model2_results[0]

#     # Draw bounding boxes on the cropped card image for visualization
#     card_with_entities = card_crop.copy()
#     entity_detections = [] # Store detected entities info

#     for i, box in enumerate(model2_result.boxes):
#         conf = float(box.conf[0])
#         if conf < CONFIDENCE_THRESHOLD:
#             continue

#         x1, y1, x2, y2 = map(int, box.xyxy[0])
#         cls_id = int(box.cls[0])
#         class_name = ENTITY_CLASSES.get(cls_id, f"entity_{cls_id}")

#         # Draw bounding box
#         color = (0, 255, 0) # Green
#         cv2.rectangle(card_with_entities, (x1, y1), (x2, y2), color, 2)
#         # Add text label
#         label = f"{class_name}: {conf:.2f}"
#         (w, h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
#         cv2.rectangle(card_with_entities, (x1, y1 - h - baseline), (x1 + w, y1), color, -1)
#         cv2.putText(card_with_entities, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

#         # Store detection for later cropping and OCR
#         entity_detections.append({
#             'bbox': (x1, y1, x2, y2),
#             'class_name': class_name,
#             'confidence': conf
#         })

#     # Save the cropped card image with entity bounding boxes drawn
#     detected_entities_filename = detected_entities_dir / f"{img_stem}_{card_name}_with_entities.jpg"
#     cv2.imwrite(str(detected_entities_filename), card_with_entities)
#     print(f"    📸 Saved image with entity boxes: {detected_entities_filename.name}")
#     print(f"    📊 Detected {len(entity_detections)} entities.")

#     # --- Step 3 & 4: Crop Entities, Preprocess, and Run OCR ---
#     print(f"  ✂️  Step 3/4: Cropping entities, preprocessing, and running OCR...")
#     extracted_entities_list: list[ExtractedEntity] = []

#     for i, detection in enumerate(entity_detections):
#         x1, y1, x2, y2 = detection['bbox']
#         class_name = detection['class_name']
#         conf = detection['confidence']

#         # Crop entity from the card_crop (which is already CV2 format)
#         entity_crop_cv2 = card_crop[y1:y2, x1:x2].copy() # Use copy

#         # Save cropped entity image
#         cropped_entity_filename_stem = f"{img_stem}_{card_name}_{class_name}_{i}_conf{int(conf*100)}"
#         cropped_entity_save_path = cropped_entities_dir / f"{cropped_entity_filename_stem}.jpg"
#         cv2.imwrite(str(cropped_entity_save_path), entity_crop_cv2)
#         # print(f"      📸 Saved cropped entity: {cropped_entity_save_path.name}") # Too verbose maybe

#         # Convert to PIL for preprocessing and OCR
#         entity_crop_pil = cv2_to_pil(entity_crop_cv2)

#         # Preprocess for OCR
#         preprocessed_entity_pil = preprocess_image_for_ocr(entity_crop_pil)

#         # Save preprocessed image
#         preprocessed_entity_save_path = preprocessed_entities_dir / f"{cropped_entity_filename_stem}_preprocessed.png"
#         preprocessed_entity_pil.save(str(preprocessed_entity_save_path))
#         # print(f"      📸 Saved preprocessed entity: {preprocessed_entity_save_path.name}") # Too verbose maybe

#         # Run OCR
#         extracted_text = run_ocr_on_image(preprocessed_entity_pil)
#         print(f"      OCR on {class_name}: \"{extracted_text.replace('\n', ' ').replace('\r', ' ').strip()[:50]}...\"" if extracted_text else f"      OCR on {class_name}: [Failed]")

#         extracted_entities_list.append(ExtractedEntity(class_name=class_name, text=extracted_text))

#     print(f"--- Finished processing {card_name} ---")
#     return extracted_entities_list

# # --- API Endpoint ---

# @app.post("/process_aadhaar", response_model=AadhaarProcessingResponse)
# async def process_aadhaar_endpoint(request: AadhaarProcessRequest):
#     """
#     Processes Aadhaar front and back images from URLs.

#     Downloads images, detects entities, crops entities, runs OCR,
#     saves all intermediate steps, and returns extracted text.
#     """
#     print(f"\n>>> Received request for user: {request.user_id}")

#     # Create a timestamped session directory for this request
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     session_dir = OUTPUT_BASE_DIR / f"session_{timestamp}"

#     try:
#         # Create base output directories
#         for d in ["1_front_back_cards", "2_detected_entities", "3_cropped_entities", "4_preprocessed_entities"]:
#             (session_dir / d).mkdir(parents=True, exist_ok=True)
#         print(f"📁 Created session directory: {session_dir}")

#         # Check if models were loaded successfully
#         if MODEL1 is None or MODEL2 is None:
#              raise RuntimeError("YOLO models failed to load at startup.")

#         # --- Download Images ---
#         print("⬇️ Downloading images...")
#         front_img_pil = download_image(request.front_url)
#         back_img_pil = download_image(request.back_url)
#         print("✅ Images downloaded.")

#         # --- Process Front Image ---
#         front_entities = process_card_image(front_img_pil, "front", session_dir)

#         # --- Process Back Image ---
#         back_entities = process_card_image(back_img_pil, "back", session_dir)

#         # --- Combine Results and Save JSON ---
#         final_results = {
#             "user_id": request.user_id,
#             "front_url": request.front_url,
#             "back_url": request.back_url,
#             "session_id": f"session_{timestamp}",
#             "front_entities": [e.dict() for e in front_entities], # Convert Pydantic models to dicts
#             "back_entities": [e.dict() for e in back_entities],
#         }

#         results_json_path = session_dir / "ocr_results.json"
#         with open(results_json_path, "w", encoding="utf-8") as f:
#             json.dump(final_results, f, indent=4)
#         print(f"\n✅ Final results saved to JSON: {results_json_path.name}")


#         # --- Return Response ---
#         response_data = AadhaarProcessingResponse(
#             session_id=f"session_{timestamp}",
#             status="success",
#             message="Aadhaar processing completed successfully.",
#             front_entities=front_entities,
#             back_entities=back_entities,
#             output_dir=str(session_dir)
#         )
#         print("<<< Request processed successfully.")
#         return response_data

#     except ValueError as ve:
#         # Specific handling for our raised ValueErrors (like Print Aadhaar)
#         print(f"🚫 Processing Error: {ve}", file=sys.stderr)
#         # Clean up the session directory if an error occurred early? Or keep for debugging?
#         # Keeping for debugging might be useful.

#         # Determine status based on error message
#         status_msg = "failed"
#         if "Print Aadhaar detected" in str(ve):
#              status_msg = "print_aadhar_detected"

#         return AadhaarProcessingResponse(
#             session_id=f"session_{timestamp}",
#             status=status_msg,
#             message=f"Processing failed: {ve}",
#             output_dir=str(session_dir)
#         )

#     except Exception as e:
#         # Catch any other unexpected errors
#         print(f"❌ An unexpected error occurred: {e}", file=sys.stderr)
#         traceback.print_exc() # Print detailed error for debugging

#         # Clean up the session directory? Or keep?

#         return AadhaarProcessingResponse(
#             session_id=f"session_{timestamp}",
#             status="failed",
#             message=f"An internal error occurred: {e}",
#             output_dir=str(session_dir)
#         )

# # --- Health Check Endpoint ---
# @app.get("/health")
# async def health_check():
#     """Basic health check to verify API is running and models are loaded."""
#     return {
#         "status": "healthy",
#         "models_loaded": MODEL1 is not None and MODEL2 is not None,
#         "tesseract_accessible": True if pytesseract.get_tesseract_version() else False # This might raise TesseractNotFoundError
#     }


# # --- How to Run ---
# # Save this file as something like `main_api.py`
# # Run from your terminal using uvicorn:
# # uvicorn main_api:app --reload
# # (The --reload flag is useful for development, removes it for production)



import os
import sys
import io
import json
import traceback
import cv2
import numpy as np
import requests
from PIL import Image
import pytesseract
from ultralytics import YOLO
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pathlib import Path
from datetime import datetime
import shutil

try:
    pytesseract.get_tesseract_version()
    print("✅ Tesseract is installed and accessible.")
except pytesseract.TesseractNotFoundError:
    print("❌ Tesseract executable not found.", file=sys.stderr)
    print("Please install Tesseract OCR and ensure it's in your PATH or set pytesseract.pytesseract.tesseract_cmd.", file=sys.stderr)

MODEL1_PATH = "/Users/yrevash/Downloads/yolov8_upload_dir/weights/best.pt" # Card detection model
MODEL2_PATH = "/Users/yrevash/Downloads/aadhar_models/aadhar_model_2/best-5.pt" 

OUTPUT_BASE_DIR = Path("api_pipeline_output")
CONFIDENCE_THRESHOLD = 0.4 # Minimum confidence for detections for BOTH models
DEBUG_CARD_DETECTION_CONF = 0.1 # Lower threshold for debugging Model 1 detections


MODEL1 = None
MODEL2 = None
try:
    print("Loading YOLOv8 models...")
    MODEL1 = YOLO(MODEL1_PATH) # Aadhaar front/back detection
    MODEL2 = YOLO(MODEL2_PATH) # Entity detection
    print("Models loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}", file=sys.stderr)
    # Models are None, which will be checked in the endpoint


# Define class names based on your actual model classes
CARD_CLASSES = {0: 'aadhar_front', 1: 'aadhar_back', 2: 'print_aadhar'} # Model1 classes
ENTITY_CLASSES = {
    0: 'aadharNumber', 1: 'address', 2: 'address_other_lang', 3: 'city',
    4: 'dob', 5: 'gender', 6: 'gender_other_lang', 7: 'mobile_no',
    8: 'name', 9: 'name_otherlang', 10: 'pincode', 11: 'state'
} # Model2 classes


app = FastAPI(
    title="Aadhaar Authentication Pipeline API",
    description="API to process Aadhaar front and back images, detect entities, and extract text via OCR."
)


class AadhaarProcessRequest(BaseModel):
    user_id: str = "default_user" # Optional user identifier
    front_url: str # URL to the front side image
    back_url: str # URL to the back side image

class ExtractedEntity(BaseModel):
    class_name: str
    text: str | None # Text will be None if OCR fails

class AadhaarProcessingResponse(BaseModel):
    session_id: str
    status: str # "success", "failed", "print_aadhar_detected"
    message: str
    front_entities: list[ExtractedEntity] | None = None
    back_entities: list[ExtractedEntity] | None = None
    output_dir: str | None = None # Path to the session output directory (Optional in case of early failure)

# --- Helper Functions ---

def download_image(url: str) -> Image.Image:
    """Downloads an image from a URL and returns a PIL Image."""
    try:
        response = requests.get(url, stream=True, timeout=15)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        img = Image.open(io.BytesIO(response.content)).convert("RGB") # Ensure RGB format
        return img
    except requests.exceptions.RequestException as e:
        print(f"Error downloading image from {url}: {e}", file=sys.stderr)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Could not download image from URL: {url}. Error: {e}")
    except Exception as e:
        print(f"Error processing image after download from {url}: {e}", file=sys.stderr)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error processing image from URL: {url}. Error: {e}")

def pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
    """Converts a PIL Image to a OpenCV image (numpy array)."""
    # OpenCV uses BGR format, PIL uses RGB
    cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return cv2_img

def cv2_to_pil(cv2_img: np.ndarray) -> Image.Image:
    """Converts a OpenCV image (numpy array) to a PIL Image."""
    # OpenCV uses BGR format, PIL uses RGB
    pil_img = Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))
    return pil_img

def preprocess_image_for_ocr(pil_img: Image.Image) -> Image.Image:
    """
    Applies basic preprocessing steps to a PIL Image for OCR.

    Args:
        pil_img (PIL.Image.Image): The input PIL image.

    Returns:
        PIL.Image.Image: The preprocessed PIL Image object.
    """
    # 1. Convert to grayscale: Often helps OCR accuracy.
    img = pil_img.convert('L')

    # 2. Resize: Large images can be slow, small text might need upscaling.
    max_side = 1200
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / float(max(w, h))
        new_size = (int(w * scale), int(h * scale))
        img = img.resize(new_size, Image.Resampling.LANCZOS) # LANCZOS is good for text

    # 3. (Optional) Binarization: Convert to strict black and white.
    # img = img.point(lambda x: 0 if x < 128 else 255, '1') # Simple threshold at 128

    return img

def run_ocr_on_image(pil_img: Image.Image) -> str | None:
    """
    Performs OCR on a preprocessed PIL image.

    Args:
        pil_img (PIL.Image.Image): The input PIL image (preferably preprocessed).

    Returns:
        str | None: The extracted text, or None if OCR fails.
    """
    try:
        # Perform OCR using pytesseract
        # config='--psm 6' suggests assuming a single uniform block of text
        text = pytesseract.image_to_string(pil_img, lang='eng', config='--psm 6')

        # Clean up extracted text (remove leading/trailing whitespace)
        cleaned_text = text.strip()
        return cleaned_text
    except pytesseract.TesseractError as te:
         print(f"  ❌ Tesseract Error during OCR: {te}", file=sys.stderr)
         return None
    except Exception as e:
        print(f"  ❌ Unexpected error during OCR: {e}", file=sys.stderr)
        # traceback.print_exc() # Uncomment for detailed exception stack
        return None

# --- Core Pipeline Logic ---

def process_card_image(
    pil_img: Image.Image,
    card_side: str, # "front" or "back"
    session_dir: Path
) -> list[ExtractedEntity]:
    """
    Processes a single card image (front or back): verifies card type,
    detects entities, crops entities, runs OCR, and saves intermediate steps.

    Args:
        pil_img (PIL.Image.Image): The input card image.
        card_side (str): The expected side of the card ("front" or "back").
        session_dir (Path): The base directory for this processing session.

    Returns:
        list[ExtractedEntity]: A list of detected entities with extracted text.
    """
    if MODEL1 is None or MODEL2 is None:
         # This check should ideally be done before calling this function,
         # but as a safeguard:
         raise RuntimeError("YOLO models failed to load.")

    card_name = f"aadhar_{card_side}"
    print(f"\n--- Processing {card_name} ---")

    # --- Directories for this side ---
    debug_dir = session_dir / "0_debug_card_detections" # New debug dir
    front_back_dir = session_dir / "1_front_back_cards"
    detected_entities_dir = session_dir / "2_detected_entities"
    cropped_entities_dir = session_dir / "3_cropped_entities"
    preprocessed_entities_dir = session_dir / "4_preprocessed_entities"

    # Ensure directories exist
    for d in [debug_dir, front_back_dir, detected_entities_dir, cropped_entities_dir, preprocessed_entities_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Convert PIL to CV2 for model inference and drawing
    cv2_img_original = pil_to_cv2(pil_img)
    # Use a unique filename stem for this card within the session
    img_stem = f"{card_side}_input"

    # --- Step 1: Verify Card Type and Crop (using Model 1) ---
    print(f"  🔍 Step 1: Verifying and cropping {card_side} card...")
    # Run model 1 on the *original* image to find the card itself
    model1_results = MODEL1(cv2_img_original, verbose=False) # Remove confidence threshold here to see all detections
    model1_result = model1_results[0]

    card_crop = None
    card_detection_found = False # Flag for finding the *required* card type with required confidence

    # --- Debug Output for Model 1 detections ---
    debug_img_model1 = cv2_img_original.copy()
    debug_filename_model1 = debug_dir / f"{img_stem}_model1_all_detections.jpg"
    debug_detections_info = []

    print(f"    Model 1 detected (all confidences):")
    for i, box in enumerate(model1_result.boxes):
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        class_name = CARD_CLASSES.get(cls_id, f"class_{cls_id}")
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        debug_detections_info.append({'class': class_name, 'confidence': conf, 'bbox': (x1, y1, x2, y2)})
        print(f"      - {class_name}: {conf:.2f}")

        # Draw bounding box on the debug image
        color = (0, 255, 0) # Green
        if class_name == card_name and conf >= CONFIDENCE_THRESHOLD:
             color = (255, 0, 0) # Blue for the one meeting criteria
        elif conf < CONFIDENCE_THRESHOLD:
             color = (128, 128, 128) # Grey for low confidence

        cv2.rectangle(debug_img_model1, (x1, y1), (x2, y2), color, 2)
        label = f"{class_name}: {conf:.2f}"
        (w, h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(debug_img_model1, (x1, y1 - h - baseline), (x1 + w, y1), color, -1)
        cv2.putText(debug_img_model1, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255) if color == (128, 128, 128) else (0, 0, 0), 1) # White text on grey, black on others

        # Check for Print Aadhaar immediately, regardless of whether it's the expected side
        if class_name == 'print_aadhar' and conf >= CONFIDENCE_THRESHOLD:
             error_msg = f"❌ PRINT AADHAAR DETECTED on {card_side} side (confidence: {conf:.2f})"
             print(error_msg, file=sys.stderr)
             # Save the debug image before raising error
             cv2.imwrite(str(debug_filename_model1), debug_img_model1)
             # Also save debug info
             debug_info_path = debug_dir / f"{img_stem}_model1_all_detections.json"
             with open(debug_info_path, 'w', encoding='utf-8') as f:
                 json.dump(debug_detections_info, f, indent=4)
             raise ValueError("Print Aadhaar detected - processing stopped for security reasons") # Caught by API endpoint


    # Save the debug image showing all Model 1 detections
    cv2.imwrite(str(debug_filename_model1), debug_img_model1)
    # Also save debug info JSON
    debug_info_path = debug_dir / f"{img_stem}_model1_all_detections.json"
    with open(debug_info_path, 'w', encoding='utf-8') as f:
        json.dump(debug_detections_info, f, indent=4)

    print(f"    📸 Saved debug image (Model 1 detections): {debug_filename_model1.name}")


    # Now, specifically look for the required card type with the REQUIRED confidence
    target_card_box = None
    for i, box in enumerate(model1_result.boxes):
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        class_name = CARD_CLASSES.get(cls_id, f"class_{cls_id}")

        if class_name == card_name and conf >= CONFIDENCE_THRESHOLD:
             target_card_box = box.xyxy[0]
             card_detection_found = True
             print(f"    ✅ Found required {card_name} detection (confidence: {conf:.2f})")
             break # Found the main card detection, stop searching

    if not card_detection_found or target_card_box is None:
        error_msg = f"❌ No strong detection for {card_name} found in the provided image with confidence >= {CONFIDENCE_THRESHOLD}. Check debug image {debug_filename_model1.name} for low-confidence detections."
        print(error_msg, file=sys.stderr)
        raise ValueError(error_msg) # Caught by API endpoint

    # Crop the card using the found box coordinates
    x1, y1, x2, y2 = map(int, target_card_box)
    card_crop = cv2_img_original[y1:y2, x1:x2].copy() # Use copy to avoid view issues

    # Save the *specifically* cropped card image
    cropped_card_filename = front_back_dir / f"{img_stem}_{card_name}_conf{int(conf*100)}.jpg" # Use conf from the target box
    cv2.imwrite(str(cropped_card_filename), card_crop)
    print(f"    📸 Saved cropped required card: {cropped_card_filename.name}")


    # --- Step 2: Detect Entities in the Cropped Card (using Model 2) ---
    print(f"  🔍 Step 2: Detecting entities in cropped {card_name}...")
    model2_results = MODEL2(card_crop, verbose=False, conf=CONFIDENCE_THRESHOLD)
    model2_result = model2_results[0]

    # Draw bounding boxes on the cropped card image for visualization
    card_with_entities = card_crop.copy()
    entity_detections = [] # Store detected entities info

    print(f"    Model 2 detected (confidence >= {CONFIDENCE_THRESHOLD}):")
    for i, box in enumerate(model2_result.boxes):
        conf = float(box.conf[0])
        # Confidence check already handled by YOLO inference setting 'conf=CONFIDENCE_THRESHOLD'
        # if conf < CONFIDENCE_THRESHOLD:
        #     continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        class_name = ENTITY_CLASSES.get(cls_id, f"entity_{cls_id}")
        print(f"      - {class_name}: {conf:.2f}")


        # Draw bounding box
        color = (0, 255, 0) # Green
        cv2.rectangle(card_with_entities, (x1, y1), (x2, y2), color, 2)
        # Add text label
        label = f"{class_name}: {conf:.2f}"
        (w, h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(card_with_entities, (x1, y1 - h - baseline), (x1 + w, y1), color, -1)
        cv2.putText(card_with_entities, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Store detection for later cropping and OCR
        entity_detections.append({
            'bbox': (x1, y1, x2, y2),
            'class_name': class_name,
            'confidence': conf
        })

    # Save the cropped card image with entity bounding boxes drawn
    detected_entities_filename = detected_entities_dir / f"{img_stem}_{card_name}_with_entities.jpg"
    cv2.imwrite(str(detected_entities_filename), card_with_entities)
    print(f"    📸 Saved image with entity boxes: {detected_entities_filename.name}")
    print(f"    📊 Detected {len(entity_detections)} entities (confidence >= {CONFIDENCE_THRESHOLD}).")

    # --- Step 3 & 4: Crop Entities, Preprocess, and Run OCR ---
    print(f"  ✂️  Step 3/4: Cropping entities, preprocessing, and running OCR...")
    extracted_entities_list: list[ExtractedEntity] = []

    for i, detection in enumerate(entity_detections):
        x1, y1, x2, y2 = detection['bbox']
        class_name = detection['class_name']
        conf = detection['confidence'] # This conf is >= CONFIDENCE_THRESHOLD

        # Crop entity from the card_crop (which is already CV2 format)
        entity_crop_cv2 = card_crop[y1:y2, x1:x2].copy() # Use copy

        # Save cropped entity image
        cropped_entity_filename_stem = f"{img_stem}_{card_name}_{class_name}_{i}_conf{int(conf*100)}"
        cropped_entity_save_path = cropped_entities_dir / f"{cropped_entity_filename_stem}.jpg"
        cv2.imwrite(str(cropped_entity_save_path), entity_crop_cv2)
        # print(f"      📸 Saved cropped entity: {cropped_entity_save_path.name}") # Too verbose maybe

        # Convert to PIL for preprocessing and OCR
        entity_crop_pil = cv2_to_pil(entity_crop_cv2)

        # Preprocess for OCR
        preprocessed_entity_pil = preprocess_image_for_ocr(entity_crop_pil)

        # Save preprocessed image
        preprocessed_entity_save_path = preprocessed_entities_dir / f"{cropped_entity_filename_stem}_preprocessed.png"
        preprocessed_entity_pil.save(str(preprocessed_entity_save_path))
        # print(f"      📸 Saved preprocessed entity: {preprocessed_entity_save_path.name}") # Too verbose maybe

        # Run OCR
        extracted_text = run_ocr_on_image(preprocessed_entity_pil)
        print(f"      OCR on {class_name}: \"{extracted_text.replace('\n', ' ').replace('\r', ' ').strip()[:50]}...\"" if extracted_text else f"      OCR on {class_name}: [Failed]")

        extracted_entities_list.append(ExtractedEntity(class_name=class_name, text=extracted_text))

    print(f"--- Finished processing {card_name} ---")
    return extracted_entities_list

# --- API Endpoint ---

@app.post("/process_aadhaar", response_model=AadhaarProcessingResponse)
async def process_aadhaar_endpoint(request: AadhaarProcessRequest):
    """
    Processes Aadhaar front and back images from URLs.

    Downloads images, detects entities, crops entities, runs OCR,
    saves all intermediate steps, and returns extracted text.
    """
    print(f"\n>>> Received request for user: {request.user_id}")

    # Create a timestamped session directory for this request
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = OUTPUT_BASE_DIR / f"session_{timestamp}"
    output_dir_str = str(session_dir) # Store as string for response

    # Ensure output base directory exists early
    OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)


    try:
        # Check if models were loaded successfully
        if MODEL1 is None or MODEL2 is None:
             raise RuntimeError("YOLO models failed to load at startup. Check server logs for details.")

        # Create base output directories for this session
        # Create debug dir first
        (session_dir / "0_debug_card_detections").mkdir(parents=True, exist_ok=True)
        for d in ["1_front_back_cards", "2_detected_entities", "3_cropped_entities", "4_preprocessed_entities"]:
            (session_dir / d).mkdir(parents=True, exist_ok=True)
        print(f"📁 Created session directory: {session_dir}")


        # --- Download Images ---
        print("⬇️ Downloading images...")
        front_img_pil = download_image(request.front_url)
        back_img_pil = download_image(request.back_url)
        print("✅ Images downloaded.")

        # --- Process Front Image ---
        front_entities = process_card_image(front_img_pil, "front", session_dir)

        # --- Process Back Image ---
        back_entities = process_card_image(back_img_pil, "back", session_dir)

        # --- Combine Results and Save JSON ---
        final_results = {
            "user_id": request.user_id,
            "front_url": request.front_url,
            "back_url": request.back_url,
            "session_id": f"session_{timestamp}",
            "front_entities": [e.dict() for e in front_entities], # Convert Pydantic models to dicts
            "back_entities": [e.dict() for e in back_entities],
        }

        results_json_path = session_dir / "ocr_results.json"
        with open(results_json_path, "w", encoding="utf-8") as f:
            json.dump(final_results, f, indent=4)
        print(f"\n✅ Final results saved to JSON: {results_json_path.name}")


        # --- Return Response ---
        response_data = AadhaarProcessingResponse(
            session_id=f"session_{timestamp}",
            status="success",
            message="Aadhaar processing completed successfully.",
            front_entities=front_entities,
            back_entities=back_entities,
            output_dir=output_dir_str
        )
        print("<<< Request processed successfully.")
        return response_data

    except ValueError as ve:
        # Specific handling for our raised ValueErrors (like Print Aadhaar, No card detected)
        print(f"🚫 Processing Error: {ve}", file=sys.stderr)
        # Keep the session directory for debugging failed attempts

        # Determine status based on error message
        status_msg = "failed"
        if "Print Aadhaar detected" in str(ve):
             status_msg = "print_aadhar_detected"

        return AadhaarProcessingResponse(
            session_id=f"session_{timestamp}",
            status=status_msg,
            message=f"Processing failed: {ve}",
            output_dir=output_dir_str # Still return the session dir path even on failure
        )

    except RuntimeError as re:
        # Handle specific runtime errors like models not loading
        print(f"🚫 Runtime Error: {re}", file=sys.stderr)
        # No session directory created yet or error before creating it
        return AadhaarProcessingResponse(
            session_id=f"N/A", # No session ID if directory creation failed
            status="failed",
            message=f"API startup/runtime error: {re}",
            output_dir=None
        )

    except Exception as e:
        # Catch any other unexpected errors
        print(f"❌ An unexpected error occurred: {e}", file=sys.stderr)
        traceback.print_exc() # Print detailed error for debugging

        # Keep the session directory for debugging failed attempts

        return AadhaarProcessingResponse(
            session_id=f"session_{timestamp}",
            status="failed",
            message=f"An internal error occurred: {e}",
            output_dir=output_dir_str # Still return the session dir path even on failure
        )

# --- Health Check Endpoint ---
@app.get("/health")
async def health_check():
    """Basic health check to verify API is running and models are loaded."""
    tesseract_ok = False
    try:
        pytesseract.get_tesseract_version()
        tesseract_ok = True
    except: # Catch any exception if tesseract_cmd is not set or tesseract isn't found
        pass

    return {
        "status": "healthy",
        "models_loaded": MODEL1 is not None and MODEL2 is not None,
        "tesseract_accessible": tesseract_ok
    }

# --- How to Run ---
# Save this file as something like `main_api.py`
# Ensure your models are at the paths specified by MODEL1_PATH and MODEL2_PATH.
# Ensure Tesseract OCR is installed and accessible.
# Install required Python packages: `pip install fastapi uvicorn ultralytics Pillow pytesseract requests opencv-python numpy`
# Run from your terminal using uvicorn:
# uvicorn main_api:app --reload
# (The --reload flag is useful for development, removes it for production)
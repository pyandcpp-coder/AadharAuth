import os
import sys
import io
import json
import traceback
import cv2
import numpy as np
from PIL import Image
import pytesseract
from ultralytics import YOLO
import argparse
from pathlib import Path
from datetime import datetime
import shutil

# --- Configuration ---
# Set the path to the Tesseract executable if it's not in your system's PATH
# Example for macOS:
# pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract'
# Example for Windows:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# Example for Linux (often in PATH, but if not):
# pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract' # or where your install places it

# Uncomment and set the path if needed based on your Tesseract installation
# try:
#     # Attempt to automatically find Tesseract (may not work on all systems)
#     tess_path = shutil.which('tesseract')
#     if tess_path:
#         pytesseract.pytesseract.tesseract_cmd = tess_path
#         print(f"✅ Tesseract executable automatically found at: {tess_path}")
#     else:
#         # Fallback to manual path if auto-detection fails
#         # Replace with your actual path if needed
#         # pytesseract.pytesseract.tesseract_cmd = r'YOUR_TESSERACT_PATH_HERE'
#         print("⚠️ Could not auto-detect Tesseract executable path. Set pytesseract.pytesseract.tesseract_cmd manually if it's not in your PATH.")
# except Exception as e:
#     print(f"⚠️ Error during Tesseract auto-detection: {e}. Set pytesseract.pytesseract.tesseract_cmd manually if needed.", file=sys.stderr)


# --- Basic Tesseract Check ---
try:
    pytesseract.get_tesseract_version()
    print("✅ Tesseract is installed and accessible.")
    TESSERACT_AVAILABLE = True
except pytesseract.TesseractNotFoundError:
    print("❌ Tesseract executable not found.", file=sys.stderr)
    print("Please install Tesseract OCR and ensure it's in your PATH or set pytesseract.pytesseract.tesseract_cmd.", file=sys.stderr)
    TESSERACT_AVAILABLE = False
except Exception as e:
     print(f"❌ Error checking Tesseract version: {e}", file=sys.stderr)
     TESSERACT_AVAILABLE = False


# --- Global Variables ---
# Define your model paths here
# Replace these with the actual paths on your system
# Use the path you provided in the last comment for Model 1
MODEL1_PATH = "/Users/yrevash/Downloads/yolov8_upload_dir/weights/best.pt" # Card detection model
MODEL2_PATH = "/Users/yrevash/Downloads/aadhar_models/aadhar_model_2/best-5.pt" # Entity detection model

OUTPUT_BASE_DIR = Path("local_pipeline_output")
CONFIDENCE_THRESHOLD = 0.4 # Minimum confidence for detections for BOTH models

# Load YOLOv8 models globally (or within main function)
# Loading here means they are loaded once when the script starts
MODEL1 = None
MODEL2 = None
MODELS_LOADED = False
try:
    print("Loading YOLOv8 models...")
    MODEL1 = YOLO(MODEL1_PATH) # Aadhaar front/back detection
    MODEL2 = YOLO(MODEL2_PATH) # Entity detection
    print("Models loaded successfully!")
    MODELS_LOADED = True
except Exception as e:
    print(f"❌ Error loading models: {e}", file=sys.stderr)
    # Models are None, MODELS_LOADED is False


# Define class names based on your actual model classes
CARD_CLASSES = {0: 'aadhar_front', 1: 'aadhar_back', 2: 'print_aadhar'} # Model1 classes
ENTITY_CLASSES = {
    0: 'aadharNumber', 1: 'address', 2: 'address_other_lang', 3: 'city',
    4: 'dob', 5: 'gender', 6: 'gender_other_lang', 7: 'mobile_no',
    8: 'name', 9: 'name_otherlang', 10: 'pincode', 11: 'state'
} # Model2 classes

# --- Helper Functions ---

def load_image_local(image_path: Path) -> Image.Image | None:
    """Loads an image from a local path and returns a PIL Image."""
    try:
        # Using PIL to load, consistent with potential preprocessing needs
        img = Image.open(image_path).convert("RGB") # Ensure RGB format
        return img
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error loading image from {image_path}: {e}", file=sys.stderr)
        return None

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
    """
    img = pil_img.convert('L')
    max_side = 1200
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / float(max(w, h))
        new_size = (int(w * scale), int(h * scale))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
    return img

def run_ocr_on_image(pil_img: Image.Image) -> str | None:
    """
    Performs OCR on a preprocessed PIL image.

    Returns:
        str | None: The extracted text, or placeholder if OCR fails or Tesseract not available.
    """
    if not TESSERACT_AVAILABLE:
        # print("  ⚠️ Tesseract not available, skipping OCR.", file=sys.stderr) # Avoid spamming console
        return "[OCR Skipped: Tesseract not available]"
    try:
        text = pytesseract.image_to_string(pil_img, lang='eng', config='--psm 6')
        cleaned_text = text.strip()
        return cleaned_text
    except pytesseract.TesseractError as te:
         print(f"  ❌ Tesseract Error during OCR: {te}", file=sys.stderr)
         return f"[OCR Failed: {te}]"
    except Exception as e:
        print(f"  ❌ Unexpected error during OCR: {e}", file=sys.stderr)
        # traceback.print_exc() # Uncomment for detailed exception stack
        return f"[OCR Failed: {e}]"


# --- Core Pipeline Logic for One Image ---

def process_single_image_pipeline(
    input_image_path: Path,
    card_side: str, # "front" or "back"
    session_dir: Path,
    confidence_threshold: float
) -> list[dict] | None:
    """
    Processes a single input image (front or back): loads image, detects card type,
    crops card, detects entities, crops entities, runs OCR, and saves intermediate steps.

    Args:
        input_image_path (Path): Path to the local input image file.
        card_side (str): The expected side of the card ("front" or "back").
        session_dir (Path): The base directory for this processing session.
        confidence_threshold (float): Confidence threshold for detections.

    Returns:
        list[dict] | None: A list of dictionaries for detected entities with extracted text
                           if successful, otherwise None (error handled internally).
                           Each dict has 'class_name' and 'text'.
    """
    if not MODELS_LOADED:
         print("Skipping processing due to model loading failure.", file=sys.stderr)
         return None # Indicate failure by returning None

    card_name = f"aadhar_{card_side}"
    print(f"\n--- Processing input image for {card_name} ---")
    print(f"Input file: {input_image_path}")

    # --- Load Image ---
    pil_img_original = load_image_local(input_image_path)
    if pil_img_original is None:
        print(f"Skipping processing for {card_name} due to image loading error.", file=sys.stderr)
        return None # Indicate failure

    # Convert PIL to CV2 for model inference and drawing
    cv2_img_original = pil_to_cv2(pil_img_original)
    img_stem = input_image_path.stem # Use input filename stem for saved files

    # --- Setup Directories for this image processing ---
    # These directories are relative to the session_dir
    debug_dir = session_dir / "0_debug_card_detections"
    front_back_dir = session_dir / "1_front_back_cards"
    detected_entities_dir = session_dir / "2_detected_entities"
    cropped_entities_dir = session_dir / "3_cropped_entities"
    preprocessed_entities_dir = session_dir / "4_preprocessed_entities"

    for d in [debug_dir, front_back_dir, detected_entities_dir, cropped_entities_dir, preprocessed_entities_dir]:
        d.mkdir(parents=True, exist_ok=True)


    # --- Step 1: Verify Card Type and Crop (using Model 1) ---
    print(f"  🔍 Step 1: Detecting and cropping {card_side} card...")
    # Run model 1 on the *original* image to find the card itself
    try:
        # Run inference with a lower confidence threshold here to ensure we capture
        # potential detections for the debug output, even if they don't meet the main threshold.
        # We will filter by confidence_threshold later when selecting the target box.
        model1_results = MODEL1(cv2_img_original, verbose=False) # Remove confidence threshold here
        model1_result = model1_results[0]
    except Exception as e:
        print(f"  ❌ Error during Model 1 inference for {card_name}: {e}", file=sys.stderr)
        traceback.print_exc()
        print(f"--- Processing {card_name} Failed: Model 1 Error ---")
        return None


    card_crop = None
    card_detection_found = False # Flag for finding the *required* card type with required confidence
    target_card_conf = 0.0

    # --- Debug Output for Model 1 detections ---
    debug_img_model1 = cv2_img_original.copy()
    debug_filename_model1 = debug_dir / f"{img_stem}_model1_all_detections.jpg"
    debug_detections_info = []

    print(f"    Model 1 detected (all confidences):")
    # Check if detections exist before iterating
    if hasattr(model1_result, 'boxes') and model1_result.boxes is not None and len(model1_result.boxes) > 0:
        # Sort detections by confidence descending for easier viewing of best detections
        sorted_boxes = sorted(model1_result.boxes, key=lambda x: float(x.conf[0]), reverse=True)

        for i, box in enumerate(sorted_boxes):
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            class_name = CARD_CLASSES.get(cls_id, f"class_{cls_id}")
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            debug_detections_info.append({'class': class_name, 'confidence': conf, 'bbox': (x1, y1, x2, y2)})
            print(f"      - {class_name}: {conf:.2f}")

            # Draw bounding box on the debug image
            color = (0, 255, 0) # Green
            if class_name == card_name and conf >= confidence_threshold:
                 color = (255, 0, 0) # Blue for the one meeting criteria
            elif conf < confidence_threshold:
                 color = (128, 128, 128) # Grey for low confidence
            else:
                 color = (0, 255, 0) # Green for other high confidence detections

            # Ensure coordinates are valid
            h_orig, w_orig = cv2_img_original.shape[:2]
            x1_draw, y1_draw, x2_draw, y2_draw = max(0, x1), max(0, y1), min(w_orig, x2), min(h_orig, y2)

            if x1_draw < x2_draw and y1_draw < y2_draw: # Only draw if box is valid
                cv2.rectangle(debug_img_model1, (x1_draw, y1_draw), (x2_draw, y2_draw), color, 2)
                label = f"{class_name}: {conf:.2f}"
                (w, h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                # Adjust text position if close to the top edge
                text_y_draw = y1_draw - 10 if y1_draw - 10 > h + baseline else y1_draw + h + baseline
                text_y_draw = max(10, text_y_draw) # Ensure text is not off the top edge
                cv2.rectangle(debug_img_model1, (x1_draw, y1_draw - h - baseline), (x1_draw + w, y1_draw), color, -1)
                cv2.putText(debug_img_model1, label, (x1_draw, text_y_draw), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255) if color == (128, 128, 128) else (0, 0, 0), 1) # White text on grey, black on others


            # Check for Print Aadhaar immediately if it meets the main threshold
            if class_name == 'print_aadhar' and conf >= confidence_threshold:
                 error_msg = f"❌ PRINT AADHAAR DETECTED on {card_side} side (confidence: {conf:.2f})"
                 print(error_msg, file=sys.stderr)
                 # Save the debug image before returning/indicating error
                 cv2.imwrite(str(debug_filename_model1), debug_img_model1)
                 debug_info_path = debug_dir / f"{img_stem}_model1_all_detections.json"
                 with open(debug_info_path, 'w', encoding='utf-8') as f:
                      json.dump(debug_detections_info, f, indent=4)
                 print(f"--- Processing {card_name} Failed: Print Aadhaar Detected ---")
                 return None # Indicate failure

    else:
        print("      (No detections found by Model 1)")


    # Save the debug image showing all Model 1 detections
    cv2.imwrite(str(debug_filename_model1), debug_img_model1)
    debug_info_path = debug_dir / f"{img_stem}_model1_all_detections.json"
    with open(debug_info_path, 'w', encoding='utf-8') as f:
        json.dump(debug_detections_info, f, indent=4)
    print(f"    📸 Saved debug image (Model 1 detections): {debug_filename_model1.name}")


    # Now, specifically look for the required card type with the REQUIRED confidence
    target_card_box = None
    if hasattr(model1_result, 'boxes') and model1_result.boxes is not None and len(model1_result.boxes) > 0:
        # Iterate through sorted boxes again to find the best *matching* detection
        for i, box in enumerate(sorted_boxes):
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            class_name = CARD_CLASSES.get(cls_id, f"class_{cls_id}")

            if class_name == card_name and conf >= confidence_threshold:
                 target_card_box = box.xyxy[0]
                 card_detection_found = True
                 target_card_conf = conf
                 print(f"    ✅ Found required {card_name} detection (confidence: {conf:.2f})")
                 break # Found the main card detection, stop searching


    if not card_detection_found or target_card_box is None:
        error_msg = f"❌ No strong detection for {card_name} found in the provided image with confidence >= {confidence_threshold}. Check debug image {debug_filename_model1.name} for all detections."
        print(error_msg, file=sys.stderr)
        print(f"--- Processing {card_name} Failed: No Required Card Detected ---")
        return None # Indicate failure

    # Crop the card using the found box coordinates
    x1, y1, x2, y2 = map(int, target_card_box)
    # Ensure coordinates are within bounds before cropping
    h_orig, w_orig = cv2_img_original.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w_orig, x2), min(h_orig, y2)

    if x1 >= x2 or y1 >= y2: # Invalid crop box after clamping
         error_msg = f"❌ Invalid bounding box coordinates after clamping for {card_name}: ({x1},{y1})-({x2},{y2})"
         print(error_msg, file=sys.stderr)
         print(f"--- Processing {card_name} Failed: Invalid Crop Box ---")
         return None

    card_crop = cv2_img_original[y1:y2, x1:x2].copy() # Use copy to avoid view issues

    # Save the *specifically* cropped card image
    cropped_card_filename = front_back_dir / f"{img_stem}_{card_name}_conf{int(target_card_conf*100)}.jpg"
    try:
        cv2.imwrite(str(cropped_card_filename), card_crop)
        print(f"    📸 Saved cropped required card: {cropped_card_filename.name}")
    except Exception as e:
        print(f"    ❌ Error saving cropped card {cropped_card_filename.name}: {e}", file=sys.stderr)


    # --- Step 2: Detect Entities in the Cropped Card (using Model 2) ---
    print(f"  🔍 Step 2: Detecting entities in cropped {card_name}...")
    entity_detections = [] # Store detected entities info
    try:
        # Run model 2 on the cropped card, filtering by confidence_threshold
        model2_results = MODEL2(card_crop, verbose=False, conf=confidence_threshold)
        model2_result = model2_results[0]

        # Draw bounding boxes on the cropped card image for visualization
        card_with_entities = card_crop.copy()

        print(f"    Model 2 detected (confidence >= {confidence_threshold}):")
        if hasattr(model2_result, 'boxes') and model2_result.boxes is not None and len(model2_result.boxes) > 0:
             # Sort entity detections by confidence
             sorted_entity_boxes = sorted(model2_result.boxes, key=lambda x: float(x.conf[0]), reverse=True)

             for i, box in enumerate(sorted_entity_boxes):
                conf = float(box.conf[0])
                # Confidence check handled by YOLO inference setting 'conf=confidence_threshold'

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                class_name = ENTITY_CLASSES.get(cls_id, f"entity_{cls_id}")
                print(f"      - {class_name}: {conf:.2f}")

                # Draw bounding box
                color = (0, 255, 0) # Green
                # Ensure coordinates are valid relative to the cropped image size
                h_crop, w_crop = card_crop.shape[:2]
                x1_draw, y1_draw, x2_draw, y2_draw = max(0, x1), max(0, y1), min(w_crop, x2), min(h_crop, y2)

                if x1_draw < x2_draw and y1_draw < y2_draw: # Only draw if box is valid
                    cv2.rectangle(card_with_entities, (x1_draw, y1_draw), (x2_draw, y2_draw), color, 2)
                    # Add text label
                    label = f"{class_name}: {conf:.2f}"
                    (w, h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    # Adjust text position relative to the cropped image bounds
                    text_y_draw = y1_draw - 10 if y1_draw - 10 > h + baseline else y1_draw + h + baseline
                    text_y_draw = max(10, text_y_draw) # Ensure text is not off the top edge
                    cv2.rectangle(card_with_entities, (x1_draw, y1_draw - h - baseline), (x1_draw + w, y1_draw), color, -1)
                    cv2.putText(card_with_entities, label, (x1_draw, text_y_draw), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                    # Store detection for later cropping and OCR (use original coords relative to crop)
                    entity_detections.append({
                        'bbox': (x1, y1, x2, y2), # These are relative to the cropped card
                        'class_name': class_name,
                        'confidence': conf
                    })
        else:
            print("      (No entities found by Model 2)")


        # Save the cropped card image with entity bounding boxes drawn
        detected_entities_filename = detected_entities_dir / f"{img_stem}_{card_name}_with_entities.jpg"
        try:
            cv2.imwrite(str(detected_entities_filename), card_with_entities)
            print(f"    📸 Saved image with entity boxes: {detected_entities_filename.name}")
        except Exception as e:
            print(f"    ❌ Error saving entities image {detected_entities_filename.name}: {e}", file=sys.stderr)

        print(f"    📊 Detected {len(entity_detections)} entities (confidence >= {confidence_threshold}).")

    except Exception as e:
        print(f"  ❌ Error during Model 2 inference or drawing for {card_name}: {e}", file=sys.stderr)
        traceback.print_exc()
        print(f"--- Processing {card_name} Failed: Model 2 Error ---")
        return None


    # --- Step 3 & 4: Crop Entities, Preprocess, and Run OCR ---
    print(f"  ✂️  Step 3/4: Cropping entities, preprocessing, and running OCR...")
    extracted_entities_list = [] # List of dicts: [{'class_name': '...', 'text': '...'}, ...]

    if not TESSERACT_AVAILABLE:
         print("  ⚠️ Tesseract not available, skipping entity cropping, preprocessing, and OCR.", file=sys.stderr)
         # Return the list of detected entities *without* text if Tesseract isn't available
         extracted_entities_list = [{"class_name": det['class_name'], "text": "[OCR Skipped: Tesseract not available]"} for det in entity_detections]
         print(f"--- Finished processing {card_name} (OCR Skipped) ---")
         return extracted_entities_list


    # If Tesseract is available, proceed with cropping and OCR
    for i, detection in enumerate(entity_detections):
        x1, y1, x2, y2 = detection['bbox'] # These are relative to the cropped card
        class_name = detection['class_name']
        conf = detection['confidence'] # This conf is >= confidence_threshold

        # Ensure coordinates are valid relative to the cropped image size
        h_crop, w_crop = card_crop.shape[:2]
        x1_crop, y1_crop, x2_crop, y2_crop = max(0, x1), max(0, y1), min(w_crop, x2), min(h_crop, y2)

        if x1_crop >= x2_crop or y1_crop >= y2_crop: # Invalid crop box after clamping
             print(f"    ❌ Invalid bounding box for entity '{class_name}' after clamping: ({x1_crop},{y1_crop})-({x2_crop},{y2_crop}). Skipping crop/OCR.", file=sys.stderr)
             extracted_entities_list.append({"class_name": class_name, "text": "[OCR Failed: Invalid Crop Box]"})
             continue # Skip to the next entity


        # Crop entity from the card_crop (which is already CV2 format)
        entity_crop_cv2 = card_crop[y1_crop:y2_crop, x1_crop:x2_crop].copy() # Use copy

        # Save cropped entity image
        cropped_entity_filename_stem = f"{img_stem}_{card_name}_{class_name}_{i}_conf{int(conf*100)}"
        cropped_entity_save_path = cropped_entities_dir / f"{cropped_entity_filename_stem}.jpg"
        try:
             cv2.imwrite(str(cropped_entity_save_path), entity_crop_cv2)
             # print(f"      📸 Saved cropped entity: {cropped_entity_save_path.name}") # Too verbose maybe
        except Exception as e:
             print(f"      ❌ Error saving cropped entity {cropped_entity_save_path.name}: {e}", file=sys.stderr)


        # Convert to PIL for preprocessing and OCR
        entity_crop_pil = cv2_to_pil(entity_crop_cv2)

        # Preprocess for OCR
        preprocessed_entity_pil = preprocess_image_for_ocr(entity_crop_pil)

        # Save preprocessed image
        preprocessed_entity_save_path = preprocessed_entities_dir / f"{cropped_entity_filename_stem}_preprocessed.png"
        try:
             preprocessed_entity_pil.save(str(preprocessed_entity_save_path))
             # print(f"      📸 Saved preprocessed entity: {preprocessed_entity_save_path.name}") # Too verbose maybe
        except Exception as e:
             print(f"      ❌ Error saving preprocessed entity {preprocessed_entity_save_path.name}: {e}", file=sys.stderr)


        # Run OCR
        extracted_text = run_ocr_on_image(preprocessed_entity_pil)
        print(f"      OCR on {class_name}: \"{extracted_text.replace('\n', ' ').replace('\r', ' ').strip()[:50]}...\"" if extracted_text and isinstance(extracted_text, str) else f"      OCR on {class_name}: {extracted_text}")

        extracted_entities_list.append({"class_name": class_name, "text": extracted_text})

    print(f"--- Finished processing {card_name} ---")
    return extracted_entities_list

# --- Main Script Execution ---

def main():
    """
    Main function to parse arguments and run the full local pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Full local pipeline for Aadhaar front and back processing."
    )
    parser.add_argument(
        "--front_path",
        "-f",
        required=True,
        type=Path, # Use Path type for automatic Path object creation
        help="Path to the local Aadhaar front image file."
    )
    parser.add_argument(
        "--back_path",
        "-b",
        required=True,
        type=Path, # Use Path type
        help="Path to the local Aadhaar back image file."
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        default="local_pipeline_output",
        type=Path, # Use Path type
        help="Base directory for saving all pipeline output (timestamped subdirectories will be created)."
    )
    parser.add_argument(
        "--confidence",
        "-c",
        type=float,
        default=0.4,
        help="Confidence threshold for object detection models (0.0 to 1.0). Defaults to 0.4."
    )


    args = parser.parse_args()

    # Update global confidence threshold from args
    global CONFIDENCE_THRESHOLD
    CONFIDENCE_THRESHOLD = args.confidence

    print("🚀 Starting local Aadhaar Processing Pipeline")
    print(f"Front Image: {args.front_path}")
    print(f"Back Image:  {args.back_path}")
    print(f"Confidence Threshold: {CONFIDENCE_THRESHOLD}")

    # Check if models are loaded
    if not MODELS_LOADED:
        print("\n❌ Models failed to load. Cannot proceed with processing.")
        print("Please check MODEL1_PATH and MODEL2_PATH in the script and ensure the files exist and are valid YOLOv8 models.")
        return # Exit main function

    # Create a timestamped session directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = args.output_dir / f"session_{timestamp}"

    try:
        # Ensure output base directory exists
        args.output_dir.mkdir(parents=True, exist_ok=True)
        # Session directory and its subdirs will be created within process_single_image_pipeline

        print(f"📁 Creating session directory: {session_dir}")
        # session_dir.mkdir(parents=True, exist_ok=True) # Subdirs created later


        # --- Process Front Image ---
        # Pass the confidence threshold to the processing function
        front_entities_data = process_single_image_pipeline(args.front_path, "front", session_dir, CONFIDENCE_THRESHOLD)

        # --- Process Back Image ---
        # Pass the confidence threshold to the processing function
        back_entities_data = process_single_image_pipeline(args.back_path, "back", session_dir, CONFIDENCE_THRESHOLD)


        # --- Save Results to JSON ---
        print("\n--- Saving Results ---")
        results_summary = {
            "session_id": f"session_{timestamp}",
            "front_image_path": str(args.front_path),
            "back_image_path": str(args.back_path),
            "confidence_threshold_used": CONFIDENCE_THRESHOLD,
            "front_entities": front_entities_data if front_entities_data is not None else "Processing Failed",
            "back_entities": back_entities_data if back_entities_data is not None else "Processing Failed",
        }

        # Save combined results JSON
        combined_json_path = session_dir / "combined_ocr_results.json"
        try:
            with open(combined_json_path, "w", encoding="utf-8") as f:
                json.dump(results_summary, f, indent=4)
            print(f"✅ Combined results saved to JSON: {combined_json_path}")
        except Exception as e:
             print(f"❌ Error saving combined JSON {combined_json_path}: {e}", file=sys.stderr)
             traceback.print_exc()


        # Save front entities JSON (only if processing succeeded)
        if front_entities_data is not None and front_entities_data != "Processing Failed":
            front_json_path = session_dir / "front_entities_ocr.json"
            try:
                with open(front_json_path, "w", encoding="utf-8") as f:
                    json.dump({"front_entities": front_entities_data}, f, indent=4)
                print(f"✅ Front entities saved to JSON: {front_json_path}")
            except Exception as e:
                print(f"❌ Error saving front entities JSON {front_json_path}: {e}", file=sys.stderr)
                traceback.print_exc()

        # Save back entities JSON (only if processing succeeded)
        if back_entities_data is not None and back_entities_data != "Processing Failed":
            back_json_path = session_dir / "back_entities_ocr.json"
            try:
                with open(back_json_path, "w", encoding="utf-8") as f:
                    json.dump({"back_entities": back_entities_data}, f, indent=4)
                print(f"✅ Back entities saved to JSON: {back_json_path}")
            except Exception as e:
                print(f"❌ Error saving back entities JSON {back_json_path}: {e}", file=sys.stderr)
                traceback.print_exc()


        print("-" * 60)
        if front_entities_data is not None and back_entities_data is not None:
             print("🎉 Pipeline completed successfully for both images!")
        else:
             print("⚠️ Pipeline completed with some processing failures. Check logs and debug output.")
        print(f"Output saved in: {session_dir}")
        print("--- Pipeline Finished ---")


    except Exception as e:
        print(f"\n❌ An unexpected error occurred during the main pipeline execution: {e}", file=sys.stderr)
        traceback.print_exc()
        print(f"Processing halted. Output directory might be partially created: {session_dir}")
        print("--- Pipeline Aborted ---")


# --- Execution Block ---
if __name__ == "__main__":
    main()
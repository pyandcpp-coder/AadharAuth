from ultralytics import YOLO
from pathlib import Path
import cv2
import os
import numpy as np
import argparse
from datetime import datetime
import shutil
import pytesseract
from PIL import Image
import json
import sys
import traceback

class ComprehensiveAadhaarPipeline:
    def __init__(self, model1_path, model2_path, output_base_dir="pipeline_output", confidence_threshold=0.4):
        """
        Initialize the comprehensive Aadhaar processing pipeline
        
        Args:
            model1_path (str): Path to the first model (front/back detection)
            model2_path (str): Path to the second model (entity detection)
            output_base_dir (str): Base output directory for saving results
            confidence_threshold (float): Minimum confidence threshold for detections
        """
        self.model1_path = model1_path
        self.model2_path = model2_path
        self.output_base_dir = Path(output_base_dir)
        self.confidence_threshold = confidence_threshold
        
        # Check Tesseract availability
        self._check_tesseract()
        
        # Load YOLOv8 models
        print("Loading YOLOv8 models...")
        self.model1 = YOLO(model1_path)  # Aadhaar front/back detection
        self.model2 = YOLO(model2_path)  # Entity detection
        print("Models loaded successfully!")
        
        # Setup directories
        self.setup_directories()
        
        # Define class names based on your actual model classes
        self.card_classes = {0: 'aadhar_front', 1: 'aadhar_back', 2: 'print_aadhar'}  # Model1 classes
        self.entity_classes = {
            0: 'aadharNumber', 1: 'address', 2: 'address_other_lang', 3: 'city',
            4: 'dob', 5: 'gender', 6: 'gender_other_lang', 7: 'mobile_no',
            8: 'name', 9: 'name_otherlang', 10: 'pincode', 11: 'state'
        }  # Model2 classes
    
    def _check_tesseract(self):
        """Check if Tesseract is available"""
        try:
            pytesseract.get_tesseract_version()
            print("✅ Tesseract is installed and accessible.")
        except pytesseract.TesseractNotFoundError:
            print("❌ Tesseract executable not found.", file=sys.stderr)
            print("Please install Tesseract OCR and ensure it's in your PATH.", file=sys.stderr)
            sys.exit(1)
    
    def setup_directories(self):
        """Create the required directory structure"""
        # Create timestamped session directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.output_base_dir / f"session_{timestamp}"
        
        # Create the directories
        self.front_back_dir = self.session_dir / "1_front_back_cards"
        self.detected_entities_dir = self.session_dir / "2_detected_entities"
        self.cropped_entities_dir = self.session_dir / "3_cropped_entities"
        self.preprocessed_entities_dir = self.session_dir / "4_preprocessed_entities"
        
        # Create all directories
        for directory in [self.front_back_dir, self.detected_entities_dir, 
                         self.cropped_entities_dir, self.preprocessed_entities_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        print(f"Created output directories in: {self.session_dir}")
    
    def detect_and_crop_cards(self, image_paths):
        """
        Step 1: Detect Aadhaar front and back cards and crop them from multiple images
        
        Args:
            image_paths (list): List of input image paths
            
        Returns:
            dict: Dictionary with 'front' and 'back' keys containing cropped card paths
        """
        print(f"\n🔍 Step 1: Detecting Aadhaar cards in {len(image_paths)} images")
        
        cropped_cards = {'front': [], 'back': []}
        
        for image_path in image_paths:
            print(f"  Processing: {Path(image_path).name}")
            
            # Run inference with model1
            results = self.model1(str(image_path))
            result = results[0]
            
            # Load original image
            img = cv2.imread(str(image_path))
            input_filename = Path(image_path).stem
            
            # Process detections
            for i, box in enumerate(result.boxes):
                conf = float(box.conf[0])
                if conf < self.confidence_threshold:
                    continue
                
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Crop the card
                crop = img[y1:y2, x1:x2]
                
                # Get class information
                cls_id = int(box.cls[0])
                class_name = self.card_classes.get(cls_id, f"class_{cls_id}")
                
                # Check if it's a print Aadhaar - raise error if detected
                if class_name == 'print_aadhar':
                    error_msg = f"❌ PRINT AADHAAR DETECTED! Cannot process print/photocopy of Aadhaar card (confidence: {conf:.2f})"
                    print(error_msg)
                    raise ValueError("Print Aadhaar detected - processing stopped for security reasons")
                
                # Only process aadhar_front and aadhar_back
                if class_name not in ['aadhar_front', 'aadhar_back']:
                    print(f"    ⚠️  Skipping unknown class: {class_name}")
                    continue
                
                # Save cropped card
                crop_filename = self.front_back_dir / f"{input_filename}_{class_name}_conf{int(conf*100)}.jpg"
                cv2.imwrite(str(crop_filename), crop)
                
                # Categorize by front/back
                if class_name == 'aadhar_front':
                    cropped_cards['front'].append(crop_filename)
                elif class_name == 'aadhar_back':
                    cropped_cards['back'].append(crop_filename)
                
                print(f"    ✅ Saved {class_name}: {crop_filename.name}")
        
        total_cards = len(cropped_cards['front']) + len(cropped_cards['back'])
        print(f"  📊 Total cards detected: {total_cards} (Front: {len(cropped_cards['front'])}, Back: {len(cropped_cards['back'])})")
        return cropped_cards
    
    def detect_entities_in_cards(self, cropped_cards):
        """
        Step 2: Detect entities in cropped cards and save images with bounding boxes
        
        Args:
            cropped_cards (dict): Dictionary with 'front' and 'back' keys containing card paths
            
        Returns:
            dict: Dictionary mapping card paths to their entity detections
        """
        all_card_paths = cropped_cards['front'] + cropped_cards['back']
        print(f"\n🔍 Step 2: Detecting entities in {len(all_card_paths)} cards")
        
        all_detections = {}
        
        for card_path in all_card_paths:
            print(f"  Processing: {card_path.name}")
            
            # Run inference with model2
            results = self.model2(str(card_path))
            result = results[0]
            
            # Load card image
            img = cv2.imread(str(card_path))
            img_with_boxes = img.copy()
            
            card_detections = []
            
            # Process entity detections
            for i, box in enumerate(result.boxes):
                conf = float(box.conf[0])
                if conf < self.confidence_threshold:
                    continue
                
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Get class information
                cls_id = int(box.cls[0])
                class_name = self.entity_classes.get(cls_id, f"entity_{cls_id}")
                
                # Draw bounding box on image
                cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img_with_boxes, f"{class_name}: {conf:.2f}", 
                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Store detection info
                card_detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'class_id': cls_id,
                    'class_name': class_name,
                    'confidence': conf
                })
            
            # Save image with bounding boxes
            card_stem = card_path.stem
            detection_filename = self.detected_entities_dir / f"{card_stem}_with_entities.jpg"
            cv2.imwrite(str(detection_filename), img_with_boxes)
            
            all_detections[card_path] = card_detections
            print(f"    ✅ Detected {len(card_detections)} entities, saved: {detection_filename.name}")
        
        return all_detections
    
    def crop_entities(self, all_detections):
        """
        Step 3: Crop individual entities and save them
        
        Args:
            all_detections (dict): Dictionary mapping card paths to their entity detections
            
        Returns:
            list: List of cropped entity file paths
        """
        print(f"\n✂️  Step 3: Cropping individual entities")
        
        total_entities = 0
        cropped_entity_paths = []
        
        for card_path, detections in all_detections.items():
            if not detections:
                continue
            
            # Load original card image
            img = cv2.imread(str(card_path))
            card_stem = card_path.stem
            
            print(f"  Processing entities from: {card_path.name}")
            
            for i, detection in enumerate(detections):
                x1, y1, x2, y2 = detection['bbox']
                class_name = detection['class_name']
                conf = detection['confidence']
                
                # Crop entity
                entity_crop = img[y1:y2, x1:x2]
                
                # Create filename for cropped entity
                entity_filename = self.cropped_entities_dir / f"{card_stem}_{class_name}_{i}_conf{int(conf*100)}.jpg"
                
                # Save cropped entity
                cv2.imwrite(str(entity_filename), entity_crop)
                cropped_entity_paths.append(entity_filename)
                total_entities += 1
                
                print(f"    ✅ Saved entity: {entity_filename.name}")
        
        print(f"  📊 Total entities cropped: {total_entities}")
        return cropped_entity_paths
    
    def preprocess_image_for_ocr(self, image_path):
        """
        Preprocess image for better OCR accuracy
        
        Args:
            image_path (Path): Path to the image file
            
        Returns:
            PIL.Image.Image | None: Preprocessed image or None if failed
        """
        try:
            # Open the image file
            img = Image.open(image_path)
            
            # Convert to grayscale
            img = img.convert('L')
            
            # Resize for better OCR (max side 1200px)
            max_side = 1200
            w, h = img.size
            if max(w, h) > max_side:
                scale = max_side / float(max(w, h))
                new_size = (int(w * scale), int(h * scale))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            return img
            
        except Exception as e:
            print(f"    ❌ Error preprocessing image {image_path.name}: {e}")
            return None
    
    def perform_ocr_on_entities(self, cropped_entity_paths):
        """
        Step 4: Perform OCR on cropped entities
        
        Args:
            cropped_entity_paths (list): List of cropped entity file paths
            
        Returns:
            dict: Dictionary mapping filenames to extracted text
        """
        print(f"\n📝 Step 4: Performing OCR on {len(cropped_entity_paths)} cropped entities")
        
        ocr_results = {}
        
        for entity_path in cropped_entity_paths:
            print(f"  Processing: {entity_path.name}")
            
            # Preprocess image
            processed_img = self.preprocess_image_for_ocr(entity_path)
            
            if processed_img:
                try:
                    # Save preprocessed image
                    preprocessed_filename = entity_path.stem + "_preprocessed.png"
                    preprocessed_save_path = self.preprocessed_entities_dir / preprocessed_filename
                    processed_img.save(str(preprocessed_save_path))
                    
                    # Perform OCR
                    text = pytesseract.image_to_string(processed_img, lang='eng', config='--psm 6')
                    extracted_text = text.strip()
                    
                    # Display snippet
                    display_text = extracted_text.replace('\n', ' ').replace('\r', ' ').strip()
                    display_snippet = (display_text[:50] + '...') if len(display_text) > 50 else display_text
                    print(f"    ✅ Extracted: \"{display_snippet}\"")
                    
                    ocr_results[entity_path.name] = extracted_text
                    
                except Exception as e:
                    print(f"    ❌ OCR failed for {entity_path.name}: {e}")
                    ocr_results[entity_path.name] = None
            else:
                print(f"    ⚠️  Skipping OCR due to preprocessing failure")
                ocr_results[entity_path.name] = None
        
        successful_ocr = sum(1 for text in ocr_results.values() if text is not None)
        print(f"  📊 OCR completed: {successful_ocr}/{len(ocr_results)} successful")
        
        return ocr_results
    
    def organize_results_by_card_type(self, cropped_cards, all_detections, ocr_results):
        """
        Organize results by front and back cards
        
        Args:
            cropped_cards (dict): Dictionary with front/back card paths
            all_detections (dict): All entity detections
            ocr_results (dict): OCR results
            
        Returns:
            dict: Organized results with front and back sections
        """
        organized_results = {
            'front': {},
            'back': {},
            'metadata': {
                'processing_timestamp': datetime.now().isoformat(),
                'session_directory': str(self.session_dir),
                'total_cards_processed': len(cropped_cards['front']) + len(cropped_cards['back']),
                'confidence_threshold': self.confidence_threshold
            }
        }
        
        # Process front cards
        for card_path in cropped_cards['front']:
            card_key = card_path.stem
            organized_results['front'][card_key] = {
                'card_info': {
                    'filename': card_path.name,
                    'card_type': 'front'
                },
                'entities': {}
            }
            
            # Get detections and OCR results for this card
            if card_path in all_detections:
                for detection in all_detections[card_path]:
                    entity_name = detection['class_name']
                    
                    # Find corresponding OCR result
                    entity_text = None
                    for filename, text in ocr_results.items():
                        if card_key in filename and entity_name in filename:
                            entity_text = text
                            break
                    
                    organized_results['front'][card_key]['entities'][entity_name] = {
                        'confidence': detection['confidence'],
                        'bbox': detection['bbox'],
                        'extracted_text': entity_text
                    }
        
        # Process back cards
        for card_path in cropped_cards['back']:
            card_key = card_path.stem
            organized_results['back'][card_key] = {
                'card_info': {
                    'filename': card_path.name,
                    'card_type': 'back'
                },
                'entities': {}
            }
            
            # Get detections and OCR results for this card
            if card_path in all_detections:
                for detection in all_detections[card_path]:
                    entity_name = detection['class_name']
                    
                    # Find corresponding OCR result
                    entity_text = None
                    for filename, text in ocr_results.items():
                        if card_key in filename and entity_name in filename:
                            entity_text = text
                            break
                    
                    organized_results['back'][card_key]['entities'][entity_name] = {
                        'confidence': detection['confidence'],
                        'bbox': detection['bbox'],
                        'extracted_text': entity_text
                    }
        
        return organized_results
    
    def save_results_to_json(self, organized_results, filename="complete_aadhaar_results.json"):
        """
        Save organized results to JSON file
        
        Args:
            organized_results (dict): Organized results dictionary
            filename (str): Output JSON filename
        """
        output_path = self.session_dir / filename
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(organized_results, f, indent=4, ensure_ascii=False)
            
            print(f"\n💾 Results saved to: {output_path}")
            return output_path
        except Exception as e:
            print(f"❌ Error saving results to JSON: {e}")
            return None
    
    def process_images(self, image_paths, verbose=True):
        """
        Main pipeline function to process multiple images
        
        Args:
            image_paths (list): List of input image paths
            verbose (bool): Whether to print detailed progress
            
        Returns:
            dict: Complete processing results
        """
        if verbose:
            print(f"🚀 Starting Comprehensive Aadhaar Processing Pipeline")
            print(f"📁 Input images: {len(image_paths)} files")
            print(f"📁 Output directory: {self.session_dir}")
            print("-" * 60)
        
        try:
            # Step 1: Detect and crop cards
            cropped_cards = self.detect_and_crop_cards(image_paths)
            
            total_cards = len(cropped_cards['front']) + len(cropped_cards['back'])
            if total_cards == 0:
                print("❌ No Aadhaar cards detected in the images!")
                return None
            
            # Step 2: Detect entities in cards
            all_detections = self.detect_entities_in_cards(cropped_cards)
            
            # Step 3: Crop entities
            cropped_entity_paths = self.crop_entities(all_detections)
            
            # Step 4: Perform OCR on entities
            ocr_results = self.perform_ocr_on_entities(cropped_entity_paths)
            
            # Step 5: Organize and save results
            organized_results = self.organize_results_by_card_type(cropped_cards, all_detections, ocr_results)
            json_path = self.save_results_to_json(organized_results)
            
            if verbose:
                print("-" * 60)
                print("🎉 Pipeline completed successfully!")
                print(f"📁 Check results in: {self.session_dir}")
                self.print_summary(organized_results)
            
            return {
                'session_dir': self.session_dir,
                'json_results_path': json_path,
                'organized_results': organized_results,
                'cropped_cards': cropped_cards,
                'ocr_results': ocr_results
            }
            
        except ValueError as ve:
            print(f"🚫 SECURITY ERROR: {str(ve)}")
            return None
        except Exception as e:
            print(f"❌ Error in pipeline: {str(e)}")
            traceback.print_exc()
            return None
    
    def print_summary(self, organized_results):
        """Print a summary of the processing results"""
        print("\n📊 PROCESSING SUMMARY:")
        
        front_cards = len(organized_results['front'])
        back_cards = len(organized_results['back'])
        total_entities = 0
        
        for card_type in ['front', 'back']:
            for card_data in organized_results[card_type].values():
                total_entities += len(card_data['entities'])
        
        print(f"  • Front cards: {front_cards}")
        print(f"  • Back cards: {back_cards}")
        print(f"  • Total entities extracted: {total_entities}")
        
        print(f"\n📂 OUTPUT STRUCTURE:")
        print(f"  {self.session_dir}/")
        print(f"  ├── 1_front_back_cards/")
        print(f"  ├── 2_detected_entities/")
        print(f"  ├── 3_cropped_entities/")
        print(f"  ├── 4_preprocessed_entities/")
        print(f"  └── complete_aadhaar_results.json")

def main():
    """Command line interface for the pipeline"""
    parser = argparse.ArgumentParser(description="Comprehensive Aadhaar Processing Pipeline")
    parser.add_argument("--images", "-i", nargs='+', required=True, help="Paths to input images (front and back)")
    parser.add_argument("--model1", "-m1", required=True, help="Path to model1 (card detection)")
    parser.add_argument("--model2", "-m2", required=True, help="Path to model2 (entity detection)")
    parser.add_argument("--output", "-o", default="pipeline_output", help="Output directory")
    parser.add_argument("--confidence", "-c", type=float, default=0.4, help="Confidence threshold")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = ComprehensiveAadhaarPipeline(
        model1_path=args.model1,
        model2_path=args.model2,
        output_base_dir=args.output,
        confidence_threshold=args.confidence
    )
    
    # Process images
    results = pipeline.process_images(args.images)
    
    if results:
        print(f"\n✅ Processing completed successfully!")
        print(f"📄 JSON results saved at: {results['json_results_path']}")
    else:
        print(f"\n❌ Processing failed!")

if __name__ == "__main__":
    # Example usage for testing (uncomment to test)
    # pipeline = ComprehensiveAadhaarPipeline(
    #     model1_path="/path/to/model1.pt",
    #     model2_path="/path/to/model2.pt"
    # )
    # 
    # # Process both front and back images
    # image_paths = [
    #     "/path/to/front_image.jpg",
    #     "/path/to/back_image.jpg"
    # ]
    # results = pipeline.process_images(image_paths)
    
    main()
from ultralytics import YOLO
from pathlib import Path
import cv2
import os
import numpy as np
import argparse
from datetime import datetime
import shutil

class AadhaarAuthenticationPipeline:
    def __init__(self, model1_path, model2_path, output_base_dir="pipeline_output", confidence_threshold=0.4):
        """
        Initialize the Aadhaar authentication pipeline using YOLOv8
        
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
    
    def setup_directories(self):
        """Create the required directory structure"""
        # Create timestamped session directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.output_base_dir / f"session_{timestamp}"
        
        # Create the three main directories
        self.front_back_dir = self.session_dir / "1_front_back_cards"
        self.detected_entities_dir = self.session_dir / "2_detected_entities"
        self.cropped_entities_dir = self.session_dir / "3_cropped_entities"
        
        # Create all directories
        for directory in [self.front_back_dir, self.detected_entities_dir, self.cropped_entities_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        print(f"Created output directories in: {self.session_dir}")
    
    def detect_and_crop_cards(self, image_path):
        """
        Step 1: Detect Aadhaar front and back cards and crop them
        
        Args:
            image_path (str): Path to input image
            
        Returns:
            list: List of cropped card image paths
        """
        print(f"\n🔍 Step 1: Detecting Aadhaar cards in {image_path}")
        
        # Run inference with model1
        results = self.model1(str(image_path))
        result = results[0]
        
        # Load original image
        img = cv2.imread(str(image_path))
        input_filename = Path(image_path).stem
        
        cropped_cards = []
        
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
                print(f"  ⚠️  Skipping unknown class: {class_name}")
                continue
            
            # Save cropped card
            crop_filename = self.front_back_dir / f"{input_filename}_{class_name}_conf{int(conf*100)}.jpg"
            cv2.imwrite(str(crop_filename), crop)
            cropped_cards.append(crop_filename)
            
            print(f"  ✅ Saved {class_name}: {crop_filename.name}")
        
        print(f"  📊 Total cards detected: {len(cropped_cards)}")
        return cropped_cards
    
    def detect_entities_in_cards(self, card_paths):
        """
        Step 2: Detect entities in cropped cards and save images with bounding boxes
        
        Args:
            card_paths (list): List of cropped card image paths
            
        Returns:
            dict: Dictionary mapping card paths to their entity detections
        """
        print(f"\n🔍 Step 2: Detecting entities in {len(card_paths)} cards")
        
        all_detections = {}
        
        for card_path in card_paths:
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
        """
        print(f"\n✂️  Step 3: Cropping individual entities")
        
        total_entities = 0
        
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
                total_entities += 1
                
                print(f"    ✅ Saved entity: {entity_filename.name}")
        
        print(f"  📊 Total entities cropped: {total_entities}")
    
    def process_image(self, image_path, verbose=True):
        """
        Main pipeline function to process a single image
        
        Args:
            image_path (str): Path to input image
            verbose (bool): Whether to print detailed progress
            
        Returns:
            dict: Processing results and file paths
        """
        if verbose:
            print(f"🚀 Starting Aadhaar Authentication Pipeline")
            print(f"📁 Input image: {image_path}")
            print(f"📁 Output directory: {self.session_dir}")
            print("-" * 60)
        
        try:
            # Step 1: Detect and crop cards
            cropped_cards = self.detect_and_crop_cards(image_path)
            
            if not cropped_cards:
                print("❌ No Aadhaar cards detected in the image!")
                return None
            
            # Step 2: Detect entities in cards
            all_detections = self.detect_entities_in_cards(cropped_cards)
            
            # Step 3: Crop entities
            self.crop_entities(all_detections)
            
            results = {
                'session_dir': self.session_dir,
                'cropped_cards': cropped_cards,
                'detections': all_detections,
                'output_dirs': {
                    'front_back': self.front_back_dir,
                    'detected_entities': self.detected_entities_dir,
                    'cropped_entities': self.cropped_entities_dir
                }
            }
            
            if verbose:
                print("-" * 60)
                print("🎉 Pipeline completed successfully!")
                print(f"📁 Check results in: {self.session_dir}")
                self.print_summary(results)
            
            return results
            
        except ValueError as ve:
            # Handle print Aadhaar detection specifically
            print(f"🚫 SECURITY ERROR: {str(ve)}")
            return None
        except Exception as e:
            print(f"❌ Error in pipeline: {str(e)}")
            return None
    
    def print_summary(self, results):
        """Print a summary of the processing results"""
        print("\n📊 PROCESSING SUMMARY:")
        print(f"  • Cards detected: {len(results['cropped_cards'])}")
        
        total_entities = sum(len(detections) for detections in results['detections'].values())
        print(f"  • Total entities detected: {total_entities}")
        
        print(f"\n📂 OUTPUT STRUCTURE:")
        print(f"  {results['session_dir']}/")
        print(f"  ├── 1_front_back_cards/     ({len(list(results['output_dirs']['front_back'].glob('*.jpg')))} files)")
        print(f"  ├── 2_detected_entities/    ({len(list(results['output_dirs']['detected_entities'].glob('*.jpg')))} files)")
        print(f"  └── 3_cropped_entities/     ({len(list(results['output_dirs']['cropped_entities'].glob('*.jpg')))} files)")

def main():
    """Command line interface for the pipeline"""
    parser = argparse.ArgumentParser(description="Aadhaar Authentication Pipeline")
    parser.add_argument("--image", "-i", required=True, help="Path to input image")
    parser.add_argument("--model1", "-m1", required=True, help="Path to model1 (card detection)")
    parser.add_argument("--model2", "-m2", required=True, help="Path to model2 (entity detection)")
    parser.add_argument("--output", "-o", default="pipeline_output", help="Output directory")
    parser.add_argument("--confidence", "-c", type=float, default=0.4, help="Confidence threshold")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = AadhaarAuthenticationPipeline(
        model1_path=args.model1,
        model2_path=args.model2,
        output_base_dir=args.output,
        confidence_threshold=args.confidence
    )
    
    # Process image
    pipeline.process_image(args.image)

if __name__ == "__main__":
    # Example usage (uncomment to test)
    pipeline = AadhaarAuthenticationPipeline(
        model1_path="/Users/yrevash/Downloads/aadhar_models/aadhar_model_1/best-4.pt",
        model2_path="/Users/yrevash/Downloads/aadhar_models/aadhar_model_2/best-5.pt"
    )
    pipeline.process_image("/Users/yrevash/India/aadhar_images_extract/_front_1632294719.jpg")
    
    main()
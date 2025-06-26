# #!/usr/bin/env python3
# """
# Aadhaar OCR Processor using Docling
# Processes cropped entity images with preprocessing and OCR extraction
# """

# import cv2
# import numpy as np
# import os
# import json
# from pathlib import Path
# from datetime import datetime
# import subprocess
# import tempfile
# import shutil
# from typing import Dict, List, Tuple, Optional
# import argparse

# class AadhaarOCRProcessor:
#     def __init__(self, input_dir: str, output_dir: str = "ocr_results", preprocessing: bool = True):
#         """
#         Initialize the OCR processor
        
#         Args:
#             input_dir (str): Directory containing cropped entity images
#             output_dir (str): Directory to save OCR results
#             preprocessing (bool): Whether to apply image preprocessing
#         """
#         self.input_dir = Path(input_dir)
#         self.output_dir = Path(output_dir)
#         self.preprocessing = preprocessing
        
#         # Create output directories
#         self.setup_directories()
        
#         # Define entity types for better organization
#         self.entity_types = [
#             'aadharNumber', 'address', 'address_other_lang', 'city',
#             'dob', 'gender', 'gender_other_lang', 'mobile_no',
#             'name', 'name_otherlang', 'pincode', 'state'
#         ]
        
#         # OCR results storage
#         self.ocr_results = {}
        
#     def setup_directories(self):
#         """Create necessary output directories"""
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         self.session_dir = self.output_dir / f"ocr_session_{timestamp}"
        
#         # Create subdirectories
#         self.preprocessed_dir = self.session_dir / "1_preprocessed_images"
#         self.ocr_outputs_dir = self.session_dir / "2_ocr_outputs"
#         self.results_dir = self.session_dir / "3_structured_results"
        
#         for directory in [self.preprocessed_dir, self.ocr_outputs_dir, self.results_dir]:
#             directory.mkdir(parents=True, exist_ok=True)
        
#         print(f"📁 Created OCR session directory: {self.session_dir}")
    
#     def preprocess_image(self, image_path: Path) -> np.ndarray:
#         """
#         Apply preprocessing to improve OCR accuracy
        
#         Args:
#             image_path (Path): Path to input image
            
#         Returns:
#             np.ndarray: Preprocessed image
#         """
#         # Read image
#         img = cv2.imread(str(image_path))
#         if img is None:
#             raise ValueError(f"Could not load image: {image_path}")
        
#         # Convert to grayscale
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
#         # Apply different preprocessing based on image characteristics
#         height, width = gray.shape
        
#         # Resize if image is too small (minimum 100px height for better OCR)
#         if height < 100:
#             scale_factor = 100 / height
#             new_width = int(width * scale_factor)
#             gray = cv2.resize(gray, (new_width, 100), interpolation=cv2.INTER_CUBIC)
        
#         # Apply preprocessing techniques
#         processed = self.apply_preprocessing_pipeline(gray)
        
#         return processed
    
#     def apply_preprocessing_pipeline(self, gray_image: np.ndarray) -> np.ndarray:
#         """
#         Apply comprehensive preprocessing pipeline
        
#         Args:
#             gray_image (np.ndarray): Grayscale input image
            
#         Returns:
#             np.ndarray: Processed image
#         """
#         # 1. Noise reduction using bilateral filter
#         denoised = cv2.bilateralFilter(gray_image, 9, 75, 75)
        
#         # 2. Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
#         clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#         enhanced = clahe.apply(denoised)
        
#         # 3. Morphological operations to clean up text
#         kernel = np.ones((1, 1), np.uint8)
#         morphed = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
        
#         # 4. Sharpening filter
#         kernel_sharpen = np.array([[-1, -1, -1],
#                                  [-1,  9, -1],
#                                  [-1, -1, -1]])
#         sharpened = cv2.filter2D(morphed, -1, kernel_sharpen)
        
#         # 5. Threshold for better text clarity
#         # Use adaptive threshold for varying light conditions
#         thresh = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
#                                      cv2.THRESH_BINARY, 11, 2)
        
#         return thresh
    
#     def save_preprocessed_image(self, processed_img: np.ndarray, original_path: Path) -> Path:
#         """
#         Save preprocessed image
        
#         Args:
#             processed_img (np.ndarray): Processed image
#             original_path (Path): Original image path
            
#         Returns:
#             Path: Path to saved preprocessed image
#         """
#         output_path = self.preprocessed_dir / f"preprocessed_{original_path.name}"
#         cv2.imwrite(str(output_path), processed_img)
#         return output_path
    
#     def run_docling_ocr(self, image_path: Path) -> Dict:
#         """
#         Run Docling OCR on the image
        
#         Args:
#             image_path (Path): Path to image file
            
#         Returns:
#             Dict: OCR results from Docling
#         """
#         try:
#             # Create temporary output directory for this image
#             temp_dir = tempfile.mkdtemp()
            
#             # Run Docling command
#             cmd = [
#                 "docling",
#                 str(image_path),
#                 "--to", "json",
#                 "--output", temp_dir,
#                 "--ocr-engine", "tesseract",  # You can change to tesseract if preferred
#                 "--ocr-lang", "en",  # Add more languages if needed (e.g., "en,hi")
#                 "--force-ocr",
#                 "--verbose"
#             ]
            
#             print(f"  🔄 Running OCR on: {image_path.name}")
#             result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
#             # Find the output JSON file
#             json_files = list(Path(temp_dir).glob("*.json"))
#             if not json_files:
#                 raise FileNotFoundError("No JSON output found from Docling")
            
#             # Read the OCR results
#             with open(json_files[0], 'r', encoding='utf-8') as f:
#                 ocr_data = json.load(f)
            
#             # Copy the JSON output to our results directory
#             output_json_path = self.ocr_outputs_dir / f"{image_path.stem}_ocr.json"
#             shutil.copy2(json_files[0], output_json_path)
            
#             # Clean up temp directory
#             shutil.rmtree(temp_dir)
            
#             return ocr_data
            
#         except subprocess.CalledProcessError as e:
#             print(f"  ❌ Docling OCR failed for {image_path.name}: {e.stderr}")
#             return {"error": str(e), "text": ""}
#         except Exception as e:
#             print(f"  ❌ OCR processing error for {image_path.name}: {str(e)}")
#             return {"error": str(e), "text": ""}
    
#     def extract_text_from_ocr_result(self, ocr_data: Dict) -> str:
#         """
#         Extract clean text from Docling OCR results
        
#         Args:
#             ocr_data (Dict): OCR result from Docling
            
#         Returns:
#             str: Extracted text
#         """
#         try:
#             if "error" in ocr_data:
#                 return ""
            
#             # Docling typically stores text in 'main_text' or similar field
#             # Check common fields where text might be stored
#             text_fields = ['main_text', 'text', 'content', 'body']
            
#             extracted_text = ""
            
#             # Try to extract text from different possible fields
#             for field in text_fields:
#                 if field in ocr_data and ocr_data[field]:
#                     extracted_text = str(ocr_data[field])
#                     break
            
#             # If no direct text field, try to extract from document structure
#             if not extracted_text and 'document' in ocr_data:
#                 doc = ocr_data['document']
#                 if isinstance(doc, dict) and 'text' in doc:
#                     extracted_text = doc['text']
#                 elif isinstance(doc, list):
#                     extracted_text = ' '.join([str(item.get('text', '')) for item in doc if isinstance(item, dict)])
            
#             # Clean up the text
#             extracted_text = extracted_text.strip()
#             extracted_text = ' '.join(extracted_text.split())  # Remove extra whitespace
            
#             return extracted_text
            
#         except Exception as e:
#             print(f"  ⚠️  Error extracting text: {str(e)}")
#             return ""
    
#     def categorize_entity(self, filename: str) -> str:
#         """
#         Determine entity type from filename
        
#         Args:
#             filename (str): Image filename
            
#         Returns:
#             str: Entity type
#         """
#         filename_lower = filename.lower()
        
#         for entity_type in self.entity_types:
#             if entity_type.lower() in filename_lower:
#                 return entity_type
        
#         return "unknown"
    
#     def process_all_images(self) -> Dict:
#         """
#         Process all images in the input directory
        
#         Returns:
#             Dict: Complete processing results
#         """
#         print(f"🚀 Starting OCR processing for images in: {self.input_dir}")
#         print(f"📁 Output directory: {self.session_dir}")
#         print(f"🔧 Preprocessing enabled: {self.preprocessing}")
#         print("-" * 60)
        
#         # Find all image files
#         image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
#         image_files = []
        
#         for ext in image_extensions:
#             image_files.extend(self.input_dir.glob(f"*{ext}"))
#             image_files.extend(self.input_dir.glob(f"*{ext.upper()}"))
        
#         if not image_files:
#             print("❌ No image files found in input directory!")
#             return {}
        
#         print(f"📊 Found {len(image_files)} images to process")
        
#         results = {
#             'session_info': {
#                 'timestamp': datetime.now().isoformat(),
#                 'input_dir': str(self.input_dir),
#                 'output_dir': str(self.session_dir),
#                 'preprocessing_enabled': self.preprocessing,
#                 'total_images': len(image_files)
#             },
#             'entities': {},
#             'summary': {}
#         }
        
#         # Process each image
#         for i, image_path in enumerate(image_files, 1):
#             print(f"\n🔍 Processing {i}/{len(image_files)}: {image_path.name}")
            
#             try:
#                 # Determine entity type
#                 entity_type = self.categorize_entity(image_path.name)
                
#                 # Preprocess image if enabled
#                 if self.preprocessing:
#                     print(f"  🔧 Preprocessing image...")
#                     processed_img = self.preprocess_image(image_path)
#                     processed_path = self.save_preprocessed_image(processed_img, image_path)
#                     ocr_input_path = processed_path
#                 else:
#                     ocr_input_path = image_path
                
#                 # Run OCR
#                 ocr_data = self.run_docling_ocr(ocr_input_path)
#                 extracted_text = self.extract_text_from_ocr_result(ocr_data)
                
#                 # Store results
#                 results['entities'][image_path.name] = {
#                     'entity_type': entity_type,
#                     'original_path': str(image_path),
#                     'preprocessed_path': str(ocr_input_path) if self.preprocessing else None,
#                     'extracted_text': extracted_text,
#                     'text_length': len(extracted_text),
#                     'processing_status': 'success' if extracted_text else 'no_text_found'
#                 }
                
#                 print(f"  ✅ Extracted text ({len(extracted_text)} chars): {extracted_text[:100]}{'...' if len(extracted_text) > 100 else ''}")
                
#             except Exception as e:
#                 print(f"  ❌ Error processing {image_path.name}: {str(e)}")
#                 results['entities'][image_path.name] = {
#                     'entity_type': entity_type,
#                     'original_path': str(image_path),
#                     'error': str(e),
#                     'processing_status': 'error'
#                 }
        
#         # Generate summary
#         successful = sum(1 for r in results['entities'].values() if r.get('processing_status') == 'success')
#         no_text = sum(1 for r in results['entities'].values() if r.get('processing_status') == 'no_text_found')
#         errors = sum(1 for r in results['entities'].values() if r.get('processing_status') == 'error')
        
#         results['summary'] = {
#             'total_processed': len(image_files),
#             'successful_extractions': successful,
#             'no_text_found': no_text,
#             'errors': errors,
#             'success_rate': f"{(successful/len(image_files)*100):.1f}%" if image_files else "0%"
#         }
        
#         # Save structured results
#         self.save_results(results)
        
#         return results
    
#     def save_results(self, results: Dict):
#         """
#         Save processing results to files
        
#         Args:
#             results (Dict): Processing results
#         """
#         # Save complete results as JSON
#         json_path = self.results_dir / "complete_ocr_results.json"
#         with open(json_path, 'w', encoding='utf-8') as f:
#             json.dump(results, f, indent=2, ensure_ascii=False)
        
#         # Save text-only results
#         text_results = {}
#         for filename, data in results['entities'].items():
#             if data.get('extracted_text'):
#                 entity_type = data.get('entity_type', 'unknown')
#                 if entity_type not in text_results:
#                     text_results[entity_type] = []
#                 text_results[entity_type].append({
#                     'filename': filename,
#                     'text': data['extracted_text']
#                 })
        
#         text_path = self.results_dir / "extracted_texts.json"
#         with open(text_path, 'w', encoding='utf-8') as f:
#             json.dump(text_results, f, indent=2, ensure_ascii=False)
        
#         # Save summary report
#         self.save_summary_report(results)
    
#     def save_summary_report(self, results: Dict):
#         """
#         Save a human-readable summary report
        
#         Args:
#             results (Dict): Processing results
#         """
#         report_path = self.results_dir / "processing_report.txt"
        
#         with open(report_path, 'w', encoding='utf-8') as f:
#             f.write("AADHAAR OCR PROCESSING REPORT\n")
#             f.write("=" * 50 + "\n\n")
            
#             # Session info
#             session = results['session_info']
#             f.write(f"Session Timestamp: {session['timestamp']}\n")
#             f.write(f"Input Directory: {session['input_dir']}\n")
#             f.write(f"Output Directory: {session['output_dir']}\n")
#             f.write(f"Preprocessing: {'Enabled' if session['preprocessing_enabled'] else 'Disabled'}\n\n")
            
#             # Summary
#             summary = results['summary']
#             f.write("PROCESSING SUMMARY:\n")
#             f.write(f"• Total Images: {summary['total_processed']}\n")
#             f.write(f"• Successful Extractions: {summary['successful_extractions']}\n")
#             f.write(f"• No Text Found: {summary['no_text_found']}\n")
#             f.write(f"• Errors: {summary['errors']}\n")
#             f.write(f"• Success Rate: {summary['success_rate']}\n\n")
            
#             # Entity details
#             f.write("EXTRACTED TEXTS BY ENTITY:\n")
#             f.write("-" * 30 + "\n")
            
#             for filename, data in results['entities'].items():
#                 f.write(f"\nFile: {filename}\n")
#                 f.write(f"Entity Type: {data.get('entity_type', 'unknown')}\n")
#                 f.write(f"Status: {data.get('processing_status', 'unknown')}\n")
                
#                 if data.get('extracted_text'):
#                     f.write(f"Text: {data['extracted_text']}\n")
#                 elif data.get('error'):
#                     f.write(f"Error: {data['error']}\n")
                
#                 f.write("-" * 30 + "\n")
        
#         print(f"📄 Summary report saved: {report_path}")

# def main():
#     """Command line interface"""
#     parser = argparse.ArgumentParser(description="Aadhaar OCR Processor using Docling")
#     parser.add_argument("--input", "-i", required=True, help="Input directory with cropped entity images")
#     parser.add_argument("--output", "-o", default="ocr_results", help="Output directory for results")
#     parser.add_argument("--no-preprocessing", action="store_true", help="Disable image preprocessing")
    
#     args = parser.parse_args()
    
#     # Initialize processor
#     processor = AadhaarOCRProcessor(
#         input_dir=args.input,
#         output_dir=args.output,
#         preprocessing=not args.no_preprocessing
#     )
    
#     # Process all images
#     results = processor.process_all_images()
    
#     if results:
#         print(f"\n🎉 OCR processing completed!")
#         print(f"📊 Summary: {results['summary']}")
#         print(f"📁 Check results in: {processor.session_dir}")
#     else:
#         print(f"\n❌ OCR processing failed!")

# if __name__ == "__main__":
#     # Example usage (uncomment to test)
#     processor = AadhaarOCRProcessor(
#         input_dir="/Users/yrevash/Qoneqt/Test_model/pipeline_output/session_20250620_093304/3_cropped_entities",
#         output_dir="ocr_results"
#     )
#     results = processor.process_all_images()
    
































#!/usr/bin/env python3
"""
Aadhaar OCR Processor using Tesseract and EasyOCR (fallback)
Processes cropped entity images with preprocessing and OCR extraction
"""

import cv2
import numpy as np
import os
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import pytesseract

# Optional: EasyOCR fallback (install with `pip install easyocr`)
try:
    import easyocr
    EASYOCR_ENABLED = True
    reader = easyocr.Reader(['en', 'hi'], gpu=False)
except ImportError:
    EASYOCR_ENABLED = False

class AadhaarOCRProcessor:
    def __init__(self, input_dir: str, output_dir: str = "ocr_results", preprocessing: bool = True):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.preprocessing = preprocessing

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.output_dir / f"ocr_session_{timestamp}"
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self.preprocessed_dir = self.session_dir / "1_preprocessed_images"
        self.ocr_results_dir = self.session_dir / "2_ocr_outputs"
        self.preprocessed_dir.mkdir(exist_ok=True)
        self.ocr_results_dir.mkdir(exist_ok=True)

        self.image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

    def preprocess_image(self, image_path: Path) -> np.ndarray:
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Cannot load image: {image_path}")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Resize for minimum height
        if gray.shape[0] < 100:
            scale = 100 / gray.shape[0]
            gray = cv2.resize(gray, (int(gray.shape[1]*scale), 100), interpolation=cv2.INTER_CUBIC)

        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # Denoise and sharpen
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        gray = cv2.filter2D(gray, -1, kernel)

        # Adaptive threshold
        final = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, 2)
        return final

    def run_tesseract_ocr(self, image: np.ndarray) -> str:
        config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(image, lang='eng+hin', config=config)
        return text.strip()

    def run_easyocr(self, image: np.ndarray) -> str:
        if not EASYOCR_ENABLED:
            return ""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        result = reader.readtext(image_rgb, detail=0)
        return " ".join(result).strip()

    def process_images(self):
        image_files = [f for f in self.input_dir.iterdir() if f.suffix.lower() in self.image_exts]
        print(f"📊 Found {len(image_files)} images in {self.input_dir}")
        results = {}

        for i, img_path in enumerate(image_files, 1):
            print(f"\n🔍 [{i}/{len(image_files)}] Processing: {img_path.name}")
            try:
                # Preprocess
                processed_img = self.preprocess_image(img_path)
                pre_out_path = self.preprocessed_dir / f"pre_{img_path.name}"
                cv2.imwrite(str(pre_out_path), processed_img)

                # OCR with Tesseract
                text = self.run_tesseract_ocr(processed_img)

                if not text and EASYOCR_ENABLED:
                    print("  ⚠️ Tesseract failed, using EasyOCR fallback...")
                    text = self.run_easyocr(processed_img)

                results[img_path.name] = {
                    "ocr_text": text,
                    "length": len(text),
                    "status": "success" if text else "no_text"
                }

                print(f"  ✅ Extracted Text ({len(text)} chars): {text[:100]}{'...' if len(text) > 100 else ''}")
            except Exception as e:
                print(f"  ❌ Error: {e}")
                results[img_path.name] = {
                    "ocr_text": "",
                    "length": 0,
                    "status": "error",
                    "error_msg": str(e)
                }

        # Save results
        out_json = self.ocr_results_dir / "ocr_output.json"
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n📁 OCR results saved to: {out_json}")

def main():
    parser = argparse.ArgumentParser(description="Best Aadhaar OCR Pipeline")
    parser.add_argument("--input", "-i", required=True, help="Path to input image folder")
    parser.add_argument("--output", "-o", default="ocr_results", help="Path to output folder")
    parser.add_argument("--no-preprocessing", action="store_true", help="Disable preprocessing")

    args = parser.parse_args()

    processor = AadhaarOCRProcessor(
        input_dir=args.input,
        output_dir=args.output,
        preprocessing=not args.no_preprocessing
    )
    processor.process_images()

if __name__ == "__main__":
    main()

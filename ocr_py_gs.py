


import os
import pytesseract
from PIL import Image
import argparse
from pathlib import Path
import sys
import traceback
import json 

try:
    pytesseract.get_tesseract_version()
    print("✅ Tesseract is installed and accessible.")
except pytesseract.TesseractNotFoundError:
    print("❌ Tesseract executable not found.", file=sys.stderr)
    print("Please install Tesseract OCR and ensure it's in your PATH or set pytesseract.pytesseract.tesseract_cmd.", file=sys.stderr)
    sys.exit(1) # Exit if Tesseract is not found

def preprocess_image_for_ocr(image_path: Path) -> Image.Image | None:
    """
    Loads an image from a given path and applies basic preprocessing steps
    to potentially improve OCR accuracy.

    Args:
        image_path (Path): The path to the image file.

    Returns:
        PIL.Image.Image | None: The preprocessed PIL Image object, or None if loading fails.
    """
    try:
        # Open the image file
        img = Image.open(image_path)

        # Basic Preprocessing:
        # 1. Convert to grayscale: Often helps OCR accuracy.
        img = img.convert('L')

        # 2. Resize: Large images can be slow, small text might need upscaling.
        # Let's resize the longest side to a reasonable maximum (e.g., 1200px)
        # while maintaining aspect ratio. This balances clarity and speed.
        max_side = 1200
        w, h = img.size
        if max(w, h) > max_side:
            scale = max_side / float(max(w, h))
            new_size = (int(w * scale), int(h * scale))
            # Use a good resampling filter for text
            # PIL.Image.LANCZOS or PIL.Image.BICUBIC are good choices
            img = img.resize(new_size, Image.Resampling.LANCZOS)

        # 3. (Optional) Binarization: Convert to strict black and white.
        # This can sometimes help, but a simple global threshold might hurt.
        # For "basic", grayscale and resize are usually sufficient starters.
        # If you need it, uncomment the line below:
        # img = img.point(lambda x: 0 if x < 128 else 255, '1') # Simple threshold at 128

        return img

    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error loading or preprocessing image {image_path}: {e}", file=sys.stderr)
        # traceback.print_exc() # Uncomment for detailed exception stack
        return None

def extract_text_from_entity_images(input_dir: Path) -> dict:
    """
    Iterates through image files in a directory, performs OCR on each
    preprocessed image, saves the preprocessed image, and collects the results.

    Args:
        input_dir (Path): The path to the directory containing the cropped entity images
                          (assumed to be like .../session_XYZ/3_cropped_entities).

    Returns:
        dict: A dictionary where keys are image filenames and values are
              the extracted text strings, or None if extraction failed for that image.
              Returns an empty dict if the directory is invalid or no images are found/processed.
    """
    if not input_dir.is_dir():
        print(f"Error: Input path {input_dir} is not a valid directory.", file=sys.stderr)
        return {}

    print(f"🔍 Starting OCR on images in: {input_dir}")

    # Determine the directory to save preprocessed images
    # Assumes input_dir is something like .../session_XYZ/3_cropped_entities
    session_dir = input_dir.parent
    preprocessed_dir = session_dir / "4_preprocessed_entities"

    # Create the preprocessed images directory
    try:
        preprocessed_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory for preprocessed images: {preprocessed_dir}")
    except Exception as e:
        print(f"❌ Error creating preprocessed images directory {preprocessed_dir}: {e}", file=sys.stderr)
        # Decide if you want to stop here or continue without saving preprocessed images
        # For now, we'll continue but warn the user.
        preprocessed_dir = None # Set to None to skip saving later

    ocr_results = {}
    # List supported image files, sort them for consistent processing order
    image_files = sorted([
        f for f in input_dir.iterdir()
        if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'] and f.is_file()
    ])

    if not image_files:
        print("No supported image files found in the directory.")
        return {}

    print(f"Found {len(image_files)} image files to process.")

    for image_path in image_files:
        print(f"\n  Processing: {image_path.name}")
        extracted_text = None # Initialize result for this image
        processed_img = None # Initialize processed image

        try:
            # Preprocess the image
            processed_img = preprocess_image_for_ocr(image_path)

            if processed_img:
                # Save the preprocessed image
                if preprocessed_dir:
                    try:
                        preprocessed_filename = image_path.stem + "_preprocessed.png"
                        preprocessed_save_path = preprocessed_dir / preprocessed_filename
                        processed_img.save(str(preprocessed_save_path))
                        print(f"    📸 Saved preprocessed image: {preprocessed_save_path.name}")
                    except Exception as e:
                        print(f"    ❌ Error saving preprocessed image {image_path.name}: {e}", file=sys.stderr)

                # Perform OCR using pytesseract
                # config='--psm 6' suggests assuming a single uniform block of text,
                # which is often suitable for cropped entities. Adjust if needed.
                text = pytesseract.image_to_string(processed_img, lang='eng', config='--psm 6')

                # Clean up extracted text (remove leading/trailing whitespace)
                extracted_text = text.strip()

                # Print a cleaned snippet for progress indication
                # Replace actual newlines/returns with spaces for display
                cleaned_text_snippet = extracted_text.replace('\n', ' ').replace('\r', ' ').strip()
                display_text = (cleaned_text_snippet[:50] + '...') if len(cleaned_text_snippet) > 50 else cleaned_text_snippet
                print(f"    ✅ Extracted text: \"{display_text}\"")

            else:
                 # preprocess_image_for_ocr already printed an error message
                 print(f"    ⚠️  Skipping OCR due to preprocessing failure.")
                 pass # extracted_text remains None

        except pytesseract.TesseractError as te:
             # Catch Tesseract-specific errors (e.g., language not found)
             print(f"  ❌ Tesseract Error during OCR for {image_path.name}: {te}", file=sys.stderr)
             # extracted_text remains None
        except Exception as e:
            print(f"  ❌ Unexpected error during OCR for {image_path.name}: {e}", file=sys.stderr)
            # traceback.print_exc() # Uncomment for detailed exception stack
            # extracted_text remains None

        # Store the result (extracted text or None)
        ocr_results[image_path.name] = extracted_text

    print(f"\n📊 OCR completed for {len(ocr_results)} files.")
    if preprocessed_dir:
         print(f"Preprocessed images saved to: {preprocessed_dir}")
    return ocr_results

def main():
    """
    Main function to parse arguments and run the OCR process.
    """
    parser = argparse.ArgumentParser(
        description="Perform OCR on cropped entity images from a directory."
    )
    parser.add_argument(
        "--input_dir",
        "-i",
        required=True,
        help="Path to the directory containing cropped entity images (e.g., path/to/session_YYYYMMDD_HHMMSS/3_cropped_entities)"
    )
    parser.add_argument(
        "--save_json",
        action="store_true", # This makes it a flag; just include it to save JSON
        help="Save the OCR results to a JSON file in the session directory."
    )
    parser.add_argument(
        "--output_json_name",
        default="ocr_results.json",
        help="Name for the output JSON file (default: ocr_results.json)."
    )


    args = parser.parse_args()

    input_directory = Path(args.input_dir)

    # Perform OCR and get results
    results = extract_text_from_entity_images(input_directory)

    # Output Results Summary
    print("\n--- Summary of Extracted Text ---")
    if results:
        processed_count = sum(1 for text in results.values() if text is not None)
        failed_count = len(results) - processed_count
        print(f"Successfully extracted text from {processed_count} files.")
        print(f"Failed to extract text from {failed_count} files.")
        print("\nDetail per file:")

        for filename, text in results.items():
             print(f"File: {filename}")
             if text is not None:
                 # Replace actual newlines/returns with escaped versions for clear printing
                 cleaned_text_display = text.replace('\n', '\\n').replace('\r', '\\r')
                 print(f"Text: \"{cleaned_text_display}\"") # Quote the text for clarity
             else:
                 print("Text: [Extraction Failed]")
             print("-" * 30)
    else:
        print("No OCR results obtained.")

    # Save results to JSON if requested
    if args.save_json and results:
        try:
            # Determine the directory to save the JSON
            # Assumes input_directory is .../session_XYZ/3_cropped_entities
            session_dir = input_directory.parent
            output_json_path = session_dir / args.output_json_name

            # Ensure the parent directory exists (session_dir should exist if input_dir does)
            output_json_path.parent.mkdir(parents=True, exist_ok=True)

            # Save the results
            with open(output_json_path, 'w', encoding='utf-8') as f:
                # JSON can handle strings and None values directly
                json.dump(results, f, indent=4)

            print(f"\n✅ OCR results saved to JSON: {output_json_path}")

        except Exception as e:
             print(f"❌ Error saving results to JSON {output_json_path}: {e}", file=sys.stderr)
             # traceback.print_exc() # Uncomment for detailed exception stack


# --- Execution Block ---
if __name__ == "__main__":
    # To run the script by providing the --input_dir argument and potentially --save_json:
    main()

    # --- Hardcoded Test Example (Uncomment to run instead of the main argparse block) ---
    # Use this if you want to quickly test a specific directory without command-line args
    # test_dir_path = Path("/Users/yrevash/Qoneqt/Test_model/pipeline_output/session_20250620_093304/3_cropped_entities")
    # if test_dir_path.exists():
    #    print("--- Running Test Example ---")
    #    test_results = extract_text_from_entity_images(test_dir_path)
    #    print("\n--- Test Run OCR Results ---")
    #    if test_results:
    #        processed_count = sum(1 for text in test_results.values() if text is not None)
    #        failed_count = len(test_results) - processed_count
    #        print(f"Successfully extracted text from {processed_count} files.")
    #        print(f"Failed to extract text from {failed_count} files.")
    #        print("\nDetail per file:")
    #        for filename, text in test_results.items():
    #             print(f"File: {filename}")
    #             if text is not None:
    #                  cleaned_text_display = text.replace('\n', '\\n').replace('\r', '\\r')
    #                  print(f"Text: \"{cleaned_text_display}\"")
    #             else:
    #                  print("Text: [Extraction Failed]")
    #             print("-" * 30)
    #    else:
    #         print("No OCR results obtained.")

    #    # Example of saving JSON in the test block (uncomment if needed)
    #    # test_session_dir = test_dir_path.parent
    #    # test_output_json_path = test_session_dir / "test_ocr_results.json"
    #    # try:
    #    #     with open(test_output_json_path, 'w', encoding='utf-8') as f:
    #    #         json.dump(test_results, f, indent=4)
    #    #     print(f"\n✅ Test results saved to JSON: {test_output_json_path}")
    #    # except Exception as e:
    #    #      print(f"❌ Error saving test results to JSON: {e}", file=sys.stderr)

    #    print("--- Test Example Finished ---\n")
    # else:
    #    print(f"Test directory not found: {test_dir_path}", file=sys.stderr)
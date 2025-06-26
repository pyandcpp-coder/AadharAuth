import cv2
import numpy as np
import os
from PIL import Image, ImageEnhance, ImageFilter
import argparse

def blur_recovery(gray_img):
    """Specialized method for recovering text from heavily blurred images"""
    # Step 1: Upscale the image significantly
    height, width = gray_img.shape
    scale_factor = 3.0  # Triple the size
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    upscaled = cv2.resize(gray_img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    # Step 2: Apply gentle denoising first
    denoised = cv2.bilateralFilter(upscaled, 9, 80, 80)
    
    # Step 3: Gentle gamma correction
    gamma = 0.7
    gamma_corrected = np.power(denoised / 255.0, gamma) * 255.0
    gamma_corrected = gamma_corrected.astype(np.uint8)
    
    # Step 4: Moderate sharpening (less aggressive)
    kernel = np.array([[-1,-1,-1],
                       [-1, 9,-1],
                       [-1,-1,-1]])
    sharpened = cv2.filter2D(gamma_corrected, -1, kernel)
    
    # Step 5: CLAHE for local contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(sharpened)
    
    # Step 6: Morphological opening to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    cleaned = cv2.morphologyEx(enhanced, cv2.MORPH_OPEN, kernel)
    
    return cleaned

def extreme_sharpening(gray_img):
    """Extreme sharpening for heavily blurred text"""
    # Upscale the image first
    height, width = gray_img.shape
    if width < 1500:
        scale_factor = 1500 / width
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        gray_img = cv2.resize(gray_img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    # Apply Richardson-Lucy deconvolution simulation
    # Multiple iterations of aggressive unsharp masking
    result = gray_img.copy().astype(np.float64)
    
    for iteration in range(5):
        # Create different blur kernels for each iteration
        sigma = 1.0 + iteration * 0.3
        blurred = cv2.GaussianBlur(result, (0, 0), sigma)
        
        # Aggressive unsharp masking
        alpha = 3.0 - iteration * 0.2  # Decrease intensity with each iteration
        result = result + alpha * (result - blurred)
        
        # Clip values to valid range
        result = np.clip(result, 0, 255)
    
    # Convert back to uint8
    result = result.astype(np.uint8)
    
    # Final contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(3, 3))
    result = clahe.apply(result)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(result, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 7, 2)
    
    return thresh

def preprocess_image_for_ocr(input_path, output_path=None, method='comprehensive'):
    """
    Preprocess a blurred image for better OCR results
    
    Args:
        input_path (str): Path to input image
        output_path (str): Path to save preprocessed image (optional)
        method (str): Preprocessing method - 'basic', 'advanced', 'comprehensive', 'extreme', or 'blur_recovery'
    
    Returns:
        str: Path to the preprocessed image
    """
    
    # Read the image
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"Could not read image from {input_path}")
    
    # Create output filename if not provided
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = f"{base_name}_preprocessed_{method}.png"
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if method == 'basic':
        processed = basic_preprocessing(gray)
    elif method == 'advanced':
        processed = advanced_preprocessing(gray)
    elif method == 'extreme':
        processed = extreme_sharpening(gray)
    elif method == 'blur_recovery':
        processed = blur_recovery(gray)
    else:  # comprehensive
        processed = comprehensive_preprocessing(gray)
    
    # Save the preprocessed image
    cv2.imwrite(output_path, processed)
    print(f"Preprocessed image saved to: {output_path}")
    
    return output_path

def basic_preprocessing(gray_img):
    """Basic preprocessing - denoising and sharpening"""
    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray_img, None, 10, 7, 21)
    
    # Sharpen using unsharp masking
    gaussian = cv2.GaussianBlur(denoised, (0, 0), 2.0)
    sharpened = cv2.addWeighted(denoised, 1.5, gaussian, -0.5, 0)
    
    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(sharpened)
    
    return enhanced

def advanced_preprocessing(gray_img):
    """Advanced preprocessing with multiple techniques"""
    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray_img, None, 10, 7, 21)
    
    # Apply bilateral filter to reduce noise while preserving edges
    bilateral = cv2.bilateralFilter(denoised, 9, 75, 75)
    
    # Sharpen using unsharp masking
    gaussian = cv2.GaussianBlur(bilateral, (0, 0), 2.0)
    sharpened = cv2.addWeighted(bilateral, 1.5, gaussian, -0.5, 0)
    
    # Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(sharpened)
    
    # Apply morphological operations to clean up text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    morphed = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
    
    return morphed

def comprehensive_preprocessing(gray_img):
    """Comprehensive preprocessing with all techniques"""
    # Step 1: Resize image for better processing (upscale if too small)
    height, width = gray_img.shape
    if width < 1000:
        scale_factor = 1000 / width
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        gray_img = cv2.resize(gray_img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    # Step 2: Initial denoising
    denoised = cv2.fastNlMeansDenoising(gray_img, None, 10, 7, 21)
    
    # Step 3: Multiple iterations of sharpening for very blurry images
    for i in range(3):
        # Unsharp masking with stronger parameters
        gaussian = cv2.GaussianBlur(denoised, (0, 0), 1.5)
        denoised = cv2.addWeighted(denoised, 2.5, gaussian, -1.5, 0)
        denoised = np.clip(denoised, 0, 255).astype(np.uint8)
    
    # Step 4: Gamma correction for brightness adjustment
    gamma = 0.8  # Make it slightly darker to enhance contrast
    gamma_corrected = np.power(denoised / 255.0, gamma) * 255.0
    gamma_corrected = gamma_corrected.astype(np.uint8)
    
    # Step 5: Aggressive contrast enhancement using CLAHE
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gamma_corrected)
    
    # Step 6: Edge enhancement using Laplacian
    laplacian = cv2.Laplacian(enhanced, cv2.CV_64F)
    enhanced = enhanced + 0.3 * laplacian
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
    
    # Step 7: Final adaptive thresholding for text clarity
    thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)
    
    return thresh

def preprocess_with_pil(input_path, output_path=None):
    """Alternative preprocessing using PIL for comparison"""
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = f"{base_name}_pil_preprocessed.png"
    
    # Open image with PIL
    img = Image.open(input_path)
    
    # Convert to grayscale
    if img.mode != 'L':
        img = img.convert('L')
    
    # Enhance sharpness
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(2.0)
    
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.5)
    
    # Apply unsharp mask filter
    img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    
    # Save the result
    img.save(output_path)
    print(f"PIL preprocessed image saved to: {output_path}")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description='Preprocess images for better OCR results')
    parser.add_argument('input', help='Input image path')
    parser.add_argument('-o', '--output', help='Output image path (optional)')
    parser.add_argument('-m', '--method', choices=['basic', 'advanced', 'comprehensive', 'extreme', 'blur_recovery'], 
                       default='comprehensive', help='Preprocessing method')
    parser.add_argument('--pil', action='store_true', help='Also create PIL-based preprocessing')
    
    args = parser.parse_args()
    
    try:
        # Main preprocessing
        output_path = preprocess_image_for_ocr(args.input, args.output, args.method)
        
        # Optional PIL preprocessing for comparison
        if args.pil:
            preprocess_with_pil(args.input)
        
        print(f"\nPreprocessing complete!")
        print(f"Original: {args.input}")
        print(f"Processed: {output_path}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Example usage if run directly
    if len(os.sys.argv) == 1:
        print("Usage examples:")
        print("python preprocess_ocr.py input_image.jpg")
        print("python preprocess_ocr.py input_image.jpg -o output.png -m advanced")
        print("python preprocess_ocr.py input_image.jpg --pil")
    else:
        main()
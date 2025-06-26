# Aadhaar Processing API

A comprehensive FastAPI-based service for processing Aadhaar cards using YOLO object detection models and multi-language OCR capabilities. This API can detect, extract, and process information from both front and back sides of Aadhaar cards with support for multiple Indian languages.

## 🚀 Features

- **Multi-Model Detection**: Uses two YOLO models for card detection and entity extraction
- **Multi-Language OCR**: Supports English, Hindi, Telugu, and Bengali text extraction
- **Async Processing**: Background task processing with status tracking
- **Security Features**: Detects and blocks printed Aadhaar cards for security
- **RESTful API**: Clean FastAPI interface with automatic documentation
- **File Management**: Organized output structure with session-based directories
- **Static File Serving**: Direct access to processed results and images

## 📋 Prerequisites

### System Requirements
- Python 3.8+
- Tesseract OCR installed and accessible in PATH
- GPU support recommended for YOLO models

### Required Libraries
```bash
pip install fastapi uvicorn
pip install ultralytics opencv-python
pip install pytesseract pillow
pip install aiohttp aiofiles
pip install pydantic
```

### Tesseract Installation

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
sudo apt-get install tesseract-ocr-hin tesseract-ocr-ben tesseract-ocr-tel
```

**macOS:**
```bash
brew install tesseract
brew install tesseract-lang
```

**Windows:**
Download from: https://github.com/UB-Mannheim/tesseract/wiki

## ⚙️ Configuration

### Environment Variables
Create a `.env` file or set the following environment variables:

```bash
MODEL1_PATH=models/model1.pt          # Path to card detection model
MODEL2_PATH=models/model2.pt          # Path to entity detection model
DOWNLOAD_DIR=downloads                # Temporary download directory
OUTPUT_DIR=pipeline_output            # Results output directory
MAX_FILE_SIZE=10485760               # Max file size (10MB)
DEFAULT_CONFIDENCE_THRESHOLD=0.4      # Default detection confidence
```

### Directory Structure
```
project/
├── comprehensive_pipeline_api.py
├── models/
│   ├── model1.pt                    # Card detection model
│   └── model2.pt                    # Entity detection model
├── downloads/                       # Temporary downloads
├── pipeline_output/                 # Processing results
└── requirements.txt
```

## 🏃‍♂️ Quick Start

### 1. Install Dependencies
```bash
pip install above mentioned libs
```

### 2. Prepare Models
Place your trained YOLO models in the `models/` directory:
- `model1.pt`: Detects Aadhaar front/back cards
- `model2.pt`: Detects entities within cards

### 3. Start the Server
```bash
python comprehensive_pipeline_api.py
```

Or using uvicorn directly:
```bash
uvicorn comprehensive_pipeline_api:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Access the API
- **API Base URL**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 📖 API Documentation

### Submit Processing Request
**POST** `/process`

Submit Aadhaar card URLs for processing.

**Request Body:**
```json
{
  "user_id": "user123",
  "front_url": "https://example.com/front.jpg",
  "back_url": "https://example.com/back.jpg",
  "confidence_threshold": 0.4
}
```

**Response:**
```json
{
  "status": "pending",
  "message": "Task received and queued for processing.",
  "task_id": "abc123def456",
  "user_id": "user123",
  "status_url": "/status/abc123def456"
}
```

### Check Processing Status
**GET** `/status/{task_id}`

Check the status of a submitted processing task.

**Response (Processing):**
```json
{
  "status": "processing",
  "message": "Processing Aadhaar cards...",
  "task_id": "abc123def456",
  "user_id": "user123"
}
```

**Response (Completed):**
```json
{
  "status": "completed",
  "message": "Processing completed successfully",
  "task_id": "abc123def456",
  "user_id": "user123",
  "session_dir": "/path/to/results",
  "json_results_url": "/results/user123/abc123def456/complete_aadhaar_results.json",
  "processing_time": 15.23,
  "results": { ... }
}
```

### Access Results
**GET** `/results/{user_id}/{task_id}/complete_aadhaar_results.json`

Download the complete JSON results file.

## 🔧 Processing Pipeline

The API follows a 4-step processing pipeline:

### Step 1: Card Detection
- Detects Aadhaar front and back cards in input images
- Crops detected cards for further processing
- Filters based on confidence threshold

### Step 2: Entity Detection
- Identifies specific entities within cropped cards
- Supports detection of: name, address, DOB, gender, mobile, etc.
- Creates bounding boxes around detected entities

### Step 3: Entity Cropping
- Extracts individual entity regions
- Saves cropped entities for OCR processing

### Step 4: Multi-Language OCR
- Performs OCR on cropped entities
- Uses English for standard fields
- Uses Hindi+Telugu+Bengali for other language fields
- Preprocesses images for better OCR accuracy

## 📊 Supported Entities

The API can detect and extract the following entities:

| Entity | Description | Language Support |
|--------|-------------|------------------|
| `aadharNumber` | 12-digit Aadhaar number | English |
| `name` | Name in English | English |
| `name_otherlang` | Name in regional language | Hindi/Telugu/Bengali |
| `address` | Address in English | English |
| `address_other_lang` | Address in regional language | Hindi/Telugu/Bengali |
| `dob` | Date of birth | English |
| `gender` | Gender in English | English |
| `gender_other_lang` | Gender in regional language | Hindi/Telugu/Bengali |
| `mobile_no` | Mobile number | English |
| `city` | City name | English |
| `state` | State name | English |
| `pincode` | PIN code | English |

## 🔒 Security Features

- **Print Detection**: Automatically detects and blocks printed Aadhaar cards
- **Input Validation**: Validates image URLs and file sizes
- **Session Isolation**: Each processing task runs in isolated directories
- **Error Handling**: Comprehensive error handling and logging

## 📁 Output Structure

Each processing session creates the following directory structure:

```
pipeline_output/
└── {user_id}/
    └── {task_id}/
        ├── 1_front_back_cards/          # Cropped card images
        ├── 2_detected_entities/         # Entity detection visualizations
        ├── 3_cropped_entities/          # Individual entity crops
        ├── 4_preprocessed_entities/     # OCR-preprocessed images
        └── complete_aadhaar_results.json # Final results JSON
```

## 🐛 Error Handling

The API provides detailed error information:

```json
{
  "status": "error",
  "message": "Task failed: No Aadhaar cards detected.",
  "task_id": "abc123def456",
  "error": "No Aadhaar cards detected.",
  "failed_step": "card_detection",
  "security_flagged": false
}
```

## 🚀 Deployment

### Docker Deployment
```dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-hin \
    tesseract-ocr-ben \
    tesseract-ocr-tel \
    libgl1-mesa-glx \
    libglib2.0-0

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application
COPY . /app
WORKDIR /app

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "comprehensive_pipeline_api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Production Considerations
- Use a production WSGI server (Gunicorn + Uvicorn)
- Implement proper logging and monitoring
- Set up reverse proxy (Nginx)
- Configure SSL/TLS certificates
- Implement rate limiting and authentication
- Regular cleanup of temporary files

## 🔧 Customization

### Adding New Languages
Modify the `other_lang_code` parameter in the pipeline initialization:

```python
pipeline = ComprehensiveAadhaarPipeline(
    other_lang_code='hin+tel+ben+tam+guj'  # Add Tamil and Gujarati
)
```

### Adjusting Confidence Thresholds
Set different default confidence levels via environment variables or request parameters.

### Custom Entity Classes
Modify the `entity_classes` dictionary to support additional entity types.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

##  Support

For issues and questions:
- Create an issue on GitHub

##  Roadmap

- [ ] Add support for more Indian languages
- [ ] Implement batch processing capabilities
- [ ] Add data validation and formatting
- [ ] Create web interface for easier testing
- [ ] Add database integration for result storage
- [ ] Implement webhook notifications for completed tasks

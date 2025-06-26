# This is a revised version of the FastAPI Aadhaar pipeline code to:
# 1. Directly return JSON response on POST /process with extracted Aadhaar fields.
# 2. Save results in both JSON and PKL formats per user in their unique output folder.
# 3. Remove the GET /status endpoint functionality.

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl, Field
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any
import uvicorn
import asyncio
import aiohttp
import aiofiles
from pathlib import Path
import hashlib
import json
import pickle
import logging
from datetime import datetime
import os

import shutil

# Import Config and ComprehensiveAadhaarPipeline from comprehensive_pipeline_api.py
from comprehensive_pipeline_api import Config, ComprehensiveAadhaarPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Assuming Config and ComprehensiveAadhaarPipeline are defined above
config = Config()
pipeline: Optional[ComprehensiveAadhaarPipeline] = None

class AadhaarProcessRequest(BaseModel):
    user_id: str = "default_user"
    front_url: HttpUrl
    back_url: HttpUrl
    confidence_threshold: float = Field(0.4, ge=0.0, le=1.0)

@app.on_event("startup")
async def startup_event():
    global pipeline
    try:
        config.DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        pipeline = ComprehensiveAadhaarPipeline(
            model1_path=str(config.MODEL1_PATH),
            model2_path=str(config.MODEL2_PATH),
            output_base_dir=str(config.OUTPUT_DIR),
            confidence_threshold=config.DEFAULT_CONFIDENCE_THRESHOLD,
            other_lang_code='hin+tel+ben'
        )
        logger.info("✅ Pipeline models loaded successfully.")
    except Exception as e:
        logger.critical(f"❌ Pipeline initialization failed: {e}")
        raise

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
                    # Take first detected entity's extracted text
                    text = card['entities'][field][0].get('extracted_text', '')
                    if text:
                        data[field] = text
    return data

@app.post("/process", response_class=JSONResponse)
async def process_aadhaar(request: AadhaarProcessRequest):
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
            raise HTTPException(status_code=400, detail="Failed to download one or both Aadhaar images")

    result = pipeline.process_images([str(front_path), str(back_path)], request.user_id, task_id, request.confidence_threshold)
    if 'error' in result:
        raise HTTPException(status_code=500, detail=result['error'])

    organized = result.get('organized_results', {})
    main_data = extract_main_fields(result['organized_results'])
    main_data['User ID'] = request.user_id

    # Save undetected front image
    if not organized.get("front") or all(
        not card or not any(card['entities'].get(field, []) for field in ['aadharNumber', 'dob', 'gender', 'name', 'address', 'pincode', 'state'])
        for card in (organized.get("front", {}) or {}).values()
    ):
        image_not_scan_dir = Path("/Users/hqpl/Desktop/aadhar_testing/AadharAuth/ImageNotScan")
        image_not_scan_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(str(front_path), image_not_scan_dir / f"{task_id}_front.jpg")
        logger.info("❗ Front image saved to ImageNotScan")

    # Save undetected back image
    if not organized.get("back") or all(
        not card or not any(card['entities'].get(field, []) for field in ['aadharNumber', 'dob', 'gender', 'name', 'address', 'pincode', 'state'])
        for card in (organized.get("back", {}) or {}).values()
    ):
        image_not_scan_dir = Path("/Users/hqpl/Desktop/aadhar_testing/AadharAuth/ImageNotScan")
        image_not_scan_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(str(back_path), image_not_scan_dir / f"{task_id}_back.jpg")
        logger.info("❗ Back image saved to ImageNotScan")

    # Save to global PKL and CSV (across all users, not per user)
    output_folder = config.OUTPUT_DIR
    output_folder.mkdir(parents=True, exist_ok=True)
    pkl_path = Path("/Users/hqpl/Desktop/aadhar_testing/AadharAuth/data/summary.pkl")
    csv_path = Path("/Users/hqpl/Desktop/aadhar_testing/AadharAuth/data/summary.csv")
    json_path = Path("/Users/hqpl/Desktop/aadhar_testing/AadharAuth/data/summary.json")

    # Check for duplicate aadharNumber in PKL (across all users)
    aadhar_number = main_data.get('aadharNumber', '').replace(' ', '')
    exists = False
    existing_data = None
    if pkl_path.exists() and aadhar_number:
        try:
            with open(pkl_path, 'rb') as pf:
                all_data = pickle.load(pf)
            for entry in all_data:
                existing_aadhar = entry.get('aadharNumber', '').replace(' ', '')
                if existing_aadhar and existing_aadhar == aadhar_number:
                    exists = True
                    existing_data = entry
                    break
        except Exception as e:
            logger.error(f"Failed to check PKL for duplicates: {e}")

    if exists:
        shutil.rmtree(user_download_dir, ignore_errors=True)
        return JSONResponse(content={"status": "aadhar number exists", "data": existing_data})

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

    # Save to CSV
    try:
        import pandas as pd
        df = pd.DataFrame([main_data])
        if csv_path.exists():
            df_existing = pd.read_csv(csv_path)
            df = pd.concat([df_existing, df], ignore_index=True)
        df.to_csv(csv_path, index=False)
    except Exception as e:
        logger.error(f"Failed to update CSV file: {e}")

    shutil.rmtree(user_download_dir, ignore_errors=True)

    # Ensure all fields are strings and not None
    for k, v in main_data.items():
        if v is None:
            main_data[k] = ""
        else:
            main_data[k] = str(v)

    return JSONResponse(content={"status": "saved", "data": main_data})

# Remove the GET status endpoint

if __name__ == "__main__":
    uvicorn.run("aadhaar_pipeline_api:app", host="0.0.0.0", port=8000, reload=True)

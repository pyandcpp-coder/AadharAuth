import requests
import cv2
import numpy as np
from ultralytics import YOLO
import torch.serialization

# === Step 1: Download Image from URL ===
def download_image(url, save_path='downloaded_image.jpg'):
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print(f"Image saved to {save_path}")
        return save_path
    else:
        raise Exception(f"Failed to download image. Status code: {response.status_code}")

# === Step 2: Load YOLOv8 model and run prediction ===
def predict_on_image(image_path, model_path='best.pt', confidence_threshold=0.45):
    # Read image as-is (no rotation)
    img = cv2.imread(image_path)

    # Allow PyTorch to unpickle the YOLO model class
    with torch.serialization.safe_globals(["ultralytics.nn.tasks.DetectionModel"]):
        model = YOLO(model_path)

    # Run inference
    results = model(image_path)

    # Show results
    for result in results:
        # Filter boxes by confidence threshold
        filtered_boxes = [box for box in result.boxes if float(box.conf[0]) >= confidence_threshold]
        # Draw only filtered boxes
        img = cv2.imread(image_path)
        for box in filtered_boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = f"{cls_id}:{conf:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        cv2.imshow('Filtered Detections', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Optionally save the output image
    results[0].save(filename='output_with_detections.jpg')
    print("Predicted image saved as 'output_with_detections.jpg'")

# === Main Execution ===
if __name__ == "__main__":
    # Use a local image path directly
    image_path = "/Users/hqpl/Desktop/aadhar_testing/AadharAuth/pipeline_output/user1a/d370c00beedf8192978e602983ea74dd/1_front_back_cards/tmp80rsgckv_aadhar_front_conf67.jpg"  # Set your local image path here
    predict_on_image(image_path, model_path='/Users/hqpl/Desktop/aadhar_testing/AadharAuth/models/best.pt', confidence_threshold=0.45)  # Update with your model path


#  from ultralytics import YOLO
# import cv2

# # Load your trained YOLOv8 model
# model = YOLO('/Users/yrevash/Downloads/best.pt')  # Make sure this path is correct

# # Path to the local image you want to test
# image_path = '/Users/yrevash/Downloads/test_image.png'  # Replace with your image path

# # Run inference
# results = model(image_path)

# # Show results
# for result in results:
#     result.show()  # This will open the image with detections in a window (requires GUI)

# # Alternatively, save the output image with detections
# results[0].save(filename='output_with_detections.jpg')



# # # from ultralytics import YOLO
# # # from pathlib import Path

# # # # === CONFIGURATION ===
# # # input_folder = Path("/Users/yrevash/India/aadhar_images_extract")
# # # output_folder = Path("/Users/yrevash/Qoneqt/Test_model")  # Save here
# # # model_path = "/Users/yrevash/Downloads/best-4.pt"
# # # max_images = 100
# # # suffix = "_pred"

# # # # === LOAD MODEL ===
# # # model = YOLO(model_path)

# # # # === GATHER IMAGES ===
# # # valid_exts = {".jpg", ".jpeg", ".png"}
# # # image_files = sorted([f for f in input_folder.iterdir() if f.suffix.lower() in valid_exts])[:max_images]

# # # print(f"🔍 Found {len(image_files)} images. Processing...")

# # # # === PREDICT AND SAVE ===
# # # for i, image_path in enumerate(image_files, 1):
# # #     print(f"[{i}/{len(image_files)}] Processing {image_path.name}...")
    
# # #     results = model(str(image_path))
    
# # #     # Define output path in script's folder
# # #     output_path = output_folder / f"{image_path.stem}{suffix}{image_path.suffix}"
# # #     results[0].save(filename=str(output_path))

# # #     print(f"✅ Saved: {output_path.name}")

# # # print("\n🎉 Done! All predictions saved to:", output_folder)

# # # # === DELETE SCRIPT (COMMENTED) ===

# # # # 🚨 DELETE PREDICTED OUTPUTS
# # # for file in output_folder.glob(f"*{suffix}.*"):
# # #     print(f"Deleting {file.name}...")
# # #     file.unlink()

# # # print("🧹 All prediction images deleted.")

# # from pathlib import Path

# # # === CONFIGURATION ===
# # target_folder = Path("/Users/yrevash/Qoneqt/Test_model")
# # suffix = "_pred"

# # # === DELETE MATCHING FILES ===
# # deleted = 0
# # for file in target_folder.glob(f"*{suffix}.*"):
# #     print(f"🗑️ Deleting {file.name}")
# #     file.unlink()
# #     deleted += 1

# # print(f"\n✅ Deleted {deleted} predicted images from {target_folder}")


# from ultralytics import YOLO
# from pathlib import Path
# import shutil


# model_path = "/Users/yrevash/Downloads/best-4.pt"
# input_folder = Path("/Users/yrevash/India/aadhar_images_extract")  
# output_base = Path("dataset")
# split = "train"  

# images_dir = output_base / "images" / split
# labels_dir = output_base / "labels" / split


# images_dir.mkdir(parents=True, exist_ok=True)
# labels_dir.mkdir(parents=True, exist_ok=True)


# model = YOLO(model_path)


# image_files = sorted([f for f in input_folder.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]])

# print(f"🔍 Found {len(image_files)} images. Generating YOLOv8-compatible annotations...")

# for i, image_path in enumerate(image_files, 1):
#     print(f"[{i}/{len(image_files)}] Annotating {image_path.name}")

#     results = model(str(image_path))
#     result = results[0]


#     target_img_path = images_dir / image_path.name
#     shutil.copy(image_path, target_img_path)


#     label_path = labels_dir / (image_path.stem + ".txt")
#     with open(label_path, "w") as f:
#         for box in result.boxes:
#             cls_id = int(box.cls[0])
#             xywh = box.xywhn[0]  #
#             f.write(f"{cls_id} {xywh[0]:.6f} {xywh[1]:.6f} {xywh[2]:.6f} {xywh[3]:.6f}\n")

# print("✅ Annotations complete. Output saved in 'dataset/images/train' and 'dataset/labels/train'")



# from ultralytics import YOLO
# from pathlib import Path
# import cv2
# import os

# # === CONFIGURATION ===
# model_path = "/Users/yrevash/Downloads/yolov8_upload_dir/weights/best.pt"
# image_path = Path("/Users/yrevash/Qoneqt/Test_model/output_with_detections.jpg")
# output_dir = Path("cropped_predictions")
# confidence_threshold = 0.4

# # Create output directory if it doesn't exist
# output_dir.mkdir(parents=True, exist_ok=True)

# # Load model
# model = YOLO(model_path)

# # Run inference
# results = model(str(image_path))
# result = results[0]

# # Load original image using OpenCV
# img = cv2.imread(str(image_path))

# # Iterate over predictions
# for i, box in enumerate(result.boxes):
#     conf = float(box.conf[0])
#     if conf < confidence_threshold:
#         continue  # skip low-confidence detections

#     # Get bounding box coordinates (xyxy format)
#     x1, y1, x2, y2 = map(int, box.xyxy[0])

#     # Crop the image
#     crop = img[y1:y2, x1:x2]

#     # Build crop filename
#     cls_id = int(box.cls[0])
#     crop_filename = output_dir / f"{image_path.stem}_obj{i}_cls{cls_id}_conf{int(conf*100)}.jpg"

#     # Save cropped image
#     cv2.imwrite(str(crop_filename), crop)
#     print(f"✅ Saved crop: {crop_filename.name}")

# print("🎯 Done cropping all objects with confidence >= 40%.")


# # import kagglehub
# # import os

# # # Login to KaggleHub
# # kagglehub.login()

# # # === CONFIG ===
# # LOCAL_MODEL_DIR = '/Users/yrevash/Downloads/yolov8_upload_dir/weights'
# # MODEL_SLUG = 'yolov8-aadhar'         # Your model name on Kaggle
# # VARIATION_SLUG = 'default'           # Variation name (like a version label)
# # USERNAME = 'yashtiwari9182'          # Your Kaggle username

# # # Create directory if not exists
# # os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

# # # Copy best.pt into that folder (optional if already placed)
# # if not os.path.exists(f"{LOCAL_MODEL_DIR}/best.pt"):
# #     import shutil
# #     shutil.copy('/Users/yrevash/Downloads/best-4.pt', f"{LOCAL_MODEL_DIR}/best.pt")

# # # Upload model
# # handle = f"{USERNAME}/{MODEL_SLUG}/keras/{VARIATION_SLUG}"
# # print(f"Uploading model to: {handle}")

# # kagglehub.model_upload(
# #     handle=handle,
# #     local_model_dir=LOCAL_MODEL_DIR,
# #     version_notes='Update 2025-06-19'
# # )

# # print("✅ Upload complete.")


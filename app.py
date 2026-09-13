import io
import re
import cv2
import torch
import sqlite3
from datetime import datetime
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
import torchvision.transforms as transforms
from ultralytics import YOLO

# Load YOLOv8 nano for vehicle bounding box isolation
try:
    yolo_vehicle_model = YOLO("yolov8n.pt")
except Exception as e:
    print(f"[WARNING] YOLO load fallback: {e}")
    yolo_vehicle_model = None

# 1. Database Setup
DB_NAME = "alpr_history.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS plate_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_number TEXT NOT NULL,
            filename TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# 2. Model Architecture
class LicensePlateCNN(nn.Module):
    def __init__(self, num_classes=26):
        super(LicensePlateCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 3. Transformations and Global Settings
DEVICE = torch.device("cpu")
CLASS_NAMES = sorted(['0', '1', '2', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'H', 'J', 'K', 'L', 'M', 'N', 'Q', 'S', 'T', 'V', 'X', 'Y', 'Z'])
LABEL_MAPPING = {
    '0': '۰', '1': '۱', '2': '۲', '4': '۴', '5': '۵', '6': '۶', '7': '۷', '8': '۸', '9': '۹',
    'A': 'الف', 'B': 'ب', 'C': 'س', 'D': 'د', 'H': 'هـ', 'J': 'ج', 'K': 'ط', 'L': 'ل', 'M': 'م', 
    'N': 'ن', 'Q': 'ا', 'S': 'ص', 'T': 'ت', 'V': 'و', 'X': 'ق', 'Y': 'ی', 'Z': 'ع'
}

ocr_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

model = LicensePlateCNN(num_classes=26).to(DEVICE)
try:
    model.load_state_dict(torch.load("license_plate_ocr_robust.pth", map_location=DEVICE))
    model.eval()
except FileNotFoundError:
    print("[WARNING] Weights file 'license_plate_ocr_robust.pth' not found.")

# 4. Post-processing & Structural Rule Enforcement
def format_persian_plate(raw_text):
    digits_map = {'0':'۰', '1':'۱', '2':'۲', '3':'۳', '4':'۴', '5':'۵', '6':'۶', '7':'۷', '8':'۸', '9':'۹'}
    cleaned_chars = []
    
    for ch in raw_text:
        if ch in digits_map:
            cleaned_chars.append(digits_map[ch])
        elif ch in ['۰', '۱', '۲', '۳', '۴', '۵', '۶', '۷', '۸', '۹']:
            cleaned_chars.append(ch)
        elif ch in LABEL_MAPPING.values() or ch.isalpha():
            cleaned_chars.append(ch)
            
    return "".join(cleaned_chars)

# 5. Two-Stage Localization & Segmentation Pipeline
def locate_plate_two_stage(img_bgr):
    h_img, w_img = img_bgr.shape[:2]
    car_crop = None
    
    # Stage 1: Isolate vehicle using YOLOv8 (COCO class 2 = car)
    if yolo_vehicle_model is not None:
        try:
            results = yolo_vehicle_model(img_bgr, verbose=False, classes=[2])
            best_car_area = 0
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    area = (x2 - x1) * (y2 - y1)
                    if area > best_car_area and area > 10000:
                        best_car_area = area
                        cx1, cy1 = max(0, x1), max(0, y1)
                        cx2, cy2 = min(w_img, x2), min(h_img, y2)
                        car_crop = img_bgr[cy1:cy2, cx1:cx2]
        except Exception:
            pass

    target_area = car_crop if (car_crop is not None and car_crop.size > 0) else img_bgr
    th_h, th_w = target_area.shape[:2]

    # Stage 2: Gradient-based plate localization inside target area (lower-middle zone)
    ymin, ymax = int(th_h * 0.4), int(th_h * 0.95)
    xmin, xmax = int(th_w * 0.1), int(th_w * 0.9)
    roi = target_area[ymin:ymax, xmin:xmax]
    
    if roi.size == 0:
        return target_area

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    gradX = cv2.Sobel(enhanced, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradX = np.absolute(gradX)
    minVal, maxVal = np.min(gradX), np.max(gradX)
    gradX = (255 * ((gradX - minVal) / (maxVal - minVal + 1e-5))).astype(np.uint8)
    
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    closed = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
    thresh = cv2.threshold(closed, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = float(w) / float(h if h > 0 else 1)
        if 2.0 < aspect_ratio < 7.0 and w > 30 and h > 8:
            candidates.append((w * h, x, y, w, h))
            
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        _, cx, cy, cw, ch = candidates[0]
        gx1 = max(0, xmin + cx - int(cw * 0.05))
        gy1 = max(0, ymin + cy - int(ch * 0.2))
        gx2 = min(th_w, xmin + cx + cw + int(cw * 0.05))
        gy2 = min(th_h, ymin + cy + ch + int(ch * 0.2))
        crop_candidate = target_area[gy1:gy2, gx1:gx2]
        if crop_candidate.size > 0:
            return crop_candidate
        
    return roi

def api_segment_and_recognize(cropped_plate_bgr):
    cropped_plate = cropped_plate_bgr if cropped_plate_bgr is not None and cropped_plate_bgr.size > 0 else np.zeros((50, 150, 3), dtype=np.uint8)
    gray = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))
    enhanced_gray = clahe.apply(gray)
    
    binary = cv2.adaptiveThreshold(enhanced_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 7)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    char_bounding_boxes = []
    plate_height, plate_width = binary.shape[:2]
    
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        aspect_ratio = bw / float(bh if bh>0 else 1)
        height_ratio = bh / float(plate_height if plate_height>0 else 1)
        if 0.05 < aspect_ratio < 0.95 and 0.2 < height_ratio < 0.98 and bw < plate_width * 0.35:
            char_bounding_boxes.append((x, y, bw, bh))
            
    char_bounding_boxes = sorted(char_bounding_boxes, key=lambda b: b[0])
    final_plate_text = ""
    
    for (x, y, bw, bh) in char_bounding_boxes:
        pad = 2
        char_crop = binary[max(0, y-pad):min(binary.shape[0], y+bh+pad), 
                           max(0, x-pad):min(binary.shape[1], x+bw+pad)]
        
        char_tensor = ocr_transform(Image.fromarray(char_crop)).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            output = model(char_tensor)
            _, predicted_idx = torch.max(output, 1)
            predicted_eng = CLASS_NAMES[predicted_idx.item()]
            
        final_plate_text += LABEL_MAPPING.get(predicted_eng, predicted_eng)
        
    return format_persian_plate(final_plate_text)

# 6. FastAPI Endpoints
app = FastAPI(title="Persian Two-Stage ALPR API")

@app.get("/history/", summary="Retrieve plate recognition history")
async def get_plate_history(limit: int = 50):
    try:
        conn = sqlite3.connect(DB_NAME)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, plate_number, filename, timestamp FROM plate_history ORDER BY id DESC LIMIT ?", 
            (limit,)
        )
        rows = cursor.fetchall()
        conn.close()
        
        return {
            "status": "success",
            "count": len(rows),
            "data": [dict(row) for row in rows]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/")
async def predict_license_plate(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    try:
        image_bytes = await file.read()
        image_array = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if img_bgr is None:
            raise ValueError("Invalid image format.")

        plate_crop = locate_plate_two_stage(img_bgr)
        final_text = api_segment_and_recognize(plate_crop)
        
        if not final_text:
            return {"status": "error", "message": "No characters detected."}
            
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO plate_history (plate_number, filename) VALUES (?, ?)", 
            (final_text, file.filename)
        )
        record_id = cursor.lastrowid
        conn.commit()
        conn.close()
            
        return {
            "status": "success",
            "record_id": record_id,
            "plate_number": final_text,
            "filename": file.filename,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

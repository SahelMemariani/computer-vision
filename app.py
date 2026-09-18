import streamlit as st
import cv2
import torch
import sqlite3
import pandas as pd
from datetime import datetime
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from ultralytics import YOLO

# --- Database Setup ---
DB_NAME = "alpr_history.db"
def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS plate_history 
                      (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                       plate_number TEXT NOT NULL, 
                       filename TEXT, 
                       timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

init_db()

# --- Load Models (Cached for max speed) ---
@st.cache_resource
def load_yolo():
    try:
        return YOLO("yolov8n.pt")
    except Exception as e:
        st.error(f"YOLO Error: {e}")
        return None

@st.cache_resource
def load_ocr_model():
    DEVICE = torch.device("cpu")
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
            
    model = LicensePlateCNN(num_classes=26).to(DEVICE)
    try:
        model.load_state_dict(torch.load("license_plate_ocr_robust.pth", map_location=DEVICE))
        model.eval()
    except Exception as e:
        st.warning(f"OCR weights not found: {e}")
    return model, DEVICE

yolo_model = load_yolo()
ocr_model, DEVICE = load_ocr_model()

# --- Constants & Transforms ---
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

# --- Two-Stage Processing Pipeline ---
def locate_plate_two_stage(img_bgr):
    h_img, w_img = img_bgr.shape[:2]
    car_crop = None
    if yolo_model is not None:
        try:
            results = yolo_model(img_bgr, verbose=False, classes=[2])
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
    ymin, ymax = int(th_h * 0.65), int(th_h * 0.86)
    xmin, xmax = int(th_w * 0.28), int(th_w * 0.72)
    crop_candidate = target_area[ymin:ymax, xmin:xmax]
    return crop_candidate if crop_candidate.size > 0 else target_area

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
        char_crop = binary[max(0, y-pad):min(binary.shape[0], y+bh+pad), max(0, x-pad):min(binary.shape[1], x+bw+pad)]
        if char_crop.size == 0: continue
        char_tensor = ocr_transform(Image.fromarray(char_crop)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = ocr_model(char_tensor)
            _, predicted_idx = torch.max(output, 1)
            predicted_eng = CLASS_NAMES[predicted_idx.item()]
        final_plate_text += LABEL_MAPPING.get(predicted_eng, predicted_eng)
        
    return format_persian_plate(final_plate_text)

# --- UI Layout ---
st.set_page_config(page_title="Persian ALPR", layout="wide")
st.title("🚗 Persian ALPR System (YOLOv8 + CNN)")
st.markdown("Powered by 16GB RAM Cloud Server")

uploaded_file = st.file_uploader("Choose a car image (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_column_width=True)
        
    with col2:
        if st.button("Process Plate", type="primary", use_container_width=True):
            with st.spinner("Detecting vehicle & recognizing plate..."):
                plate_crop = locate_plate_two_stage(img_bgr)
                st.image(cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB), caption="Localized Plate Region", width=300)
                
                final_text = api_segment_and_recognize(plate_crop)
                
                if final_text:
                    st.success(f"### Detected Plate: {final_text}")
                    # Save to DB
                    conn = sqlite3.connect(DB_NAME)
                    cursor = conn.cursor()
                    cursor.execute("INSERT INTO plate_history (plate_number, filename) VALUES (?, ?)", (final_text, uploaded_file.name))
                    conn.commit()
                    conn.close()
                else:
                    st.error("No characters detected.")

st.divider()
st.subheader("📋 Recognition History")
conn = sqlite3.connect(DB_NAME)
df = pd.read_sql_query("SELECT id, plate_number, filename, timestamp FROM plate_history ORDER BY id DESC LIMIT 10", conn)
conn.close()
st.dataframe(df, use_container_width=True, hide_index=True)

Persian License Plate Recognition (ALPR)
An end-to-end computer vision application that detects a vehicle in an image, localizes its license plate, and reads the Persian characters using a custom CNN built in PyTorch. Every recognition is stored in a database and can be reviewed from a web dashboard.
How it works
Image ──► YOLOv8 (vehicle detection) ──► plate localization ──► OpenCV preprocessing
      ──► character segmentation ──► PyTorch CNN (character OCR) ──► Persian plate string ──► SQLite history

Vehicle detection. YOLOv8n (COCO, class car) finds the vehicles in the image. The largest bounding box is selected.
Plate localization. The plate region is cropped from the lower-central area of the vehicle box.
Preprocessing. Grayscale conversion, CLAHE contrast enhancement and Gaussian adaptive thresholding.
Character segmentation. External contours are filtered by size and aspect ratio, then sorted left to right.
Character recognition. Each character is resized to 32×32 and classified by a CNN.
Post-processing. Predictions are mapped to Persian digits and letters and formatted as a plate number.
Persistence. Results are saved to SQLite with filename and timestamp.
OCR model
Layer
Details
Input
1 × 32 × 32 grayscale
Conv blocks
3 × (Conv2d → ReLU → MaxPool), 16 / 32 / 64 filters
Classifier
FC 128 → Dropout 0.5 → FC 26
Weights
license_plate_ocr_robust.pth

Applications
File
Description
app.py
Standalone Streamlit app. It runs the full pipeline locally and shows the uploaded image, the localized plate and the recognized number, plus the recognition history.
dashboard.py
Streamlit client for the deployed FastAPI backend. It sends images to POST /predict/ and reads GET /history/?limit=50.

Tech stack
Python 3.11 · PyTorch · torchvision · Ultralytics YOLOv8 · OpenCV · Streamlit · FastAPI · SQLite · pandas · NumPy · Pillow
Getting started
git clone https://github.com/SahelMemariani/computer-vision.git
cd computer-vision
pip install -r requirements.txt

# Run the standalone app
streamlit run app.py

# Or run the dashboard that calls the FastAPI backend
streamlit run dashboard.py

YOLOv8n weights (yolov8n.pt) are downloaded automatically by Ultralytics on first run.
Repository structure
computer-vision/
├── app.py                         # Standalone Streamlit ALPR app (full pipeline)
├── dashboard.py                   # Streamlit client for the FastAPI backend
├── license_plate_ocr_robust.pth   # Trained CNN weights for character OCR
├── requirements.txt
└── Runtime.txt                    # Python version for deployment
Possible improvements
Replace the fixed-ratio plate crop with a dedicated plate detector.
Report character-level and plate-level accuracy on a labelled test set.
Add the FastAPI backend source code and a Dockerfile to this repository.
Author
Sahel Memariani. LinkedIn · GitHub

import streamlit as st
import requests
import pandas as pd
from PIL import Image
import io

# API Base URL (must match your running FastAPI server)
API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Persian ALPR Dashboard",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Persian License Plate Recognition (ALPR) Dashboard")
st.markdown("End-to-End AI-powered vehicle plate detection, OCR, and history tracking.")

# Sidebar navigation
page = st.sidebar.selectbox("Navigation", ["Upload & Predict", "Recognition History"])

if page == "Upload & Predict":
    st.header("Upload Vehicle Image")
    uploaded_file = st.file_uploader("Choose a car image (JPG, PNG)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)
            
        with col2:
            st.subheader("Recognition Result")
            if st.button("Process Plate", type="primary"):
                with st.spinner("Analyzing image through FastAPI backend..."):
                    # Reset stream position and send to FastAPI
                    uploaded_file.seek(0)
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    
                    try:
                        response = requests.post(f"{API_URL}/predict/", files=files)
                        if response.status_code == 200:
                            res_json = response.json()
                            if res_json.get("status") == "success":
                                st.success("Recognition Complete!")
                                st.metric(label="Detected Persian Plate", value=res_json["plate_number"])
                                st.json({
                                    "Record ID": res_json.get("record_id"),
                                    "Filename": res_json.get("filename"),
                                    "Timestamp": res_json.get("timestamp")
                                })
                            else:
                                st.warning(res_json.get("message", "No characters detected."))
                        else:
                            st.error(f"API Error ({response.status_code}): {response.text}")
                    except requests.exceptions.ConnectionError:
                        st.error("Could not connect to FastAPI server. Make sure Uvicorn is running on port 8000!")

elif page == "Recognition History":
    st.header("Database Traffic History")
    
    col_ref, col_lim = st.columns([1, 3])
    with col_ref:
        if st.button("Refresh History"):
            st.rerun()
            
    try:
        response = requests.get(f"{API_URL}/history/?limit=50")
        if response.status_code == 200:
            data = response.json().get("data", [])
            if data:
                df = pd.DataFrame(data)
                # Reorder columns for clean display
                df = df[['id', 'plate_number', 'filename', 'timestamp']]
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.info("No records found in database yet.")
        else:
            st.error(f"Failed to fetch history: {response.text}")
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to FastAPI server at http://127.0.0.1:8000/docs")
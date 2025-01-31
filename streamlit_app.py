import streamlit as st
import os
import cv2
import numpy as np
from ultralytics import YOLO
import boto3
from PIL import Image

# AWS S3 Configuration
BUCKET_NAME = "osas"  # Replace with your S3 Bucket Name
LOCAL_MODEL_PATH = "models"  # Local directory to store the downloaded model
S3_MODEL_PATH = "ml-models/yolov8-facemask/yolov8.pt"  # Path to the model in S3

def download_model(local_path, s3_path):
    """Downloads the YOLOv8 model from S3."""
    s3 = boto3.client("s3")
    os.makedirs(local_path, exist_ok=True)
    local_file = os.path.join(local_path, os.path.basename(s3_path))
    
    if not os.path.exists(local_file):
        with st.spinner("Downloading model... Please wait!"):
            s3.download_file(BUCKET_NAME, s3_path, local_file)
            st.success("Model downloaded successfully!")
    else:
        st.info("Model already exists locally.")
    
    return local_file

# Streamlit UI Title
st.title("Face Mask Detection using YOLOv8")

# Select Mode
mode = st.radio("Choose Detection Mode", ("Upload Image", "Real-time Detection"))

# Button to download the model
if st.button("Download Model"):
    model_path = download_model(LOCAL_MODEL_PATH, S3_MODEL_PATH)

if mode == "Upload Image":
    # Upload Image
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Convert image to OpenCV format
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Load model
        model_file = os.path.join(LOCAL_MODEL_PATH, "last.pt")
        if os.path.exists(model_file):
            model = YOLO(model_file)
            
            # Perform detection
            results = model(image)
            
            # Draw bounding boxes
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = f"{model.names[int(box.cls)]}: {box.conf[0]:.2f}"
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Convert back to RGB for display
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            st.image(image, caption="Detection Results", use_column_width=True)
        else:
            st.warning("Model file not found. Please download the model first.")

elif mode == "Real-time Detection":
    st.write("Starting real-time webcam detection...")
    
    model_file = os.path.join(LOCAL_MODEL_PATH, "last.pt")
    if os.path.exists(model_file):
        model = YOLO(model_file)
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("Error: Could not access the webcam.")
        else:
            stframe = st.empty()
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture frame from webcam.")
                    break
                
                results = model(frame)
                
                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        label = f"{model.names[int(box.cls)]}: {box.conf[0]:.2f}"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                stframe.image(frame, channels="RGB")
                
            cap.release()
    else:
        st.warning("Model file not found. Please download the model first.")

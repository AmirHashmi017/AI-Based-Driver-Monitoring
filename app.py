import streamlit as st
import cv2
import numpy as np
from PIL import Image
from src.detector import YoloDriverDetector
import os

st.set_page_config(
    page_title="Driver Monitoring AI",
    page_icon="🚗",
    layout="wide"
)

st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stAlert {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🔍 Live Detection", "📊 Model Metrics"])

with tab1:
    st.write("Upload an image to detect driver drowsiness or distraction using AI.")
    
    st.sidebar.header("Configuration")
    model_path = st.sidebar.selectbox(
        "Select Model Weights",
        ["models/best_dms_phone_model.pt", "yolov8n.pt"],
        index=0
    )

    if not os.path.exists(model_path):
        st.sidebar.error(f"Model not found: {model_path}")
        st.info("Please make sure you have downloaded the weights to the 'models/' folder.")
    else:
        st.sidebar.success("Model ready!")

    @st.cache_resource
    def load_detector(path):
        return YoloDriverDetector(path)

    detector = load_detector(model_path)

    uploaded_file = st.file_uploader("Choose a driver photo...", type=["jpg", "jpeg", "png"], key="detector_upload")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        frame = np.array(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        with st.spinner("Analyzing driver state..."):
            detections = detector.detect(frame)

        result_frame = detector.draw_detections(frame.copy(), detections)
        result_image = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
        
        # 3. Save result to outputs folder
        if not os.path.exists("outputs"):
            os.makedirs("outputs")
        save_path = os.path.join("outputs", f"result_{uploaded_file.name}")
        cv2.imwrite(save_path, result_frame)

        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)
            
        with col2:
            st.subheader("Inference Results")
            st.image(result_image, use_container_width=True)

        st.subheader("AI Analysis Report")
        if not detections:
            st.info("No driver detected in the image.")
        else:
            for i, det in enumerate(detections):
                label = det['label']
                conf = det['conf']
                
                if "Drowsy" in label:
                    st.error(f"🔴 ALERT: Driver {i+1} appears DROWSY (Confidence: {conf:.2f})")
                elif "Phone" in label:
                    st.warning(f"🟠 WARNING: Driver {i+1} is using a SMARTPHONE (Confidence: {conf:.2f})")
                elif "Awake" in label:
                    st.success(f"🟢 SAFE: Driver {i+1} is AWAKE and focused (Confidence: {conf:.2f})")
                else:
                    st.info(f"⚪ Detected: {label} (Confidence: {conf:.2f})")

with tab2:
    st.header("Model Performance & Training Metrics")
    st.write("These graphs are generated during the 100-epoch training phase on the Vigil Drive dataset.")
    
    metrics_path = "metrics" 
    
    col_m1, col_m2 = st.columns(2)
    
    with col_m1:
        st.subheader("Confusion Matrix")
        cm_img = os.path.join(metrics_path, "confusion_matrix.png")
        if os.path.exists(cm_img):
            st.image(cm_img, caption="Per-class Accuracy Breakdown", use_container_width=True)
        else:
            st.info("💡 Tip: Download 'confusion_matrix.png' from your Google Drive 'train_runs' folder and place it in 'models/metrics/' to see it here.")
            
    with col_m2:
        st.subheader("Training Results")
        res_img = os.path.join(metrics_path, "results.png")
        if os.path.exists(res_img):
            st.image(res_img, caption="Loss and mAP Trends over 100 Epochs", use_container_width=True)
        else:
            st.info("💡 Tip: Place 'results.png' in 'models/metrics/' to see your training curves.")

    st.subheader("F1 and PR Curves")
    f1_img = os.path.join(metrics_path, "F1_curve.png")
    if os.path.exists(f1_img):
        st.image(f1_img, caption="Precision-Recall Tradeoff", use_container_width=True)

st.divider()
st.markdown("Developed for University Computer Vision Project - 2026")

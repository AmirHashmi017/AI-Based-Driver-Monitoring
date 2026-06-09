# 🚗 AI-Based Driver Monitoring System (DMS)
### Modern Deep Learning Solution for Real-Time Safety

This project implements a professional **Driver Monitoring System (DMS)** utilizing the **YOLOv8** computer vision architecture. The system provides real-time detection and alerting for fatigue, phone distraction, and seatbelt compliance, ensuring a safer driving environment.

---

## 🌟 Intelligent Class Detection
The model has been optimized on the **Vigil Drive Dataset** (10,000+ frames) to identify:
- **Awake**: Driver is alert and looking forward.
- **Drowsy**: Eyes closed or head nodding (High-priority alert).
- **Phone**: Smartphone talking or texting detected.
- **Seat Belt**: Safety belt engagement check.

## 📊 Performance Statistics
| Metric | Result |
| :--- | :--- |
| **mAP@50 (Mean Average Precision)** | **90.8%** |
| **Inference Speed (CPU)** | ~25 FPS |
| **Inference Speed (T4 GPU)** | ~140 FPS |

---

## 📂 Project Organization
```text
├── app.py              # Main Web Dashboard (Streamlit)
├── main.py             # Live Webcam Monitoring Application
├── demo.py             # Performance testing for video files
├── src/
│   ├── detector.py     # Core YOLOv8 inference & HUD logic
│   └── train.py        # Unified training pipeline with .env support
├── models/             # Folder for trained weight files (.pt)
├── metrics/            # Project performance charts and CMs
├── outputs/            # Processed videos and demo results
├── requirements.txt    # Project dependencies
└── README.md           # Documentation
```

---

## 🛠️ Setup & Installation

### 1. Environment Setup
Create a virtual environment and install the required AI stack:
```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Secure Configuration
Create a `.env` file in the root directory to store your Roboflow credentials:
```env
ROBOFLOW_API_KEY="your_api_key_here"
```

---

## 🚀 How to Run

### I. The "ics UI" Web Dashboard
Recommended for professional demonstrations. Shows image uploads and model metrics.
```bash
streamlit run app.py
```

### II. Live Real-Time Monitoring
Uses your webcam to monitor driver status with a color-coded HUD.
```bash
python main.py
```

### III. Processing Test Videos
Runs the AI on pre-recorded footage and saves the result to the `outputs/` folder.
```bash
python demo.py --input test_assets/video.mp4 --model models/best_dms_phone_model.pt
```

### IV. Retraining the Model
If you wish to retrain the model on your local hardware:
```bash
python src/train.py
```

---
**Developed for University Computer Vision Open House - 2026**

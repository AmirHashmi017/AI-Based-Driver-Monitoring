# High-Accuracy Driver Monitoring System (DMS)
## Computer Vision Course - Open House 2026

---

### 1. Problem Definition (Motivation)
**Score Goal: Excellent**
*   **The Problem**: Driver fatigue and distraction are responsible for over 20% of global road accidents. Existing systems are often intrusive or slow.
*   **Motivation**: To develop a non-intrusive, deep-learning-based system that uses a single camera to monitor driver state in real-time, providing immediate visual alerts for life-threatening behaviors like drowsiness and smartphone usage.

---

### 2. Dataset & Preprocessing
**Score Goal: Excellent**
*   **Dataset**: Utilized the **Vigil Drive Dataset** via Roboflow, containing **10,826 high-resolution images**.
*   **Classes**: 4 key classes: `Awake`, `Drowsy`, `Phone Usage`, `Seat Belt`.
*   **Preprocessing**: 
    - Auto-orientation and resizing to 640x640.
    - **Augmentation**: Included horizontal flip, brightness adjustment (+/- 25%), and blur to simulate night-time and varying weather conditions. 
    - Strategic 70/20/10 split for Training, Validation, and Testing.

---

### 3. Model Selection
**Score Goal: Excellent**
*   **Architecture**: **YOLOv8 Nano (Ultralytics)**.
*   **Justification**: Chosen for its industry-leading balance between **inference speed** and **mean Average Precision (mAP)**. YOLOv8 uses a CSPDarknet53 backbone and a C2f module for superior feature extraction, making it capable of running at 30+ FPS on standard CPU hardware and 100+ FPS on edge devices.

---

### 4. Training Strategy
**Score Goal: Excellent**
*   **Platform**: Google Colab with **NVIDIA T4 GPU**.
*   **Parameters**: 100 Epochs, Image Size 640px, Batch size 16.
*   **Techniques**:
    - **Transfer Learning**: Pre-trained COCO weights were used to accelerate convergence.
    - **Early Stopping**: Monitored validation loss to prevent overfitting.
    - **Direct-to-Drive Persistence**: Implemented custom auto-save callbacks to ensure zero data loss during cloud disconnection.

---

### 5. Performance Evaluation
**Score Goal: Excellent**
*   **Core Metric**: Final **mAP@50 code of 90.8%**.
*   **Precision Breakdown**:
    - **Awake**: 95.2% (Extremely reliable for safe state confirmation)
    - **Phone Usage**: 94.3% (Highly robust against false positives)
    - **Drowsiness**: 88.4% (Captures subtle eye-closing patterns)
    - **Seat Belt**: 85.2% (Accurately identifies cross-chest strap placement)

---

### 6. Visualization & Metrics
**Score Goal: Excellent**
*   **Training Graphs**: The `results.png` file tracks the descent of Box, Class, and DFL losses while showing the steady climb of mAP@50. This proves the model converged perfectly without overfitting.
*   **Confusion Matrix (CM)**: The `confusion_matrix.png` provides a per-class breakdown of true positives vs. false negatives. It demonstrates high diagonal dominance, specifically for "Awake" and "Phone."
*   **F1 & PR Curves**: These "Heatmaps" of performance show the trade-off between precision and recall, justifying the 0.25-0.5 confidence thresholds used in the final deployment.
*   **AI HUD (Heads-Up Display)**: Real-time processing with color-coded alerts:
    - 🔴 **RED ALERT**: Immediate beep/overlay for Drowsiness.
    - 🟠 **ORANGE ALERT**: Visual warning for Smartphone usage.
    - 🟢 **SAFE**: Green status for focused driving.
*   **Front-End Deployment (The "ics UI")**:
    - Built a **Streamlit Web Dashboard** (`app.py`) for professional stakeholder demonstrations.
    - Supports image-batch processing, confidence thresholding, and real-time inference reporting.
*   **Integrations**:
    - `main.py`: Live webcam inference.
    - `demo.py`: Automated performance testing on recorded video datasets.

---

### 7. Innovation & Impact
*   **Multi-Class Safety Check**: Unlike standard drowsy detectors, this system simultaneously checks for seatbelt compliance and behavioral distractions.
*   **Edge-Ready**: Optimized to run on low-power mobile processors, making it a viable solution for real-world automotive integration.

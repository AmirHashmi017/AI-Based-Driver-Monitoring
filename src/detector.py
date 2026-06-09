import cv2
import numpy as np
from ultralytics import YOLO

class YoloDriverDetector:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)
        
        self.class_names = self.model.names

    def detect(self, frame):
        results = self.model(frame, verbose=False)[0]
        detections = []
        
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            label = self.class_names[cls]
            
            detections.append({
                "box": [int(x1), int(y1), int(x2), int(y2)],
                "conf": conf,
                "class": cls,
                "label": label
            })
            
        return detections

    def draw_detections(self, frame, detections):
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            label = f"{det['label']} {det['conf']:.2f}"
            label_lower = det['label'].lower()
            if "drowsy" in label_lower or "sleep" in label_lower:
                color = (0, 0, 255) 
            elif "phone" in label_lower:
                color = (0, 165, 255) 
            elif "belt" in label_lower:
                color = (0, 255, 0) 
            elif "awake" in label_lower:
                color = (0, 255, 0)
            else:
                color = (0, 255, 255) 

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame

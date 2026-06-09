import cv2
import time
from src.detector import YoloDriverDetector

def main():
    detector = YoloDriverDetector("models/best_dms_phone_model.pt") 

    video_source = 0 
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    print("YOLO Driver Monitoring System Started...")
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        detections = detector.detect(frame)

        frame = detector.draw_detections(frame, detections)

        status = "Normal"
        for det in detections:
            label = det['label'].lower()
            if "drowsy" in label or "close" in label:
                status = "DROWSY ALERT!"

                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), 10)
                cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
            elif "phone" in label:
                status = "PHONE USAGE DETECTED"

        cv2.putText(frame, f"STATUS: {status}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        try:
            cv2.imshow("YOLO Driver Monitor", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except cv2.error:
            print("Error: Could not display window. Make sure you have 'opencv-python' installed and not 'opencv-python-headless'.")
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


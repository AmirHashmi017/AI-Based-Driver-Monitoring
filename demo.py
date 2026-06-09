import cv2
import time
import argparse
from src.detector import YoloDriverDetector

def run_demo(video_path, model_path="yolov8n.pt", output_path="outputs/output_demo.mp4", show_gui=True):

    detector = YoloDriverDetector(model_path)
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Starting demo on {video_path}...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        start_time = time.time()

        detections = detector.detect(frame)

        frame = detector.draw_detections(frame, detections)

        curr_fps = 1.0 / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {curr_fps:.1f}", (10, height - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        out.write(frame)
        
        if show_gui:
            try:
                cv2.imshow("Driver Monitoring Demo", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except cv2.error:
                print("Warning: GUI window not available. Continuing in background mode...")
                show_gui = False # Disable for future frames
            
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Demo complete. Output saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to input video")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Path to .pt weights")
    parser.add_argument("--no-gui", action="store_true", help="Run without showing a window")
    args = parser.parse_args()
    
    run_demo(args.input, args.model, show_gui=not args.no_gui)

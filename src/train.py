from ultralytics import YOLO
from roboflow import Roboflow
import os
import shutil
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("ROBOFLOW_API_KEY")

def train(epochs=100):
    rf = Roboflow(api_key=API_KEY)
    project = rf.workspace("amirs-workspace-tyygo").project("vigil-drive-3mjq0")
    dataset = project.version(1).download("yolov8")

    model = YOLO("yolov8n.pt")
    model.train(
        data=os.path.join(dataset.location, "data.yaml"),
        epochs=epochs,
        imgsz=640,
        project="runs",
        name="dms_train"
    )

    if not os.path.exists("models"):
        os.makedirs("models")
    if not os.path.exists("metrics"):
        os.makedirs("metrics")

    shutil.copy("runs/dms_train/weights/best.pt", "models/best_dms_phone_model.pt")
    
    for f in ["results.png", "confusion_matrix.png", "F1_curve.png"]:
        src = os.path.join("runs/dms_train", f)
        if os.path.exists(src):
            shutil.copy(src, os.path.join("metrics", f))

if __name__ == "__main__":
    train()

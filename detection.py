# detection.py - YOLOv5 detection
import torch
import numpy as np
from PIL import Image

_yolo_model = None  # Singleton pattern for lazy loading

def load_yolo_model(weights_path="best.pt"):
    global _yolo_model
    if _yolo_model is None:
        _yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)
    return _yolo_model

def detect_scale_bars(model, image):
    results = model(image)
    print("YOLO raw output:", results)
    detections = results.xyxy[0].cpu().numpy()
    return [] if detections.size == 0 else detections

# NEW: Agent-compatible version using just the image path
def detect_scale_bars_from_path(image_path: str) -> str:
    """
    Detects scale bars in an image using YOLOv5.
    Returns a stringified list of bounding boxes for LLM use.
    """
    model = load_yolo_model("best.pt")
    image = Image.open(image_path).convert("RGB")
    detections = detect_scale_bars(model, np.array(image))
    
    formatted = [
        [float(x1), float(y1), float(x2), float(y2), float(conf)]
        for x1, y1, x2, y2, conf, _ in detections
    ]
    return str(formatted)

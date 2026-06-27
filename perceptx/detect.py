from ultralytics import YOLO
import supervision as sv
import numpy as np

ADAS_classes = [0,1,2,3,5,7,9,11]
class Detector:
    def __init__(self, model_path = "yolov8s.pt" , confidence =0.4, device ="cuda"):
        self.model = YOLO(model_path)
        self.confidence =confidence
        self.device = device

    def detect(self, frame):
        results = self.model(frame, conf = self.confidence, device = self.device, verbose = False)[0]
        detections = sv.Detections.from_ultralytics(results)
        mask = np.isin(detections.class_id, ADAS_classes)
        detections = detections[mask]
        return detections
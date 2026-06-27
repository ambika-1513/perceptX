import cv2
import numpy as np
import supervision as sv
from pathlib import Path
from tqdm import tqdm

from perceptx.detect import Detector
from perceptx.segment import Segmentor
from perceptx.track import Tracker

class PerceptXPipeline:
    def __init__(self, detector, segmentor, tracker, output_dir = "outputs"):

        self.detector = detector
        self.segmentor = segmentor
        self.tracker = tracker
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_scale=0.5)
        self.mask_annotator = sv.MaskAnnotator(opacity=0.4)

    def process_video(self, input_path):
        cap = cv2.VideoCapture(input_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        out_path = str(self.output_dir / f"{Path(input_path).stem}_perceptx.mp4")
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

        for _ in tqdm(range(total), desc="PerceptX"):
            ok, frame = cap.read()
            if not ok:
                break
            annotated = self.process_frame(frame)
            writer.write(annotated)

        cap.release()
        writer.release()
        return out_path

    def process_frame(self, frame):
        annotated = frame.copy()

        detections = self.detector.detect(frame)

        if len(detections) == 0:
            return annotated

        detections = self.tracker.update(detections)

        if self.segmentor is not None:
            masks = self.segmentor.segment(frame, detections.xyxy)
            if masks:
                detections.mask = np.array(masks)
            annotated = self.mask_annotator.annotate(annotated, detections)

        annotated = self.box_annotator.annotate(annotated, detections)
        annotated = self.label_annotator.annotate(annotated, detections)

        return annotated

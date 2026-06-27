import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor

class Segmentor:
    def __init__(self , checkpoint_path, model_type = "vit_b", device = "cuda"):
        sam = sam_model_registry[model_type](checkpoint = checkpoint_path)
        sam.to(device = device)
        self.predictor = SamPredictor(sam)
        self.device = device
    def segment(self, frame, boxes_xyxy):
        if len(boxes_xyxy) == 0: #if YOLOv8 found nothing
            return []
        rgb_frame = frame[:, :, ::-1] #BGR to RGB
        self.predictor.set_image(rgb_frame) #SAM processes the whole image once and stores internal features. Then we can prompt it with multiple boxes cheaply.
        boxes_tensor = torch.tensor(boxes_xyxy, dtype = torch.float32, device = self.device) #tensorconversion
        transformed_boxes = self.predictor.transform.apply_boxes_torch(boxes_tensor, rgb_frame.shape[:2]) #SAM internally resizes images to a fixed size (1024x1024). So our box coordinates need to be scaled to match. This line does that scaling autmatically
        masks, _, _ = self.predictor.predict_torch(point_coords =None, point_labels= None, boxes = transformed_boxes, multimask_output= False) #masks shape is (N, 1, H, W) — N objects, 1 mask each, H×W pixels.
        return [masks[i,0].cpu().numpy() for i in range(len(masks))]
import supervision as sv

class Tracker:
    def __init__(self): 
        self.tracker = sv.ByteTracker(
            minimum_matching_threshold=0.8,
        )
    def update(self, detections):
        if len(detections) == 0:
            return detections
        return self.tracker.update_with_detections(detections)

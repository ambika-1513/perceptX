import supervision as sv

class Tracker:
    def __init__(self, frame_rate =30, track_buffer =30): #keep a lost track alive for 30 frames before deleting it
        self.tracker = sv.ByteTrack(frame_rate = frame_rate, track_buffer = track_buffer)
    def update(self, detections):
        if len(detections) == 0:
            return detections
        return self.tracker.update_with_detections(detections)

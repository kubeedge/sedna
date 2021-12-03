from datetime import datetime
from typing import List


class DetTrackResult:
    def __init__(self, bbox : List = None, scene = None, confidence : List = None, detection_time : List = None, camera : int = 0, bbox_coord : List = [], tracking_ids : List = [], features : List = [], is_target=False, ID : List = []):
        self.bbox : List = bbox
        self.tracklets : List = tracking_ids
        self.bbox_coord : List = bbox_coord
        self.scene = scene
        self.confidence : List = confidence
        self.detection_time : List = detection_time
        self.camera : List = camera
        self.features : List = features
        self.is_target = is_target
        self.ID : List = ID
        try:
            self.image_key = f'{datetime.strptime(self.detection_time[0], "%a, %d %B %Y %H:%M:%S").timestamp()}_{self.camera[0]}' 
        except:
            self.image_key = "0"

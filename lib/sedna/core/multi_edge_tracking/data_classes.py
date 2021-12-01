from datetime import datetime
import json
from typing import List


class DetTrackResult:
    def __init__(self, bbox : List, scene, confidence : List, detection_time : List, camera : int, bbox_coord : List = [], tracking_ids : List = [], features : List = [], is_target=False, ID : List = []):
        self.bbox = bbox
        self.tracklets = tracking_ids
        self.bbox_coord = bbox_coord
        self.scene = scene
        self.confidence = confidence
        self.detection_time = detection_time
        self.camera = camera
        self.features = features
        self.is_target = is_target
        self.ID = ID
        try:
            self.image_key = f'{datetime.strptime(self.detection_time[0], "%a, %d %B %Y %H:%M:%S").timestamp()}_{self.camera[0]}' 
        except:
            self.image_key = 0

    def to_json(self):
        return json.dumps(self.__dict__)

    @classmethod
    def from_json(cls, json_str):
        json_dict = json.loads(json_str)
        return cls(**json_dict)

    # def build_copy(self, obj):
    #     return DetTrackResult(
    #     bbox = obj.bbox
    #     scene = obj.scene
    #         self.confidence = confidence
    #     self.detection_time = detection_time
    #     self.camera = camera
    #     self.features = features
    #     self.is_target = is_target
    #     self.ID = ID
    #     )

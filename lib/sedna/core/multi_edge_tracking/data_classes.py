from datetime import datetime
import json, numpy
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
        t = self

        t.scene = t.scene.tolist()
        t.confidence = [elem.tolist() if isinstance(elem, (numpy.generic) ) else elem for elem in t.confidence]
        t.features = [elem.tolist() if isinstance(elem, (numpy.ndarray, numpy.generic) ) else elem for elem in t.features]

        # t.features = [elem.tolist() if isinstance(elem, (numpy.ndarray, numpy.generic) ) else elem for elem in t.features]
        # t.features = [elem.tolist() if isinstance(elem, (numpy.ndarray, numpy.generic) ) else elem for elem in t.features]
        # t.features = [elem.tolist() if isinstance(elem, (numpy.ndarray, numpy.generic) ) else elem for elem in t.features]
         
        return json.dumps(t.__dict__)

    @classmethod
    def from_json(cls, json_str):
        json_dict = json.loads(json_str)

        t = cls(**json_dict)
        t.scene = numpy.asarray(t.scene)
        t.features = [numpy.asarray(elem) for elem in t.confidence]
        t.features = [numpy.asarray(elem) for elem in t.features]

        return t

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

from datetime import datetime
from typing import Dict, List
from enum import Enum

# Class defining the possible operational modes of the service.
class OP_MODE(Enum):
    DETECTION = "detection"
    TRACKING = "tracking"
    NOOP = "noop"

# Class defining the output of a ReID service.
class DetTrackResult:
    def __init__(self, bbox : List = None, scene = None, confidence : List = None, detection_time : List = None, camera : int = 0, bbox_coord : List = [], tracking_ids : List = [], features : List = [], is_target=False, ID : List = []):
        self.userID = "DEFAULT"
        self.bbox : List = bbox
        self.tracklets : List = tracking_ids
        self.bbox_coord : List = bbox_coord
        self.scene = scene
        self.confidence : List = confidence
        self.detection_time = detection_time
        self.camera = camera
        self.features : List = features
        self.is_target = is_target
        self.targetID : List = ID

        try:
            self.image_key = f'{datetime.strptime(self.detection_time, "%a, %d %B %Y %H:%M:%S.%f").timestamp()}_{self.camera}' 
        except:
            self.image_key = "0"

# Class used for internal synchronization among the pods of the ReID service.
class SyncDS:
    def __init__(self) -> None:
        self.op_mode = OP_MODE.DETECTION
        self.threshold = 0.75
        self.last_update = datetime.now().strftime("%a, %d %B %Y %H:%M:%S.%f")
        self.targets_collection : List[TargetImages] = [] # A list of targets, for each userid.

class TargetImages:
    def __init__(self, userid, targets = []) -> None:
        self.userid = userid
        self.targets = targets
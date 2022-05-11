# Copyright 2021 The KubeEdge Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from datetime import datetime
from typing import List
from enum import Enum

# Class defining the possible operational modes of the service.
class OP_MODE(Enum):
    DETECTION = "detection"
    TRACKING = "tracking"
    COVID19 = "covid19"
    NOOP = "noop"

# Class defining the output of a ReID service.
class DetTrackResult:
    def __init__(self, frame_index : int = 0, bbox : List = None, scene = None, confidence : List = None, detection_time : List = None, camera : int = 0, bbox_coord : List = [], tracking_ids : List = [], features : List = [], is_target=False, ID : List = []):
        self.userID = "DEFAULT" # Name of the enduser using the application, used to bound the user to the results
        self.frame_index = frame_index # Video frame index number
        self.bbox : List = bbox # List of bounding boxes containing, for example, pedestrians
        self.tracklets : List = tracking_ids # List of tracking IDs, one per bbox
        self.bbox_coord : List = bbox_coord # Coordinates of each bbox
        self.scene = scene # Original video frame
        self.confidence : List = confidence # Confidence value of the detection pass
        self.detection_time = detection_time # When detection was triggered (date)
        self.camera = camera # ID of the camera  where the video stream was acquired
        self.features : List = features # List of features extracted for each bbox
        self.is_target = is_target # Index of the target in the list of features
        self.targetID : List = ID # List of subjects IDs associated to the list of features. For example ['0002', '0001']. It depends on the ReID gallery.
        
        # Image key is used to uniquely identify the video frame associated with this object
        try:
            self.image_key = f'{datetime.strptime(self.detection_time, "%a, %d %B %Y %H:%M:%S.%f").timestamp()}_{self.camera}' 
        except:
            self.image_key = "0"

class TargetImages:
    def __init__(self, userid, targets = []) -> None:
        self.userid = userid
        self.targets = targets

class Target:
    def __init__(self, _userid, _features, _targetid="0000", _tracking_id=None, _location = None, _frame_index = 0) -> None:
        self.userid : str = _userid
        self.features : List = _features
        self.targetid : str = _targetid
        self.tracking_id : str = _tracking_id
        self.location : str = _location
        self.frame_index : int = _frame_index

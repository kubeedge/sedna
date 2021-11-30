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


    def find_target(self, target_id):
        for idx, elem in enumerate(self.ID):
            if int(elem) == int(target_id):
                return DetTrackResult(
                    self.bbox[idx],
                    self.scene,
                    self.confidence[idx],
                    self.detection_time[idx],
                    self.camera[idx],
                    self.features[idx],
                    self.is_target,
                    elem
                )

        return None

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

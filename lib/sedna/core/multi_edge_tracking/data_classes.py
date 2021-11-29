class DetTrackResult:
    def __init__(self) -> None:
        self.frame_nr = frame_nr
        self.bbox = bbox
        self.scene = scene
        self.confidence = confidence
        self.detection_time = detection_time
        self.camera = camera

class FEResult:
    def __init__(self) -> None:
        self.features = features
        self.is_target = is_target
        self.detection_time = detection_time
        self.camera = camera

class ReIDResult:
    def __init__(self) -> None:
        self.ID = ID
        self.features = features
        self.detection_time = detection_time
        self.camera = camera

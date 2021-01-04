import json


class ServiceInfo:
    def __init__(self):
        self.startTime = ''
        self.updateTime = ''
        self.inferenceNumber = 0
        self.hardExampleNumber = 0
        self.uploadCloudRatio = 0

    @staticmethod
    def from_json(json_str):
        info = ServiceInfo()
        info.__dict__ = json.loads(json_str)
        return info

    def to_json(self):
        info = json.dumps(self.__dict__)
        return info

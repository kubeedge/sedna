import os
import time
import json
import requests
import threading

from sedna.common.file_ops import FileOps
from sedna.common.class_factory import ClassFactory, ClassType

__all__ = ('UnseenSampleDetection', )


@ClassFactory.register(ClassType.UTD)
class UnseenSampleDetection(threading.Thread):
    '''
    Divide inference samples into seen samples and unseen samples

    Parameters
    ----------
    task_index: str or dict
        knowledge base index which includes indexes of tasks, samples and etc.
    '''
    # MODEL_MANIPULATION_SEM = threading.Semaphore(1)

    def __init__(self, edge_knowledge_management, **kwargs):
        self.get_environ_varia()
        self.unseen_save_url = edge_knowledge_management.local_unseen_save_url
        self.check_time = 1
        self.current_status = "False" # 默认机器狗从未跌倒状态开始
        self.current_sample_num = 0
        super(UnseenSampleDetection, self).__init__()

    def run(self):
        while True:
            self.get_environ_varia()
            time.sleep(self.check_time)
            try:
                check_request = requests.post(
                    url=self.status_service_ip
                )
                status_dict = json.loads(check_request.text)
                if status_dict["if_fall"] == "False":
                    continue
                elif status_dict["if_fall"] == "True" and self.current_status == "False":
                    self.current_status = "True" # 修改机器狗目前状态为摔倒
                    samples = os.listdir(self.local_image_url)
                    samples.sort(reverse=True)
                    # start_idx = status_dict["time_stamp"]
                    # # 这里缺少原时间戳到目标图片名称的映射，以及到目标图片的idx的映射
                    # end_idx = 10
                    if len(samples) > 0:
                        start_idx, end_idx = self.get_index(samples, status_dict["time_stamp"])
                        
                        for sample in samples[start_idx:end_idx]:
                            local_sample_url = FileOps.join_path(self.local_image_url, sample)
                            dest_sample_url = FileOps.join_path(self.unseen_save_url, sample)
                            FileOps.upload(local_sample_url, dest_sample_url, clean=False)
                else:
                    continue
            except Exception as e:
                continue

    def get_index(self, samples, timestamp):
        # 在该函数中根据状态服务返回的时间戳判断摔倒时的时间戳，并返回
        # status_dict["time_stamp"]的格式： 16XXXXX.XX
        time_stamp = int(timestamp)
        for i in range(len(samples)):
            if int(samples[i].split(".")[0]) <= time_stamp:
                start_idx = i
                end_idx = min(i + 10, len(samples) - 1)
                return start_idx, end_idx
            else:
                continue
        return 0, len(samples) - 1

    def get_environ_varia(self):
        try:
            self.status_service_ip = os.environ["STATUS_IP"]
        except:
            self.status_service_ip = "127.0.0.1"
        self.query_url = "http://" + self.status_service_ip + ":8000/robot_status/query/"
        try:
            self.local_image_url = os.environ["IMAGE_TOPIC_URL"]
        except:
            self.local_image_url = "/tmp/"



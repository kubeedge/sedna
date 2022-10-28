from prometheus_client import start_http_server,Gauge
import os
import time


JointInference_num = Gauge('JointInference_InferenceCount','InferenceCount for Joint Inference',['type'])


def get_joint_inference_metrics(cloud_inference_path, output_path):
    JointInference_num.labels(type='Edge').set(len(os.listdir(output_path)))
    JointInference_num.labels(type='Cloud').set(len(os.listdir(cloud_inference_path)))


if __name__ == "__main__":
    '''
    These are paths in demo test cases.
    If you have run your own tasks, please change the following paths to the paths you used.
    If you want to monitor multiple tasks, you need to change this exporter a little.
    When Sedna is available to show metrics like inference count in -oyaml, there is no need to run this exporter.
    '''
    # joint inference
    cloud_inference_path = "/joint_inference/output/hard_example_cloud_inference_output"
    output_path = "/joint_inference/output/output"

    start_http_server(8001)

    while True:
        get_joint_inference_metrics(cloud_inference_path, output_path)
        time.sleep(10)

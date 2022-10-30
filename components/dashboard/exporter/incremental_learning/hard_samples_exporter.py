from prometheus_client import start_http_server,Gauge
import os
import time


Incremental_num = Gauge('Incremental_HardSamples', 'Hard Samples for Incremental', ['hardSample'])


def get_incremental_learning_metrics(hard_samples_path):
    Incremental_num.labels(hardSample=True).set(len(os.listdir(hard_samples_path)))


if __name__ == "__main__":
    '''
    These are paths in demo test cases.
    If you have run your own tasks, please change the following paths to the paths you used.
    If you want to monitor multiple tasks, you need to change this exporter a little.
    When Sedna is available to show metrics like inference count in -oyaml, there is no need to run this exporter.
    '''
    # incremental learning
    hard_samples_path = "/incremental_learning/he"

    start_http_server(8000)

    while True:
        get_incremental_learning_metrics(hard_samples_path)
        time.sleep(10)

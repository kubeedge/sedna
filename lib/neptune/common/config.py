import os


class BaseConfig:
    """The base config, the value can not be changed."""
    # dataset
    train_dataset_url = os.getenv("TRAIN_DATASET_URL")
    test_dataset_url = os.getenv("TEST_DATASET_URL")
    data_path_prefix = os.getenv("DATA_PATH_PREFIX", "/home/data")
    # k8s crd info
    namespace = os.getenv("NAMESPACE", "")
    worker_name = os.getenv("WORKER_NAME", "")
    # the name of JointInferenceService and others Service
    service_name = os.getenv("SERVICE_NAME", "")
    # the name of FederatedLearningJob and others Job
    job_name = os.getenv("JOB_NAME", "")

    model_url = os.getenv("MODEL_URL")

    # user parameter
    parameters = os.getenv("PARAMETERS")
    # Hard Example Mining Algorithm
    hem_name = os.getenv("HEM_NAME")
    hem_parameters = os.getenv("HEM_PARAMETERS")

    def __init__(self):
        pass

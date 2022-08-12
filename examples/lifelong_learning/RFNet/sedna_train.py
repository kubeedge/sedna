import os
os.environ['BACKEND_TYPE'] = 'PYTORCH'
os.environ["OUTPUT_URL"] = "./cloud_kb/"
# os.environ['CLOUD_KB_INDEX'] = "./cloud_kb/index.pkl"
os.environ["TRAIN_DATASET_URL"] = "./data_txt/sedna_data.txt"
os.environ["KB_SERVER"] = "http://0.0.0.0:9020"
os.environ["HAS_COMPLETED_INITIAL_TRAINING"] = "false"

from sedna.common.file_ops import FileOps
from sedna.datasources import IndexDataParse
from sedna.common.config import Context, BaseConfig
from sedna.core.lifelong_learning import LifelongLearning

from basemodel import Model

def _load_txt_dataset(dataset_url):
    # use original dataset url
    original_dataset_url = Context.get_parameters('original_dataset_url')
    return os.path.join(os.path.dirname(original_dataset_url), dataset_url)

def train(estimator, train_data):
    task_definition = {
        "method": "TaskDefinitionByOrigin",
        "param": {
            "origins": ["real", "sim"]
        }
    }

    task_allocation = {
        "method": "TaskAllocationByOrigin",
        "param": {
            "origins": ["real", "sim"]
        }
    }

    ll_job = LifelongLearning(estimator,
                              task_definition=task_definition,
                              task_relationship_discovery=None,
                              task_allocation=task_allocation,
                              task_remodeling=None,
                              inference_integrate=None,
                              task_update_decision=None,
                              unseen_task_allocation=None,
                              unseen_sample_recognition=None,
                              unseen_sample_re_recognition=None
                              )

    ll_job.train(train_data)

def update(estimator, train_data):
    ll_job = LifelongLearning(estimator,
                              task_definition=None,
                              task_relationship_discovery=None,
                              task_allocation=None,
                              task_remodeling=None,
                              inference_integrate=None,
                              task_update_decision=None,
                              unseen_task_allocation=None,
                              unseen_sample_recognition=None,
                              unseen_sample_re_recognition=None
                              )

    ll_job.update(train_data)

def run():
    estimator = Model()
    train_dataset_url = BaseConfig.train_dataset_url
    train_data = IndexDataParse(data_type="train")
    train_data.parse(train_dataset_url, use_raw=False)

    is_completed_initilization = str(Context.get_parameters("HAS_COMPLETED_INITIAL_TRAINING", "false")).lower()
    if is_completed_initilization == "false":
        train(estimator, train_data)
    else:
        update(estimator, train_data)

if __name__ == '__main__':
    run()

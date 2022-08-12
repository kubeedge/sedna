import os
os.environ['BACKEND_TYPE'] = 'PYTORCH'
# os.environ["KB_SERVER"] = "http://0.0.0.0:9020"
# os.environ["test_dataset_url"] = "./data_txt/sedna_data.txt"
# os.environ["MODEL_URLS"] = "./cloud_next_kb/index.pkl"
# os.environ["operator"] = "<"
# os.environ["model_threshold"] = "0"

from sedna.core.lifelong_learning import LifelongLearning
from sedna.datasources import IndexDataParse
from sedna.common.config import Context

from accuracy import accuracy
from basemodel import Model

def _load_txt_dataset(dataset_url):
    # use original dataset url
    original_dataset_url = Context.get_parameters('original_dataset_url')
    return os.path.join(os.path.dirname(original_dataset_url), dataset_url)

def eval():
    estimator = Model()
    eval_dataset_url = Context.get_parameters("test_dataset_url")
    eval_data = IndexDataParse(data_type="eval", func=_load_txt_dataset)
    eval_data.parse(eval_dataset_url, use_raw=False)

    task_allocation = {
        "method": "TaskAllocationByOrigin",
        "param": {
            "origins": ["real", "sim"]
        }
    }

    ll_job = LifelongLearning(estimator,
                              task_definition=None,
                              task_relationship_discovery=None,
                              task_allocation=task_allocation,
                              task_remodeling=None,
                              inference_integrate=None,
                              task_update_decision=None,
                              unseen_task_allocation=None,
                              unseen_sample_recognition=None,
                              unseen_sample_re_recognition=None
                              )

    ll_job.evaluate(eval_data, metrics=accuracy)


if __name__ == '__main__':
    print(eval())

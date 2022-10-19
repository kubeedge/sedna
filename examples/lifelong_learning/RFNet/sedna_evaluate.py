import os
# os.environ["KB_SERVER"] = "http://0.0.0.0:9020"
# os.environ["OUTPUT_URL"] = "./cloud_kb/"
# os.environ["test_dataset_url"] = "./data_txt/sedna_test.txt"
# os.environ["MODEL_URLS"] = "./cloud_kb/index.pkl"
# os.environ["operator"] = "<"
# os.environ["model_threshold"] = "0.01"

from basemodel import Model
from sedna.core.lifelong_learning import LifelongLearning
from sedna.datasources import TxtDataParse
from sedna.common.config import Context

from accuracy import accuracy, robo_accuracy

def _load_txt_dataset(dataset_url):
    # use original dataset url
    original_dataset_url = Context.get_parameters('original_dataset_url', "")
    dataset_urls = dataset_url.split()
    dataset_urls = [
        os.path.join(
            os.path.dirname(original_dataset_url),
            dataset_url) for dataset_url in dataset_urls]
    return dataset_urls[:-1], dataset_urls[-1]

def eval():
    estimator = Model(num_class=31)
    eval_dataset_url = Context.get_parameters("test_dataset_url")
    eval_data = TxtDataParse(data_type="eval", func=_load_txt_dataset)
    eval_data.parse(eval_dataset_url, use_raw=False)

    task_allocation = {
        "method": "TaskAllocationByOrigin"
    }

    inference_integrate = {
        "method": "InferenceIntegrateByType"
    }

    ll_job = LifelongLearning(estimator,
                              task_definition=None,
                              task_relationship_discovery=None,
                              task_allocation=task_allocation,
                              task_remodeling=None,
                              inference_integrate=inference_integrate,
                              task_update_decision=None,
                              unseen_task_allocation=None,
                              unseen_sample_recognition=None,
                              unseen_sample_re_recognition=None
                              )

    ll_job.evaluate(eval_data, metrics=robo_accuracy)


if __name__ == '__main__':
    print(eval())

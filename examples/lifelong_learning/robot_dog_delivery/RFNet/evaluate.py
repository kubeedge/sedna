import os

from sedna.core.lifelong_learning import LifelongLearning
from sedna.datasources import TxtDataParse
from sedna.common.config import Context

from accuracy import robo_accuracy
from basemodel import Model


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
        "method": "TaskAllocationSimple"
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

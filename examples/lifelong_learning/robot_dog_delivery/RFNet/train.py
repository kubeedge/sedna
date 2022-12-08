import os
os.environ["TRAIN_DATASET_URL"] = "/home/lsq/RFNet/data_index/train.txt"
os.environ["OUTPUT_URL"] = "s3://kubeedge/sedna-robo/kb/"
os.environ["S3_ENDPOINT_URL"] = "https://obs.cn-north-1.myhuaweicloud.com"
os.environ["SECRET_ACCESS_KEY"] = "OYPxi4uD9k5E90z0Od3Ug99symbJZ0AfyB4oveQc"
os.environ["ACCESS_KEY_ID"] = "EMPTKHQUGPO2CDUFD2YR"
os.environ["KB_SERVER"] = "http://0.0.0.0:9020"

from sedna.core.lifelong_learning import LifelongLearning
from sedna.common.config import Context, BaseConfig
from sedna.datasources import TxtDataParse

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


def train(estimator, train_data):
    task_definition = {
        "method": "TaskDefinitionSimple"
    }

    task_allocation = {
        "method": "TaskAllocationSimple"
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
    estimator = Model(num_class=31, epochs=1)
    train_dataset_url = BaseConfig.train_dataset_url
    train_data = TxtDataParse(data_type="train", func=_load_txt_dataset)
    train_data.parse(train_dataset_url, use_raw=False)

    train(estimator, train_data)


if __name__ == '__main__':
    run()

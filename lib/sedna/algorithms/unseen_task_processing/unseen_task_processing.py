import json

from sedna.backend import set_backend
from sedna.common.file_ops import FileOps
from sedna.common.constant import KBResourceConstant
from sedna.common.class_factory import ClassFactory, ClassType
from sedna.common.log import LOGGER
from sedna.algorithms.unseen_task_processing.GANwithSelfTaughtLearning.deeplabv3.train import train_deepblabv3
from sedna.algorithms.unseen_task_processing.GANwithSelfTaughtLearning.GAN.train import train as trainGAN
from sedna.algorithms.unseen_task_processing.GANwithSelfTaughtLearning.selftaughtlearning.train import train as trainAutoEncoder

__all__ = ('UnseenTaskProcessing', )


class UnseenTaskProcessing:
    '''
    Process unseen tasks given task update strategies

    Parameters:
    ----------
    estimator: Instance
        An instance with the high-level API that greatly simplifies
        machine learning programming. Estimators encapsulate training,
        evaluation, prediction, and exporting for your model.
    cloud_knowledge_management: Instance of class CloudKnowledgeManagement
    unseen_task_allocation: Dict
        Mining tasks of unseen inference sample.
    '''

    def __init__(self, estimator, config,
                 cloud_knowledge_management,
                 edge_knowledge_management,
                 unseen_task_allocation,
                 **kwargs):
        self.estimator = set_backend(estimator=estimator, config=config)
        self.cloud_knowledge_management = cloud_knowledge_management
        self.edge_knowledge_management = edge_knowledge_management

        self.unseen_task_allocation = unseen_task_allocation or {
            "method": "UnseenTaskAllocationDefault"
        }
        self.unseen_models = None
        self.unseen_extractor = None
        self.unseen_task_groups = None
        self.unseen_task_key = KBResourceConstant.UNSEEN_TASK.value
        self.task_group_key = KBResourceConstant.TASK_GROUPS.value
        self.extractor_key = KBResourceConstant.EXTRACTOR.value

    @staticmethod
    def _parse_param(param_str):
        if not param_str:
            return {}
        if isinstance(param_str, dict):
            return param_str
        try:
            raw_dict = json.loads(param_str, encoding="utf-8")
        except json.JSONDecodeError:
            raw_dict = {}
        return raw_dict

    def _unseen_task_allocation(self, samples):
        """
        Mining unseen tasks of inference sample base on task attribute extractor
        """
        method_name = self.unseen_task_allocation.get("method")
        extend_param = self._parse_param(
            self.unseen_task_allocation.get("param")
        )

        method_cls = ClassFactory.get_cls(ClassType.UTP, method_name)(
            task_extractor=self.unseen_extractor, **extend_param
        )
        return method_cls(samples=samples)

    def initialize(self):
        """
        Intialize unseen task groups

        Returns:
        res: Dict
            evaluation result.
        task_index: Dict or str
            unseen task index which includes models, samples, extractor and etc.
        """
        task_index = {
            self.extractor_key: None,
            self.task_group_key: []
        }

        res = {}
        return res, task_index

    def update(self, tasks, task_update_strategies, **kwargs):
        """
        Parameters:
        ----------
        tasks: List[Task]
            from the output of module task_update_decision
        task_update_strategies: Dict
            from the output of module task_update_decision

        Returns
        -------
        task_index : Dict
            updated unseen task index of knowledge base
        """
        task_index = {
            self.extractor_key: None,
            self.task_group_key: []
        }

        unseen_samples = kwargs['unseen_samples']
        is_autoencoder_trained = kwargs['is_autoencoder_trained']
        self.log.info(self.TAG + 'start processing unseen task')
        # If this unseen task has no corresponding model, then we need to use GAN plus self-taught learning to train a encoder model.
        # Here, I use `IS_AUTOENCODER_TRAINED` to let developer himself/herself decide whether to train an encoder model from scrach.
        if not is_autoencoder_trained:
            self.log.info(
                self.TAG + 'No autoencoder model is found. So starting training...')
            self.log.info(self.TAG + 'firstly, process training GAN')
            # use GAN model to train GAN for generating more samples
            trainGAN(unseen_samples)
            self.log.info(self.TAG + 'secondly, process self-taught module')
            # train aotuencoder in self-taught learning module
            trainAutoEncoder(unseen_samples)
        self.log.info(
            'load trained autoencoder model from self-taught learning module')
        self.log.info('encoder unseen samples')
        #
        # start training
        train_deepblabv3(unseen_samples)

        return task_index

    def predict(self, data, post_process=None, **kwargs):
        """
        Predict the result for unseen data.

        Parameters
        ----------
        data : BaseDataSource
            inference sample, see `sedna.datasources.BaseDataSource` for
            more detail.
        post_process: function
            function or a registered method,  effected after `estimator`
            prediction, like: label transform.

        Returns
        -------
        result : array_like
            results array, contain all inference results in each sample.
        tasks : List
            tasks assigned to each sample.
        """
        if not self.unseen_task_groups and not self.unseen_models:
            self.load(self.edge_knowledge_management.task_index)

        samples, mappings = self._unseen_task_allocation(data)
        result = {}
        tasks = []

        return result, tasks

    def load(self, task_index):
        """
        load task_detail (tasks/models etc ...) from task index file.
        It'll automatically loaded during `inference` phases.

        Parameters
        ----------
        task_index_url : str
            task index file path.
        """
        assert task_index is not None, "task index url is None!!!"
        if isinstance(task_index, str):
            task_index = FileOps.load(task_index)

        self.unseen_extractor = task_index[self.unseen_task_key][self.extractor_key]
        if isinstance(self.unseen_extractor, str):
            self.unseen_extractor = FileOps.load(self.unseen_extractor)
        self.unseen_task_groups = task_index[self.unseen_task_key][self.task_group_key]
        self.unseen_models = [task.model for task in self.unseen_task_groups]

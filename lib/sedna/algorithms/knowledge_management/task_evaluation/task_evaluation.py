from sedna.common.config import Context
from sedna.common.log import LOGGER
from sedna.common.class_factory import ClassType, ClassFactory

__all__ = ('TaskEvaluationDefault', )


@ClassFactory.register(ClassType.KM)
class TaskEvaluationDefault:
    """
    Evaluated the performance of each seen task and filter seen tasks
    based on defined rules.

    Parameters
    ----------
    estimator: Instance
        An instance with the high-level API that greatly simplifies
        machine learning programming. Estimators encapsulate training,
        evaluation, prediction, and exporting for your model.
    """

    def __init__(self, **kwargs):
        self.log = LOGGER

    def __call__(self, tasks_detail, **kwargs):
        """
        Parameters
        ----------
        tasks_detail: List[Task]
            output of module task_update_decision, consisting of results of evaluation.
        metrics : function / str
            Metrics to assess performance on the task by given prediction.
        metrics_param : Dict
            parameter for metrics function.
        kwargs: Dict
            parameters for `estimator` evaluate.

        Returns
        -------
        drop_task: List[str]
            names of the tasks which will not to be deployed to the edge.
        """

        self.model_filter_operator = Context.get_parameters("operator", ">")
        self.model_threshold = float(
            Context.get_parameters(
                "model_threshold", 0.1))

        drop_tasks = []

        operator_map = {
            ">": lambda x, y: x > y,
            "<": lambda x, y: x < y,
            "=": lambda x, y: x == y,
            ">=": lambda x, y: x >= y,
            "<=": lambda x, y: x <= y,
        }
        if self.model_filter_operator not in operator_map:
            self.log.warn(
                f"operator {self.model_filter_operator} use to "
                f"compare is not allow, set to <"
            )
            self.model_filter_operator = "<"
        operator_func = operator_map[self.model_filter_operator]

        for detail in tasks_detail:
            scores = detail.scores
            entry = detail.entry
            self.log.info(f"{entry} scores: {scores}")
            if any(map(lambda x: operator_func(float(x),
                                               self.model_threshold),
                       scores.values())):
                self.log.warn(
                    f"{entry} will not be deploy because all "
                    f"scores {self.model_filter_operator} {self.model_threshold}")
                drop_tasks.append(entry)
                continue
        drop_task = ",".join(drop_tasks)

        return drop_task

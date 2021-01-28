import logging
from enum import Enum

LOG = logging.getLogger(__name__)


class ModelType(Enum):
    GlobalModel = 1
    PersonalizedModel = 2


class Framework(Enum):
    Tensorflow = "tensorflow"
    Pytorch = "pytorch"
    Mindspore = "mindspore"


class K8sResourceKind(Enum):
    JOINT_INFERENCE_SERVICE = "jointinferenceservice"
    FEDERATED_LEARNING_JOB = "federatedlearningjob"
    INCREMENTAL_JOB = "incrementallearningjob"


class K8sResourceKindStatus(Enum):
    COMPLETED = "completed"
    FAILED = "failed"
    RUNNING = "running"


FRAMEWORK = Framework.Tensorflow  # TODO: should read from env.

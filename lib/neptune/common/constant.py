import logging
from enum import Enum

LOG = logging.getLogger(__name__)


class Framework(Enum):
    Tensorflow = "tensorflow"
    Pytorch = "pytorch"
    Mindspore = "mindspore"


class K8sResourceKind(Enum):
    JOINT_INFERENCE_SERVICE = "jointinferenceservice"


class K8sResourceKindStatus(Enum):
    COMPLETED = "completed"
    FAILED = "failed"
    RUNNING = "running"


FRAMEWORK = Framework.Tensorflow  # TODO: should read from env.

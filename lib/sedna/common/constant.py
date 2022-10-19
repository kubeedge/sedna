# Copyright 2021 The KubeEdge Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from enum import Enum


class K8sResourceKind(Enum):
    """
    Sedna job/service kind
    """

    DEFAULT = "default"
    REID_JOB = "reidjob"
    VIDEO_ANALYTICS_JOB = "videoanalyticsjob"
    FEATURE_EXTRACTION_SERVICE = "featureextractionservice"
    JOINT_INFERENCE_SERVICE = "jointinferenceservice"
    FEDERATED_LEARNING_JOB = "federatedlearningjob"
    INCREMENTAL_JOB = "incrementallearningjob"
    LIFELONG_JOB = "lifelonglearningjob"


class K8sResourceKindStatus(Enum):
    """
    Job/Service status
    """

    COMPLETED = "completed"
    FAILED = "failed"
    RUNNING = "running"


class KBResourceConstant(Enum):
    """
    Knowledge used constant
    """

    MIN_TRAIN_SAMPLE = 10
    KB_INDEX_NAME = "index.pkl"
    TASK_EXTRACTOR_NAME = "task_attr_extractor.pkl"
    SEEN_TASK = "seen_task"
    UNSEEN_TASK = "unseen_task"
    TASK_GROUPS = "task_groups"
    EXTRACTOR = "extractor"
    EDGE_KB_DIR = "/var/lib/sedna/kb"

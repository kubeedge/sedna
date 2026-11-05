# Copyright 2026 The KubeEdge Authors.
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

"""Regression tests for lifelong-learning task update decisions."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from sedna.algorithms.seen_task_learning.artifact import Task, TaskGroup
from sedna.common.class_factory import ClassFactory, ClassType
from sedna.common.config import BaseConfig
from sedna.common.file_ops import FileOps
from sedna.core.lifelong_learning import LifelongLearning
from sedna.datasources import BaseDataSource
from sedna.service.client import KBClient, LCClient


def make_samples(features, labels):
    """Create a labeled dataset without a task-specific schema."""
    samples = BaseDataSource()
    samples.x = np.array(features)
    samples.y = np.array(labels)
    return samples


class LocalEstimator:
    """Small persistent estimator for exercising the real learning pipeline."""

    def __init__(self):
        self.state = {"rounds": 0}

    def train(self, train_data):
        self.state = {
            "rounds": self.state["rounds"] + 1,
            "features": np.asarray(train_data.x).tolist(),
            "labels": np.asarray(train_data.y).tolist(),
        }
        return {"sample_count": len(train_data)}

    def save(self, model_path):
        joblib.dump(self.state, model_path)

    def load(self, model_url):
        self.state = joblib.load(model_url)


class TaskUpdateRegistrationTest(unittest.TestCase):
    """The public imports must register both built-in update strategies."""

    def test_public_import_registers_update_strategies(self):
        imports = (
            "import sedna.algorithms.seen_task_learning",
            "from sedna.core.lifelong_learning import LifelongLearning",
        )
        for statement in imports:
            with self.subTest(statement=statement):
                result = subprocess.run(
                    [sys.executable, "-c", statement + "\n" +
                     "from sedna.common.class_factory import "
                     "ClassFactory, ClassType\n"
                     "ClassFactory.get_cls(ClassType.KM, "
                     "'UpdateStrategyDefault')\n"
                     "ClassFactory.get_cls(ClassType.KM, "
                     "'UpdateStrategyByFinetune')\n"],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    universal_newlines=True, check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_strategy_modules_export_their_actual_classes(self):
        modules = {
            "task_update_decision": "UpdateStrategyDefault",
            "task_update_decision_finetune": "UpdateStrategyByFinetune",
        }
        for module, class_name in modules.items():
            with self.subTest(module=module):
                namespace = {}
                exec("from sedna.algorithms.seen_task_learning."
                     "task_update_decision." + module + " import *",
                     namespace)
                self.assertIs(namespace[class_name], ClassFactory.get_cls(
                    ClassType.KM, class_name))


class DefaultTaskUpdateDecisionTest(unittest.TestCase):
    """The default must supply incoming data to every selected task."""

    def setUp(self):
        self.samples = make_samples([[1], [2]], [0, 1])
        self.samples.meta_attr = {"source": "new samples"}
        self.seen_tasks = [
            Task("north", make_samples([[10]], [0]), meta_attr="north"),
            Task("south", make_samples([[20]], [1]), meta_attr="south"),
            Task("west", make_samples([[30]], [0]), meta_attr="west"),
        ]
        for task in self.seen_tasks:
            task.model = task.entry + ".model"
        self.unseen_task = Task("unseen", make_samples([[40]], [1]))
        self.task_index = {
            "seen_task": {
                "extractor": {"domain": "test"},
                "task_groups": [
                    TaskGroup("paired", self.seen_tasks[:2]),
                    TaskGroup("empty", []),
                    TaskGroup("single", self.seen_tasks[2:]),
                ],
            },
            "unseen_task": {
                "extractor": None,
                "task_groups": [TaskGroup("unseen", [self.unseen_task])],
            },
        }
        self.strategy_class = ClassFactory.get_cls(
            ClassType.KM, "UpdateStrategyDefault")

    def test_updates_every_seen_task_with_incoming_samples(self):
        strategy = self.strategy_class(self.task_index)
        tasks, decisions = strategy(self.samples, task_type="seen_task")

        self.assertEqual(tasks, self.seen_tasks)
        self.assertEqual(set(decisions), {"north", "south", "west"})
        for task in tasks:
            self.assertIs(task.samples, self.samples)
            self.assertEqual(task.model, task.entry + ".model")
            self.assertEqual(task.meta_attr, task.entry)
            self.assertIs(decisions[task.entry]["raw_data_update"],
                          self.samples)
            self.assertIs(decisions[task.entry]["target_model_update"], True)
            self.assertIsNone(decisions[task.entry]["task_attr_update"])
        self.assertEqual(self.samples.meta_attr, {"source": "new samples"})
        self.assertEqual(self.unseen_task.samples.x.tolist(), [[40]])

    def test_selects_unseen_tasks_without_changing_seen_samples(self):
        strategy = self.strategy_class(self.task_index)
        tasks, decisions = strategy(self.samples, task_type="unseen_task")

        self.assertEqual(tasks, [self.unseen_task])
        self.assertEqual(set(decisions), {"unseen"})
        self.assertIs(self.unseen_task.samples, self.samples)
        self.assertIs(decisions["unseen"]["raw_data_update"], self.samples)
        self.assertIs(decisions["unseen"]["target_model_update"], True)
        self.assertIsNone(decisions["unseen"]["task_attr_update"])
        self.assertEqual(self.seen_tasks[0].samples.x.tolist(), [[10]])

    def test_empty_task_groups_need_no_update(self):
        for task_type in ("seen_task", "unseen_task"):
            for groups in ([], [TaskGroup("empty", [])]):
                with self.subTest(task_type=task_type, groups=groups):
                    self.task_index[task_type]["task_groups"] = groups
                    strategy = self.strategy_class(self.task_index)
                    self.assertEqual(strategy(self.samples, task_type),
                                     ([], {}))

    def test_loads_a_saved_task_index(self):
        with tempfile.TemporaryDirectory() as directory:
            index_path = FileOps.dump(
                self.task_index, str(Path(directory) / "index.pkl"))
            strategy = self.strategy_class(index_path)
            tasks, decisions = strategy(self.samples, task_type="seen_task")

        self.assertEqual([task.entry for task in tasks],
                         ["north", "south", "west"])
        self.assertEqual(set(decisions), {"north", "south", "west"})
        for task in tasks:
            self.assertIs(task.samples, self.samples)
            self.assertEqual(task.model, task.entry + ".model")

    def test_repeated_calls_replace_previous_training_samples(self):
        strategy = self.strategy_class(self.task_index)
        strategy(self.samples, task_type="seen_task")
        new_samples = make_samples([[3]], [1])
        tasks, decisions = strategy(new_samples, task_type="seen_task")

        for task in tasks:
            self.assertIs(task.samples, new_samples)
            self.assertIs(decisions[task.entry]["raw_data_update"],
                          new_samples)
        self.assertEqual(self.samples.x.tolist(), [[1], [2]])


class LifelongLearningUpdateTest(unittest.TestCase):
    """Exercise training and repeated updates using real local model files."""

    def test_default_update_reloads_and_retrains_existing_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_directory = str(root / "models")
            environment = {
                "BACKEND_TYPE": "UNKNOWN",
                "OUTPUT_URL": str(root / "cloud"),
                "EDGE_OUTPUT_URL": str(root / "edge"),
                "CLOUD_KB_INDEX": "cloud/index.pkl",
                "MIN_TRAIN_SAMPLE": "1",
                "HAS_COMPLETED_INITIAL_TRAINING": "false",
                "MODEL_HOT_UPDATE": "false",
            }
            with patch.dict(os.environ, environment), \
                    patch.object(BaseConfig, "data_path_prefix", directory), \
                    patch.object(BaseConfig, "model_url", model_directory), \
                    patch.object(KBClient, "upload_file", autospec=True,
                                 side_effect=lambda client, path: path), \
                    patch.object(KBClient, "update_db", autospec=True,
                                 side_effect=lambda client, path: path), \
                    patch.object(LCClient, "send"):
                job = LifelongLearning(
                    LocalEstimator,
                    task_definition={
                        "method": "TaskDefinitionByDataAttr",
                        "param": {"attribute": ["domain"]},
                    },
                )
                job.cloud_knowledge_management.local_task_index_url = str(
                    root / "downloaded-index.pkl")
                initial_samples = BaseDataSource()
                initial_samples.x = pd.DataFrame({
                    "domain": ["north", "north", "south", "south"],
                    "feature": [1, 3, 10, 20],
                })
                initial_samples.y = pd.Series([0, 1, 0, 1])
                job.train(initial_samples)

                index_path = str(root / "cloud" / "index.pkl")
                initial_index = FileOps.load(index_path)
                groups = initial_index["seen_task"]["task_groups"]
                self.assertEqual([group.entry for group in groups],
                                 ["north", "south"])
                for group in groups:
                    self.assertEqual(joblib.load(group.model.model)["rounds"],
                                     1)

                # A fresh job must load the prior index and each saved model.
                job = LifelongLearning(LocalEstimator)
                job.cloud_knowledge_management.local_task_index_url = str(
                    root / "downloaded-index.pkl")
                updates = (
                    make_samples([[5], [6], [7], [8]], [1, 0, 1, 0]),
                    make_samples([[9], [11]], [0, 1]),
                )
                os.environ["HAS_COMPLETED_INITIAL_TRAINING"] = "true"
                for round_number, samples in enumerate(updates, start=2):
                    if round_number == 2:
                        result = job.train(samples)
                    else:
                        result = job.update(samples)
                    self.assertEqual(result, index_path)
                    index = FileOps.load(result)
                    groups = index["seen_task"]["task_groups"]
                    self.assertEqual([group.entry for group in groups],
                                     ["north", "south"])
                    self.assertEqual(index["unseen_task"]["task_groups"], [])
                    for group in groups:
                        state = joblib.load(group.model.model)
                        self.assertEqual(state["rounds"], round_number)
                        self.assertEqual(state["features"], samples.x.tolist())
                        self.assertEqual(state["labels"], samples.y.tolist())
                        self.assertEqual(group.tasks[0].meta_attr,
                                         [group.entry])


if __name__ == "__main__":
    unittest.main()

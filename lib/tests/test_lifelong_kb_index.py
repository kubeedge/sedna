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

"""Regression tests for resolving the previous lifelong-learning KB index."""

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from sedna.common.config import BaseConfig, Context
from sedna.common.constant import K8sResourceKindStatus
from sedna.common.file_ops import FileOps
from sedna.core.lifelong_learning.lifelong_learning import LifelongLearning


class LifelongKBIndexTest(unittest.TestCase):
    """Exercise update() with real paths and isolated learning stages."""

    def setUp(self):
        self.job = LifelongLearning.__new__(LifelongLearning)
        self.manager = Mock()
        self.job.cloud_knowledge_management = self.manager
        self.job.log = Mock()
        self.job.config = SimpleNamespace(data_path_prefix="/home/data")
        self.job.report_task_info = Mock()
        self.job.unseen_sample_re_recognition = {"method": "recognition"}
        self.job.unseen_sample_re_recognition_param = {}
        self.job.task_update_decision = {"method": "decision"}
        self.job.task_update_decision_param = {}
        self.manager.local_task_index_url = "index.pkl"
        self.manager.update_kb.return_value = "updated/index.pkl"

        def samples():
            return SimpleNamespace(x=np.array([1]), y=np.array([0]))

        self.recognition = Mock(side_effect=lambda _: (samples(), samples()))
        self.decision = Mock(return_value=(["task"], ["update"]))
        self.recognition_factory = Mock(return_value=self.recognition)
        self.decision_factory = Mock(return_value=self.decision)
        factory = patch(
            "sedna.core.lifelong_learning.lifelong_learning."
            "ClassFactory.get_cls",
            side_effect=lambda kind, name: {
                "recognition": self.recognition_factory,
                "decision": self.decision_factory}[name])
        factory.start()
        self.addCleanup(factory.stop)

    def test_index_path_resolution(self):
        """Prefix relative local paths, including the fallback index."""
        cases = [
            ("/workspace/output/train/1/index.pkl",
             "/home/data", "/workspace/output/train/1/index.pkl"),
            ("/home/data/output/index.pkl",
             "/home/data", "/home/data/output/index.pkl"),
            ("output/train/1/index.pkl",
             "/home/data", "/home/data/output/train/1/index.pkl"),
            ("./output/index.pkl", "/mnt/data/",
             "/mnt/data/./output/index.pkl"),
            ("index.pkl", "", "index.pkl"),
            ("/workspace/index.pkl", "", "/workspace/index.pkl"),
            ("s3://bucket/output/index.pkl", "/home/data",
             "s3://bucket/output/index.pkl"),
            ("gs://bucket/output/index.pkl", "/home/data",
             "gs://bucket/output/index.pkl"),
            ("https://example.com/output/index.pkl", "/home/data",
             "https://example.com/output/index.pkl"),
            ("http://example.com/output/index.pkl", "/home/data",
             "http://example.com/output/index.pkl"),
        ]
        for configured in (True, False):
            for source, prefix, expected in cases:
                with self.subTest(configured=configured, source=source,
                                  prefix=prefix):
                    with patch.object(
                        Context, "parameters",
                        {"CLOUD_KB_INDEX": source} if configured else {}
                    ), patch.object(
                        BaseConfig, "data_path_prefix", prefix
                    ), patch.object(FileOps, "download") as download:
                        self.manager.task_index = (
                            "unused/index.pkl" if configured else source)
                        result = self.job.update(Mock())
                    download.assert_called_once_with(expected, "index.pkl")
                    self.assertEqual(result, "updated/index.pkl")

    def test_local_index_is_loaded_and_update_completes(self):
        """Copy and deserialize an actual index before running both updates."""
        for relative in (False, True):
            for configured in (False, True):
                with self.subTest(relative=relative, configured=configured):
                    with tempfile.TemporaryDirectory() as directory:
                        source = os.path.join(directory, "previous/index.pkl")
                        destination = os.path.join(directory,
                                                   "local/index.pkl")
                        previous_index = {"seen_task": {}, "unseen_task": {}}
                        FileOps.dump(previous_index, source)
                        self.manager.local_task_index_url = destination
                        parameter = ("previous/index.pkl" if relative
                                     else source)
                        self.manager.task_index = parameter

                        def recognize(_):
                            self.assertEqual(FileOps.load(destination),
                                             previous_index)
                            return (SimpleNamespace(x=np.array([1]),
                                                    y=np.array([0])),
                                    SimpleNamespace(x=np.array([2]),
                                                    y=np.array([1])))

                        self.recognition.side_effect = recognize
                        with patch.object(
                            Context, "parameters",
                            {"CLOUD_KB_INDEX": parameter} if configured else {}
                        ), patch.object(
                            BaseConfig, "data_path_prefix", directory
                        ):
                            result = self.job.update(Mock())
                        self.assertEqual(result, "updated/index.pkl")
                        seen = self.manager.seen_estimator
                        unseen = self.manager.unseen_estimator
                        for estimator in (seen, unseen):
                            estimator.update.assert_called_with(
                                ["task"], ["update"], task_index=destination)
                        self.manager.update_kb.assert_called_with({
                            "seen_task": seen.update.return_value,
                            "unseen_task": unseen.update.return_value
                        })
                        self.job.report_task_info.assert_called_with(
                            None, K8sResourceKindStatus.COMPLETED.value,
                            seen.model_info.return_value)
                        self.assertTrue(os.path.isfile(source))


if __name__ == "__main__":
    unittest.main()

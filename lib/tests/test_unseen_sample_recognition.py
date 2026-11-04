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

"""Regression tests for default unseen-sample recognition and routing."""

import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from sedna.algorithms.seen_task_learning.artifact import Model, TaskGroup
from sedna.algorithms.unseen_task_detection.unseen_sample_recognition.\
    unseen_sample_recognition import SampleRegonitionDefault
from sedna.common.class_factory import ClassFactory, ClassType
from sedna.common.file_ops import FileOps
from sedna.core.lifelong_learning import LifelongLearning
from sedna.datasources import BaseDataSource


def make_samples(features):
    """Build inference data without requiring a dataset or model."""
    samples = BaseDataSource(data_type="test")
    samples.x = features
    return samples


class EchoEstimator:
    """A local model that makes input order and keyword forwarding visible."""

    def predict(self, data, **kwargs):
        return ["{}:{}".format(kwargs.get("mode"), row[0]) for row in data]


class TestSampleRegonitionDefault(unittest.TestCase):
    def setUp(self):
        self.recognizer = SampleRegonitionDefault(task_index={})

    def test_all_samples_are_seen(self):
        # Force the old coin flip to take its unseen branch, avoiding flakes.
        with patch("random.randint", return_value=0):
            for size in (0, 1, 2, 3, 4):
                for container in (list, np.array):
                    with self.subTest(size=size, container=container):
                        features = container([
                            ["image-{}.jpg".format(i)] for i in range(size)
                        ])
                        samples = make_samples(features)

                        seen, unseen = self.recognizer(samples)

                        self.assertIsInstance(seen, BaseDataSource)
                        self.assertIsInstance(unseen, BaseDataSource)
                        self.assertEqual(seen.data_type, "test")
                        self.assertEqual(unseen.data_type, "test")
                        self.assertEqual(seen.num_examples(), size)
                        self.assertEqual(unseen.num_examples(), 0)
                        np.testing.assert_array_equal(seen.x, features)
                        self.assertIs(samples.x, features)

    def test_accepts_inference_keywords(self):
        samples = make_samples([["image.jpg"]])

        seen, unseen = self.recognizer(samples, mode="predict", threshold=0.5)

        np.testing.assert_array_equal(seen.x, samples.x)
        self.assertEqual(unseen.num_examples(), 0)

    def test_repeated_calls_are_consistent(self):
        samples = make_samples([["image.jpg"]])
        with patch("random.randint", side_effect=[0, 1] * 10):
            for _ in range(20):
                seen, unseen = self.recognizer(samples)
                np.testing.assert_array_equal(seen.x, [["image.jpg"]])
                self.assertEqual(unseen.num_examples(), 0)

    def test_result_containers_are_independent(self):
        samples = make_samples([["image.jpg"]])

        seen, unseen = self.recognizer(samples)
        unseen.x.append(["new.jpg"])
        _, next_unseen = self.recognizer(samples)

        self.assertIsNot(seen, samples)
        self.assertIsNot(seen, unseen)
        self.assertEqual(next_unseen.num_examples(), 0)
        np.testing.assert_array_equal(samples.x, [["image.jpg"]])

    def test_default_remains_registered(self):
        recognizer = ClassFactory.get_cls(
            ClassType.UTD, "SampleRegonitionDefault")
        self.assertIs(recognizer, SampleRegonitionDefault)


class TestLifelongRecognitionRouting(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        index = Path(directory.name) / "index.pkl"
        FileOps.dump({}, str(index))

        # Supply model/service boundaries while using the real inference and
        # recognizer initialization paths; no background services are needed.
        self.job = LifelongLearning.__new__(LifelongLearning)
        self.job.edge_knowledge_management = SimpleNamespace(
            task_index=str(index),
            current_index_version="1",
            lastest_index_version="1",
            seen_estimator=Mock(),
            unseen_estimator=Mock(),
            save_unseen_samples=Mock(),
            pinned_service_start=True,
            log=Mock(),
        )
        self.job.start_inference_service = False
        self.job.unseen_sample_detection = object()
        self.job.recognize_unseen_samples = None
        self.job.unseen_sample_recognition = {
            "method": "SampleRegonitionDefault"
        }
        self.job.unseen_sample_recognition_param = {}

    def test_default_routes_every_sample_to_seen_estimator(self):
        edge = self.job.edge_knowledge_management
        for size in (1, 3, 4):
            with self.subTest(size=size):
                samples = make_samples([
                    ["image-{}.jpg".format(i)] for i in range(size)
                ])
                predictions = ["seen"] * size
                edge.seen_estimator.predict.return_value = (
                    predictions, ["seen-task"])

                result, is_unseen, tasks = self.job.inference(
                    samples, mode="predict")

                self.assertEqual(result, predictions)
                self.assertEqual(tasks, ["seen-task"])
                expected = False if size == 1 else {
                    "image-{}.jpg".format(i): False for i in range(size)
                }
                self.assertEqual(is_unseen, expected)
                call = edge.seen_estimator.predict.call_args[1]
                np.testing.assert_array_equal(call["data"].x, samples.x)
                self.assertEqual(call["mode"], "predict")
                edge.unseen_estimator.predict.assert_not_called()
                edge.save_unseen_samples.assert_not_called()

    def test_empty_input_does_not_call_either_estimator(self):
        result, is_unseen, tasks = self.job.inference(
            make_samples([]), mode="predict")

        self.assertIsNone(result)
        self.assertFalse(is_unseen)
        self.assertEqual(tasks, [])
        edge = self.job.edge_knowledge_management
        edge.seen_estimator.predict.assert_not_called()
        edge.unseen_estimator.predict.assert_not_called()
        edge.save_unseen_samples.assert_not_called()

    def test_custom_recognizer_can_still_route_unseen_samples(self):
        samples = make_samples([["novel.jpg"]])
        self.job.recognize_unseen_samples = Mock(
            return_value=(make_samples([]), samples))
        edge = self.job.edge_knowledge_management
        edge.unseen_estimator.predict.return_value = (["unseen"], ["new-task"])

        result, is_unseen, tasks = self.job.inference(samples, mode="predict")

        self.assertEqual(result, ["unseen"])
        self.assertTrue(is_unseen)
        self.assertEqual(tasks, ["new-task"])
        self.job.recognize_unseen_samples.assert_called_once_with(
            samples, mode="predict")
        edge.seen_estimator.predict.assert_not_called()
        edge.unseen_estimator.predict.assert_called_once_with(
            samples, task_index=edge.task_index)
        edge.save_unseen_samples.assert_called_once_with(
            samples, post_process=None)


class TestLifelongDefaultInference(unittest.TestCase):
    def test_omitted_recognizer_uses_seen_pipeline(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
                "os.environ", {"BACKEND_TYPE": "UNKNOWN",
                               "OUTPUT_URL": directory,
                               "EDGE_OUTPUT_URL": directory,
                               "MODEL_HOT_UPDATE": "false"}):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="UNKNOWN Not Support yet, use itself")
                job = LifelongLearning(seen_estimator=EchoEstimator())

            group = TaskGroup(entry="seen", tasks=[])
            group.model = Model(index=0, entry="seen",
                                model=EchoEstimator(), result=None)
            edge = job.edge_knowledge_management
            FileOps.dump({"seen_task": {"extractor": {"default": 0},
                                        "task_groups": [group]}},
                         edge.task_index)
            # Represent services already running; inference, index loading,
            # task allocation, model prediction and integration remain real.
            job.unseen_sample_detection = object()
            edge.pinned_service_start = True

            for size in (1, 3, 4):
                with self.subTest(size=size):
                    samples = make_samples(np.array([
                        ["image-{}.jpg".format(i)] for i in range(size)
                    ]))

                    result, is_unseen, tasks = job.inference(
                        samples, mode="predict")

                    np.testing.assert_array_equal(result, [
                        "predict:image-{}.jpg".format(i) for i in range(size)
                    ])
                    self.assertEqual(len(tasks), 1)
                    self.assertEqual(tasks[0].samples.num_examples(), size)
                    expected = False if size == 1 else {
                        "image-{}.jpg".format(i): False for i in range(size)
                    }
                    self.assertEqual(is_unseen, expected)
                    self.assertEqual(list(
                        Path(edge.local_unseen_save_url).iterdir()), [])


if __name__ == "__main__":
    unittest.main()

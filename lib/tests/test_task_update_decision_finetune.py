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

"""Regression tests for the fine-tuning update strategy configuration."""

from pathlib import Path
import tempfile
import unittest

import joblib

from sedna.algorithms.seen_task_learning.task_update_decision import (
    task_update_decision_finetune,
)
from sedna.common.class_factory import ClassFactory, ClassType
from sedna.common.constant import KBResourceConstant


UpdateStrategyByFinetune = (
    task_update_decision_finetune.UpdateStrategyByFinetune
)


class TestUpdateStrategyByFinetune(unittest.TestCase):
    def test_missing_attribute_uses_defaults(self):
        strategy = UpdateStrategyByFinetune({})

        self.assertEqual(strategy.attribute, ("real", "sim"))
        self.assertEqual(strategy.city, "berlin")

    def test_none_attribute_uses_defaults(self):
        strategy = UpdateStrategyByFinetune({}, attribute=None)

        self.assertEqual(strategy.attribute, ("real", "sim"))

    def test_empty_attribute_uses_defaults(self):
        strategy = UpdateStrategyByFinetune({}, attribute="")

        self.assertEqual(strategy.attribute, ("real", "sim"))

    def test_explicit_attributes_are_split(self):
        for attribute, expected in (
            ("real, sim", ["real", "sim"]),
            ("source, target", ["source", "target"]),
        ):
            with self.subTest(attribute=attribute):
                strategy = UpdateStrategyByFinetune({}, attribute=attribute)

                self.assertEqual(strategy.attribute, expected)

    def test_explicit_separator_is_preserved(self):
        strategy = UpdateStrategyByFinetune({}, attribute="source,target")

        self.assertEqual(strategy.attribute, ["source,target"])

    def test_custom_city_with_default_and_explicit_attributes(self):
        for params in ({}, {"attribute": "source, target"}):
            with self.subTest(params=params):
                strategy = UpdateStrategyByFinetune(
                    {}, city="hamburg", **params)

                self.assertEqual(strategy.city, "hamburg")

    def test_registry_construction_preserves_task_index(self):
        task_index = {
            KBResourceConstant.SEEN_TASK.value: {},
            KBResourceConstant.UNSEEN_TASK.value: {},
        }
        strategy_class = ClassFactory.get_cls(
            ClassType.KM, "UpdateStrategyByFinetune")

        strategy = strategy_class(task_index)

        self.assertIsInstance(strategy, UpdateStrategyByFinetune)
        self.assertIs(strategy.task_index, task_index)
        self.assertEqual(strategy.attribute, ("real", "sim"))
        self.assertEqual(strategy.seen_task_key,
                         KBResourceConstant.SEEN_TASK.value)
        self.assertEqual(strategy.unseen_task_key,
                         KBResourceConstant.UNSEEN_TASK.value)
        self.assertEqual(strategy.task_group_key,
                         KBResourceConstant.TASK_GROUPS.value)

    def test_registry_construction_loads_saved_index_without_attribute(self):
        task_index = {
            KBResourceConstant.SEEN_TASK.value: {},
            KBResourceConstant.UNSEEN_TASK.value: {},
        }
        strategy_class = ClassFactory.get_cls(
            ClassType.KM, "UpdateStrategyByFinetune")
        with tempfile.TemporaryDirectory() as directory:
            index_path = str(Path(directory) / "task-index.pkl")
            joblib.dump(task_index, index_path)

            strategy = strategy_class(index_path)

        self.assertEqual(strategy.task_index, task_index)
        self.assertEqual(strategy.attribute, ("real", "sim"))
        self.assertEqual(strategy.city, "berlin")


if __name__ == "__main__":
    unittest.main()

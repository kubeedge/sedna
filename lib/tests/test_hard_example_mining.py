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

"""Unit tests of the hard example mining algorithms."""

import unittest

from sedna.algorithms.hard_example_mining.hard_example_mining import (
    MarginFilter)
from sedna.common.class_factory import ClassFactory, ClassType


class MarginFilterTest(unittest.TestCase):
    def test_registered_as_margin(self):
        hem = ClassFactory.get_cls(ClassType.HEM, "Margin")
        self.assertIs(hem, MarginFilter)

    def test_small_margin_is_hard(self):
        self.assertTrue(MarginFilter(0.2)([0.45, 0.4, 0.15]))

    def test_large_margin_is_not_hard(self):
        self.assertFalse(MarginFilter(0.2)([0.1, 0.8, 0.1]))

    def test_margin_equal_to_threshold_is_not_hard(self):
        self.assertFalse(MarginFilter(0.25)([0.5, 0.25, 0.25]))

    def test_margin_uses_two_highest_scores_in_any_order(self):
        self.assertTrue(MarginFilter(0.1)([0.1, 0.46, 0.04, 0.4]))

    def test_threshold_is_configurable_from_string(self):
        # CRD parameters are passed to the algorithm as strings
        scores = [0.5, 0.375, 0.125]  # margin of 0.125
        self.assertFalse(MarginFilter("0.1")(scores))
        self.assertTrue(MarginFilter("0.2")(scores))

    def test_default_threshold(self):
        self.assertEqual(MarginFilter().threshold_margin, 0.2)

    def test_invalid_input_is_not_hard(self):
        f = MarginFilter(0.2)
        self.assertFalse(f())
        self.assertFalse(f([]))
        self.assertFalse(f([0.5]))
        self.assertFalse(f([0.5, 1.5]))
        self.assertFalse(f([-0.1, 0.5]))


if __name__ == "__main__":
    unittest.main()

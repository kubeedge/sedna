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
    CrossEntropyFilter)


class CrossEntropyFilterTest(unittest.TestCase):
    def test_uniform_distribution_is_hard(self):
        self.assertTrue(CrossEntropyFilter(0.5)([0.5, 0.5]))

    def test_confident_distribution_is_not_hard(self):
        self.assertFalse(CrossEntropyFilter(0.5)([0.98, 0.01, 0.01]))

    def test_zero_probability_does_not_raise(self):
        # a saturated softmax output contains exact zeros
        self.assertFalse(CrossEntropyFilter(0.5)([1.0, 0.0, 0.0]))
        self.assertTrue(CrossEntropyFilter(0.6)([0.5, 0.5, 0.0, 0.0]))

    def test_single_class_is_not_hard(self):
        self.assertFalse(CrossEntropyFilter(0.5)([1.0]))

    def test_invalid_input_is_not_hard(self):
        f = CrossEntropyFilter(0.5)
        self.assertFalse(f([]))
        self.assertFalse(f(None))
        self.assertFalse(f([0.5, 1.5]))


if __name__ == "__main__":
    unittest.main()

import unittest

from sedna.backend.base import BackendBase


class FitOnlyEstimator:

    def fit(self, value):
        return "fit", value


class TrainOnlyEstimator:

    def train(self, value):
        return "train", value


class UpdateOnlyEstimator:

    def update(self, value):
        return "update", value


class BackendBaseFallbackTest(unittest.TestCase):

    def test_train_uses_fit_without_eager_train_lookup(self):
        backend = BackendBase(FitOnlyEstimator())
        self.assertEqual(backend.train("sample"), ("fit", "sample"))

    def test_train_falls_back_to_train(self):
        backend = BackendBase(TrainOnlyEstimator())
        self.assertEqual(backend.train("sample"), ("train", "sample"))

    def test_update_uses_fit_without_eager_update_lookup(self):
        backend = BackendBase(FitOnlyEstimator())
        self.assertEqual(backend.update("sample"), ("fit", "sample"))

    def test_update_falls_back_to_update(self):
        backend = BackendBase(UpdateOnlyEstimator())
        self.assertEqual(backend.update("sample"), ("update", "sample"))


if __name__ == "__main__":
    unittest.main()

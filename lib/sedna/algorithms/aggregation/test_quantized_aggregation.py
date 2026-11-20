"""Unit tests for block-wise quantized aggregation in Sedna."""

import unittest

import numpy as np
from sedna.algorithms.aggregation import (
    AggClient,
    FedAvg,
    QuantizedFedAvg,
    dequantize_weights,
    quantize_weights,
)
from sedna.common.class_factory import ClassFactory, ClassType


class TestQuantizedAggregation(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.layer1 = np.random.randn(64, 32).astype(np.float32)
        self.layer2 = np.random.randn(32).astype(np.float32)
        self.weights = [self.layer1.tolist(), self.layer2.tolist()]

    def test_class_factory_registration(self):
        agg_cls = ClassFactory.get_cls(ClassType.FL_AGG, "QuantizedFedAvg")
        self.assertEqual(agg_cls.__name__, "QuantizedFedAvg")

    def test_quantize_dequantize_int8(self):
        quantized = quantize_weights(self.weights, nbits=8, block_size=64)
        self.assertEqual(len(quantized), 2)
        self.assertEqual(quantized[0]["nbits"], 8)

        reconstructed = dequantize_weights(quantized)
        recon_arr = np.array(reconstructed[0], dtype=np.float32)

        orig_flat = self.layer1.flatten()
        recon_flat = recon_arr.flatten()
        cosine_sim = np.dot(orig_flat, recon_flat) / (np.linalg.norm(orig_flat) * np.linalg.norm(recon_flat))
        self.assertGreater(cosine_sim, 0.99)

    def test_quantize_dequantize_int4(self):
        quantized = quantize_weights(self.weights, nbits=4, block_size=64)
        self.assertEqual(len(quantized), 2)
        self.assertEqual(quantized[0]["nbits"], 4)

        # Packed size check: 64*32 = 2048 parameters -> 1024 bytes packed
        self.assertEqual(len(quantized[0]["data"]), 1024)

        reconstructed = dequantize_weights(quantized)
        recon_arr = np.array(reconstructed[0], dtype=np.float32)

        orig_flat = self.layer1.flatten()
        recon_flat = recon_arr.flatten()
        cosine_sim = np.dot(orig_flat, recon_flat) / (np.linalg.norm(orig_flat) * np.linalg.norm(recon_flat))
        self.assertGreater(cosine_sim, 0.95)

    def test_quantized_fedavg_unquantized_clients(self):
        c1 = AggClient()
        c1.num_samples = 100
        c1.weights = [self.layer1.tolist(), self.layer2.tolist()]

        c2 = AggClient()
        c2.num_samples = 200
        layer1_c2 = (self.layer1 * 1.5).tolist()
        layer2_c2 = (self.layer2 * 1.5).tolist()
        c2.weights = [layer1_c2, layer2_c2]

        fedavg = FedAvg()
        res_fedavg = fedavg.aggregate([c1, c2])

        qfedavg = QuantizedFedAvg()
        res_qfedavg = qfedavg.aggregate([c1, c2])

        np.testing.assert_allclose(res_fedavg[0], res_qfedavg[0], rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(res_fedavg[1], res_qfedavg[1], rtol=1e-5, atol=1e-5)

    def test_quantized_fedavg_quantized_clients(self):
        c1 = AggClient()
        c1.num_samples = 100
        c1.weights = quantize_weights([self.layer1, self.layer2], nbits=4, block_size=64)

        c2 = AggClient()
        c2.num_samples = 100
        c2.weights = quantize_weights([self.layer1 * 2, self.layer2 * 2], nbits=4, block_size=64)

        qfedavg = QuantizedFedAvg(nbits=4, block_size=64)
        aggregated = qfedavg.aggregate([c1, c2])

        expected_layer1 = (self.layer1 + self.layer1 * 2) / 2.0
        agg_arr = np.array(aggregated[0], dtype=np.float32)
        orig_flat = expected_layer1.flatten()
        agg_flat = agg_arr.flatten()
        cosine_sim = np.dot(orig_flat, agg_flat) / (np.linalg.norm(orig_flat) * np.linalg.norm(agg_flat))
        self.assertGreater(cosine_sim, 0.95)

    def test_invalid_nbits(self):
        with self.assertRaises(ValueError):
            quantize_weights(self.weights, nbits=3)


if __name__ == "__main__":
    unittest.main()

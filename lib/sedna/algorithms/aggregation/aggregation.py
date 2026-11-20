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

"""Aggregation algorithms"""

import abc
from copy import deepcopy
from typing import List

from sedna.common.class_factory import ClassFactory, ClassType

__all__ = ('AggClient', 'FedAvg', 'FedAvgV2', 'MistNet', 'QuantizedFedAvg',
           'quantize_weights', 'dequantize_weights')


class AggClient:
    """
    Client that interacts with cloud aggregator

    Parameters
    ----------
    num_samples: int
        number of samples for the current weights
    weights: List
        weights of the layer as a list of number-like array,
        such as [[0, 0, 0, 0], [0, 0, 0, 0] ... ]
    """

    num_samples: int
    weights: List


class BaseAggregation(metaclass=abc.ABCMeta):
    """
    Abstract class of aggregator
    """

    def __init__(self):
        self.total_size = 0
        self.weights = None

    @abc.abstractmethod
    def aggregate(self, clients: List[AggClient]):
        """
        Some algorithms can be aggregated in sequence,
        but some can be calculated only after all aggregated data is uploaded.
        therefore, this abstractmethod should consider that all weights are
        uploaded.

        Parameters
        ----------
        clients: List
            All clients in federated learning job

        Returns
        -------
        Array-like
            final weights use to update model layer
        """


@ClassFactory.register(ClassType.FL_AGG)
class FedAvg(BaseAggregation, abc.ABC):
    """
    Federated averaging algorithm
    """

    def aggregate(self, clients: List[AggClient]):
        """
        Calculate the average weight according to the number of samples

        Parameters
        ----------
        clients: List
            All clients in federated learning job

        Returns
        -------
        update_weights : Array-like
            final weights use to update model layer
        """
        import numpy as np

        if not len(clients):
            return self.weights
        self.total_size = sum([c.num_samples for c in clients])
        old_weight = [np.zeros(np.array(c).shape) for c in
                      next(iter(clients)).weights]
        updates = []
        for inx, row in enumerate(old_weight):
            for c in clients:
                row += (np.array(c.weights[inx]) * c.num_samples
                        / self.total_size)
            updates.append(row.tolist())
        self.weights = deepcopy(updates)
        return updates


@ClassFactory.register(ClassType.FL_AGG)
class MistNet(BaseAggregation, abc.ABC):
    def __init__(self, cut_layer, epsilon=100):
        super().__init__()
        self.parameters = {
            "type": "mistnet",
            "cut_layer": cut_layer,
            "epsilon": epsilon
        }
        if isinstance(self.parameters["cut_layer"], str):
            if self.parameters["cut_layer"].isdigit():
                self.parameters["cut_layer"] = int(cut_layer)

        if isinstance(self.parameters["epsilon"], str):
            if self.parameters["epsilon"].isdigit():
                self.parameters["epsilon"] = int(cut_layer)

    def aggregate(self, clients: List[AggClient]):
        pass


@ClassFactory.register(ClassType.FL_AGG)
class FedAvgV2(BaseAggregation, abc.ABC):
    def __init__(self):
        super().__init__()
        self.parameters = {
            "type": "fedavg",
        }

    def aggregate(self, clients: List[AggClient]):
        pass


def quantize_weights(weights: List, nbits: int = 8, block_size: int = 64) -> List[dict]:
    """Quantize model weights using block-wise symmetric quantization.

    Supports INT8 and packed INT4 with block scale factors.

    Parameters
    ----------
    weights: List
        List of layer weights (numpy arrays or nested lists).
    nbits: int
        Bit precision, 4 or 8. Default is 8.
    block_size: int
        Block size for scaling. Default is 64.

    Returns
    -------
    List[dict]
        List of quantized layer dictionaries with metadata and scale factors.
    """
    import numpy as np

    if nbits not in (4, 8):
        raise ValueError(f"Unsupported nbits: {nbits}. Choose 4 or 8.")

    max_val = 7.0 if nbits == 4 else 127.0
    quantized_layers = []

    for w in weights:
        arr = np.asarray(w, dtype=np.float32)
        orig_shape = arr.shape
        flat = arr.flatten()
        orig_len = len(flat)

        pad_len = (block_size - (orig_len % block_size)) % block_size
        if pad_len > 0:
            flat_padded = np.pad(flat, (0, pad_len), mode="constant")
        else:
            flat_padded = flat

        blocks = flat_padded.reshape(-1, block_size)
        scales = np.max(np.abs(blocks), axis=1) / max_val
        scales[scales == 0] = 1.0

        q = np.clip(np.round(blocks / scales[:, None]), -max_val, max_val).astype(np.int8)

        if nbits == 4:
            q_flat = q.flatten()
            even = q_flat[0::2] & 0x0F
            odd = q_flat[1::2] & 0x0F
            packed = ((even << 4) | odd).astype(np.uint8)
            data = packed.tolist()
        else:
            data = q.flatten().tolist()

        quantized_layers.append({
            "data": data,
            "scale": scales.tolist(),
            "shape": list(orig_shape),
            "orig_len": orig_len,
            "pad_len": pad_len,
            "nbits": nbits,
            "block_size": block_size,
        })

    return quantized_layers


def dequantize_weights(quantized_layers: List[dict]) -> List:
    """Dequantize block-wise quantized layers back to floating-point weights.

    Parameters
    ----------
    quantized_layers: List[dict]
        List of dictionaries produced by quantize_weights.

    Returns
    -------
    List
        List of reconstructed float layers as nested lists or arrays.
    """
    import numpy as np

    reconstructed_layers = []
    for q_dict in quantized_layers:
        nbits = q_dict["nbits"]
        scales = np.array(q_dict["scale"], dtype=np.float32)
        block_size = q_dict["block_size"]
        orig_len = q_dict["orig_len"]
        shape = q_dict["shape"]

        if nbits == 4:
            packed = np.array(q_dict["data"], dtype=np.uint8)
            even = (packed >> 4) & 0x0F
            odd = packed & 0x0F
            even = np.where(even >= 8, even - 16, even).astype(np.int8)
            odd = np.where(odd >= 8, odd - 16, odd).astype(np.int8)
            unpacked = np.empty(len(packed) * 2, dtype=np.int8)
            unpacked[0::2] = even
            unpacked[1::2] = odd
            q_blocks = unpacked.reshape(-1, block_size)
        else:
            q_flat = np.array(q_dict["data"], dtype=np.int8)
            q_blocks = q_flat.reshape(-1, block_size)

        recon_blocks = q_blocks.astype(np.float32) * scales[:, None]
        recon_flat = recon_blocks.flatten()[:orig_len]
        reconstructed_layers.append(recon_flat.reshape(shape).tolist())

    return reconstructed_layers


@ClassFactory.register(ClassType.FL_AGG)
class QuantizedFedAvg(BaseAggregation, abc.ABC):
    """Federated averaging algorithm supporting quantized updates from edge clients.

    Edge clients can send quantized model updates (INT4 or INT8) to reduce uplink
    transmission bandwidth. QuantizedFedAvg transparently reconstructs the float
    weights before computing sample-weighted aggregation.
    """

    def __init__(self, nbits: int = 8, block_size: int = 64):
        super().__init__()
        self.nbits = nbits
        self.block_size = block_size
        self.parameters = {
            "type": "quantized_fedavg",
            "nbits": nbits,
            "block_size": block_size,
        }

    def aggregate(self, clients: List[AggClient]):
        """Calculate weighted average across clients, handling quantized or float weights.

        Parameters
        ----------
        clients: List[AggClient]
            List of client objects participating in this aggregation round.

        Returns
        -------
        update_weights: List
            Aggregated float weights formatted as nested lists.
        """
        import numpy as np

        if not len(clients):
            return self.weights

        self.total_size = sum(c.num_samples for c in clients)

        unpacked_client_weights = []
        for c in clients:
            weights = c.weights
            if weights and isinstance(weights[0], dict) and "data" in weights[0]:
                unpacked = dequantize_weights(weights)
            else:
                unpacked = weights
            unpacked_client_weights.append(unpacked)

        first_weights = unpacked_client_weights[0]
        accumulators = [np.zeros(np.array(layer).shape, dtype=np.float32) for layer in first_weights]

        for c_idx, c in enumerate(clients):
            sample_ratio = c.num_samples / self.total_size
            weights = unpacked_client_weights[c_idx]
            for inx, layer in enumerate(weights):
                accumulators[inx] += np.asarray(layer, dtype=np.float32) * sample_ratio

        updates = [acc.tolist() for acc in accumulators]
        self.weights = deepcopy(updates)
        return updates

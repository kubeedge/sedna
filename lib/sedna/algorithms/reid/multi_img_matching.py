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

import torch
from typing import List, Any, Tuple
import numpy as np


def cosine_similarity_score(
        query: np.ndarray = None,
        candidates: np.ndarray = None):
    """ Computes the cosine similarity score between the
        query feature and the candidate features.
    @param query: Feature map of dimension
        [1, n_feat_dim] representing the query.
    @param candidates: Feature map of dimension
        [n_candidates, n_feat_dim] representing the candidate for match.
    """
    sim_measure = np.matmul(query, candidates.T)
    return sim_measure


def tensor_reshape(data: Any) -> torch.Tensor:

    if isinstance(data, torch.Tensor):
        if len(data.shape) > 2:
            data = data.squeeze(0)

    if isinstance(data, List):
        if len(data[0].shape) > 2:
            temp = [x.squeeze(0) for x in data]
            data = torch.cat(temp, dim=0)
        else:
            data = torch.cat(data, dim=0)

    return data


def match_query_to_targets(query_feats: List,
                           candidate_feats: List,
                           avg_mode: bool = False) -> Tuple[int, float]:
    """
    Query features refer to the features of
    the person we are looking for in the video.
    Candidate features refers to features of the
    persons found by the detector in the current scene.
    Args:
        query_feats: [M x d] M being the number of target images in the query
        candidate_feats:
            [N x d] N is the number of persons detected in the scene
        avg_mode: If set, use an average representation of the query.
            Query feats becomes [1 x d]
    Returns:
        Id of the candidate which best matches the query
    """
    query_feats, candidate_feats = \
        tensor_reshape(query_feats), tensor_reshape(candidate_feats)

    if avg_mode:
        # average query_feats
        query_feats = torch.mean(query_feats, dim=0).unsqueeze(0)

    # compare features
    sim_dist = torch.mm(query_feats, candidate_feats.t())
    _, idx = (sim_dist == torch.max(sim_dist)).nonzero()[0]
    match_id = idx.item()

    return match_id, torch.max(sim_dist).item()

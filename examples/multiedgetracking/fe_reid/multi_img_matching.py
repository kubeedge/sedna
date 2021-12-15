# from sedna.algorithms.reid.mAP import R1_mAP
# from sedna.common.log import LOGGER

import torch
from typing import List, Any, Tuple
import numpy as np


def cosine_similarity_score(
        query: np.ndarray = None,
        candidates: np.ndarray = None):
    """ Computes the cosine similarity score between the query feature and the candidate features.
    @param query: Feature map of dimension [1, n_feat_dim] representing the query.
    @param candidates: Feature map of dimension [n_candidates, n_feat_dim] representing the candidate for match.
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


def match_query_to_targets(query_feat: Any,
                           candidate_feats: Any,
                           avg_mode: bool = False) -> Tuple[int, int]:

    query_feat, candidate_feats = tensor_reshape(query_feat), tensor_reshape(candidate_feats)
    query_feat_np = query_feat.cpu().numpy()
    candidate_feats_np = candidate_feats.cpu().numpy()

    match_id = -1
    if avg_mode:
        query_feat_np = np.mean(query_feat_np, axis=0)
        sim_measure = cosine_similarity_score(query_feat_np, candidate_feats_np)
        match_id = np.argsort(sim_measure)[-1]
        match_score = max(sim_measure.flatten())
    else:
        sim_measure = cosine_similarity_score(query_feat_np, candidate_feats_np)
        match_score = max(sim_measure.flatten())
        _, y = np.where(sim_measure == match_score)
        match_id = y[0]

    return match_id, match_score


# def inference(
#         model,
#         features,
#         num_query):

#     LOGGER.info("Enter inferencing")

#     evaluator = R1_mAP(num_query, max_rank=50)

#     evaluator.reset()

#     model.eval()
#     img_path_list = []

#     for i, feat in enumerate(features):
#         evaluator.update((feat, 0, 0))
#         img_path_list.extend("0001_0_0.smh")
#         LOGGER.info('Done update: ', i)

#     cmc, mAP, distmat, pids, camids, qfeats, gfeats = evaluator.compute()
#     LOGGER.info('Done compute evaluator!')
    
#     return gfeats, img_path_list
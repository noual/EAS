"""Weighted-Tchebycheff scalarization, matching PMOCO's own sign convention.

PMOCO (MOTSPTrainer._train_one_batch / MOTSPTester._test_one_batch) computes:
    cost = -reward                      # reward is negative distance -> cost is positive
    tch  = max_obj( pref * (cost - z) )  # z is the (zero) utopia point
    tch_reward = -tch                   # back to "reward" convention: higher is better

Reusing this exact formula (rather than inventing a new scalarization) keeps
the gradient signal consistent with what the PMOCO backbone was pretrained
under.
"""

import torch


def tch_scalarize(reward: torch.Tensor, pref: torch.Tensor, z: float = 0.0) -> torch.Tensor:
    """Tchebycheff-scalarize a vector reward into a scalar "reward" (higher is better).

    reward: (..., n_obj), negative distances (PMOCO env convention).
    pref: (n_obj,), on the simplex (pref.sum() == 1, pref >= 0).
    Returns: (...), same sign convention as `reward` (higher is better).
    """
    cost = -reward
    weighted = pref * (cost - z)
    tch, _ = weighted.max(dim=-1)
    return -tch

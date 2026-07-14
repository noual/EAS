"""CLI entry point for MO-EAS-Emb on MOTSP (PMOCO backbone).

Mirrors run_search.py's style (argparse, one output dir per run) but sweeps a
grid of preference vectors per instance and aggregates the incumbents from
each ω-run into a per-instance Pareto front + normalized hypervolume.

Example:
    python run_search_mo.py -problem_size 20 -n_instances 20 -n_prefs 11 -max_iter 50
"""

import argparse
import datetime
import logging
import os
import pickle
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from source.motsp_pmoco.eas_emb_mo import run_eas_emb_mo
from source.motsp_pmoco.pareto import non_dominated_filter, normalized_hypervolume
from source.motsp_pmoco.pmoco_bridge import REF_POINT_BY_SIZE, get_random_problems, load_motsp_model


def get_config():
    parser = argparse.ArgumentParser(description="MO-EAS-Emb (MOTSP, PMOCO backbone)")

    parser.add_argument("-problem_size", default=20, type=int, choices=[20, 50, 100])
    parser.add_argument("-epoch", default=200, type=int, help="Checkpoint epoch to load")

    parser.add_argument("-n_instances", default=20, type=int)
    parser.add_argument("-n_prefs", default=11, type=int, help="Number of ω on the 2-simplex grid")

    parser.add_argument("-max_iter", default=200, type=int, help="EAS iterations per (instance, ω)")
    parser.add_argument("-param_lr", default=0.0041, type=float)
    parser.add_argument("-param_lambda", default=0.013, type=float, help="Imitation-loss weight")

    parser.add_argument("-seed", default=0, type=int)
    parser.add_argument("-log_every", default=0, type=int, help="0 disables per-iteration logging")
    parser.add_argument("-output_path", default="", type=str)

    return parser.parse_args()


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_pref_grid(n_prefs: int, device: torch.device) -> torch.Tensor:
    """n_prefs evenly-spaced points on the 2-obj simplex, matching PMOCO's own
    test scripts (test_motsp_n*.py): pref = [1 - i/(n-1), i/(n-1)]."""
    if n_prefs == 1:
        return torch.tensor([[0.5, 0.5]], device=device)
    w1 = torch.linspace(1.0, 0.0, n_prefs, device=device)
    w2 = 1.0 - w1
    return torch.stack([w1, w2], dim=1)


def main():
    config = get_config()

    use_cuda = torch.cuda.is_available()
    torch.set_default_tensor_type("torch.cuda.FloatTensor" if use_cuda else "torch.FloatTensor")
    device = torch.device("cuda" if use_cuda else "cpu")
    set_global_seed(config.seed)

    now = datetime.datetime.now()
    run_id = f"{config.problem_size}_{now.strftime('%H%M%S%f')}"
    if config.output_path == "":
        config.output_path = os.getcwd()
    output_path = os.path.join(
        config.output_path, "runs_mo", f"run_{now.day}.{now.month}.{now.year}_{run_id}"
    )
    os.makedirs(output_path)

    logging.basicConfig(
        filename=os.path.join(output_path, "log.txt"),
        filemode="w",
        level=logging.INFO,
        format="[%(levelname)s]%(message)s",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logger = logging.getLogger(__name__)
    logger.info("Call: %s", " ".join(sys.argv))

    model = load_motsp_model(config.problem_size, device, epoch=config.epoch)
    problems = get_random_problems(config.n_instances, config.problem_size).to(device)
    prefs = build_pref_grid(config.n_prefs, device)

    logger.info(
        "Starting MO-EAS-Emb: problem_size=%d, n_instances=%d, n_prefs=%d, max_iter=%d",
        config.problem_size, config.n_instances, config.n_prefs, config.max_iter,
    )

    t_start = time.time()
    result = run_eas_emb_mo(
        model,
        problems,
        prefs,
        device,
        max_iter=config.max_iter,
        lr=config.param_lr,
        imitation_lambda=config.param_lambda,
        log_every=config.log_every,
    )
    runtime = time.time() - t_start

    ref_point = np.array(REF_POINT_BY_SIZE[config.problem_size])

    # Each of the max_iter iterations, for each of n_prefs ω, constructs
    # group_s = problem_size + 1 tours (teacher-forced lane included) — see
    # "Matching evaluation counts" in nrpa-eas.md.
    group_s = config.problem_size + 1
    n_evaluations_per_instance = config.n_prefs * config.max_iter * group_s

    per_instance_fronts = []
    per_instance_hv = np.zeros(config.n_instances)
    objectives = result["objectives"]  # (n_prefs, n_instances, 2)
    for i in range(config.n_instances):
        points = objectives[:, i, :]
        front = non_dominated_filter(points)
        per_instance_fronts.append(front)
        per_instance_hv[i] = normalized_hypervolume(front, ref_point)

    logger.info("Runtime: %.2fs", runtime)
    logger.info(
        "Mean normalized HV: %.4f (+/- %.4f) over %d instances",
        per_instance_hv.mean(), per_instance_hv.std(), config.n_instances,
    )
    logger.info(
        "Objective evaluations: %d per instance, %d total over %d instances",
        n_evaluations_per_instance, n_evaluations_per_instance * config.n_instances, config.n_instances,
    )

    pickle.dump(
        {
            "config": vars(config),
            "runtime": runtime,
            "n_evaluations_per_instance": n_evaluations_per_instance,
            "n_evaluations_total": n_evaluations_per_instance * config.n_instances,
            "objectives": objectives,
            "tours": result["tours"],
            "prefs": prefs.cpu().numpy(),
            "fronts": per_instance_fronts,
            "normalized_hv": per_instance_hv,
            "ref_point": ref_point,
        },
        open(os.path.join(output_path, "results_mo.pkl"), "wb"),
    )

    return per_instance_hv.mean()


if __name__ == "__main__":
    main()

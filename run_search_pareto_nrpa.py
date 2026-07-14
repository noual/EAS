"""CLI entry point for nested Pareto-NRPA over EAS-Emb on MOTSP (PMOCO backbone).

Mirrors run_search_mo.py's style, but replaces the flat 101-ω sweep with the
nested Pareto-NRPA search in `source/motsp_pmoco/pareto_nrpa_eas_emb.py`.
Runs on a single instance per invocation (see nrpa-eas.md: per-instance, not
batched across instances).

Example:
    python run_search_pareto_nrpa.py -problem_size 20 -level 2 -n_iter 10
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

from source.motsp_pmoco.pareto_nrpa_eas_emb import ParetoNRPA_EAS_Emb
from source.motsp_pmoco.pareto import non_dominated_filter, normalized_hypervolume
from source.motsp_pmoco.pmoco_bridge import REF_POINT_BY_SIZE, get_random_problems, load_motsp_model


def get_config():
    parser = argparse.ArgumentParser(description="Pareto-NRPA over MO-EAS-Emb (MOTSP, PMOCO backbone)")

    parser.add_argument("-problem_size", default=20, type=int, choices=[20, 50, 100])
    parser.add_argument("-epoch", default=200, type=int, help="Checkpoint epoch to load")

    parser.add_argument("-level", default=2, type=int, help="NRPA recursion depth")
    parser.add_argument("-n_iter", default=10, type=int, help="Iterations per NRPA level")
    parser.add_argument(
        "-leaf_pomo_size", default=0, type=int,
        help="Multi-start rollouts per leaf call; 0 defaults to problem_size",
    )
    parser.add_argument("-top_k_sequences", default=10, type=int, help="Max sequences teacher-forced per Adapt call")

    parser.add_argument("-param_lr", default=0.0041, type=float)

    parser.add_argument("-seed", default=0, type=int)
    parser.add_argument("-output_path", default="", type=str)

    return parser.parse_args()


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
        config.output_path, "runs_nrpa", f"run_{now.day}.{now.month}.{now.year}_{run_id}"
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
    problem = get_random_problems(1, config.problem_size).to(device)

    leaf_pomo_size = config.leaf_pomo_size or config.problem_size

    logger.info(
        "Starting Pareto-NRPA-EAS-Emb: problem_size=%d, level=%d, n_iter=%d, leaf_pomo_size=%d, top_k=%d",
        config.problem_size, config.level, config.n_iter, leaf_pomo_size, config.top_k_sequences,
    )

    search = ParetoNRPA_EAS_Emb(
        model,
        problem,
        device,
        level=config.level,
        n_iter=config.n_iter,
        leaf_pomo_size=leaf_pomo_size,
        top_k_sequences=config.top_k_sequences,
        lr=config.param_lr,
    )

    t_start = time.time()
    search.run()
    runtime = time.time() - t_start

    global_F = search.global_pareto_front.get("F")
    front = non_dominated_filter(global_F)
    ref_point = np.array(REF_POINT_BY_SIZE[config.problem_size])
    hv = normalized_hypervolume(front, ref_point)

    logger.info("Runtime: %.2fs", runtime)
    logger.info("Objective evaluations: %d, Adapt calls: %d", search.n_evaluations, search.n_adapt_calls)
    logger.info("Global Pareto front size: %d, normalized HV: %.4f", len(front), hv)

    pickle.dump(
        {
            "config": vars(config),
            "runtime": runtime,
            "n_evaluations": search.n_evaluations,
            "n_adapt_calls": search.n_adapt_calls,
            "front": front,
            "normalized_hv": hv,
            "ref_point": ref_point,
            "problem": problem.cpu().numpy(),
        },
        open(os.path.join(output_path, "results_nrpa.pkl"), "wb"),
    )

    return hv


if __name__ == "__main__":
    main()

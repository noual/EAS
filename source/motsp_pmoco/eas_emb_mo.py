"""EAS-Emb, ported to multi-objective MOTSP via an outer preference sweep.

Ports `source/eas_emb.py`'s algorithm (fine-tune only the decoder's
`single_head_key` embedding; REINFORCE with mean-reward baseline + an
incumbent-teacher-forcing imitation loss) onto PMOCO's preference-conditioned
`MOTSPModel`/`MOTSPEnv`. One independent EAS-Emb run per preference vector ω;
the scalar reward driving that run is the Tchebycheff scalarization PMOCO
itself uses for training/testing (see `scalarization.py`).

Kept deliberately close to the original `run_eas_emb` control flow so the
diff is easy to audit: same `group_s = problem_size + 1` teacher-forcing
trick, same two loss terms, same incumbent-by-argmax bookkeeping — just with
a vector reward scalarized by ω instead of a scalar reward, and a fresh
optimizer per ω over the PMOCO model's `single_head_key` instead of EAS's
own `node_prob_calculator.single_head_key`.
"""

import logging

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from .pmoco_bridge import TSPEnv, _get_encoding
from .scalarization import tch_scalarize

logger = logging.getLogger(__name__)


def run_eas_emb_mo(
    model,
    problems: torch.Tensor,
    prefs: torch.Tensor,
    device: torch.device,
    max_iter: int = 200,
    lr: float = 0.0041,
    imitation_lambda: float = 0.013,
    weight_decay: float = 1e-6,
    log_every: int = 0,
) -> dict:
    """Run EAS-Emb-MO: one preference-conditioned EAS-Emb search per ω in `prefs`.

    model: pretrained PMOCO TSPModel, frozen, eval mode, already on `device`.
    problems: (batch_size, problem_size, 4) raw MOTSP instances (2 coord pairs).
    prefs: (n_prefs, 2) preference vectors on the simplex.
    Returns dict with:
        "objectives": (n_prefs, batch_size, 2) float64 ndarray, raw (positive)
            incumbent tour lengths per objective, per instance, per ω.
        "tours": (n_prefs, batch_size, problem_size) int ndarray, incumbent tours.
    """
    problems = problems.to(device)
    prefs = prefs.to(device)

    batch_size, problem_size, _ = problems.shape
    n_prefs = prefs.shape[0]
    group_s = problem_size + 1  # last lane is teacher-forced to the incumbent

    all_objectives = np.zeros((n_prefs, batch_size, 2), dtype=np.float64)
    all_tours = np.zeros((n_prefs, batch_size, problem_size), dtype=np.int64)

    pref_bar = tqdm(range(n_prefs), desc="ω sweep", position=0)
    for pref_idx in pref_bar:
        pref = prefs[pref_idx]

        env = TSPEnv(problem_size=problem_size, pomo_size=group_s)
        env.batch_size = batch_size
        env.problems = problems
        env.BATCH_IDX = torch.arange(batch_size, device=device)[:, None].expand(batch_size, group_s)
        env.POMO_IDX = torch.arange(group_s, device=device)[None, :].expand(batch_size, group_s)

        with torch.no_grad():
            reset_state, _, _ = env.reset()
            model.decoder.assign(pref)
            model.pre_forward(reset_state)  # frozen encoder + decoder.set_kv

        # EAS-Emb: only the single_head_key embedding is fine-tuned, everything
        # else (encoder, hypernetwork-derived decoder weights) stays frozen.
        model.decoder.single_head_key = model.decoder.single_head_key.clone().requires_grad_(True)
        optimizer = optim.Adam([model.decoder.single_head_key], lr=lr, weight_decay=weight_decay)

        incumbent_tch = torch.full((batch_size,), float("-inf"), device=device)
        incumbent_tour = torch.zeros((batch_size, problem_size), dtype=torch.long, device=device)
        incumbent_obj = torch.zeros((batch_size, 2), device=device)

        iter_bar = tqdm(
            range(max_iter), desc=f"pref {pref_idx + 1}/{n_prefs}", position=1, leave=False
        )
        for it in iter_bar:
            state, _, done = env.reset()

            first_action = (torch.arange(group_s, device=device) % problem_size)[None, :].expand(
                batch_size, group_s
            ).clone()
            if it > 0:
                first_action[:, -1] = incumbent_tour[:, 0]

            with torch.no_grad():
                encoded_first_node = _get_encoding(model.encoded_nodes, first_action)
                model.decoder.set_q1(encoded_first_node)

            state, reward, done = env.step(first_action)
            solutions = [first_action.unsqueeze(2)]
            log_probs = []

            step_idx = 1
            while not done:
                encoded_last_node = _get_encoding(model.encoded_nodes, state.current_node)
                probs = model.decoder(encoded_last_node, ninf_mask=state.ninf_mask)
                # shape: (batch, group_s, problem_size)

                action = probs.reshape(batch_size * group_s, -1).multinomial(1).reshape(
                    batch_size, group_s
                )
                if it > 0:
                    action[:, -1] = incumbent_tour[:, step_idx]

                state, reward, done = env.step(action)
                solutions.append(action.unsqueeze(2))

                chosen_prob = probs.gather(2, action.unsqueeze(2)).squeeze(2)
                log_probs.append(chosen_prob.log())
                step_idx += 1

            solutions = torch.cat(solutions, dim=2)  # (batch, group_s, problem_size)
            log_prob = torch.stack(log_probs, dim=2).sum(dim=2)  # (batch, group_s)
            tch_reward = tch_scalarize(reward, pref)  # (batch, group_s), higher is better

            # Incumbent update (per instance, over the group_s rollouts of this iter)
            max_tch_iter, best_idx = tch_reward.max(dim=1)
            improved = max_tch_iter > incumbent_tch
            if improved.any():
                incumbent_tch[improved] = max_tch_iter[improved]
                tour_gather_idx = best_idx[improved].view(-1, 1, 1).expand(-1, 1, problem_size)
                incumbent_tour[improved] = solutions[improved].gather(1, tour_gather_idx).squeeze(1)
                obj_gather_idx = best_idx[improved].view(-1, 1, 1).expand(-1, 1, 2)
                # reward is negative distance; negate back to positive objective values.
                incumbent_obj[improved] = -reward[improved].gather(1, obj_gather_idx).squeeze(1)

            # LEARNING — REINFORCE on the free lanes + imitation loss on the
            # teacher-forced lane, both driven by the TCH-scalarized reward.
            rl_tch = tch_reward[:, : group_s - 1]
            rl_logp = log_prob[:, : group_s - 1]
            advantage = rl_tch - rl_tch.mean(dim=1, keepdim=True)
            rl_loss = (-advantage * rl_logp).mean()
            imitation_loss = -log_prob[:, group_s - 1].mean()
            loss = rl_loss + imitation_lambda * imitation_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            mean_tch = incumbent_tch.mean().item()
            iter_bar.set_postfix(mean_tch=f"{mean_tch:.4f}")

            if log_every and (it % log_every == 0 or it == max_iter - 1):
                logger.info(
                    "pref %d/%d iter %d/%d: mean incumbent TCH=%.4f",
                    pref_idx + 1, n_prefs, it + 1, max_iter, mean_tch,
                )

        iter_bar.close()
        pref_bar.set_postfix(mean_tch=f"{incumbent_tch.mean().item():.4f}")

        all_objectives[pref_idx] = incumbent_obj.detach().cpu().numpy()
        all_tours[pref_idx] = incumbent_tour.detach().cpu().numpy()

    return {"objectives": all_objectives, "tours": all_tours}

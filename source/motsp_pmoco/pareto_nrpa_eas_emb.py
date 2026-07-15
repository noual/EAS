"""EAS-Emb as the search algorithm of a nested Pareto-NRPA, on MOTSP (PMOCO backbone).

Restructures `eas_emb_mo.py`'s flat "one independent EAS-Emb run per ω-grid-point"
sweep into the actual nested-rollout + policy-adaptation loop of Pareto-NRPA
(ported from `src/search/neural_nrpa/neural_nrpa.py`), using PMOCO's
`decoder.single_head_key` embedding as the only adapted "policy" state — the
same tensor EAS-Emb already fine-tunes, just threaded through recursion levels
instead of optimized flat against a fixed ω.

See `nrpa-eas.md` (next to this file's package, at the EAS repo root) for the
full design log: why REINFORCE is dropped in favor of pure crowding-weighted
imitation here, the `set_kv`-clobbers-`single_head_key` gotcha and its fix,
and why teacher-forcing is done per-sequence rather than batched.
"""

import logging

import numpy as np
import torch
import torch.optim as optim
from pymoo.core.individual import Individual
from pymoo.core.population import Population
from pymoo.core.problem import Problem
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from .pmoco_bridge import TSPEnv, _get_encoding

logger = logging.getLogger(__name__)


def _softmax_temp(x: np.ndarray, temp: float) -> np.ndarray:
    x = x / temp
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class ParetoNRPA_EAS_Emb:
    """Nested Pareto-NRPA over a single MOTSP instance, adapting EAS-Emb's
    `single_head_key` embedding.

    One instance per object — no cross-instance batching (this project's
    per-instance claim). The pomo/multi-start dimension is used only for
    free parallelism *within* a single leaf call.
    """

    def __init__(
        self,
        model,
        problem: torch.Tensor,  # (1, problem_size, 4)
        device: torch.device,
        level: int = 2,
        n_iter: int = 15,
        leaf_pomo_size: int | None = None,
        top_k_sequences: int = 10,
        lr: float = 0.0041,
        weight_decay: float = 1e-6,
        dirichlet_alpha: float = 1.0,
        omega_true_temp: float = 0.5,
        omega_true_eps: float = 1e-8,
    ) -> None:
        self.model = model
        self.device = device
        self.problem = problem.to(device)
        self.problem_size = problem.shape[1]
        self.level = level
        self.n_iter = n_iter
        self.leaf_pomo_size = leaf_pomo_size or self.problem_size
        self.top_k_sequences = top_k_sequences
        self.lr = lr
        self.weight_decay = weight_decay
        self.dirichlet = torch.distributions.Dirichlet(
            torch.full((2,), dirichlet_alpha, device=device)
        )
        self.omega_true_temp = omega_true_temp
        self.omega_true_eps = omega_true_eps

        self.global_pareto_front = Population()
        self.n_evaluations = 0
        self.n_adapt_calls = 0

        with torch.no_grad():
            self.encoded_nodes = model.encoder(self.problem)  # frozen, ω-independent

    # ------------------------------------------------------------------
    # Decoder conditioning
    # ------------------------------------------------------------------

    def _use_pref_and_key(self, pref: torch.Tensor, key: torch.Tensor) -> None:
        """Condition the decoder on `pref`, then restore `key` as single_head_key.

        `decoder.set_kv` recomputes the ω-dependent k/v *and*, as a side
        effect, resets `single_head_key` to the raw (unadapted) encoder
        output — restore our tracked `key` right after, every time ω changes.
        """
        with torch.no_grad():
            self.model.decoder.assign(pref)
            self.model.decoder.set_kv(self.encoded_nodes)
        self.model.decoder.single_head_key = key

    def _root_key(self) -> torch.Tensor:
        """single_head_key = encoded_nodes.transpose(1, 2) is ω-independent
        by construction (see `set_kv`) — no arbitrary bootstrap ω needed."""
        return self.encoded_nodes.transpose(1, 2).clone().requires_grad_(True)

    # ------------------------------------------------------------------
    # Level 0: one playout, no gradient step
    # ------------------------------------------------------------------

    def _sample_rollout(self, pref: torch.Tensor, key: torch.Tensor, pomo_size: int):
        """One stochastic multi-start rollout under (pref, key). No grad."""
        env = TSPEnv(problem_size=self.problem_size, pomo_size=pomo_size)
        env.batch_size = 1
        env.problems = self.problem
        env.BATCH_IDX = torch.zeros((1, pomo_size), dtype=torch.long, device=self.device)
        env.POMO_IDX = torch.arange(pomo_size, device=self.device)[None, :].expand(1, pomo_size)

        with torch.no_grad():
            self._use_pref_and_key(pref, key)
            env.reset()

            first_action = (torch.arange(pomo_size, device=self.device) % self.problem_size)[
                None, :
            ].clone()
            encoded_first_node = _get_encoding(self.encoded_nodes, first_action)
            self.model.decoder.set_q1(encoded_first_node)
            state, reward, done = env.step(first_action)
            tours = [first_action.unsqueeze(2)]

            while not done:
                encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
                probs = self.model.decoder(encoded_last_node, ninf_mask=state.ninf_mask)
                action = probs.reshape(pomo_size, -1).multinomial(1).reshape(1, pomo_size)
                state, reward, done = env.step(action)
                tours.append(action.unsqueeze(2))

        tours = torch.cat(tours, dim=2).squeeze(0)  # (pomo_size, problem_size)
        objectives = -reward.squeeze(0)  # (pomo_size, 2), positive distances
        return tours.cpu().numpy(), objectives.cpu().numpy().astype(np.float64)

    def _level0(self, key: torch.Tensor) -> Population:
        pref = self.dirichlet.sample()
        tours, objectives = self._sample_rollout(pref, key, self.leaf_pomo_size)
        self.n_evaluations += self.leaf_pomo_size

        pref_np = pref.detach().cpu().numpy()
        individuals = []
        for i in range(self.leaf_pomo_size):
            ind = Individual()
            ind.set("X", tours[i])
            ind.set("F", objectives[i])
            ind.set("W", pref_np)
            individuals.append(ind)
        pop = Population.merge(Population(), *individuals)

        self.global_pareto_front = Population.merge(self.global_pareto_front, pop)
        fronts = NonDominatedSorting().do(self.global_pareto_front.get("F"))
        self.global_pareto_front = self.global_pareto_front[fronts[0]]

        return pop

    # ------------------------------------------------------------------
    # Pareto-Adapt: crowding-weighted imitation, ω_true from achieved F
    # ------------------------------------------------------------------

    def _teacher_force_logprob(self, pref: torch.Tensor, key: torch.Tensor, tour: np.ndarray) -> torch.Tensor:
        """Log-probability of `tour` under (pref, key), WITH grad w.r.t. key.

        Teacher-forcing a *known* tour has no autoregressive dependency between
        steps (mask/last-node at step t is a deterministic function of
        tour[:t]), so all T-1 decode steps are computed as a single batched
        decoder call on the pomo axis instead of a T-1-iteration Python loop —
        the same trick PMOCO uses for pomo/multi-start rollouts, just applied
        to time steps here. This was previously the dominant cost: ~T decoder
        forward+step calls at batch size 1 per sequence, per Adapt call.
        """
        tour_t = torch.as_tensor(tour, dtype=torch.long, device=self.device).view(1, -1)  # (1, T)
        problem_len = tour_t.shape[1]

        self._use_pref_and_key(pref, key)

        first_action = tour_t[:, :1]
        with torch.no_grad():
            encoded_first_node = _get_encoding(self.encoded_nodes, first_action)
            self.model.decoder.set_q1(encoded_first_node)

        prev_nodes = tour_t[:, : problem_len - 1]  # (1, T-1): tour[0..T-2], the "last visited" at each step
        next_actions = tour_t[:, 1:]  # (1, T-1): tour[1..T-1], the target action at each step
        encoded_last_nodes = _get_encoding(self.encoded_nodes, prev_nodes)  # (1, T-1, embedding)

        # ninf_mask[0, s, :] excludes every node visited in tour[0..s] (inclusive),
        # matching TSPEnv.step's cumulative visited-node masking.
        visited_one_hot = torch.nn.functional.one_hot(prev_nodes[0], num_classes=self.problem_size).float()
        cum_visited = visited_one_hot.cumsum(dim=0)  # (T-1, problem_size)
        ninf_mask = torch.where(cum_visited > 0, float("-inf"), 0.0).unsqueeze(0)  # (1, T-1, problem_size)

        probs = self.model.decoder(encoded_last_nodes, ninf_mask=ninf_mask)  # (1, T-1, problem_size)
        chosen_probs = probs.gather(2, next_actions.unsqueeze(2)).squeeze(2)  # (1, T-1)
        log_prob = chosen_probs.log().sum()

        return log_prob

    def _pareto_adapt(self, key: torch.Tensor, optimizer: torch.optim.Optimizer, optimal_set: Population) -> float:
        if len(optimal_set) == 0:
            return 0.0

        F_values = np.array([ind.get("F") for ind in optimal_set], dtype=np.float64)
        global_F = np.array([ind.get("F") for ind in self.global_pareto_front], dtype=np.float64)
        f_min = global_F.min(axis=0)
        f_max = global_F.max(axis=0)

        if len(optimal_set) == 1:
            weights = np.array([1.0])  # crowding distance undefined for a single point
        else:
            crowding = RankAndCrowding()
            distances = crowding.do(problem=Problem(n_constr=0), pop=optimal_set)
            weights = np.where(distances.get("crowding") == np.inf, 2.0, distances.get("crowding"))

        sequences = [ind.get("X") for ind in optimal_set]
        if len(sequences) > self.top_k_sequences:
            top_idx = np.argsort(weights)[-self.top_k_sequences :]
            sequences = [sequences[i] for i in top_idx]
            weights = weights[top_idx]
            F_values = F_values[top_idx]

        total_loss = torch.zeros((), device=self.device)
        for seq, f_val, w in zip(sequences, F_values, weights):
            omega_true = _softmax_temp(
                (f_max - f_val) / (f_max - f_min + self.omega_true_eps), temp=self.omega_true_temp
            )
            pref_t = torch.as_tensor(omega_true, dtype=torch.float32, device=self.device)
            log_prob = self._teacher_force_logprob(pref_t, key, seq)
            total_loss = total_loss - w * log_prob
        total_loss = total_loss / len(sequences)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        self.n_adapt_calls += 1
        return total_loss.item()

    # ------------------------------------------------------------------
    # Nested recursion
    # ------------------------------------------------------------------

    def _pareto_nrpa(self, level: int, key: torch.Tensor, optimizer: torch.optim.Optimizer) -> Population:
        if level == 0:
            return self._level0(key)

        optimal_set = Population()

        from tqdm import tqdm

        pbar = tqdm(
            range(self.n_iter),
            desc=f"Pareto-NRPA level {level}",
            position=self.level - level,
            leave=False,
        )

        for i in pbar:
            if level == 1:
                result = self._pareto_nrpa(level - 1, key, optimizer)
            else:
                child_key = key.detach().clone().requires_grad_(True)
                child_optimizer = optim.Adam([child_key], lr=self.lr, weight_decay=self.weight_decay)
                result = self._pareto_nrpa(level - 1, child_key, child_optimizer)

            optimal_set = Population.merge(optimal_set, result)
            fronts = NonDominatedSorting().do(optimal_set.get("F"))
            optimal_set = optimal_set[fronts[0]]

            # De-dup identical objective vectors (as neural_nrpa.py does).
            seen: set = set()
            unique_inds = []
            for ind in optimal_set:
                fkey = tuple(ind.get("F"))
                if fkey not in seen:
                    seen.add(fkey)
                    unique_inds.append(ind)
            optimal_set = Population.merge(Population(), *unique_inds)

            loss = self._pareto_adapt(key, optimizer, optimal_set)

            pbar.set_postfix(
                front=len(optimal_set),
                global_front=len(self.global_pareto_front),
                loss=f"{loss:.4f}",
            )
            logger.debug(
                "level %d iter %d/%d: |optimal_set|=%d |global_front|=%d loss=%.4f",
                level, i + 1, self.n_iter, len(optimal_set), len(self.global_pareto_front), loss,
            )

        pbar.close()
        return optimal_set

    def run(self) -> Population:
        root_key = self._root_key()
        root_optimizer = optim.Adam([root_key], lr=self.lr, weight_decay=self.weight_decay)
        return self._pareto_nrpa(self.level, root_key, root_optimizer)

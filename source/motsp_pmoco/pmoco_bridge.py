"""Wires extern/PMOCO's MOTSP model/env into EAS.

PMOCO's MOTSP code (extern/PMOCO/MOTSP/...) is a set of top-level modules, not
an installable package, and it's where the preference-conditioned hypernetwork
decoder lives (`MOTSPModel.TSPModel.decoder.assign(pref)`). This module does
the same `sys.path` wiring PMOCO's own scripts do (`sys.path.insert(0, "..")`
from within POMO/) so we can `import MOTSPModel, MOTSPEnv, MOTSProblemDef`
from here, then re-exports what EAS-MO needs.
"""

import sys
from pathlib import Path

import torch

_PMOCO_ROOT = Path(__file__).resolve().parents[3] / "PMOCO"
_MOTSP_DIR = _PMOCO_ROOT / "MOTSP"
_MOTSP_POMO_DIR = _MOTSP_DIR / "POMO"

for _p in (str(_MOTSP_DIR), str(_MOTSP_POMO_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from MOTSPModel import TSPModel, _get_encoding  # noqa: E402
from MOTSPEnv import TSPEnv  # noqa: E402
from MOTSProblemDef import get_random_problems  # noqa: E402

# Checkpoint directory naming used by PMOCO's own test scripts, e.g.
# extern/PMOCO/MOTSP/POMO/result/saved_TSP100_model/checkpoint_motsp-200.pt
_CHECKPOINT_DIR_BY_SIZE = {
    20: "saved_TSP20_model",
    50: "saved_TSP50_model",
    100: "saved_TSP100_model",
}

# Reference points used by PMOCO's own test scripts (test_motsp_n{20,50,100}.py)
# for hypervolume computation. Fixed per problem size, per instance-generation
# convention (CLAUDE.md: "compute once ... reuse across runs").
REF_POINT_BY_SIZE = {
    20: (15.0, 15.0),
    50: (30.0, 30.0),
    100: (60.0, 60.0),
}


def default_model_params() -> dict:
    """Model hyper-parameters matching the released PMOCO MOTSP checkpoints."""
    return {
        "embedding_dim": 128,
        "sqrt_embedding_dim": 128 ** (1 / 2),
        "encoder_layer_num": 6,
        "qkv_dim": 16,
        "head_num": 8,
        "logit_clipping": 10,
        "ff_hidden_dim": 512,
        "eval_type": "argmax",
    }


def checkpoint_path(problem_size: int, epoch: int = 200) -> Path:
    if problem_size not in _CHECKPOINT_DIR_BY_SIZE:
        raise ValueError(
            f"No released PMOCO MOTSP checkpoint for problem_size={problem_size}; "
            f"available: {sorted(_CHECKPOINT_DIR_BY_SIZE)}"
        )
    ckpt_dir = _MOTSP_POMO_DIR / "result" / _CHECKPOINT_DIR_BY_SIZE[problem_size]
    return ckpt_dir / f"checkpoint_motsp-{epoch}.pt"


def load_motsp_model(problem_size: int, device: torch.device, epoch: int = 200) -> TSPModel:
    """Load the frozen, pretrained PMOCO MOTSP model (encoder + hypernetwork decoder)."""
    model = TSPModel(**default_model_params())
    ckpt = torch.load(checkpoint_path(problem_size, epoch), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


__all__ = [
    "TSPModel",
    "TSPEnv",
    "get_random_problems",
    "_get_encoding",
    "default_model_params",
    "checkpoint_path",
    "load_motsp_model",
    "REF_POINT_BY_SIZE",
]

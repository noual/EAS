"""Non-dominated filtering and normalized hypervolume for the ω-sweep results.

Conventions (per project's dominance/HV rules):
- Objectives are minimized (raw tour distances).
- Cast to float64 before any dominance/HV computation.
- A fixed reference point per instance/problem size, reused across all methods.
"""

import numpy as np
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


def non_dominated_filter(points: np.ndarray) -> np.ndarray:
    """Keep only the non-dominated rows of `points` (n_points, n_obj), minimization.

    Ties/duplicates are kept as-is (pymoo's first front already excludes
    dominated points; it does not collapse duplicates).
    """
    points = np.asarray(points, dtype=np.float64)
    if len(points) <= 1:
        return points
    fronts = NonDominatedSorting().do(points, only_non_dominated_front=True)
    return points[fronts]


def hypervolume(points: np.ndarray, ref_point: np.ndarray) -> float:
    """Hypervolume of a (non-dominated) point set w.r.t. a fixed reference point."""
    points = np.asarray(points, dtype=np.float64)
    ref_point = np.asarray(ref_point, dtype=np.float64)
    if len(points) == 0:
        return 0.0
    return float(HV(ref_point=ref_point)(points))


def normalized_hypervolume(points: np.ndarray, ref_point: np.ndarray) -> float:
    """HV normalized by the reference-point box volume (prod of ref_point)."""
    box_volume = float(np.prod(np.asarray(ref_point, dtype=np.float64)))
    if box_volume <= 0.0:
        raise ValueError("ref_point must be strictly positive to normalize HV")
    return hypervolume(points, ref_point) / box_volume


from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple
import numpy as np


@dataclass
class ReachabilityResult:
    reachable_indices: np.ndarray
    unreachable_indices: np.ndarray
    labels: np.ndarray


def build_reachability_inputs(
    grasp_transforms: np.ndarray,
    contact_points: np.ndarray,
    fc_passing_indices: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if fc_passing_indices is None:
        subset_indices = np.arange(len(grasp_transforms))
    else:
        subset_indices = np.asarray(fc_passing_indices, dtype=int)

    return subset_indices, grasp_transforms[subset_indices], contact_points[subset_indices]


def run_reachability_check(
    grasp_transforms: np.ndarray,
    contact_points: np.ndarray,
    fc_passing_indices: Optional[Sequence[int]] = None,
    object_filename: Optional[str] = None,#obj to be input for mujoco check
    object_scale: Optional[float] = None,#obj scale
    mujoco_model_path: Optional[str] = None,#arm setup
    **kwargs,
) -> ReachabilityResult:
    #placeholder for mujoco
    subset_indices, _, _ = build_reachability_inputs(
        grasp_transforms=grasp_transforms,
        contact_points=contact_points,
        fc_passing_indices=fc_passing_indices,
    )
    # ----TODO: mujoco calls here and fill reach_mask---
    reachable_mask = np.zeros(len(subset_indices), dtype=bool)

    reachable_indices = subset_indices[reachable_mask]
    unreachable_indices = subset_indices[~reachable_mask]

    labels = np.full(len(grasp_transforms), -1, dtype=int)
    labels[reachable_indices] = 1
    labels[unreachable_indices] = 0

    return ReachabilityResult(reachable_indices=reachable_indices,unreachable_indices=unreachable_indices,labels=labels)

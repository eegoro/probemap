"""Landmark fits map moving physical ZYX coordinates into reference physical ZYX."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import ndimage

from .volumes import Volume, spacing_zyx


def validate_affine(matrix) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("Expected a finite 4×4 homogeneous affine matrix.")
    if not np.allclose(matrix[3], (0, 0, 0, 1)) or abs(np.linalg.det(matrix[:3, :3])) < 1e-10:
        raise ValueError("The affine must be invertible with last row [0, 0, 0, 1].")
    return matrix


def transform_points(points_zyx_um, moving_to_reference) -> np.ndarray:
    points = np.asarray(points_zyx_um, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Points must be a finite N×3 array in ZYX order.")
    matrix = validate_affine(moving_to_reference)
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def fit_landmarks(reference_zyx_um, moving_zyx_um, model="affine", allow_reflection=False) -> dict:
    ref, mov = (np.asarray(p, dtype=float) for p in (reference_zyx_um, moving_zyx_um))
    if ref.ndim != 2 or ref.shape[1:] != (3,) or ref.shape != mov.shape:
        raise ValueError("Reference and moving landmarks must have matching N×3 shapes and order.")
    if not np.isfinite(ref).all() or not np.isfinite(mov).all():
        raise ValueError("Landmarks must be finite.")
    if model not in ("rigid", "similarity", "affine"):
        raise ValueError("model must be rigid, similarity, or affine.")
    minimum = 4 if model == "affine" else 3
    if len(ref) < minimum:
        raise ValueError(f"{model} requires at least {minimum} landmark pairs.")
    ref_center, mov_center = ref.mean(0), mov.mean(0)
    ref_c, mov_c = ref - ref_center, mov - mov_center
    required_rank = 3 if model == "affine" else 2
    if min(np.linalg.matrix_rank(ref_c), np.linalg.matrix_rank(mov_c)) < required_rank:
        raise ValueError("Landmarks are degenerate. Spread pairs across the brain and several Z planes.")
    matrix = np.eye(4)
    if model == "affine":
        # Centering avoids numerical problems from fitting an intercept in micrometres.
        coefficients, _, _, _ = np.linalg.lstsq(mov_c, ref_c, rcond=None)
        matrix[:3, :3] = coefficients.T
    else:
        u, singular, vt = np.linalg.svd(mov_c.T @ ref_c)
        signs = np.ones(3)
        if np.linalg.det(vt.T @ u.T) < 0:
            signs[-1] = -1
        rotation = vt.T @ np.diag(signs) @ u.T
        scale = float(np.sum(singular * signs) / np.sum(mov_c ** 2)) if model == "similarity" else 1.0
        matrix[:3, :3] = scale * rotation
    matrix[:3, 3] = ref_center - matrix[:3, :3] @ mov_center
    validate_affine(matrix)
    determinant = float(np.linalg.det(matrix[:3, :3]))
    if determinant < 0 and not allow_reflection:
        raise ValueError("The fit mirrors anatomy. Check pair order/orientation; allow_reflection=True only if intentional.")
    errors = np.linalg.norm(transform_points(mov, matrix) - ref, axis=1)
    return {"schema_version": 1, "model": model,
            "coordinate_order": "ZYX", "units": "um", "origin": "TIFF-local zero",
            "direction": "moving_to_reference",
            "moving_to_reference_zyx_um": matrix.tolist(),
            "reference_points_zyx_um": ref.tolist(), "moving_points_zyx_um": mov.tolist(),
            "residuals_um": errors.tolist(), "rms_um": float(np.sqrt(np.mean(errors ** 2))),
            "determinant": determinant, "linear_condition_number": float(np.linalg.cond(matrix[:3, :3]))}


def save_landmarks(path: Path | str, reference_zyx_um, moving_zyx_um, **fit_options) -> dict:
    result = fit_landmarks(reference_zyx_um, moving_zyx_um, **fit_options)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding='utf-8')
    return result


def resample_manual(moving: Volume, reference: Volume, moving_to_reference, order=1) -> np.ndarray:
    """Sample the original moving grid onto the reference grid (inverse/pull map)."""
    inverse = np.linalg.inv(validate_affine(moving_to_reference))
    ref_scale = np.diag(spacing_zyx(reference.spacing_um_zyx))
    inv_mov_scale = np.diag(1 / spacing_zyx(moving.spacing_um_zyx))
    return ndimage.affine_transform(
        moving.data.astype(np.float32), inv_mov_scale @ inverse[:3, :3] @ ref_scale,
        offset=inv_mov_scale @ inverse[:3, 3], output_shape=reference.data.shape,
        order=order, mode="constant", cval=0, prefilter=False)


def write_ants_initial_transform(moving_to_reference, path: Path | str) -> Path:
    """ANTs image transforms pull fixed XYZ physical points into moving XYZ.

    The napari fit is a forward ZYX physical map, so both inversion and axis
    reversal are required. This does not assert an anatomical LPS orientation.
    """
    import ants

    pull_zyx = np.linalg.inv(validate_affine(moving_to_reference))
    reverse = np.eye(3)[::-1]
    transform = ants.create_ants_transform(
        transform_type="AffineTransform", precision="double", dimension=3,
        matrix=reverse @ pull_zyx[:3, :3] @ reverse,
        offset=reverse @ pull_zyx[:3, 3], center=(0, 0, 0))
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    ants.write_transform(transform, str(path))
    return path

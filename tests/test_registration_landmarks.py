import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from probemap.registration.landmarks import (
    fit_landmarks, resample_manual, transform_points, write_ants_initial_transform,
)
from probemap.registration.volumes import Volume, read_volume, write_volume


@pytest.mark.parametrize("model", ["rigid", "similarity", "affine"])
def test_fit_known_3d_mapping(model):
    rng = np.random.default_rng(12)
    points = rng.normal(size=(15, 3)) * [100, 200, 400]
    angle = 0.4
    matrix = np.eye(4)
    matrix[:3, :3] = [[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]]
    if model == "similarity":
        matrix[:3, :3] *= 1.4
    if model == "affine":
        matrix[:3, :3] @= np.array([[1.2, 0.1, 0], [0, 0.9, 0.2], [0, 0, 0.8]])
    matrix[:3, 3] = [140, -60, 290]
    result = fit_landmarks(transform_points(points, matrix), points, model=model)
    assert_allclose(result["moving_to_reference_zyx_um"], matrix, atol=1e-10)
    assert result["rms_um"] < 1e-10


def test_invalid_pairs_and_reflection():
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [2, 2, 2.]])
    with pytest.raises(ValueError, match="matching"):
        fit_landmarks(points, points[:-1])
    with pytest.raises(ValueError, match="degenerate"):
        fit_landmarks(points[:4] * [0, 1, 1], points[:4])
    with pytest.raises(ValueError, match="mirrors"):
        fit_landmarks(points * [-1, 1, 1], points)
    result = fit_landmarks(points * [-1, 1, 1], points, allow_reflection=True)
    assert result["determinant"] < 0


def test_spacing_aware_pull_resampling():
    moving = np.zeros((20, 24, 26), np.float32)
    moving[3, 5, 7] = 1
    mov = Volume(moving, (2, 4, 6))
    ref = Volume(np.zeros((40, 60, 80), np.float32), (1, 2, 3))
    matrix = np.eye(4)
    matrix[:3, 3] = [4, 6, 9]
    warped = resample_manual(mov, ref, matrix)
    assert np.unravel_index(warped.argmax(), warped.shape) == (10, 13, 17)


def test_tiff_stride_preserves_physical_coordinates(tmp_path):
    original = np.arange(7 * 11 * 13, dtype=np.uint16).reshape(7, 11, 13)
    path = write_volume(tmp_path / "original.tif", original, (2, 3, 4))
    preview = read_volume(path, (2, 3, 4), (2, 3, 4))
    assert preview.spacing_um_zyx == (4, 9, 16)
    assert_allclose(preview.data, original[::2, ::3, ::4])


def test_manual_ants_image_and_point_directions(tmp_path):
    import ants
    from probemap.registration.ants_workflow import to_ants, from_ants

    source = np.zeros((20, 24, 26), np.float32)
    source[3, 5, 7] = 1
    mov = Volume(source, (2, 4, 6))
    ref = Volume(np.zeros((40, 60, 80), np.float32), (1, 2, 3))
    matrix = np.eye(4)
    matrix[:3, 3] = [4, 6, 9]
    path = write_ants_initial_transform(matrix, tmp_path / "initial.mat")
    warped = ants.apply_transforms(to_ants(ref), to_ants(mov), transformlist=[str(path)], whichtoinvert=[False])
    assert np.unravel_index(from_ants(warped).data.argmax(), ref.data.shape) == (10, 13, 17)
    # The stored pull transform maps the reference physical point into moving.
    pull = ants.read_transform(str(path), precision="double")
    assert_allclose(pull.apply_to_point((51, 26, 10)), (42, 20, 6), atol=1e-5)


def test_nondiagonal_affine_agrees_between_scipy_and_ants(tmp_path):
    import ants
    from probemap.registration.ants_workflow import to_ants, from_ants

    rng = np.random.default_rng(15)
    source = Volume(rng.random((25, 27, 29), dtype=np.float32), (2, 3, 4))
    target = Volume(np.zeros((30, 32, 34), np.float32), (1.8, 2.8, 3.8))
    matrix = np.array([[0.9, 0.1, 0.05, 2], [-0.08, 1.1, 0.02, 4],
                       [0.03, -0.02, 1.05, 6], [0, 0, 0, 1.]])
    path = write_ants_initial_transform(matrix, tmp_path / "affine.mat")
    scipy_result = resample_manual(source, target, matrix)
    ants_result = from_ants(ants.apply_transforms(to_ants(target), to_ants(source),
                                                 transformlist=[str(path)], whichtoinvert=[False])).data
    # ITK and scipy treat samples within half a voxel of image boundaries differently.
    inside = scipy_result > 0
    assert_allclose(ants_result[inside], scipy_result[inside], atol=4e-5)


def test_composite_initialization_and_point_roundtrip(tmp_path):
    """Real ANTs zero-iteration stages must retain the full manual/affine/SyN chain."""
    from probemap.registration.ants_workflow import (
        load_registration, run_affine_registration, run_syn_registration, transfer_points,
        warp_volume, save_jacobian,
    )
    z, y, x = np.indices((32, 32, 32))
    data = np.exp(-((z - 12) ** 2 + (y - 14) ** 2 + (x - 17) ** 2) / 24).astype(np.float32)
    path = write_volume(tmp_path / "brain.tif", data, (2, 3, 4))
    project = {"reference": {"path": path, "spacing_um_zyx": [2, 3, 4]},
               "moving": {"path": path, "spacing_um_zyx": [2, 3, 4]},
               "ants_reference_downsample_zyx": [1, 1, 1],
               "ants_moving_downsample_zyx": [1, 1, 1], "output_dir": tmp_path / "output"}
    matrix = np.eye(4)
    matrix[:3, 3] = [2, 6, -4]
    manual = {"direction": "moving_to_reference", "units": "um", "moving_to_reference_zyx_um": matrix.tolist()}
    manual_path = tmp_path / "manual.json"
    manual_path.write_text(json.dumps(manual))
    affine_path = run_affine_registration(project, manual_path, aff_iterations=(0, 0, 0, 0))
    syn_path = run_syn_registration(affine_path, reg_iterations=(0, 0, 0))
    points = np.array([[10, 12, 14], [14, 14, 17.]])
    for manifest in (affine_path, syn_path):
        load_registration(manifest)
        transformed = transfer_points(points, manifest)
        assert_allclose(transformed, points + [1, 2, -1], atol=2e-3)
        assert_allclose(transfer_points(transformed, manifest, direction="reference_to_moving"), points, atol=2e-3)
        warped = warp_volume(manifest, Volume(data, (2, 3, 4)), Volume(data, (2, 3, 4)))
        assert np.unravel_index(warped.data.argmax(), data.shape) == (13, 16, 16)
        labels = np.zeros(data.shape, dtype=np.uint16)
        labels[8:14, 10:18, 12:20] = 101
        labels[16:20, 20:24, 20:24] = 207
        shifted_labels = warp_volume(manifest, Volume(labels, (2, 3, 4)), Volume(data, (2, 3, 4)),
                                     direction="reference_to_moving", is_label=True)
        assert shifted_labels.data.dtype == np.uint16
        assert set(np.unique(shifted_labels.data)) == {0, 101, 207}
    jacobian = save_jacobian(syn_path)
    assert jacobian["nonpositive_fraction"] == 0
    assert abs(jacobian["minimum"] - 1) < 0.01

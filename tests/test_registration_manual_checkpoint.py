import json

import ants
import numpy as np
import tifffile
from numpy.testing import assert_allclose

from probemap.registration.ants_workflow import (
    load_registration, save_manual_registration, run_syn_registration, transfer_points,
)
from probemap.registration.landmarks import fit_landmarks
from probemap.registration.volumes import write_volume


def test_manual_checkpoint_preserves_maps_and_skips_optimization(tmp_path, monkeypatch):
    z, y, x = np.indices((32, 32, 32))
    data = np.exp(-((z - 12) ** 2 + (y - 14) ** 2 + (x - 17) ** 2) / 24).astype(np.float32)
    source = write_volume(tmp_path / "placed.tif", data, (2, 3, 4))
    project = {"reference": {"path": source, "spacing_um_zyx": [2, 3, 4]},
               "moving": {"path": source, "spacing_um_zyx": [2, 3, 4]},
               "ants_reference_downsample_zyx": [2, 2, 2],
               "ants_moving_downsample_zyx": [2, 2, 2], "output_dir": tmp_path / "output"}
    points = np.array([[8, 10, 12], [12, 16, 18], [18, 12, 16], [10, 20, 15], [20, 18, 20.]])
    moving_um = points * [2, 3, 4]
    linear = np.array([[1.0, 0.02, 0], [0.01, 0.98, 0.03], [0, -0.01, 1.02]])
    reference_um = moving_um @ linear.T + [2, 6, -4]
    fit = fit_landmarks(reference_um, moving_um)
    manual = tmp_path / "manual.json"
    manual.write_text(json.dumps(fit))
    calls = []
    registration = ants.registration

    def tracked_registration(*args, **kwargs):
        calls.append(kwargs["type_of_transform"])
        return registration(*args, **kwargs)

    monkeypatch.setattr(ants, "registration", tracked_registration)
    manifest_path = save_manual_registration(project, manual)
    assert calls == []
    checkpoint = load_registration(manifest_path)
    assert checkpoint["stage"] == "manual"
    assert checkpoint["parameters"]["orientation_stage"] is False
    assert checkpoint["parameters"]["automatic_optimization"] is False
    folder = manifest_path.parent
    for name in ("manual_image_moving_to_reference", "manual_image_reference_to_moving"):
        for suffix in (".mat", ".h5"):
            assert (folder / (name + suffix)).is_file()
    matrices = json.loads((folder / "affine_matrices.json").read_text())
    direct = np.asarray(matrices["points_moving_to_reference_zyx_voxel"])
    inverse = np.asarray(matrices["points_reference_to_moving_zyx_voxel"])
    assert_allclose(direct @ inverse, np.eye(4), atol=1e-12)
    expected = reference_um / [2, 3, 4]
    assert_allclose(points @ direct[:3, :3].T + direct[:3, 3], expected, atol=1e-10)
    assert_allclose(transfer_points(points, manifest_path), expected, atol=2e-4)
    assert_allclose(transfer_points(expected, manifest_path, direction="reference_to_moving"), points, atol=2e-4)
    assert tifffile.imread(folder / "moving_in_reference_manual.tif").shape == (16, 16, 16)
    csv = np.loadtxt(folder / "landmark_pairs.csv", delimiter=",", skiprows=1)
    assert_allclose(csv[:, 10:13], points)
    assert_allclose(csv[:, 7:10], expected)
    # A separate call starts a fresh snapshot and leaves the previous run intact.
    second = save_manual_registration(project, manual, export_preview=False)
    assert second != manifest_path
    syn_path = run_syn_registration(manifest_path, reg_iterations=(0, 0, 0))
    assert calls == ["SyNOnly"]
    assert_allclose(transfer_points(points, syn_path), expected, atol=2e-3)

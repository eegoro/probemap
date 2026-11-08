import json
from pathlib import Path

import numpy as np
import pytest
import tifffile
from numpy.testing import assert_allclose

from probemap.registration.coarse import (
    bounding_grid, compose_coarse_and_landmarks, export_coarse_alignment,
    pose_matrix, resample_tiff, sample_transformed_slice, transformed_corners,
)
from probemap.registration.landmarks import fit_landmarks, resample_manual, transform_points
from probemap.registration.volumes import Volume, load_project, write_volume


def test_pose_axes_scale_and_centre():
    shape, spacing = (8, 10, 12), (2, 3, 4)
    centre = (np.asarray(shape) - 1) / 2 * spacing
    matrix = pose_matrix(shape, spacing, (5, 6, 7), scale_xyz=(0.5, 2, 1.5))
    assert_allclose(transform_points(centre[None], matrix)[0], centre + [5, 6, 7])
    assert_allclose(matrix[:3, :3], np.diag([1.5, 2, 0.5]))
    matrix = pose_matrix(shape, spacing, rotation_xyz_deg=(0, 0, 90))
    assert_allclose(transform_points((centre + [0, 0, 1])[None], matrix)[0], centre + [0, 1, 0])
    with pytest.raises(ValueError, match="positive"):
        pose_matrix(shape, spacing, scale_xyz=(1, 0, 1))


def test_nonuniform_scale_uses_atlas_axes_after_rotation():
    shape, spacing = (8, 10, 12), (25, 25, 25)
    centre = (np.asarray(shape) - 1) / 2 * spacing
    rotated = pose_matrix(shape, spacing, rotation_xyz_deg=(0, 0, 90))
    scaled = pose_matrix(shape, spacing, rotation_xyz_deg=(0, 0, 90), scale_xyz=(0.5, 2, 1.5))
    # Raw X now points along atlas Y: its length doubles with atlas-Y scale,
    # while atlas-X scale only changes the rotated raw-Y direction.
    raw_x = (centre + [0, 0, 25])[None]
    assert_allclose(transform_points(raw_x, scaled)[0] - centre, [0, 50, 0], atol=1e-10)
    assert_allclose(scaled[:3, :3], np.diag([1.5, 2, 0.5]) @ rotated[:3, :3], atol=1e-10)
    assert bounding_grid(shape, spacing, scaled, spacing)["shape_zyx"] == [12, 24, 5]


def test_bounds_change_voxel_counts_and_keep_every_corner():
    shape, spacing = (8, 10, 12), (25, 25, 25)
    identity = bounding_grid(shape, spacing, np.eye(4), spacing)
    assert identity["shape_zyx"] == list(shape)
    assert identity["origin_um_zyx"] == [0, 0, 0]
    translated = bounding_grid(shape, spacing, pose_matrix(shape, spacing, (3.7, -9.8, 12.3)), spacing)
    assert translated["shape_zyx"] == list(shape)
    assert_allclose(translated["origin_um_zyx"], [3.7, -9.8, 12.3])
    matrix = pose_matrix(shape, spacing, scale_xyz=(0.5, 2, 1.5))
    assert bounding_grid(shape, spacing, matrix, spacing)["shape_zyx"] == [12, 20, 6]
    rotation = pose_matrix(shape, spacing, (45, -38, 53), rotation_xyz_deg=(35, 60, 23))
    grid = bounding_grid(shape, spacing, rotation, (16.84, 16.84, 16.84))
    corners = transformed_corners(shape, spacing, rotation)
    origin, output_spacing = np.asarray(grid["origin_um_zyx"]), np.asarray(grid["spacing_um_zyx"])
    assert np.all(corners >= origin - output_spacing / 2 - 1e-8)
    assert np.all(corners <= origin + (np.asarray(grid["shape_zyx"]) - 0.5) * output_spacing + 1e-8)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_reslicing_agrees_with_full_3d_transform(axis):
    rng = np.random.default_rng(4)
    moving = Volume(rng.random((17, 21, 25), dtype=np.float32), (2, 3, 4))
    reference = Volume(np.zeros((23, 24, 27), np.float32), (1.8, 2.4, 3.2))
    matrix = pose_matrix(moving.data.shape, moving.spacing_um_zyx, (2, 4, 6),
                         rotation_xyz_deg=(12, 25, -16), scale_xyz=(0.8, 1.2, 0.9))
    index = reference.data.shape[axis] // 2
    expected = np.take(resample_manual(moving, reference, matrix), index, axis=axis)
    assert_allclose(sample_transformed_slice(moving, reference, matrix, axis, index), expected, atol=2e-6)


def test_chunked_native_export_has_new_shape_and_real_resampled_data(tmp_path):
    data = np.arange(8 * 10 * 12, dtype=np.uint16).reshape(8, 10, 12)
    source = write_volume(tmp_path / "source.tif", data, (25, 25, 25))
    matrix = pose_matrix(data.shape, (25, 25, 25), scale_xyz=(0.5, 2, 1.5))
    grid = bounding_grid(data.shape, (25, 25, 25), matrix, (25, 25, 25))
    path = resample_tiff(source, (25, 25, 25), matrix, grid, tmp_path / "adapted.tif", chunk_depth=3)
    result = tifffile.imread(path)
    assert result.shape == (12, 20, 6)
    assert result.dtype == data.dtype
    assert_allclose(tifffile.imread(source), data)
    # Independent scipy pull map also accounts for the nonzero saved origin.
    from scipy import ndimage
    inverse = np.linalg.inv(matrix)
    expected = ndimage.affine_transform(data, inverse[:3, :3],
        offset=(inverse[:3, :3] @ grid["origin_um_zyx"] + inverse[:3, 3]) / 25,
        output_shape=result.shape, order=1, prefilter=False)
    assert_allclose(result, expected, atol=1)
    with tifffile.TiffFile(path) as tif:
        assert tif.imagej_metadata["spacing"] == 25
        assert json.loads(tif.imagej_metadata["origin_um_zyx"]) == grid["origin_um_zyx"]
    assert not list(tmp_path.glob("*.partial.tif"))


def test_export_and_composition_map_original_raw_coordinates(tmp_path):
    from probemap.registration.ants_workflow import transfer_points

    shape = (24, 28, 32)
    z, y, x = np.indices(shape)
    data = np.exp(-((z - 12) ** 2 + (y - 14) ** 2 + (x - 16) ** 2) / 35).astype(np.float32)
    raw = write_volume(tmp_path / "raw.tif", data, (2, 3, 4))
    atlas = write_volume(tmp_path / "atlas.tif", data, (3, 3, 3))
    project = {"reference": {"path": atlas, "spacing_um_zyx": [3, 3, 3]},
               "moving": {"path": raw, "spacing_um_zyx": [2, 3, 4]},
               "output_dir": tmp_path / "output", "manual_downsample_zyx": [2, 2, 2],
               "ants_reference_downsample_zyx": [2, 2, 2], "ants_moving_downsample_zyx": [2, 2, 2]}
    coarse = pose_matrix(shape, (2, 3, 4), (5, -4, 8), (10, -15, 20), (0.8, 1.2, 0.9))
    pose = {"direction": "moving_to_reference", "coordinate_order": "ZYX", "units": "um",
            "model": "interactive_coarse", "moving_to_reference_zyx_um": coarse.tolist()}
    adaptation_path = export_coarse_alignment(project, pose, (3, 3, 3))
    adaptation = json.loads(adaptation_path.read_text())
    assert tifffile.imread(adaptation["scan_on_atlas_grid"]).shape == shape
    assert tifffile.imread(adaptation["scan_adapted"]).shape == tuple(adaptation["adapted_grid"]["shape_zyx"])
    fine_project = load_project(adaptation["refinement_config"])
    assert fine_project["moving"]["path"] == Path(adaptation["scan_on_atlas_grid"])
    raw_voxels = np.array([[6, 9, 12], [10, 12, 18], [16, 10, 15], [12, 18, 22], [18, 20, 20.]])
    adapted_um = transform_points(raw_voxels * [2, 3, 4], coarse)
    fine_matrix = pose_matrix(shape, (3, 3, 3), (1, 2, 3), (2, -3, 4), (1.02, 0.98, 1))
    target_um = transform_points(adapted_um, fine_matrix)
    fine = fit_landmarks(target_um, adapted_um)
    fine["sources"] = {name: {"path": str(fine_project[name]["path"])} for name in ("reference", "moving")}
    fine_path = tmp_path / "fine.json"
    fine_path.write_text(json.dumps(fine))
    final = compose_coarse_and_landmarks(project, adaptation_path, fine_path)
    assert_allclose(transfer_points(raw_voxels, final), target_um / 3, atol=1e-4)
    assert_allclose(transfer_points(target_um / 3, final, direction="reference_to_moving"), raw_voxels, atol=1e-4)
    local_map = np.asarray(adaptation["adapted_grid"]["raw_scan_to_adapted_voxel_zyx"])
    expected_local = (adapted_um - adaptation["adapted_grid"]["origin_um_zyx"]) / 3
    assert_allclose(transform_points(raw_voxels, local_map), expected_local, atol=1e-10)

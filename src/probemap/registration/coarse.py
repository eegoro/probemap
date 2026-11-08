"""Interactive pose geometry, reslicing and lossless-extent TIFF exports."""
from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage
from scipy.spatial.transform import Rotation

from .landmarks import transform_points, validate_affine
from .volumes import Volume, spacing_zyx, tiff_info


def pose_matrix(shape_zyx, voxel_um_zyx, translation_zyx_um=(0, 0, 0),
                rotation_xyz_deg=(0, 0, 0), scale_xyz=(1, 1, 1)) -> np.ndarray:
    """Rotate the scan, scale along fixed atlas axes, then translate its centre.

    Returned map is raw scan physical ZYX → atlas physical ZYX. Euler order is
    extrinsic X then Y then Z. Scaling acts *after* rotation: S_atlas @ R.
    Both operations pivot at the scan centre; translation uses atlas axes.
    """
    translation = np.asarray(translation_zyx_um, dtype=float)
    angles = np.asarray(rotation_xyz_deg, dtype=float)
    scales = np.asarray(scale_xyz, dtype=float)
    if any(v.shape != (3,) or not np.isfinite(v).all() for v in (translation, angles, scales)):
        raise ValueError("Translation, angles and scales must each have three finite values.")
    if (scales <= 0).any():
        raise ValueError("Scale factors must be positive.")
    centre = (np.asarray(shape_zyx) - 1) / 2 * spacing_zyx(voxel_um_zyx)
    reverse = np.eye(3)[::-1]
    linear = reverse @ np.diag(scales) @ Rotation.from_euler("xyz", angles, degrees=True).as_matrix() @ reverse
    result = np.eye(4)
    result[:3, :3] = linear
    result[:3, 3] = centre + translation - linear @ centre
    return validate_affine(result)


def transformed_corners(shape_zyx, voxel_um_zyx, matrix) -> np.ndarray:
    # Include voxel-cell edges, so even a rotated/scaled edge voxel is retained.
    corners = np.asarray(list(product(*[(-0.5, n - 0.5) for n in shape_zyx])))
    return transform_points(corners * spacing_zyx(voxel_um_zyx), matrix)


def bounding_grid(shape_zyx, voxel_um_zyx, matrix, output_spacing_um_zyx) -> dict:
    """A grid enclosing all transformed source voxel cells, with explicit origin."""
    spacing = spacing_zyx(output_spacing_um_zyx)
    corners = transformed_corners(shape_zyx, voxel_um_zyx, matrix)
    lower, upper = corners.min(0), corners.max(0)
    # This independent bounding grid need not snap to the atlas lattice. Keeping
    # its fractional origin prevents a pure translation from adding extra voxels.
    origin = lower + 0.5 * spacing
    shape = np.maximum(1, np.ceil((upper - lower) / spacing - 1e-10)).astype(int)
    return {"shape_zyx": shape.tolist(), "spacing_um_zyx": spacing.tolist(),
            "origin_um_zyx": origin.tolist(),
            "transformed_bounds_um_zyx": [lower.tolist(), upper.tolist()]}


def sample_transformed_slice(moving: Volume, reference: Volume, matrix, axis: int, index: int) -> np.ndarray:
    """Reslice through the transformed 3D volume, including out-of-plane rotations.

    Only the requested plane is evaluated, making angle/scale controls interactive
    without interpolating a whole 3D stack at each mouse movement.
    """
    if axis not in (0, 1, 2) or not 0 <= index < reference.data.shape[axis]:
        raise ValueError("Invalid slice axis or index.")
    others = [a for a in range(3) if a != axis]
    grid = np.indices(tuple(reference.data.shape[a] for a in others), dtype=float)
    coordinates = np.empty((3, grid.shape[1] * grid.shape[2]), dtype=float)
    coordinates[axis] = index
    for position, a in enumerate(others):
        coordinates[a] = grid[position].ravel()
    inverse = np.linalg.inv(validate_affine(matrix))
    physical = coordinates * np.asarray(reference.spacing_um_zyx)[:, None]
    source = (inverse[:3, :3] @ physical + inverse[:3, 3, None]) / np.asarray(moving.spacing_um_zyx)[:, None]
    return ndimage.map_coordinates(moving.data, source, order=1, prefilter=False,
                                   mode="constant", cval=0).reshape(grid.shape[1:])


def resample_tiff(source_path, source_spacing_um_zyx, matrix, target_grid: dict,
                   output_path, chunk_depth=8, progress=None, *, overwrite=False,
                   order=1, ome=False) -> Path:
    """Apply an affine to original intensities using disk-backed Z slabs.

    Output voxel counts and samples change when scale/angles change. This is not
    a spacing-only metadata edit. The original intensity dtype is preserved.
    """
    if chunk_depth < 1:
        raise ValueError("chunk_depth must be positive.")
    if order not in (0, 1):
        raise ValueError('Interpolation must be nearest or linear.')
    tiff_info(source_path)
    try:
        source = tifffile.memmap(source_path, mode="r")
    except ValueError:
        source = tifffile.imread(source_path, out="memmap")
    spacing = spacing_zyx(target_grid["spacing_um_zyx"])
    shape = tuple(int(v) for v in target_grid["shape_zyx"])
    origin = np.asarray(target_grid["origin_um_zyx"], dtype=float)
    if len(shape) != 3 or min(shape) < 1 or origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError("Invalid output grid.")
    inverse = np.linalg.inv(validate_affine(matrix))
    source_inverse_scale = np.diag(1 / spacing_zyx(source_spacing_um_zyx))
    pull_linear = source_inverse_scale @ inverse[:3, :3] @ np.diag(spacing)
    pull_offset = source_inverse_scale @ (inverse[:3, :3] @ origin + inverse[:3, 3])
    # Least-squares fits can put an exact boundary at -1e-15. Constant padding
    # would then erase a whole edge plane even for a pure translation.
    for values in (pull_linear,pull_offset):
        rounded=np.rint(values)
        values[np.abs(values-rounded)<1e-12]=rounded[np.abs(values-rounded)<1e-12]
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output == Path(source_path).resolve():
        raise ValueError("The output TIFF must differ from the source scan.")
    if output.exists() and not overwrite:
        raise FileExistsError(output)
    temporary = output.with_name(output.stem + ".partial.tif")
    big = int(np.prod(shape, dtype=np.int64)) * source.dtype.itemsize > 4_000_000_000
    metadata = {"axes": "ZYX", "unit": "um", "spacing": float(spacing[0]),
                "origin_um_zyx": json.dumps(origin.tolist())}
    if ome:
        metadata = {'axes':'ZYX', **{f'PhysicalSize{axis}':float(value)
                    for axis,value in zip('ZYX',spacing)},
                    **{f'PhysicalSize{axis}Unit':'µm' for axis in 'ZYX'}}
    target = None
    try:
        target = tifffile.memmap(
            temporary, shape=shape, dtype=source.dtype, bigtiff=big, imagej=not big and not ome, ome=ome,
            photometric="minisblack", resolution=(10_000 / spacing[2], 10_000 / spacing[1]),
            resolutionunit="CENTIMETER", metadata=metadata)
        for start in range(0, shape[0], chunk_depth):
            stop = min(start + chunk_depth, shape[0])
            ndimage.affine_transform(
                source, pull_linear, offset=pull_offset + pull_linear[:, 0] * start,
                output_shape=(stop - start, *shape[1:]), output=target[start:stop],
                order=order, prefilter=False, mode="constant", cval=0)
            if progress:
                progress(stop, shape[0])
        target.flush()
        from ..platform_io import close_memmap
        close_memmap(target)
        del target
        target = None
        temporary.replace(output)
    except BaseException:
        if target is not None:
            target.flush()
            from ..platform_io import close_memmap
            close_memmap(target)
            del target
        temporary.unlink(missing_ok=True)
        raise
    return output


def export_coarse_alignment(project: dict, pose: dict, adapted_spacing_um_zyx=(25, 25, 25)) -> Path:
    """Save native-scale adapted TIFF, atlas-grid TIFF, geometry and raw checkpoint."""
    from .ants_workflow import save_manual_registration

    matrix = validate_affine(pose["moving_to_reference_zyx_um"])
    for name in ("reference", "moving"):
        source_info = pose.get("sources", {}).get(name, {})
        saved = source_info.get("path")
        if saved and Path(saved).resolve() != Path(project[name]["path"]).resolve():
            raise ValueError(f"Saved GUI pose belongs to a different {name} file.")
        if "spacing_um_zyx" in source_info and not np.allclose(
                source_info["spacing_um_zyx"], project[name]["spacing_um_zyx"]):
            raise ValueError(f"The {name} spacing changed after selecting the GUI pose.")
        if "shape_zyx" in source_info and source_info["shape_zyx"] != tiff_info(project[name]["path"])["shape_zyx"]:
            raise ValueError(f"The {name} shape changed after selecting the GUI pose.")
    state_path = Path(project["output_dir"]) / "coarse_pose.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(pose, indent=2) + "\n", encoding='utf-8')
    coarse_project = {**project, "output_dir": Path(project["output_dir"]) / "coarse"}
    manifest = save_manual_registration(coarse_project, state_path, export_preview=False)
    folder = manifest.parent
    (folder / "coarse_pose.json").write_text(json.dumps(pose, indent=2) + "\n", encoding='utf-8')
    reference_info = tiff_info(project["reference"]["path"])
    moving_info = tiff_info(project["moving"]["path"])
    fitted_grid = bounding_grid(moving_info["shape_zyx"], project["moving"]["spacing_um_zyx"],
                                matrix, adapted_spacing_um_zyx)
    atlas_grid = {"shape_zyx": reference_info["shape_zyx"],
                  "spacing_um_zyx": list(project["reference"]["spacing_um_zyx"]),
                  "origin_um_zyx": [0.0, 0.0, 0.0]}
    for filename, grid in (("scan_adapted.tif", fitted_grid), ("scan_on_atlas_grid.tif", atlas_grid)):
        print(f"Export {filename}: {grid['shape_zyx']}", flush=True)
        last_percent = [-1]
        def progress(done, total):
            percent = int(100 * done / total) // 20 * 20
            if percent != last_percent[0]:
                print(f"  {percent}%", flush=True)
                last_percent[0] = percent
        resample_tiff(project["moving"]["path"], project["moving"]["spacing_um_zyx"],
                       matrix, grid, folder / filename, progress=progress)
    source_scale, target_scale, shift = np.eye(4), np.eye(4), np.eye(4)
    source_scale[:3, :3] = np.diag(project["moving"]["spacing_um_zyx"])
    target_scale[:3, :3] = np.diag(fitted_grid["spacing_um_zyx"])
    shift[:3, 3] = -np.asarray(fitted_grid["origin_um_zyx"])
    voxel_map = np.linalg.inv(target_scale) @ shift @ matrix @ source_scale
    fitted_grid.update(raw_scan_to_adapted_voxel_zyx=voxel_map.tolist(),
                       adapted_voxel_to_raw_scan_zyx=np.linalg.inv(voxel_map).tolist())
    (folder / "scan_adapted_geometry.json").write_text(json.dumps(fitted_grid, indent=2) + "\n", encoding='utf-8')
    refinement_project = {
        "reference": {"path": str(Path(project["reference"]["path"]).resolve()),
                      "spacing_um_zyx": atlas_grid["spacing_um_zyx"]},
        "moving": {"path": str(folder / "scan_on_atlas_grid.tif"),
                   "spacing_um_zyx": atlas_grid["spacing_um_zyx"]},
        "output_dir": str(folder / "landmark_refinement"),
        "manual_downsample_zyx": list(project.get("manual_downsample_zyx", (2, 2, 2))),
        "ants_reference_downsample_zyx": [2, 2, 2], "ants_moving_downsample_zyx": [2, 2, 2],
    }
    refinement_config = folder / "refinement_project.json"
    refinement_config.write_text(json.dumps(refinement_project, indent=2) + "\n", encoding='utf-8')
    record = {"raw_coarse_checkpoint": str(manifest), "pose": str(folder / "coarse_pose.json"),
              "scan_adapted": str(folder / "scan_adapted.tif"), "adapted_grid": fitted_grid,
              "scan_on_atlas_grid": str(folder / "scan_on_atlas_grid.tif"), "atlas_grid": atlas_grid,
              "refinement_config": str(refinement_config), "original_inputs": {
                  name: {"path": str(Path(project[name]["path"]).resolve()),
                         "spacing_um_zyx": list(project[name]["spacing_um_zyx"])}
                  for name in ("reference", "moving")}}
    path = folder / "adaptation.json"
    path.write_text(json.dumps(record, indent=2) + "\n", encoding='utf-8')
    return path


def compose_coarse_and_landmarks(project: dict, adaptation_path, landmark_path) -> Path:
    """Compose F_landmark @ F_coarse into a raw-scan → native-atlas checkpoint."""
    from .ants_workflow import save_manual_registration

    adaptation = json.loads(Path(adaptation_path).read_text(encoding='utf-8'))
    coarse = json.loads(Path(adaptation["pose"]).read_text(encoding='utf-8'))
    fine = json.loads(Path(landmark_path).read_text(encoding='utf-8'))
    for name in ("reference", "moving"):
        source = adaptation["original_inputs"][name]
        if (Path(source["path"]).resolve() != Path(project[name]["path"]).resolve()
                or not np.allclose(source["spacing_um_zyx"], project[name]["spacing_um_zyx"])):
            raise ValueError(f"Original {name} geometry changed after the coarse export.")
    expected = {"reference": project["reference"]["path"], "moving": adaptation["scan_on_atlas_grid"]}
    for name, path in expected.items():
        if Path(fine.get("sources", {}).get(name, {}).get("path", "")).resolve() != Path(path).resolve():
            raise ValueError(f"Landmarks refer to a different {name} file.")
    coarse_matrix = validate_affine(coarse["moving_to_reference_zyx_um"])
    fine_matrix = validate_affine(fine["moving_to_reference_zyx_um"])
    matrix = fine_matrix @ coarse_matrix
    combined = {**fine, "model": "interactive_coarse_then_landmark_affine",
                "moving_to_reference_zyx_um": matrix.tolist(),
                "moving_points_zyx_um": transform_points(fine["moving_points_zyx_um"],
                                                         np.linalg.inv(coarse_matrix)).tolist(),
                "sources": {name: {"path": str(Path(project[name]["path"]).resolve())}
                            for name in ("reference", "moving")},
                "determinant": float(np.linalg.det(matrix[:3, :3])),
                "linear_condition_number": float(np.linalg.cond(matrix[:3, :3])),
                "composition": "landmark_affine @ coarse_GUI_affine",
                "coarse_stage_zyx_um": coarse_matrix.tolist(),
                "landmark_stage_zyx_um": fine_matrix.tolist()}
    final_project = {**project, "output_dir": Path(project["output_dir"]) / "final"}
    path = Path(final_project["output_dir"]) / "combined_manual_transform.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(combined, indent=2) + "\n", encoding='utf-8')
    manifest = save_manual_registration(final_project, path, export_preview=True)
    (manifest.parent / "adaptation_snapshot.json").write_text(json.dumps(adaptation, indent=2) + "\n", encoding='utf-8')
    (manifest.parent / "landmark_transform_snapshot.json").write_text(json.dumps(fine, indent=2) + "\n", encoding='utf-8')
    return manifest

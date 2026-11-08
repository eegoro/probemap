"""Checkpointed manual → rigid → affine → optional SyN registration.

TIFF arrays are ZYX. ANTs images are explicitly transposed to XYZ; physical
coordinates remain in micrometres in a local zero-origin, identity-direction
frame. Native NRRD world orientation is intentionally not inferred from TIFF.
"""
from __future__ import annotations

import json
import os
import platform
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import numpy as np

from .landmarks import validate_affine, write_ants_initial_transform
from .volumes import Volume, read_volume, tiff_info, write_volume


def _ants():
    # Set before ANTs/ITK initialization. The notebooks do this in the first cell too.
    os.environ.setdefault("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "4")
    import ants
    return ants


def to_ants(volume: Volume):
    return _ants().from_numpy(np.ascontiguousarray(volume.data.transpose(2, 1, 0), dtype=np.float32),
                              spacing=tuple(volume.spacing_um_zyx[::-1]),
                              origin=(0, 0, 0), direction=np.eye(3))


def from_ants(image, path=None) -> Volume:
    return Volume(np.ascontiguousarray(image.numpy().transpose(2, 1, 0)),
                  tuple(image.spacing[::-1]), path)


def prepare_pair(project: dict) -> tuple[Volume, Volume]:
    return tuple(read_volume(project[key]["path"], project[key]["spacing_um_zyx"],
                             project[f"ants_{key}_downsample_zyx"], normalize=True)
                 for key in ("reference", "moving"))


def _new_run(output_dir, stage):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = Path(output_dir) / "runs" / f"{stamp}_{stage}_{uuid4().hex[:6]}"
    path.mkdir(parents=True, exist_ok=False)
    return path.resolve()


def _inputs(project):
    return {key: {**tiff_info(project[key]["path"]),
                  "spacing_um_zyx": list(project[key]["spacing_um_zyx"]),
                  "downsample_zyx": list(project[f"ants_{key}_downsample_zyx"])}
            for key in ("reference", "moving")}


def _prepare_from_manifest(manifest):
    return tuple(read_volume(manifest["inputs"][key]["path"],
                             manifest["inputs"][key]["spacing_um_zyx"],
                             manifest["inputs"][key]["downsample_zyx"], normalize=True)
                 for key in ("reference", "moving"))


def _save_manifest(run_dir, result, inputs, stage, parameters, parent=None, manual=None):
    # Composite HDF5 files contain the initialization as well as the fitted stage.
    # Persist explicit filenames returned by ANTs instead of searching with glob.
    manifest = {"schema_version": 1, "stage": stage, "inputs": inputs,
                "coordinate_convention": "TIFF-local physical XYZ in ANTs, ZYX in numpy; um; origin zero",
                "fwd_composite": str(Path(result["fwdtransforms"]).resolve()),
                "inv_composite": str(Path(result["invtransforms"]).resolve()),
                "parameters": parameters, "parent_manifest": str(parent) if parent else None,
                "manual_transform": str(manual) if manual else None,
                "versions": {key: version(key) for key in ("antspyx", "numpy", "scipy", "tifffile")},
                "platform": platform.platform(),
                "itk_threads": os.environ.get("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS")}
    path = run_dir / "registration.json"
    from ..project_paths import relative_path_links
    links={"reference":inputs.get('reference'),"moving":inputs.get('moving'),
           "exports":[{"path":manifest[k]} for k in ('fwd_composite','inv_composite','parent_manifest','manual_transform') if manifest.get(k)]}
    manifest['relative_paths']=relative_path_links(links,path)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding='utf-8')
    return path


def load_registration(path: Path | str) -> dict:
    path = Path(path).resolve()
    manifest = json.loads(path.read_text(encoding='utf-8'))
    from ..project_paths import relocate_project_paths
    keys=[k for k in ('fwd_composite','inv_composite','parent_manifest','manual_transform') if manifest.get(k)]
    inputs=manifest.get('inputs',{})
    links={'reference':dict(inputs['reference']) if inputs.get('reference') else None,
           'moving':dict(inputs['moving']) if inputs.get('moving') else None,
           'exports':[{'path':manifest[k]} for k in keys],
           'relative_paths':manifest.get('relative_paths',{})}
    relocate_project_paths(links,path)
    for key,entry in zip(keys,links['exports']): manifest[key]=entry['path']
    for key in ('reference','moving'):
        if links.get(key): inputs[key]['path']=links[key]['path']
    for key in ("fwd_composite", "inv_composite"):
        if not Path(manifest[key]).is_file():
            raise FileNotFoundError(manifest[key])
    return manifest


def save_manual_registration(project: dict, manual_path: Path | str, export_preview=True,
                             *, output_folder: Path | str | None = None) -> Path:
    """Checkpoint an existing landmark affine without estimating any new transform.

    Save both point-map matrices (physical and original-TIFF voxels), ANTs image
    transforms in both directions, and the exact landmark/configuration snapshot.
    The checkpoint can be used directly by warp_volume, transfer_points and
    run_syn_registration; no rigid or automatic affine stage is run here.
    """
    ants = _ants()
    manual_path = Path(manual_path).resolve()
    manual = json.loads(manual_path.read_text(encoding='utf-8'))
    if (manual.get("direction") != "moving_to_reference" or manual.get("units") != "um"
            or manual.get("coordinate_order", "ZYX") != "ZYX"):
        raise ValueError("Expected a manual moving→reference transform in ZYX micrometres.")
    matrix = validate_affine(manual["moving_to_reference_zyx_um"])
    for key in ("reference", "moving"):
        saved = manual.get("sources", {}).get(key, {}).get("path")
        if saved and Path(saved).resolve() != Path(project[key]["path"]).resolve():
            raise ValueError(f"Manual transform belongs to a different {key} file.")
    inputs = _inputs(project)
    run_dir = Path(output_folder).resolve() if output_folder is not None else _new_run(project["output_dir"], "manual")
    run_dir.mkdir(parents=True, exist_ok=True)
    snapshot = run_dir / "manual_transform.json"
    snapshot.write_text(json.dumps(manual, indent=2, ensure_ascii=False) + "\n", encoding='utf-8')
    (run_dir / "project_config.json").write_text(json.dumps(project, indent=2, default=str) + "\n", encoding='utf-8')
    # Derive the snapshot from the fitted points, rather than copying a possibly
    # newer/older landmark_points.json from an interactive editing session.
    points = {key: manual[key] for key in (
        "reference_points_zyx_um", "moving_points_zyx_um", "sources") if key in manual}
    points.update(coordinate_order="ZYX", units="um")
    (run_dir / "landmark_points.json").write_text(json.dumps(points, indent=2) + "\n", encoding='utf-8')
    inverse = np.linalg.inv(matrix)
    reference_scale, moving_scale = np.eye(4), np.eye(4)
    reference_scale[:3, :3] = np.diag(inputs["reference"]["spacing_um_zyx"])
    moving_scale[:3, :3] = np.diag(inputs["moving"]["spacing_um_zyx"])
    voxel_matrix = np.linalg.inv(reference_scale) @ matrix @ moving_scale
    matrices = {
        "points_moving_to_reference_zyx_um": matrix,
        "points_reference_to_moving_zyx_um": inverse,
        "points_moving_to_reference_zyx_voxel": voxel_matrix,
        "points_reference_to_moving_zyx_voxel": np.linalg.inv(voxel_matrix),
    }
    for name, value in matrices.items():
        np.save(run_dir / f"{name}.npy", value)
    (run_dir / "affine_matrices.json").write_text(json.dumps(
        {name: value.tolist() for name, value in matrices.items()}, indent=2) + "\n", encoding='utf-8')
    if all(key in manual for key in ("reference_points_zyx_um", "moving_points_zyx_um")):
        reference_points = np.asarray(manual["reference_points_zyx_um"]).reshape(-1, 3)
        moving_points = np.asarray(manual["moving_points_zyx_um"]).reshape(-1, 3)
        table = np.column_stack((np.arange(1, len(reference_points) + 1), reference_points,
                                 moving_points,
                                 reference_points / inputs["reference"]["spacing_um_zyx"],
                                 moving_points / inputs["moving"]["spacing_um_zyx"]))
        np.savetxt(run_dir / "landmark_pairs.csv", table, delimiter=",", comments="",
                   header="pair,ref_z_um,ref_y_um,ref_x_um,mov_z_um,mov_y_um,mov_x_um,"
                          "ref_z_voxel,ref_y_voxel,ref_x_voxel,mov_z_voxel,mov_y_voxel,mov_x_voxel")
    result = {}
    for key, point_matrix, stem in (
        ("fwdtransforms", matrix, "manual_image_moving_to_reference"),
        ("invtransforms", inverse, "manual_image_reference_to_moving"),
    ):
        path = write_ants_initial_transform(point_matrix, run_dir / f"{stem}.mat")
        transform = ants.read_transform(str(path), precision="double")
        composite = run_dir / f"{stem}.h5"
        ants.write_transform(transform, str(composite))
        result[key] = str(composite)
    manifest_path = _save_manifest(
        run_dir, result, inputs, "manual",
        {"model": manual.get("model", "affine"), "automatic_optimization": False,
         "orientation_stage": False, "spacing_assumption": project.get("spacing_assumption")},
        manual=snapshot)
    (run_dir / "transform_conventions.txt").write_text(
        "Input/output grids are the supplied TIFFs, with origin zero.\n"
        "points_*.npy and affine_matrices.json are direct ZYX coordinate maps.\n"
        "Voxel matrices refer to the full input TIFFs, not downsampled previews.\n"
        "manual_image_*.mat/.h5 are ANTs XYZ physical pull/resampling maps.\n"
        "For moving→reference images use manual_image_moving_to_reference.h5.\n"
        "For moving→reference points use manual_image_reference_to_moving.h5.\n"
        "warp_volume and transfer_points select the correct direction automatically.\n"
        "Preprocessing before the supplied TIFFs is outside this checkpoint.\n", encoding='utf-8')
    if export_preview:
        reference, moving = prepare_pair(project)
        warped = warp_volume(manifest_path, moving, reference)
        write_volume(run_dir / "moving_in_reference_manual.tif", warped.data, reference.spacing_um_zyx)
    return manifest_path


def run_affine_registration(project: dict, manual_path: Path | str | None = None,
                            aff_iterations=(500, 250, 100, 0), verbose=False) -> Path:
    """Fit rigid then affine. None explicitly selects automatic center-of-mass initialization."""
    ants = _ants()
    reference, moving = prepare_pair(project)
    fixed, mov = to_ants(reference), to_ants(moving)
    run_dir = _new_run(project["output_dir"], "affine")
    initial = None
    if manual_path is not None:
        manual_path = Path(manual_path).resolve()
        manual = json.loads(manual_path.read_text(encoding='utf-8'))
        if manual.get("direction") != "moving_to_reference" or manual.get("units") != "um":
            raise ValueError("Manual transform must map moving→reference in physical micrometres.")
        for key in ("reference", "moving"):
            saved = manual.get("sources", {}).get(key, {}).get("path")
            if saved and Path(saved).resolve() != Path(project[key]["path"]).resolve():
                raise ValueError(f"Manual transform belongs to a different {key} file.")
        initial = str(write_ants_initial_transform(manual["moving_to_reference_zyx_um"],
                                                   run_dir / "manual_initial.mat"))
        # Keep a reviewable snapshot even if landmarks are edited later.
        (run_dir / "manual_transform.json").write_text(json.dumps(manual, indent=2) + "\n", encoding='utf-8')
        manual_path = run_dir / "manual_transform.json"
    options = dict(aff_metric="mattes", aff_sampling=32, aff_random_sampling_rate=1.0,
                   aff_iterations=tuple(aff_iterations), aff_shrink_factors=(8, 4, 2, 1),
                   aff_smoothing_sigmas=(3, 2, 1, 0), singleprecision=True,
                   write_composite_transform=True, verbose=verbose)
    print("Rigid registration…", flush=True)
    rigid = ants.registration(fixed=fixed, moving=mov, type_of_transform="Rigid",
                              initial_transform=initial, outprefix=str(run_dir / "rigid_"), **options)
    print("Affine registration…", flush=True)
    affine = ants.registration(fixed=fixed, moving=mov, type_of_transform="Affine",
                               initial_transform=rigid["fwdtransforms"],
                               outprefix=str(run_dir / "affine_"), **options)
    write_volume(run_dir / "moving_in_reference_affine.tif",
                 from_ants(affine["warpedmovout"]).data, reference.spacing_um_zyx)
    parameters = {"aff_iterations": list(aff_iterations), "aff_metric": "mattes",
                  "aff_sampling": 32, "aff_random_sampling_rate": 1.0,
                  "aff_shrink_factors": [8, 4, 2, 1], "aff_smoothing_sigmas": [3, 2, 1, 0]}
    return _save_manifest(run_dir, affine, _inputs(project), "affine", parameters, manual=manual_path)


def run_syn_registration(affine_manifest_path: Path | str, reg_iterations=(40, 20, 0),
                         grad_step=0.1, flow_sigma=3.0, total_sigma=0.0, verbose=False) -> Path:
    """Resume from a reviewed manual/affine checkpoint without repeating affine fitting."""
    ants = _ants()
    parent_path = Path(affine_manifest_path).resolve()
    parent = load_registration(parent_path)
    reference, moving = _prepare_from_manifest(parent)
    run_dir = _new_run(parent_path.parent.parent.parent, "syn")
    parameters = {"reg_iterations": list(reg_iterations), "grad_step": grad_step,
                  "flow_sigma": flow_sigma, "total_sigma": total_sigma,
                  "syn_metric": "mattes", "syn_sampling": 32}
    print("SyN registration…", flush=True)
    result = ants.registration(fixed=to_ants(reference), moving=to_ants(moving),
                               type_of_transform="SyNOnly", initial_transform=parent["fwd_composite"],
                               outprefix=str(run_dir / "syn_"), write_composite_transform=True,
                               singleprecision=True, verbose=verbose, **parameters)
    write_volume(run_dir / "moving_in_reference_syn.tif", from_ants(result["warpedmovout"]).data,
                 reference.spacing_um_zyx)
    return _save_manifest(run_dir, result, parent["inputs"], "syn", parameters,
                          parent=parent_path, manual=parent.get("manual_transform"))


def warp_volume(manifest_path: Path | str, source: Volume, target: Volume,
                 direction="moving_to_reference", is_label=False) -> Volume:
    """Apply a saved image transform; source/target must use their original local grids."""
    if direction not in ("moving_to_reference", "reference_to_moving"):
        raise ValueError("Unknown transform direction.")
    ants = _ants()
    manifest = load_registration(manifest_path)
    key = "fwd_composite" if direction == "moving_to_reference" else "inv_composite"
    result = ants.apply_transforms(fixed=to_ants(target), moving=to_ants(source),
                                   transformlist=[manifest[key]], whichtoinvert=[False],
                                   interpolator="genericLabel" if is_label else "linear",
                                   singleprecision=True)
    data = from_ants(result).data
    if is_label:
        data = np.rint(data).astype(source.data.dtype)
    return Volume(data, target.spacing_um_zyx)


def transfer_points(points_zyx, manifest_path: Path | str, direction="moving_to_reference", units="voxel") -> np.ndarray:
    """Transfer points between original TIFF grids, in native voxels or physical um.

    Point transforms use the OPPOSITE composite from image resampling. Input and
    output both use ZYX, with no extra flip/permutation required by the caller.
    """
    import pandas as pd

    ants = _ants()
    manifest = load_registration(manifest_path)
    points = np.asarray(points_zyx, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Expected finite N×3 ZYX points.")
    if units not in ("voxel", "um"):
        raise ValueError("units must be voxel or um.")
    if direction == "moving_to_reference":
        source, target, key = "moving", "reference", "inv_composite"
    elif direction == "reference_to_moving":
        source, target, key = "reference", "moving", "fwd_composite"
    else:
        raise ValueError("Unknown transform direction.")
    if len(points) == 0:
        return points.copy()
    if units == "voxel":
        points = points * np.asarray(manifest["inputs"][source]["spacing_um_zyx"])
    frame = pd.DataFrame(points[:, ::-1], columns=["x", "y", "z"])
    result = ants.apply_transforms_to_points(3, frame, transformlist=[manifest[key]], whichtoinvert=[False])
    transformed = result[["x", "y", "z"]].to_numpy()[:, ::-1]
    if units == "voxel":
        transformed = transformed / np.asarray(manifest["inputs"][target]["spacing_um_zyx"])
    return transformed


def save_qc(manifest_path: Path | str) -> Path:
    """Save orthogonal central-slice overlays; quantitative success still needs anatomy review."""
    import matplotlib.pyplot as plt

    manifest = load_registration(manifest_path)
    reference, moving = _prepare_from_manifest(manifest)
    warped = warp_volume(manifest_path, moving, reference).data
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for axis in range(3):
        index = reference.data.shape[axis] // 2
        ref_slice, mov_slice = (np.take(v, index, axis=axis) for v in (reference.data, warped))
        axes[0, axis].imshow(ref_slice, cmap="gray", vmin=0, vmax=1)
        overlay = np.zeros((*ref_slice.shape, 3), dtype=np.float32)
        overlay[..., 1], overlay[..., 0], overlay[..., 2] = ref_slice, mov_slice, mov_slice
        axes[1, axis].imshow(np.clip(overlay, 0, 1))
        axes[0, axis].set_title(f"Reference: {'ZYX'[axis]}={index}")
        axes[1, axis].set_title("Green reference / magenta moving")
    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    path = Path(manifest_path).parent / "registration_qc.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_jacobian(manifest_path: Path | str) -> dict:
    """Jacobian of the total fixed→moving physical map, including the affine part."""
    ants = _ants()
    manifest = load_registration(manifest_path)
    reference, moving = _prepare_from_manifest(manifest)
    out = Path(manifest_path).parent
    field = ants.apply_transforms(fixed=to_ants(reference), moving=to_ants(moving),
                                  transformlist=[manifest["fwd_composite"]], whichtoinvert=[False],
                                  compose=str(out / "total_"), singleprecision=True)
    jac = ants.create_jacobian_determinant_image(to_ants(reference), field, do_log=False, geom=True)
    data = from_ants(jac).data
    write_volume(out / "total_pull_jacobian.tif", data, reference.spacing_um_zyx)
    report = {"map": "total fixed-to-moving physical pull map, including affine",
              "minimum": float(np.min(data)), "maximum": float(np.max(data)),
              "nonpositive_fraction": float(np.mean(data <= 0))}
    (out / "jacobian_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding='utf-8')
    return report

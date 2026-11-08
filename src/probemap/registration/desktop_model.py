"""UI-independent project, calibrated TIFF IO and affine export for ProbeMap."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import tifffile
from scipy import ndimage

from ..branding import APP_NAME, LEGACY_APP_NAME
from ..project_paths import project_file_path, relative_path_links, relocate_project_paths
from .coarse import bounding_grid, resample_tiff, transformed_corners
from .landmarks import fit_landmarks, transform_points, validate_affine
from .volumes import Volume, normalize_intensity, spacing_zyx, tiff_info


def _unit_um(unit):
    return {"um": 1, "µm": 1, "μm": 1, "micron": 1, "micrometer": 1,
            "micrometers": 1, "nm": 0.001, "mm": 1000, "cm": 10000,
            "m": 1e6, "inch": 25400}.get(str(unit).lower())


def inspect_tiff(path) -> dict:
    """Read calibrated OME/ImageJ/TIFF tags; never invent missing axis spacing."""
    path = Path(path).resolve()
    info = tiff_info(path)
    values, origins, evidence = [None] * 3, [0.0] * 3, []
    with tifffile.TiffFile(path) as tif:
        ij = tif.imagej_metadata or {}
        unit = _unit_um(ij.get("unit", ""))
        if unit is not None and "spacing" in ij:
            values[0] = float(ij["spacing"]) * unit
            evidence.append("ImageJ Z spacing")
        page = tif.pages[0]
        resolution_unit = int(page.tags["ResolutionUnit"].value) if "ResolutionUnit" in page.tags else 1
        physical_unit = {2: 25400, 3: 10000}.get(resolution_unit, unit)
        if physical_unit is not None:
            for index, name in ((2, "XResolution"), (1, "YResolution")):
                if name in page.tags:
                    numerator, denominator = page.tags[name].value
                    if numerator > 0:
                        values[index] = physical_unit * denominator / numerator
            evidence.append("TIFF XY resolution tags")
        if tif.ome_metadata:
            root = ElementTree.fromstring(tif.ome_metadata)
            pixels = next((node for node in root.iter() if node.tag.split("}")[-1] == "Pixels"), None)
            if pixels is not None:
                for index, axis in enumerate("ZYX"):
                    key = f"PhysicalSize{axis}"
                    if key in pixels.attrib:
                        factor = _unit_um(pixels.attrib.get(key + "Unit", "µm"))
                        if factor is not None:
                            values[index] = float(pixels.attrib[key]) * factor
                evidence.append("OME physical size")
        if "origin_um_zyx" in ij:
            value = ij["origin_um_zyx"]
            origins = json.loads(value) if isinstance(value, str) else list(value)
    # ProbeMap sidecars take precedence over generic tags.
    for sidecar in (path.with_suffix(".geometry.json"), path.with_name(path.stem + "_geometry.json")):
        if sidecar.is_file():
            side = json.loads(sidecar.read_text(encoding='utf-8'))
            if "spacing_um_zyx" in side:
                values = list(spacing_zyx(side["spacing_um_zyx"]))
            origins = side.get("origin_um_zyx", origins)
            evidence.append(sidecar.name)
            break
    for i, value in enumerate(values):
        if value is not None and (not np.isfinite(value) or value <= 0):
            values[i] = None
    if np.asarray(origins).shape != (3,) or not np.isfinite(origins).all():
        raise ValueError("Invalid TIFF origin metadata.")
    return {**info, "spacing_um_zyx": values, "origin_um_zyx": list(origins),
            "calibration_source": "; ".join(evidence) or "No physical voxel size found"}


@dataclass
class ImageSpec:
    path: Path
    shape_zyx: tuple[int, int, int]
    spacing_um_zyx: tuple[float, float, float]
    origin_um_zyx: tuple[float, float, float] = (0, 0, 0)
    initial_spacing_um_zyx: tuple[float, float, float] | None = None

    def __post_init__(self):
        self.path = Path(self.path).resolve()
        self.shape_zyx = tuple(int(n) for n in self.shape_zyx)
        if len(self.shape_zyx) != 3 or min(self.shape_zyx) < 1:
            raise ValueError("Image shape must contain three positive dimensions.")
        self.spacing_um_zyx = tuple(spacing_zyx(self.spacing_um_zyx))
        if self.initial_spacing_um_zyx is not None:
            self.initial_spacing_um_zyx=tuple(spacing_zyx(self.initial_spacing_um_zyx))
        self.origin_um_zyx = tuple(float(v) for v in self.origin_um_zyx)
        if len(self.origin_um_zyx) != 3 or not np.isfinite(self.origin_um_zyx).all():
            raise ValueError("Image origin must contain three finite values.")

    def record(self):
        record={"path": str(self.path), "shape_zyx": list(self.shape_zyx),
                "spacing_um_zyx": list(self.spacing_um_zyx), "origin_um_zyx": list(self.origin_um_zyx)}
        if self.initial_spacing_um_zyx is not None:
            record['initial_spacing_um_zyx']=list(self.initial_spacing_um_zyx)
        return record


def load_preview(spec: ImageSpec, max_dimension=256) -> Volume:
    factors = np.maximum(1, np.ceil(np.array(spec.shape_zyx) / max_dimension).astype(int))
    # Bound CPU preview memory even when the user requests fine landmark views.
    # The preview dimensions shown in the app are the actual sampled grid.
    while np.prod(np.ceil(np.array(spec.shape_zyx)/factors)) > 32_000_000:
        axis = int(np.argmax(np.array(spec.shape_zyx)/factors))
        factors[axis] += 1
    with tifffile.TiffFile(spec.path) as tif:
        if tuple(tif.series[0].shape) != spec.shape_zyx:
            raise ValueError(f"Image shape changed: {spec.path}")
        mapped = tif.asarray(out="memmap")
        data = np.array(mapped[tuple(slice(None, None, int(n)) for n in factors)], copy=True)
    from .volumes import intensity_limits
    return Volume(normalize_intensity(data), tuple(np.array(spec.spacing_um_zyx) * factors), spec.path,intensity_limits(data))


def full_view_grid(reference: ImageSpec, moving: ImageSpec, matrix, preview_spacing,
                   max_voxels=32_000_000, max_dimension=768, *, additional_matrices=()) -> dict:
    """Atlas-axis lattice enclosing the reference and all requested scan poses.

    Display padding is independent of the exported atlas grid. A translated or
    rotated scan must never be cropped merely to prepare landmark previews.
    """
    corners = np.vstack([
        transformed_corners(reference.shape_zyx, reference.spacing_um_zyx, np.eye(4)),
        *[transformed_corners(moving.shape_zyx, moving.spacing_um_zyx, transform)
          for transform in (matrix, *additional_matrices)],
    ])
    lower, upper = corners.min(0), corners.max(0)
    spacing = spacing_zyx(preview_spacing).copy()
    def geometry():
        origin = np.floor(lower/spacing + .5 + 1e-10)*spacing
        shape = np.maximum(1, np.ceil((upper-origin)/spacing + .5 - 1e-10)).astype(int)
        return origin, shape
    origin, shape = geometry()
    while int(np.prod(shape, dtype=np.int64)) > max_voxels or max(shape) > max_dimension:
        spacing *= 1.15
        origin, shape = geometry()
    return {"shape_zyx": shape.tolist(), "spacing_um_zyx": spacing.tolist(),
            "origin_um_zyx": origin.tolist(), "axes": "ZYX", "frame": "atlas TIFF-local physical"}


def resample_preview_grid(volume: Volume, matrix, grid):
    inverse = np.linalg.inv(validate_affine(matrix))
    inv_spacing = np.diag(1/np.asarray(volume.spacing_um_zyx))
    linear = inv_spacing @ inverse[:3,:3] @ np.diag(grid["spacing_um_zyx"])
    offset = inv_spacing @ (inverse[:3,:3] @ grid["origin_um_zyx"] + inverse[:3,3])
    return ndimage.affine_transform(volume.data, linear, offset=offset,
                                   output_shape=tuple(grid["shape_zyx"]), order=1,
                                   prefilter=False, mode="constant", cval=0)


def sample_preview_grid_slice(volume: Volume, matrix, grid, axis, index, *, order=1):
    """Reslice a physical plane with an arbitrary (possibly negative) origin."""
    others = [a for a in range(3) if a != axis]
    shape = tuple(grid["shape_zyx"][a] for a in others)
    plane = np.indices(shape, dtype=float)
    coords = np.empty((3, int(np.prod(shape))))
    coords[axis] = index
    for i,a in enumerate(others):
        coords[a] = plane[i].ravel()
    world = coords*np.asarray(grid["spacing_um_zyx"])[:,None] + np.asarray(grid["origin_um_zyx"])[:,None]
    inverse = np.linalg.inv(validate_affine(matrix))
    source = (inverse[:3,:3]@world + inverse[:3,3,None])/np.asarray(volume.spacing_um_zyx)[:,None]
    return ndimage.map_coordinates(volume.data, source, order=order, prefilter=False,
                                   mode="constant", cval=0).reshape(shape)


def prepare_landmark_views(reference_volume, moving_volume, reference, moving, matrix):
    grid = full_view_grid(reference, moving, matrix, reference_volume.spacing_um_zyx)
    return (resample_preview_grid(reference_volume, np.eye(4), grid),
            resample_preview_grid(moving_volume, matrix, grid), grid)


@dataclass
class AlignmentProject:
    reference: ImageSpec
    moving: ImageSpec
    output_dir: Path
    coarse: np.ndarray = field(default_factory=lambda: np.eye(4))
    residual: np.ndarray = field(default_factory=lambda: np.eye(4))
    pairs: list = field(default_factory=list)
    fitted_pairs: list = field(default_factory=list)
    fitted_coarse: list = field(default_factory=list)
    fit_result: dict | None = None
    display: dict = field(default_factory=dict)
    preview_dimension: int = 256
    track: ImageSpec | None = None
    labels_dir: Path | None = None
    track_analysis: dict = field(default_factory=dict)
    labels_spacing_um_zyx: tuple[float,float,float] | None = None
    exports: list = field(default_factory=list)
    ephys: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.labels_spacing_um_zyx is not None:
            self.labels_spacing_um_zyx=tuple(spacing_zyx(self.labels_spacing_um_zyx))

    @property
    def label_spacing(self):
        if self.labels_spacing_um_zyx is not None:
            return self.labels_spacing_um_zyx
        if self.labels_dir is not None:
            files=sorted(p for p in Path(self.labels_dir).iterdir() if p.suffix.lower() in ('.tif','.tiff'))
            if files:
                values=inspect_tiff(files[0])['spacing_um_zyx']
                return tuple(v if v is not None else fallback for v,fallback in zip(values,self.reference.spacing_um_zyx))
        return self.reference.spacing_um_zyx

    def validate_track(self):
        if self.track is None:
            return
        if not np.allclose(self.track.origin_um_zyx,self.moving.origin_um_zyx,rtol=0,atol=1e-5):
            raise ValueError("Brain and electrodes scans must share the same physical origin and axis directions.")

    @property
    def total(self):
        return validate_affine(self.residual @ self.coarse)

    @property
    def fit_is_current(self):
        return (self.fit_result is not None and self.pairs == self.fitted_pairs
                and np.allclose(self.coarse, self.fitted_coarse, rtol=0, atol=1e-9))

    def fit(self, model="affine"):
        complete = [p for p in self.pairs if p.get("reference") is not None and p.get("moving") is not None]
        result = fit_landmarks([p["reference"] for p in complete], [p["moving"] for p in complete], model=model)
        self.residual = np.array(result["moving_to_reference_zyx_um"])
        self.fit_result = result
        self.fitted_pairs = json.loads(json.dumps(self.pairs))
        self.fitted_coarse = self.coarse.tolist()
        return result

    def as_record(self):
        return {"application": APP_NAME, "schema_version": 1,
                "coordinate_order": "ZYX", "units": "um",
                "coordinate_frame": "Atlas TIFF-local axes, zero origin; source origins recorded separately",
                "reference": self.reference.record(), "moving": self.moving.record(),
                "output_dir": str(Path(self.output_dir).resolve()), "coarse": self.coarse.tolist(),
                "residual": self.residual.tolist(), "total": self.total.tolist(), "pairs": self.pairs,
                "fitted_pairs": self.fitted_pairs, "fitted_coarse": self.fitted_coarse,
                "fit_result": self.fit_result, "display": self.display,
                "preview_dimension": self.preview_dimension,
                "track":self.track.record() if self.track is not None else None,
                "labels_dir":str(Path(self.labels_dir).resolve()) if self.labels_dir else None,
                "labels_spacing_um_zyx":list(self.label_spacing) if self.labels_dir else None,
                "track_analysis":self.track_analysis,"exports":self.exports,"ephys":self.ephys}

    def save(self, path):
        from .insertion_model import sync_measurements
        for session in self.track_analysis.get('sessions',[]):
            sync_measurements(session)
        path = project_file_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.output_dir = path.parent
        record=self.as_record()
        record['relative_paths']=relative_path_links(record,path)
        data=json.dumps(record,indent=2)+"\n"
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(data, encoding='utf-8')
        temporary.replace(path)
        return path

    @classmethod
    def load(cls, path):
        path = project_file_path(path)
        record = json.loads(path.read_text(encoding='utf-8'))
        if record.get("application") not in (APP_NAME,LEGACY_APP_NAME) or record.get("schema_version") != 1:
            raise ValueError("Choose a ProbeMap project JSON file.")
        relocate_project_paths(record,path)
        specs = [ImageSpec(**record[key]) for key in ("reference", "moving")]
        for spec in specs:
            if tuple(tiff_info(spec.path)["shape_zyx"]) != spec.shape_zyx:
                raise ValueError(f"Image shape changed: {spec.path}")
        project = cls(*specs, Path(record["output_dir"]),
                      coarse=validate_affine(record["coarse"]), residual=validate_affine(record["residual"]),
                      pairs=record.get("pairs", []), fitted_pairs=record.get("fitted_pairs", []),
                      fitted_coarse=record.get("fitted_coarse", []), fit_result=record.get("fit_result"),
                      display=record.get("display", {}), preview_dimension=int(record.get("preview_dimension", 256)),
                      track=ImageSpec(**record["track"]) if record.get("track") else None,
                      labels_dir=Path(record["labels_dir"]) if record.get("labels_dir") else None,
                      track_analysis=record.get("track_analysis",{}),
                      labels_spacing_um_zyx=record.get('labels_spacing_um_zyx'),exports=record.get('exports',[]),
                      ephys=record.get('ephys',{}))
        project.validate_track()
        if project.track and tuple(tiff_info(project.track.path)["shape_zyx"]) != project.track.shape_zyx:
            raise ValueError(f"Track image shape changed: {project.track.path}")
        for pair in project.pairs:
            for key in ("reference", "moving"):
                value = pair.get(key)
                if value is not None and (np.asarray(value).shape != (3,) or not np.isfinite(value).all()):
                    raise ValueError("Invalid saved landmark coordinates.")
        return project


def orientation_linear(order="XYZ", flips=(False, False, False)):
    """Explicit scan-axis reassignment; order lists source axes for atlas XYZ."""
    if sorted(order) != list("XYZ"):
        raise ValueError("Axis order must be a permutation of XYZ.")
    xyz = np.eye(3)[["XYZ".index(axis) for axis in order]]
    xyz = np.diag([-1 if value else 1 for value in flips]) @ xyz
    reverse = np.eye(3)[::-1]
    return reverse @ xyz @ reverse


def matrix_summary(matrix, moving):
    linear = validate_affine(matrix)[:3, :3]
    sample_vectors = linear @ np.diag(moving.spacing_um_zyx)
    return {"determinant": float(np.linalg.det(linear)),
            "transformed_source_sample_lengths_um_zyx": np.linalg.norm(sample_vectors, axis=0).tolist(),
            "transformed_source_sample_vectors_in_atlas_zyx_um": sample_vectors.tolist()}


def export_project(project: AlignmentProject, stage="affine", adapted_spacing=None, include_ants=True,
                   progress=None, cancelled=lambda: False) -> Path:
    """Export directly from the input scan, with exact atlas grid and a full extent copy.

    All matrices refer to the original input TIFF, never a downsampled preview.
    Scaling is baked into the output samples. Output spacing is explicitly chosen
    and is not guessed from a rotated/sheared linear map.
    """
    if stage not in ("placement", "affine"):
        raise ValueError("Unknown export stage.")
    if stage == "affine" and not project.fit_is_current:
        raise ValueError("Fit the current landmark pairs before exporting the affine result.")
    project.validate_track()
    matrix = project.coarse if stage == "placement" else project.total
    folder = Path(project.output_dir).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    project.save(folder / "project.json")
    atlas_grid = {"shape_zyx": list(project.reference.shape_zyx),
                  "spacing_um_zyx": list(project.reference.spacing_um_zyx), "origin_um_zyx": [0., 0., 0.],
                  "axes": "ZYX", "reference_path": str(project.reference.path)}
    grids = [("scan_on_atlas_grid.tif", atlas_grid, project.moving, "brain")]
    if adapted_spacing is not None:
        grids.append(("scan_adapted.tif", bounding_grid(project.moving.shape_zyx,
                     project.moving.spacing_um_zyx, matrix, adapted_spacing), project.moving, "brain"))
    if project.track is not None:
        grids.append(("track_on_atlas_grid.tif", atlas_grid, project.track, "track"))
        if adapted_spacing is not None:
            grids.append(("track_adapted.tif", bounding_grid(project.track.shape_zyx,
                         project.track.spacing_um_zyx,matrix,adapted_spacing), project.track, "track"))
    input_paths={spec.path for spec in (project.reference,project.moving,project.track) if spec is not None}
    if any((folder/filename).resolve() in input_paths for filename,*_ in grids):
        raise ValueError("Choose a results folder that will not overwrite an input TIFF.")
    transforms = {"placement_moving_to_atlas_zyx_um": project.coarse,
                  "landmark_placed_to_atlas_zyx_um": project.residual if stage == "affine" else np.eye(4),
                  "moving_to_atlas_zyx_um": matrix, "atlas_to_moving_zyx_um": np.linalg.inv(matrix)}
    source_scale, atlas_scale = np.eye(4), np.eye(4)
    source_scale[:3, :3] = np.diag(project.moving.spacing_um_zyx)
    atlas_scale[:3, :3] = np.diag(project.reference.spacing_um_zyx)
    transforms["moving_to_atlas_zyx_voxel"] = np.linalg.inv(atlas_scale) @ matrix @ source_scale
    transforms["atlas_to_moving_zyx_voxel"] = np.linalg.inv(transforms["moving_to_atlas_zyx_voxel"])
    if project.track is not None:
        transforms["track_to_atlas_zyx_um"] = matrix
        transforms["atlas_to_track_zyx_um"] = np.linalg.inv(matrix)
        track_scale=np.eye(4)
        track_scale[:3,:3]=np.diag(project.track.spacing_um_zyx)
        transforms["track_to_atlas_zyx_voxel"] = np.linalg.inv(atlas_scale) @ matrix @ track_scale
        transforms["atlas_to_track_zyx_voxel"] = np.linalg.inv(transforms["track_to_atlas_zyx_voxel"])
    for name, value in transforms.items():
        np.save(folder / f"{name}.npy", value)
    (folder / "transforms.json").write_text(json.dumps({key: value.tolist() for key, value in transforms.items()}, indent=2) + "\n", encoding='utf-8')
    manifest = {"application": APP_NAME, "schema_version": 1, "stage": stage, "status": "in_progress",
                "reference": project.reference.record(), "moving": project.moving.record(),
                "track":project.track.record() if project.track else None,
                "outputs": [], "coordinate_order": "ZYX", "units": "um",
                "direction": "moving_to_reference", "moving_to_reference_zyx_um": matrix.tolist(),
                "geometry_summary": matrix_summary(matrix, project.moving),
                "track_geometry_summary":matrix_summary(matrix,project.track) if project.track else None,
                "interpolation": "linear; source intensity dtype retained",
                "note": "Preview contrast/opacity are not applied to exported intensities."}
    manifest_path = folder / "export.json"
    try:
        for number, (filename, grid, source, role) in enumerate(grids):
            def update(done, total):
                if cancelled():
                    raise InterruptedError("Export cancelled. Completed files remain in the results folder.")
                if progress:
                    progress(int(90 * (number + done / total) / len(grids)), f"Writing {filename}: {done}/{total} Z slices")
            resample_tiff(source.path, source.spacing_um_zyx, matrix, grid,
                          folder / filename, progress=update, overwrite=True)
            grid = {**grid, "axes": "ZYX", "unit": "um", "reference_path": str(project.reference.path),
                    "source_path":str(source.path),"source_role":role,"source_to_atlas_zyx_um":matrix.tolist()}
            origin_shift = np.eye(4)
            origin_shift[:3, 3] = -np.asarray(grid["origin_um_zyx"])
            target_scale = np.eye(4)
            target_scale[:3, :3] = np.diag(grid["spacing_um_zyx"])
            current_source_scale=np.eye(4)
            current_source_scale[:3,:3]=np.diag(source.spacing_um_zyx)
            voxel_map = np.linalg.inv(target_scale) @ origin_shift @ matrix @ current_source_scale
            grid.update(moving_to_output_voxel_zyx=voxel_map.tolist(),
                        output_voxel_to_moving_zyx=np.linalg.inv(voxel_map).tolist())
            (folder / filename).with_suffix(".geometry.json").write_text(json.dumps(grid, indent=2) + "\n", encoding='utf-8')
            manifest["outputs"].append({"filename": filename, **grid})
        if include_ants:
            if progress:
                progress(92, "Saving ANTs transforms")
            from .ants_workflow import save_manual_registration
            manual = {"model": "composed_manual_affine", "direction": "moving_to_reference",
                      "coordinate_order": "ZYX", "units": "um", "moving_to_reference_zyx_um": matrix.tolist(),
                      "sources": {key: getattr(project, key).record() for key in ("reference", "moving")}}
            if stage == "affine":
                ref = project.fit_result["reference_points_zyx_um"]
                raw = transform_points(project.fit_result["moving_points_zyx_um"], np.linalg.inv(project.coarse))
                manual.update(reference_points_zyx_um=ref, moving_points_zyx_um=raw.tolist())
            manual_path = folder / "manual_transform.json"
            manual_path.write_text(json.dumps(manual, indent=2) + "\n", encoding='utf-8')
            config = {key: getattr(project, key).record() for key in ("reference", "moving")}
            config.update(output_dir=folder, ants_reference_downsample_zyx=[1]*3,
                          ants_moving_downsample_zyx=[1]*3)
            manifest["ants_checkpoint"] = str(save_manual_registration(
                config, manual_path, export_preview=False, output_folder=folder))
        if stage == "affine":
            result = project.fit_result
            ref, placed = np.asarray(result["reference_points_zyx_um"]), np.asarray(result["moving_points_zyx_um"])
            raw = transform_points(placed, np.linalg.inv(project.coarse))
            ids = [i+1 for i,pair in enumerate(project.pairs)
                   if pair.get("reference") is not None and pair.get("moving") is not None]
            table = np.column_stack((ids, ref, placed, raw, result["residuals_um"]))
            np.savetxt(folder / "landmark_pairs.csv", table, delimiter=",", comments="",
                       header="pair,atlas_z_um,atlas_y_um,atlas_x_um,placed_z_um,placed_y_um,placed_x_um,raw_z_um,raw_y_um,raw_x_um,residual_um")
        if stage == 'affine' and project.track_analysis.get('sessions'):
            from .neuropixels_model import write_reports
            write_reports(project,folder)
        manifest["status"] = "complete"
        if progress:
            progress(100, "Export complete")
    except BaseException as error:
        manifest.update(status="incomplete", error=str(error))
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding='utf-8')
    return manifest_path

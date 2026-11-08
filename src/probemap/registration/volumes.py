"""TIFF-local physical grids used by the Lavian-style registration workflow."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile


def spacing_zyx(value) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.isfinite(result).all() or (result <= 0).any():
        raise ValueError("spacing_um_zyx must contain three finite positive values.")
    return result


@dataclass
class Volume:
    data: np.ndarray
    spacing_um_zyx: tuple[float, float, float]
    path: Path | None = None
    intensity_limits: tuple[float,float] | None = None

    def __post_init__(self):
        if self.data.ndim != 3:
            raise ValueError(f"Expected a 3D ZYX volume, got {self.data.shape}.")
        self.spacing_um_zyx = tuple(spacing_zyx(self.spacing_um_zyx))


def load_project(config_path: Path | str) -> dict:
    """Resolve config paths relative to the project, independent of notebook cwd."""
    path = Path(config_path).resolve()
    config = json.loads(path.read_text(encoding='utf-8'))
    root = path.parent.parent
    for key in ("reference", "moving"):
        config[key]["path"] = (root / config[key]["path"]).resolve()
        spacing_zyx(config[key]["spacing_um_zyx"])
        if not config[key]["path"].is_file():
            raise FileNotFoundError(config[key]["path"])
    config["output_dir"] = (root / config["output_dir"]).resolve()
    if config.get("atlas_labels"):
        config["atlas_labels"] = (root / config["atlas_labels"]).resolve()
    config["config_path"] = path
    return config


def tiff_info(path: Path | str) -> dict:
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        if series.ndim != 3 or series.axes not in ("ZYX", "QYX", "IYX"):
            raise ValueError(f"Expected scalar ZYX TIFF, got {series.axes}: {series.shape}.")
        return {"path": str(Path(path).resolve()), "shape_zyx": list(series.shape),
                "dtype": str(series.dtype), "axes": series.axes,
                "imagej_metadata": tif.imagej_metadata}


def read_volume(path: Path | str, spacing_um_zyx, downsample_zyx=(1, 1, 1),
                normalize: bool = False) -> Volume:
    """Stride a disk-backed TIFF and keep the exact physical spacing of the grid.

    Decimation is for interactive/registration previews, not quantitative analysis.
    A compressed TIFF may need a temporary decompressed memory-mapped file.
    """
    tiff_info(path)
    factors = np.asarray(downsample_zyx)
    if factors.shape != (3,) or (factors < 1).any() or not np.equal(factors, factors.astype(int)).all():
        raise ValueError("downsample_zyx must contain three positive integers.")
    factors = factors.astype(int)
    with tifffile.TiffFile(path) as tif:
        mapped = tif.asarray(out="memmap")
        data = np.array(mapped[tuple(slice(None, None, int(f)) for f in factors)], copy=True)
    if normalize:
        data = normalize_intensity(data)
    spacing = spacing_zyx(spacing_um_zyx) * factors
    return Volume(data, tuple(spacing), Path(path).resolve())


def intensity_limits(data):
    sample = data.ravel()[::max(1, data.size // 500_000)]
    sample = sample[np.isfinite(sample)]
    if sample.size == 0:
        raise ValueError("Volume has no finite intensities.")
    lo, hi = np.percentile(sample, (1, 99.8))
    if hi <= lo:
        lo, hi = float(sample.min()), float(sample.max())
    if hi <= lo:
        raise ValueError("Volume is constant; registration needs intensity variation.")
    return float(lo),float(hi)


def normalize_intensity(data: np.ndarray) -> np.ndarray:
    lo,hi=intensity_limits(data)
    result = np.nan_to_num(data.astype(np.float32), nan=float(lo), posinf=float(hi), neginf=float(lo))
    result -= lo
    result /= hi - lo
    return np.clip(result, 0, 1, out=result)


def write_volume(path: Path | str, data: np.ndarray, spacing_um_zyx) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    z, y, x = spacing_zyx(spacing_um_zyx)
    tifffile.imwrite(path, data, imagej=True, photometric="minisblack",
                     resolution=(10_000 / x, 10_000 / y), resolutionunit="CENTIMETER",
                     metadata={"axes": "ZYX", "unit": "um", "spacing": z})
    return path

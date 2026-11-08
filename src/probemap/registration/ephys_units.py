"""Stable raster lanes: display positions never replace anatomical unit depth."""
import numpy as np


def raster_positions(units):
    """Spread colliding/nearby units within neighbouring depth gaps, by ID."""
    ordered = sorted(units, key=lambda u: (float(u['height_um']), int(u['id'])))
    if not ordered: return {}
    depths = np.asarray([u['height_um'] for u in ordered], float)
    # Resolve crowded runs together; isolated units retain their true depth.
    groups = np.split(np.arange(len(depths)), np.flatnonzero(np.diff(depths) >= 15) + 1)
    result = {}
    for group in groups:
        center = float(np.mean(depths[group]))
        half = max(7.5, (len(group)-1)*7.5, float(np.ptp(depths[group])) / 2)
        if group[0]: half = min(half, max(0.01, (center-depths[group[0]-1])*.45))
        if group[-1] < len(depths)-1:
            half = min(half, max(0.01, (depths[group[-1]+1]-center)*.45))
        values = [center] if len(group)==1 else np.linspace(max(0,center-half),center+half,len(group))
        result.update((int(ordered[i]['id']),float(y)) for i,y in zip(group,values))
    return result

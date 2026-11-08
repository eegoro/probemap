"""Separate editable region paths, preserving disconnected islands and holes."""
import numpy as np


def region_contours(mask):
    from skimage.measure import find_contours
    # Pad so regions touching the image edge have closed boundaries.
    return [c - 1 for c in find_contours(np.pad(mask.astype(np.uint8), 1), .5)]


def matplotlib_region(ax, mask, extent, color, gid):
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch
    height, width = mask.shape
    left, right, bottom, top = extent
    vertices, codes = [], []
    for contour in region_contours(mask):
        xy = np.column_stack((left+(contour[:,1]+.5)*(right-left)/width,
                              top+(contour[:,0]+.5)*(bottom-top)/height))
        vertices.extend(xy)
        codes.extend([Path.MOVETO]+[Path.LINETO]*(len(xy)-2)+[Path.CLOSEPOLY])
    if not vertices: return
    patch = PathPatch(Path(vertices,codes),facecolor=color,edgecolor='none',zorder=2)
    patch.set_gid(gid); ax.add_patch(patch)
    return patch


def qt_region(mask, scale):
    from qtpy.QtCore import Qt
    from qtpy.QtGui import QPainterPath
    path=QPainterPath(); path.setFillRule(Qt.FillRule.OddEvenFill)
    for contour in region_contours(mask):
        path.moveTo((contour[0,1]+.5)/scale,(contour[0,0]+.5)/scale)
        for y,x in contour[1:]: path.lineTo((x+.5)/scale,(y+.5)/scale)
        path.closeSubpath()
    return path

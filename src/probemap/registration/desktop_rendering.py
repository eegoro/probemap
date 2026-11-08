"""Per-volume GPU rendering settings for the standalone alignment viewer."""
from __future__ import annotations

import copy

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (QComboBox, QDialog, QDoubleSpinBox, QFormLayout, QHBoxLayout,
                           QLabel, QPushButton, QSizePolicy, QSlider, QTabWidget, QVBoxLayout, QWidget)

from .desktop_widgets import PopupComboBox


METHODS = {"Maximum intensity (MIP)":"mip", "Attenuated MIP":"attenuated_mip",
           "Minimum intensity (MinIP)":"minip", "Average intensity":"average",
           "Translucent volume":"translucent", "Additive volume":"additive", "Isosurface":"iso"}
BLENDING = {"Additive":"additive", "Translucent":"translucent",
            "Translucent (no depth)":"translucent_no_depth", "Opaque":"opaque",'Minimum':'minimum','Multiplicative':'multiplicative'}
COLORS = {"green":(.2,1,.45,1), "magenta":(1,.18,.85,1), "gray":(1,1,1,1),
          "red":(1,.1,.1,1), "blue":(.2,.45,1,1), "cyan":(0,1,1,1), "yellow":(1,1,.1,1)}
from napari.utils.colormaps import AVAILABLE_COLORMAPS
from napari.layers.image._image_constants import Interpolation
COLORMAPS = tuple(AVAILABLE_COLORMAPS)
INTERPOLATIONS=tuple(i.value for i in Interpolation if i.value!='custom')


def default_render_settings(layer_count=2):
    return [{"rendering":"mip", "blending":"additive", "contrast":[0.,1.], "gamma":1.,
             "colormap":color, "interpolation":"linear", "iso_threshold":.3,
             "attenuation":.05, "sampling_step":1.,'auto_contrast':False,'depiction':'volume',
             'plane_position_um_zyx':None,'plane_normal_um_zyx':[1,0,0],'plane_thickness_um':None,'projection_mode':'mean'}
             for color in ("green","magenta","gray")[:layer_count]]


def validate_render_settings(records,layer_count=2):
    if not isinstance(records,list) or len(records)!=layer_count:
        raise ValueError(f"3D rendering settings must contain {layer_count} layer settings.")
    result=default_render_settings(layer_count)
    for target,record in zip(result,records):
        for key,choices in (("rendering",METHODS.values()),("blending",BLENDING.values()),
                            ("interpolation",INTERPOLATIONS),('depiction',('volume','plane'))):
            if record.get(key,target[key]) not in choices:
                raise ValueError(f"Unknown 3D {key}.")
            target[key]=record.get(key,target[key])
        for key,low,high in (("gamma",.1,5),("iso_threshold",-1e12,1e12),("attenuation",0,10),("sampling_step",.1,3)):
            value=float(record.get(key,target[key]))
            if not np.isfinite(value) or not low<=value<=high:
                raise ValueError(f"Invalid 3D {key}.")
            target[key]=value
        contrast=np.asarray(record.get("contrast",target["contrast"]),dtype=float)
        if contrast.shape!=(2,) or not np.isfinite(contrast).all() or not contrast[0]<contrast[1]:
            raise ValueError("Contrast minimum must be below maximum.")
        target["contrast"]=contrast.tolist()
        for key in ('colormap','colormap_colors','auto_contrast','plane_position_um_zyx','plane_normal_um_zyx','plane_thickness_um','projection_mode'):
            if key in record: target[key]=copy.deepcopy(record[key])
    return result


def apply_render_settings(visual, settings, previous=None):
    from napari.utils.colormaps import ensure_colormap,Colormap
    from napari._vispy.layers.image import _napari_cmap_to_vispy
    from napari._vispy.utils.gl import BLENDING_MODES
    previous=previous or {}
    if previous.get("colormap")!=settings["colormap"] or previous.get('colormap_colors')!=settings.get('colormap_colors'):
        try: cmap=ensure_colormap(settings['colormap'])
        except (ValueError,KeyError): cmap=Colormap(settings['colormap_colors'],name=settings['colormap'])
        visual.cmap=_napari_cmap_to_vispy(cmap)
    for key,attribute in (("rendering","method"),("contrast","clim"),("gamma","gamma"),
        ("interpolation","interpolation"),("iso_threshold","threshold"),
        ("attenuation","attenuation"),("sampling_step","relative_step_size")):
        if previous.get(key)!=settings[key]:
            setattr(visual,attribute,tuple(settings[key]) if key=="contrast" else settings[key])
    blend=settings['blending']; visual.set_gl_state(**BLENDING_MODES[blend])
    visual.mip_cutoff=visual._texture.clim_normalized[0] if blend in ('translucent','translucent_no_depth') else None
    visual.minip_cutoff=visual._texture.clim_normalized[1] if blend in ('translucent','translucent_no_depth') else None
    visual.raycasting_mode=settings.get('depiction','volume')
    spacing=np.asarray(getattr(visual,'source_spacing_zyx',np.ones(3)))
    position=settings.get('plane_position_um_zyx')
    visual.plane_position=(np.asarray(visual._vol_shape)-1)/2 if position is None else np.asarray(position)/spacing
    normal=np.asarray(settings.get('plane_normal_um_zyx',[1,0,0]))*spacing; normal/=np.linalg.norm(normal)
    visual.plane_normal=normal
    thickness=settings.get('plane_thickness_um')
    visual.plane_thickness=max(1.,thickness*np.linalg.norm(normal/spacing)) if thickness is not None else 1.


def volume_visual(volume,parent,settings):
    from napari._vispy.visuals.volume import Volume
    visual=Volume(volume.data,clim=(0,1),method='mip',texture_format='auto',parent=parent)
    visual.unfreeze(); visual.source_spacing_zyx=volume.spacing_um_zyx; visual.freeze()
    apply_render_settings(visual,settings)
    return visual


from .desktop_render_panel import RenderSettingsPanel


class RenderSettingsDialog(QDialog):
    def __init__(self,owner):
        super().__init__(owner)
        self.setWindowTitle("3D rendering settings")
        self.resize(490,650)
        layout=QVBoxLayout(self)
        self.panel=RenderSettingsPanel(owner)
        layout.addWidget(self.panel)
        self.tabs,self.controls=self.panel.tabs,self.panel.controls

    def sync(self):
        self.panel.sync()

    def _auto(self,index,auto):
        self.panel._auto(index,auto)

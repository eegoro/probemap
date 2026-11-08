"""The actual napari image controls, bound to ProbeMap's image settings."""
import copy
import numpy as np
from qtpy.QtCore import Signal,Qt
from qtpy.QtWidgets import QWidget,QVBoxLayout,QComboBox,QListView
from napari.layers import Image
from napari._qt.layer_controls.qt_image_controls import QtImageControls


class NapariImageControls(QWidget):
    changed=Signal(object)

    def __init__(self,ndisplay=3):
        super().__init__(); self.ndisplay=ndisplay; self._syncing=False; self.spacing=np.ones(3)
        self.layer=Image(np.linspace(0,1,8 if ndisplay==3 else 4,dtype=np.float32).reshape((2,)*ndisplay),
            contrast_limits=(0,1),rgb=False,cache=False)
        self.panel=QtImageControls(self.layer); self.panel.ndisplay=ndisplay
        self.panel.panzoom_button.hide(); self.panel.transform_button.hide()
        # Thick projections along extra, non-displayed dimensions do not apply
        # to these 2D slices and complete 3D spatial volumes.
        for label,widget in self.panel._projection_mode_control.get_widget_controls():
            label.hide(); widget.hide()
        layout=QVBoxLayout(self); layout.setContentsMargins(0,0,0,0); layout.addWidget(self.panel)
        for combo in self.panel.findChildren(QComboBox):
            view=QListView(combo); view.setTextElideMode(Qt.TextElideMode.ElideNone); combo.setView(view)
            view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            view.setMinimumWidth(view.sizeHintForColumn(0)+view.verticalScrollBar().sizeHint().width()+12)
            combo.setMinimumWidth(0); combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(8)
        for name in ('opacity','blending','contrast_limits','gamma','colormap','interpolation2d','interpolation3d','rendering','attenuation','iso_threshold','depiction','projection_mode'):
            getattr(self.layer.events,name).connect(self._changed)
        for name in ('position','normal','thickness'): getattr(self.layer.plane.events,name).connect(self._changed)
        self.auto_button=self.panel._contrast_limits_control.auto_scale_bar._auto_btn
        self.auto_button.toggled.connect(self._changed)
        plane=self.panel._depiction_control.plane_normal_buttons
        for button,normal in ((plane.x_button,[0,0,1]),(plane.y_button,[0,1,0]),(plane.z_button,[1,0,0])):
            button.clicked.disconnect(); button.clicked.connect(lambda checked=False,n=normal:setattr(self.layer.plane,'normal',n))
        plane.oblique_button.clicked.disconnect()
        plane.oblique_button.clicked.connect(self._oblique)
        self.view_direction=None
        for combo in (self.panel._render_control.render_combobox,self.panel._interpolation_control.interpolation_combobox):
            for row in range(combo.count()): combo.setItemData(row,combo.itemText(row))

    def _oblique(self):
        direction=self.view_direction() if self.view_direction else [1,1,1]
        self.layer.plane.normal=direction

    def record(self):
        layer=self.layer; normal=np.asarray(layer.plane.normal)/self.spacing
        normal/=np.linalg.norm(normal)
        result=dict(contrast=list(layer.contrast_limits),gamma=layer.gamma,colormap=layer.colormap.name,
            colormap_colors=layer.colormap.colors.tolist(),blending=str(layer.blending),opacity=layer.opacity,
            interpolation=str(layer.interpolation3d if self.ndisplay==3 else layer.interpolation2d),
            auto_contrast=self.auto_button.isChecked(),projection_mode=str(layer.projection_mode))
        if self.ndisplay==3:
            n=np.asarray(layer.plane.normal)
            result.update(rendering=str(layer.rendering),iso_threshold=layer.iso_threshold,attenuation=layer.attenuation,
                depiction=str(layer.depiction),plane_position_um_zyx=(np.asarray(layer.plane.position)*self.spacing).tolist(),
                plane_normal_um_zyx=normal.tolist(),plane_thickness_um=float(layer.plane.thickness/np.linalg.norm(n/self.spacing)))
        return result

    def _changed(self,*args):
        if not self._syncing: self.changed.emit(self.record())

    def set_data(self,data,spacing=None):
        if spacing is not None: self.spacing=np.asarray(spacing,float)
        if self.layer.data is data: return
        self._syncing=True
        try:
            self.layer.data=data
            if self.ndisplay==3:
                from napari.components import Dims
                self.layer._slice_dims(Dims(ndisplay=3))
        finally: self._syncing=False
        if self.auto_button.isChecked():
            self.layer.reset_contrast_limits(); self._changed()

    def restore(self,record,opacity=None):
        from napari.utils.colormaps import ensure_colormap,Colormap
        self._syncing=True
        try:
            layer=self.layer
            for key in ('gamma','blending','projection_mode'):
                if key in record: setattr(layer,key,record[key])
            layer.opacity=record.get('opacity',1) if opacity is None else opacity
            layer.contrast_limits=record.get('contrast',(0,1))
            name=record.get('colormap','gray')
            try: layer.colormap=ensure_colormap(name)
            except (ValueError,KeyError): layer.colormap=Colormap(record['colormap_colors'],name=name)
            setattr(layer,'interpolation3d' if self.ndisplay==3 else 'interpolation2d',record.get('interpolation','linear'))
            if self.ndisplay==3:
                for key in ('rendering','attenuation','iso_threshold','depiction'):
                    if key in record: setattr(layer,key,record[key])
                position=record.get('plane_position_um_zyx')
                layer.plane.position=((np.asarray(layer.data.shape)-1)/2 if position is None else np.asarray(position)/self.spacing)
                n=np.asarray(record.get('plane_normal_um_zyx',[1,0,0]))*self.spacing; n/=np.linalg.norm(n)
                layer.plane.normal=n
                thickness=record.get('plane_thickness_um')
                layer.plane.thickness=max(1.,thickness*np.linalg.norm(n/self.spacing)) if thickness is not None else 1.
            self.auto_button.blockSignals(True); self.auto_button.setChecked(record.get('auto_contrast',False)); self.auto_button.blockSignals(False)
            layer._keep_auto_contrast=record.get('auto_contrast',False)
        finally: self._syncing=False


class SettingControl:
    """Preserve the small programmatic settings interface used by workspaces."""
    def __init__(self,get,set,choices=None): self.get,self.set,self.choices=get,set,choices
    def value(self): return self.get()
    def __getattr__(self,name):
        widget=self.__dict__.get('widget')
        if widget is None: raise AttributeError(name)
        return getattr(widget,name)
    def currentIndex(self): return self.choices.index(self.get())
    def setValue(self,value): self.set(value)
    def setChecked(self,value): self.set(bool(value))
    def isChecked(self): return bool(self.get())
    def findData(self,value): return self.choices.index(value)
    def setCurrentIndex(self,index): self.set(self.choices[index])
    def currentData(self): return self.get()

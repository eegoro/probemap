"""napari's real image-layer controls for the standalone volume views."""
import copy
import numpy as np
from qtpy.QtWidgets import QWidget,QVBoxLayout,QTabWidget
from .desktop_napari import NapariImageControls,SettingControl


class RenderSettingsPanel(QWidget):
    def __init__(self,owner,include_opacity=True,compact=False,*,minimal=False,layer_count=None):
        super().__init__(owner); self.owner=owner; self.layer_count=layer_count or len(owner.layer_titles)
        self._updating=False; self.native_controls=[]; self.controls=[]; self.auto_buttons=[]
        layout=QVBoxLayout(self); layout.setContentsMargins(0,0,0,0)
        self.tabs=QTabWidget(); layout.addWidget(self.tabs)
        if minimal:
            for slider,amount in zip(owner.layers.sliders,owner.layers.amounts): slider.hide(); amount.hide()
        from .desktop_rendering import METHODS,BLENDING,COLORMAPS
        for i,title in enumerate(owner.layer_titles[:self.layer_count]):
            controls=NapariImageControls(3); self.native_controls.append(controls); self.tabs.addTab(controls,title)
            controls.changed.connect(lambda record,i=i:self._changed(i,record))
            controls.view_direction=lambda i=i:self._view_direction(i)
            aliases={}
            for key in ('opacity','gamma','iso_threshold','attenuation','sampling_step','low','high','rendering','blending','colormap','interpolation'):
                choices={'rendering':list(METHODS.values()),'blending':list(BLENDING.values()),'colormap':list(COLORMAPS),'interpolation':['nearest','linear','cubic']}.get(key)
                aliases[key]=SettingControl(lambda i=i,k=key:self._value(i,k),lambda value,i=i,k=key:self._set(i,k,value),choices)
            self.controls.append(aliases)
            aliases.update(rendering=controls.panel._render_control.render_combobox,
                blending=controls.panel._opacity_blending_controls.blend_combobox,
                colormap=controls.panel._colormap_control.colormap_combobox,
                interpolation=controls.panel._interpolation_control.interpolation_combobox)
            self.auto_buttons.append(controls.panel._contrast_limits_control.auto_scale_bar._once_btn)
        owner.renderSettingsChanged.connect(self.sync)
        owner.opacityChanged.connect(self.sync)
        self.sync()

    def _view_direction(self,index):
        if self.owner.view is None: return [1,1,1]
        direction=np.asarray(self.owner.view.camera.transform.map([0,0,1,0]))[:3][::-1]
        matrix=np.eye(3) if index==0 else self.owner.matrix[:3,:3]
        volumes=getattr(self.owner,'render_volumes',())
        spacing=np.array(volumes[index].spacing_um_zyx) if index<len(volumes) else np.ones(3)
        return np.linalg.solve(matrix,direction)/spacing

    def _value(self,index,key):
        if key=='opacity': return self.owner.layers.sliders[index].value()
        record=self.owner.render_settings[index]
        return record['contrast'][key=='high'] if key in ('low','high') else record[key]

    def _set(self,index,key,value):
        if key=='opacity': self.owner.layers.sliders[index].setValue(round(value)); return
        record=copy.deepcopy(self.owner.render_settings[index])
        if key in ('low','high'): record['contrast'][key=='high']=value
        else: record[key]=value
        self.owner.set_render_settings(index,record)

    def _changed(self,index,record):
        if self._updating: return
        target=copy.deepcopy(self.owner.render_settings[index]); opacity=record.pop('opacity')
        target.update(record)
        self.owner.layers.sliders[index].setValue(round(opacity*100))
        self.owner.set_render_settings(index,target)

    def sync(self,*args):
        if self._updating: return
        self._updating=True
        try:
            volumes=getattr(self.owner,'render_volumes',())
            for i,controls in enumerate(self.native_controls):
                if i<len(volumes): controls.set_data(volumes[i].data,volumes[i].spacing_um_zyx)
                controls.restore(self.owner.render_settings[i],self.owner.layers.sliders[i].value()/100)
        finally: self._updating=False

    def _auto(self,index,auto):
        if auto: self.native_controls[index].layer.reset_contrast_limits()
        else: self._set(index,'low',0.); self._set(index,'high',1.)

    def reset_settings(self):
        from .desktop_rendering import default_render_settings
        records=self.owner.render_settings_record(); records[:self.layer_count]=default_render_settings(self.layer_count)
        self.owner.restore_render_settings(records)

"""Native intensity controls, shared by the orthogonal and oblique slices."""
import copy

import numpy as np
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (QCheckBox,QDoubleSpinBox,QFormLayout,QGroupBox,
                           QHBoxLayout,QPushButton,QSizePolicy,QVBoxLayout,QWidget)

from .desktop_widgets import PopupComboBox


class SliceDisplayControls(QGroupBox):
    changed=Signal()
    autoRequested=Signal(int)

    def __init__(self,channels=('Track scan','Brain scan','Atlas'),colors=None,minimal=False):
        super().__init__('Appearance' if minimal else 'Slice appearance')
        from .desktop_napari import NapariImageControls,SettingControl
        from .desktop_rendering import COLORMAPS
        self.colors=colors; self.records=[]; self.defaults=[]; self._updating=False; self._data={}
        layout=QVBoxLayout(self); layout.setContentsMargins(6,8,6,6)
        self.channel=PopupComboBox(); self.channel.addItems(channels); self.channel.currentIndexChanged.connect(self.sync)
        channel_row=QHBoxLayout(); channel_row.addWidget(self.channel,1)
        from .desktop_widgets import view_button
        channel_row.addWidget(view_button('reset','Reset image appearance',self.reset_channel)); layout.addLayout(channel_row)
        self.native=NapariImageControls(2); layout.addWidget(self.native)
        self.native.changed.connect(self._native_changed)
        for key in ('gamma','brightness','invert','opacity','low','high','interpolation','colormap'):
            choices={'interpolation':['linear','nearest','cubic'],'colormap':list(COLORMAPS)}.get(key)
            setattr(self,key,SettingControl(lambda k=key:self._value(k),lambda value,k=key:self._edit(k,value),choices))
        self.form=self.native.panel.layout()
        self.interpolation.widget=self.native.panel._interpolation_control.interpolation_combobox
        self.colormap.widget=self.native.panel._colormap_control.colormap_combobox

    def _value(self,key):
        record=self.records[self.channel.currentIndex()]
        return record['contrast'][key=='high'] if key in ('low','high') else record.get(key)

    def _edit(self,key,value):
        record=self.records[self.channel.currentIndex()]
        if key in ('low','high'): record['contrast'][key=='high']=value
        else: record[key]=value
        self.sync(); self.changed.emit()

    def initialize(self,limits,saved=None):
        self.defaults=[dict(contrast=list(pair),gamma=1.,brightness=0.,invert=False,interpolation='linear',
            colormap='gray',opacity=1.,blending='translucent',auto_contrast=False) for pair in limits]
        if self.colors is not None:
            for record,color in zip(self.defaults,self.colors): record.update(colormap=color,blending='additive')
        self.records=copy.deepcopy(self.defaults)
        if isinstance(saved,list):
            for target,record in zip(self.records,saved): target.update(copy.deepcopy(record))
        self.sync()

    def sync(self,*args):
        if not self.records: return
        self._updating=True
        try:
            index=self.channel.currentIndex()
            if index in self._data: self.native.set_data(self._data[index])
            self.native.restore(self.records[index])
        finally: self._updating=False

    def observe(self,index,pixels):
        self._data[index]=pixels
        if index!=self.channel.currentIndex() or not self.records: return
        self._updating=True
        try:
            self.native.set_data(pixels)
            if self.records[index].get('auto_contrast',False):
                self.records[index]['contrast']=list(self.native.layer.contrast_limits)
        finally: self._updating=False

    def _native_changed(self,record):
        if self._updating or not self.records: return
        self.records[self.channel.currentIndex()].update(record); self.changed.emit()

    def reset_channel(self):
        if not self.records: return
        index=self.channel.currentIndex(); self.records[index]=copy.deepcopy(self.defaults[index])
        self.sync(); self.changed.emit()


def slice_grayscale(pixels,settings):
    low,high=settings['contrast']
    if settings.get('auto_contrast',False):
        from napari.layers.utils.layer_utils import calc_data_range
        low,high=calc_data_range(np.asarray(pixels))
    data=np.nan_to_num(np.clip((pixels-low)/max(high-low,1e-12),0,1),nan=0)
    data=np.clip(np.power(data,settings['gamma'])+settings.get('brightness',0),0,1)
    return 1-data if settings.get('invert',False) else data


def frozen_slice_settings(pixels,settings):
    record=copy.deepcopy(settings)
    if record.get('auto_contrast',False):
        from napari.layers.utils.layer_utils import calc_data_range
        record['contrast']=list(calc_data_range(np.asarray(pixels)))
    record['auto_contrast']=False
    return record


def slice_rgb(grayscale,colormap,colors=None):
    from napari.utils.colormaps import ensure_colormap,Colormap
    try: cmap=ensure_colormap(colormap)
    except (ValueError,KeyError): cmap=Colormap(colors,name=colormap)
    return cmap.map(grayscale.ravel())[:,:3].reshape((*grayscale.shape,3))


def slice_color(pixels,settings):
    rgb=slice_rgb(slice_grayscale(pixels,settings),settings.get('colormap','gray'),settings.get('colormap_colors'))
    return blend_rgb(np.zeros_like(rgb),rgb,settings.get('opacity',1),settings.get('blending','translucent'))


def blend_rgb(background,source,alpha,mode):
    if alpha<=0: return background
    if mode=='opaque': return source
    if mode=='minimum': return np.minimum(background,source)
    if mode=='multiplicative': return background*source
    if mode=='additive': return np.clip(background+alpha*source,0,1)
    return background*(1-alpha)+source*alpha

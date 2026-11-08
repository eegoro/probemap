"""Shared 3D layer controls and bounded atlas-region surface previews."""
from __future__ import annotations

import numpy as np
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QCheckBox,QGridLayout,QLabel,QSlider,QWidget


class VolumeLayers(QWidget):
    visibilityChanged=Signal()
    opacityChanged=Signal(int,int)
    labelsRequested=Signal(bool)

    def __init__(self,titles):
        super().__init__()
        self.grid=QGridLayout(self)
        self.grid.setContentsMargins(0,0,0,0)
        self.checks=[]
        self.sliders=[]
        self.amounts=[]
        self.titles=[]
        self.labels_check=QCheckBox('Labels')
        self.labels_check.setChecked(False)
        self.labels_check.setToolTip('Atlas region surfaces in the same physical coordinates as the scans')
        self.labels_opacity=QSlider(Qt.Orientation.Horizontal)
        self.labels_opacity.setRange(0,100)
        self.labels_opacity.setValue(25)
        self.labels_amount=QLabel('25%')
        self.labels_amount.setFixedWidth(36)
        self.labels_opacity.valueChanged.connect(lambda value:self.labels_amount.setText(f'{value}%'))
        self.labels_check.toggled.connect(self.labelsRequested)
        self.labels_check.toggled.connect(self.visibilityChanged)
        self.labels_opacity.valueChanged.connect(self.visibilityChanged)
        self.set_titles(titles)

    def set_titles(self,titles):
        while len(self.checks)>len(titles):
            self.checks.pop().deleteLater()
            self.sliders.pop().deleteLater()
            self.amounts.pop().deleteLater()
        for index,title in enumerate(titles):
            if index==len(self.checks):
                check=QCheckBox(title)
                check.setChecked(index<2)
                check.toggled.connect(self.visibilityChanged)
                slider=QSlider(Qt.Orientation.Horizontal)
                slider.setRange(0,100)
                slider.setValue((65,90,25)[index])
                slider.valueChanged.connect(lambda value,i=index:self.opacityChanged.emit(i,value))
                self.checks.append(check)
                self.sliders.append(slider)
                amount=QLabel(f'{slider.value()}%')
                amount.setFixedWidth(36)
                slider.valueChanged.connect(lambda value,amount=amount:amount.setText(f'{value}%'))
                self.amounts.append(amount)
                self.grid.addWidget(amount,index,2)
            self.checks[index].setText(title)
            self.grid.addWidget(self.checks[index],index,0)
            self.grid.addWidget(self.sliders[index],index,1)
        self.titles=list(titles)
        self.grid.addWidget(self.labels_check,len(titles),0)
        self.grid.addWidget(self.labels_opacity,len(titles),1)
        self.grid.addWidget(self.labels_amount,len(titles),2)
        self.grid.setColumnStretch(1,1)

    def record(self):
        return {'volumes':{title:{'visible':check.isChecked(),'opacity':slider.value()}
                for title,check,slider in zip(self.titles,self.checks,self.sliders)},
                'labels':{'visible':self.labels_check.isChecked(),'opacity':self.labels_opacity.value()}}

    def restore(self,record):
        for title,check,slider in zip(self.titles,self.checks,self.sliders):
            item=record.get('volumes',{}).get(title,{})
            for widget,value in ((check,item.get('visible',check.isChecked())),(slider,item.get('opacity',slider.value()))):
                widget.blockSignals(True)
                widget.setChecked(value) if widget is check else widget.setValue(value)
                widget.blockSignals(False)
            self.amounts[self.titles.index(title)].setText(f'{slider.value()}%')
        for widget,key,default in ((self.labels_check,'visible',False),(self.labels_opacity,'opacity',25)):
            widget.blockSignals(True)
            value=record.get('labels',{}).get(key,default)
            widget.setChecked(value) if widget is self.labels_check else widget.setValue(value)
            widget.blockSignals(False)
        self.labels_amount.setText(f'{self.labels_opacity.value()}%')

    def show_registration_layers(self,only):
        for index,widgets in enumerate(zip(self.checks,self.sliders,self.amounts)):
            for widget in widgets: widget.setVisible(not only or index<2)
        for widget in (self.labels_check,self.labels_opacity,self.labels_amount):
            widget.setVisible(not only)


def label_surfaces(labels,max_dimension=112):
    """Categorical surfaces for display only; native labels remain quantitative."""
    from skimage.measure import marching_cubes
    factors=np.maximum(1,np.ceil(np.asarray(labels.data.shape)/max_dimension).astype(int))
    data=np.asarray(labels.data[tuple(slice(None,None,int(n)) for n in factors)])
    spacing=np.asarray(labels.spacing_um_zyx)*factors
    meshes=[]
    for region in labels.regions:
        mask=data==region['id']
        if not mask.any(): continue
        padded=np.pad(mask,1)
        vertices,faces,_,_=marching_cubes(padded.astype(np.float32),.5,spacing=tuple(spacing),allow_degenerate=False)
        vertices-=spacing
        meshes.append({'id':region['id'],'vertices_zyx_um':vertices.astype(np.float32),'faces':faces,
                       'color_rgb':region['color_rgb']})
    return meshes

"""The three placement tools: physical position, volume rendering and slices."""
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (QCheckBox, QDoubleSpinBox, QFormLayout, QGridLayout,
    QGroupBox, QHBoxLayout, QLabel, QScrollArea, QSlider, QTabWidget, QVBoxLayout, QWidget)

from .desktop_project import icon_button
from .desktop_rendering import RenderSettingsPanel
from .desktop_slice_display import SliceDisplayControls
from .desktop_widgets import PopupComboBox,flat_tabs


class DragSlider(QSlider):
    def wheelEvent(self,event):
        event.ignore()


class EditedSpin(QDoubleSpinBox):
    def wheelEvent(self,event):
        event.ignore()

    def textFromValue(self,value):
        return f'{value:.{self.decimals()}f}'.rstrip('0').rstrip('.')


class SelectedCombo(PopupComboBox):
    def wheelEvent(self,event):
        event.ignore()


class PlacementTools(QTabWidget):
    def __init__(self,owner):
        super().__init__()
        self.owner=owner
        flat_tabs(self)
        self.setObjectName('placementTools')
        self._attached=False
        for title,build in (('Position',self._position),('3D',self._volume),('Slices',self._slices)):
            page=QWidget()
            layout=QVBoxLayout(page)
            layout.setContentsMargins(8,10,8,10)
            build(layout)
            layout.addStretch()
            scroll=QScrollArea()
            scroll.setFrameShape(QScrollArea.Shape.NoFrame)
            scroll.setWidgetResizable(True)
            scroll.setWidget(page)
            self.addTab(scroll,title)

    def _position(self,layout):
        owner=self.owner
        toolbar=QHBoxLayout()
        toolbar.addStretch()
        self.reset_button=icon_button('reset','Reset rotation and scale',owner.reset_pose)
        toolbar.addWidget(self.reset_button)
        layout.addLayout(toolbar)
        owner.pose_controls={}
        for name,title,low,high,step,decimals in (
            ('translation','Position · µm',-1e6,1e6,25,5),
            ('rotation','Rotation · °',-180,180,1,2),
            ('scale','Scale',.02,10,.01,4)):
            group=QGroupBox(title)
            grid=QGridLayout(group)
            for axis,axis_name in enumerate('XYZ'):
                grid.addWidget(QLabel(axis_name),axis,0)
                slider=DragSlider(Qt.Orientation.Horizontal)
                slider.setAccessibleName(f'{title} {axis_name}')
                factor=1 if name=='translation' else (10 if name=='rotation' else 1000)
                slider.setRange(round(low*factor),round(high*factor))
                spin=EditedSpin()
                spin.setRange(low,high)
                spin.setDecimals(decimals)
                spin.setSingleStep(step)
                spin.setKeyboardTracking(False)
                spin.setValue(1 if name=='scale' else 0)
                spin.setAccessibleName(f'{title} {axis_name}')
                slider.setValue(round(spin.value()*factor))
                slider.valueChanged.connect(lambda value,spin=spin,factor=factor:spin.setValue(value/factor))
                spin.valueChanged.connect(lambda value,name=name,axis=axis:owner._pose_changed(name,axis,value))
                owner.pose_controls[name,axis]=(slider,spin,factor)
                grid.addWidget(slider,axis,1)
                grid.addWidget(spin,axis,2)
            layout.addWidget(group)
        axes=QGroupBox('Axes')
        form=QFormLayout(axes)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        owner.axis_order=SelectedCombo()
        owner.axis_order.addItems(['XYZ','XZY','YXZ','YZX','ZXY','ZYX'])
        owner.axis_order.setToolTip('Scan axes assigned to atlas X, Y, Z')
        owner.axis_order.currentIndexChanged.connect(lambda _:owner._pose_changed())
        form.addRow('Order',owner.axis_order)
        flips=QWidget()
        row=QHBoxLayout(flips)
        row.setContentsMargins(0,0,0,0)
        owner.flips=[]
        for name in 'XYZ':
            check=QCheckBox('Flip '+name)
            check.toggled.connect(lambda _:owner._pose_changed())
            row.addWidget(check)
            owner.flips.append(check)
        form.addRow(flips)
        layout.addWidget(axes)

    def _volume(self,layout):
        view=self.owner.volume_view
        self.volume_layout=layout
        self.rendering=RenderSettingsPanel(view,include_opacity=False,compact=True,minimal=True,layer_count=2)
        layout.addWidget(self.rendering)
        view.renderSettingsChanged.connect(self.rendering.sync)

    def _slices(self,layout):
        owner=self.owner
        modes=QGroupBox('Comparison')
        form=QFormLayout(modes)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        for title,pane in zip(('XY','XZ','YZ'),owner.overlay_panes):
            form.addRow(title,pane.mode_selector)
        layout.addWidget(modes)
        owner._appearance(layout,'placement')
        self.slice_display=SliceDisplayControls(channels=('Atlas','Brain scan'),colors=('green','magenta'),minimal=True)
        self.slice_display.changed.connect(owner.request_render)
        self.slice_display.autoRequested.connect(self.auto_contrast)
        self.slice_display.initialize(((0,1),(0,1)))
        layout.addWidget(self.slice_display)

    def auto_contrast(self,index):
        import numpy as np
        volume=(self.owner.reference,self.owner.moving)[index]
        if volume is None: return
        data=volume.data.ravel()
        sample=data[::max(1,len(data)//500000)]
        sample=sample[np.isfinite(sample)&(sample>0)]
        limits=np.percentile(sample,(1,99.8)).tolist() if len(sample) else [0,1]
        if limits[1]-limits[0]<1e-6: limits=[0,1]
        self.slice_display.records[index]['contrast']=limits
        self.slice_display.sync()
        self.owner.request_render()

    def attach_volume_controls(self,enabled):
        """Move the existing controls; do not duplicate live layer state."""
        view=self.owner.volume_view
        if enabled!=self._attached:
            for widget in (view.layers,view.plane_controls):
                if enabled:
                    view.layout.removeWidget(widget)
                    self.volume_layout.insertWidget(0 if widget is view.layers else 1,widget)
                else:
                    self.volume_layout.removeWidget(widget)
                    view.layout.insertWidget(1 if widget is view.layers else view.layout.indexOf(view.zoom_controls),widget)
                widget.show()
            self._attached=enabled
        view.set_placement_mode(enabled)
        if enabled: self.rendering.sync()

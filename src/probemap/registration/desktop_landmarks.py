"""Landmark editing tools and shared, independently adjustable slice views."""
import numpy as np
from qtpy.QtCore import QEvent, QPointF, Qt, Signal
from qtpy.QtGui import QIntValidator
from qtpy.QtWidgets import (QButtonGroup, QCheckBox, QFormLayout, QGridLayout,
    QHBoxLayout, QLabel, QLineEdit, QScrollArea, QSpinBox, QTabWidget, QVBoxLayout, QWidget)

from .desktop_placement import DragSlider, SelectedCombo
from .desktop_project import icon_button
from .desktop_slice_display import SliceDisplayControls
from .desktop_views import SliceCanvas, SlicePane
from .desktop_widgets import flat_tabs, view_button


class LandmarkCanvas(SliceCanvas):
    pairSelected=Signal(int)
    deleteRequested=Signal()
    viewReset=Signal()

    def __init__(self,axis):
        super().__init__(axis)
        self.navigation_mode=True
        self.show_axes=False
        self.background_color='#000000'
        self.set_tool('add')
        self.setToolTip('Pinch or wheel: zoom · Shift/Space + drag: pan · Right-drag: zoom')

    def set_tool(self,tool):
        self.tool=tool
        self.mode='landmarks' if tool=='add' else 'view'
        self.left_drag_pans=tool=='pan'
        self.pan_mode=tool=='pan'
        self.setCursor({'pan':Qt.CursorShape.OpenHandCursor,'add':Qt.CursorShape.CrossCursor,
                        'select':Qt.CursorShape.ArrowCursor}[tool])

    def mouseReleaseEvent(self,event):
        if (self.tool=='select' and self._button==Qt.MouseButton.LeftButton and not self._panning
                and self._start_pixel is not None
                and (event.position()-self._start_pixel).manhattanLength()<4):
            hits=[((self._pixel(position)-event.position()).manhattanLength(),number-1)
                  for number,position,_ in self.visible_points()]
            distance,row=min(hits,default=(float('inf'),-1))
            self.pairSelected.emit(row if distance<=14 else -1)
        super().mouseReleaseEvent(event)
        self.set_tool(self.tool)

    def keyPressEvent(self,event):
        if event.key() in (Qt.Key.Key_Backspace,Qt.Key.Key_Delete):
            if not event.isAutoRepeat(): self.deleteRequested.emit()
            event.accept()
        else: super().keyPressEvent(event)

    def keyReleaseEvent(self,event):
        super().keyReleaseEvent(event)
        if event.key()==Qt.Key.Key_Space: self.set_tool(self.tool)

    def mouseDoubleClickEvent(self,event):
        if self.tool=='pan': self.fit_image()
        else: self.mousePressEvent(event)

    def fit_image(self):
        super().fit_image()
        self.viewReset.emit()

    def center_on(self,point):
        if self.image is None: return
        self.pan+=QPointF(self.width()/2,self.height()/2)-self._pixel(point)
        self.update()
        self.viewChanged.emit()


class LandmarkPane(SlicePane):
    def __init__(self,axis,title,heading):
        super().__init__(axis,title,canvas_type=LandmarkCanvas,compact=True)
        self.setObjectName('registrationPane')
        layout=self.layout()
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(3)
        layout.removeWidget(self.label)
        self.label.setText(heading)
        self.label.setObjectName('muted')
        self.label.setVisible(bool(heading))
        self.slider.hide()
        self.adjust_range=15
        self._adjust_anchor=0
        self._adjusting=False
        self.adjust_slider=DragSlider(Qt.Orientation.Horizontal)
        self.adjust_slider.setObjectName('landmarkFineAdjust')
        self.adjust_slider.setFixedHeight(20)
        self.adjust_slider.setStyleSheet('''
            QSlider#landmarkFineAdjust {background:transparent;}
            QSlider#landmarkFineAdjust::groove:horizontal {height:2px;background:#3d5063;border-radius:1px;}
            QSlider#landmarkFineAdjust::handle:horizontal {background:#94b2c9;width:8px;margin:-3px 0;border-radius:4px;}
            QSlider#landmarkFineAdjust::handle:horizontal:hover {background:#bed7e6;}
            QSlider#landmarkFineAdjust::handle:horizontal:focus {background:#bed7e6;}
        ''')
        self.adjust_slider.setAccessibleName(f'Adjust {title} {("XY","XZ","YZ")[axis]} slice')
        self.adjust_slider.setTickPosition(DragSlider.TickPosition.NoTicks)
        self.header=QWidget()
        self.header.setObjectName('registrationHeader')
        self.header.setFixedHeight(26)
        row=QHBoxLayout(self.header)
        row.setContentsMargins(2,0,2,0)
        row.setSpacing(4)
        row.addWidget(self.label)
        row.addStretch()
        row.addWidget(self.adjust_slider)
        self.fit_button=view_button('fit','Reset atlas and scan views',self.canvas.fit_image)
        self.rotate_button=view_button('rotate','Rotate atlas and scan views 90°',self.canvas.rotate_view)
        row.addWidget(self.fit_button); row.addWidget(self.rotate_button)
        from .desktop_figures import figure_button
        row.addWidget(figure_button(self.canvas))
        layout.insertWidget(0,self.header)
        self.adjust_slider.valueChanged.connect(self._adjust_depth)
        self.slider.valueChanged.connect(self._depth_changed)
        self.slider.installEventFilter(self)
        self.reset_adjust()

    def resizeEvent(self,event):
        super().resizeEvent(event)
        self.adjust_slider.setFixedWidth(min(120,max(32,self.width()//3-8)))

    def eventFilter(self,watched,event):
        if watched is self.slider and event.type()==QEvent.Type.Wheel:
            event.ignore()
            return True
        return super().eventFilter(watched,event)

    def configure(self,*args,**kwargs):
        super().configure(*args,**kwargs)
        self.reset_adjust()

    def reset_adjust(self):
        self._adjust_anchor=self.slider.value()
        self.adjust_slider.blockSignals(True)
        self.adjust_slider.setRange(max(-self.adjust_range,self.slider.minimum()-self._adjust_anchor),
                                    min(self.adjust_range,self.slider.maximum()-self._adjust_anchor))
        self.adjust_slider.setTickInterval(self.adjust_range)
        self.adjust_slider.setValue(0)
        self.adjust_slider.blockSignals(False)
        self._adjust_tooltip()

    def set_adjust_range(self,value):
        self.adjust_range=value
        self.reset_adjust()

    def _adjust_tooltip(self):
        self.adjust_slider.setToolTip(f'Adjust {self.title} slice · {self.adjust_slider.value():+d} · range ±{self.adjust_range} slices')

    def _depth_changed(self):
        if not self._adjusting: self.reset_adjust()

    def _adjust_depth(self,value):
        self._adjusting=True
        try: self.slider.setValue(self._adjust_anchor+value)
        finally: self._adjusting=False
        self._adjust_tooltip()

    def display(self,pixels):
        self.canvas.set_image(pixels,self.spacing,self.slider.value(),self.origin)

    def set_world_depth(self,position):
        # Use the same half-open depth bins as SliceCanvas.visible_points().
        self.slider.setValue(int(np.floor((position-self.origin[self.axis])/self.spacing[self.axis]+.5+1e-10)))
        self.reset_adjust()


class LandmarkViews(QWidget):
    def __init__(self,owner):
        super().__init__()
        self.owner=owner
        self._syncing_views=False
        self._view_states={}
        self.setObjectName('registrationViews')
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground,True)
        self.setStyleSheet('''
            QWidget#registrationViews, QWidget#registrationPane, QWidget#registrationHeader {background:#242b36;}
            QLabel, QSlider {background:transparent;}
        ''')
        layout=QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        self.toolbar=QWidget()
        self.toolbar.setAccessibleName('Registration tools')
        self.toolbar.setStyleSheet('QToolButton:checked {background:#285b7a;border:1px solid #70c2e7;}')
        toolbar=QHBoxLayout(self.toolbar)
        toolbar.setContentsMargins(0,0,0,0)
        toolbar.setSpacing(4)
        self.tool_buttons={}
        group=QButtonGroup(self)
        group.setExclusive(True)
        for tool,title in (('pan','Pan'),('add','Add landmarks'),('select','Select a pair · Delete/Backspace removes both points')):
            b=icon_button(tool,title,lambda tool=tool:owner.set_landmark_tool(tool))
            b.setCheckable(True)
            group.addButton(b)
            self.tool_buttons[tool]=b
            toolbar.addWidget(b)
        grid=QGridLayout()
        grid.setContentsMargins(0,0,0,0)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)
        layout.addLayout(grid,1)
        owner.landmark_panes={'reference':[],'moving':[]}
        for row,(name,title) in enumerate((('reference','Atlas'),('moving','Brain scan'))):
            for axis in range(3):
                heading=title if axis==0 else ''
                if row==0:
                    plane=('XY','XZ','YZ')[axis]
                    heading=f'{heading} · {plane}' if heading else plane
                pane=LandmarkPane(axis,title,heading)
                pane.canvas.pointClicked.connect(lambda point,name=name:owner.add_point(name,point))
                pane.canvas.pairSelected.connect(owner.select_landmark)
                pane.canvas.deleteRequested.connect(owner.delete_pair)
                pane.changed.connect(lambda name=name,axis=axis:owner._landmark_slice_changed(name,axis))
                owner.landmark_panes[name].append(pane)
                grid.addWidget(pane,row,axis)
                grid.setColumnStretch(axis,1)
            grid.setRowStretch(row,1)
        for name,panes in owner.landmark_panes.items():
            for axis,pane in enumerate(panes):
                self._view_states[name,axis]=pane.canvas.view_record()
                pane.canvas.viewChanged.connect(lambda name=name,axis=axis:self._sync_view(name,axis))
                pane.canvas.viewReset.connect(lambda name=name,axis=axis:self._reset_views(name,axis))
        self.shared_sliders=[]
        for axis in range(3):
            slider=DragSlider(Qt.Orientation.Horizontal)
            slider.setAccessibleName(f'Shared {("XY","XZ","YZ")[axis]} slice')
            slider.setToolTip('Move atlas and scan slices together, keeping individual adjustments')
            slider.valueChanged.connect(lambda value,axis=axis:self.move_shared(axis,value))
            self.shared_sliders.append(slider)
            grid.addWidget(slider,2,axis)

    def sync_slices(self):
        for axis,slider in enumerate(self.shared_sliders):
            ref=self.owner.landmark_panes['reference'][axis].slider
            mov=self.owner.landmark_panes['moving'][axis].slider
            offset=mov.value()-ref.value()
            slider.blockSignals(True)
            slider.setRange(max(0,-offset),min(ref.maximum(),mov.maximum()-offset))
            slider.setValue(ref.value())
            slider.blockSignals(False)

    def move_shared(self,axis,value):
        ref=self.owner.landmark_panes['reference'][axis].slider
        delta=value-ref.value()
        for name in ('reference','moving'):
            slider=self.owner.landmark_panes[name][axis].slider
            slider.blockSignals(True)
            slider.setValue(slider.value()+delta)
            slider.blockSignals(False)
            self.owner.landmark_panes[name][axis].reset_adjust()
        self.sync_slices()
        self.owner.draw_landmarks()

    def set_adjust_range(self,value):
        for panes in self.owner.landmark_panes.values():
            for pane in panes: pane.set_adjust_range(value)

    def _sync_view(self,name,axis):
        source=self.owner.landmark_panes[name][axis].canvas
        previous=self._view_states[name,axis]
        current=source.view_record()
        self._view_states[name,axis]=current
        if self._syncing_views: return
        other=self.owner.landmark_panes['moving' if name=='reference' else 'reference'][axis].canvas
        # Propagate the gesture, preserving the independent centers found for
        # a landmark pair. Absolute camera copying would undo point search.
        factor=(current['zoom']/previous['zoom'] *
                source.fit_scale(current['rotation'])/source.fit_scale(previous.get('rotation',0)))
        turns=((current['rotation']-previous.get('rotation',0))%360)//90
        rotation=np.linalg.matrix_power(np.array([[0,-1],[1,0]]),turns)
        shift=np.asarray(current['pan'])-factor*(rotation@np.asarray(previous['pan']))
        self._syncing_views=True
        try:
            other.restore_view({'zoom':current['zoom'],
                                'rotation':current['rotation'],
                                'pan':(factor*(rotation@np.array([other.pan.x(),other.pan.y()]))+shift).tolist()})
        finally: self._syncing_views=False

    def _reset_views(self,name,axis):
        self._syncing_views=True
        try:
            other=self.owner.landmark_panes['moving' if name=='reference' else 'reference'][axis].canvas
            other.restore_view({'zoom':1.,'pan':[0,0],'rotation':0})
        finally: self._syncing_views=False

    def center_pair(self,pair):
        self._syncing_views=True
        try:
            for name,panes in self.owner.landmark_panes.items():
                if pair.get(name) is not None:
                    for pane in panes: pane.canvas.center_on(pair[name])
        finally: self._syncing_views=False

    def restore_views(self,records):
        self._syncing_views=True
        try:
            for name,panes in self.owner.landmark_panes.items():
                saved=records.get(name,[])
                for axis,pane in enumerate(panes):
                    pane.canvas.restore_view(saved[axis] if axis<len(saved) else {})
            # Older projects allowed independent zooms; keep each image's
            # center while normalizing the two magnifications on reopening.
            for axis in range(3):
                ref=self.owner.landmark_panes['reference'][axis].canvas
                mov=self.owner.landmark_panes['moving'][axis].canvas
                factor=ref.zoom/mov.zoom
                mov.restore_view({'zoom':ref.zoom,'pan':[mov.pan.x()*factor,mov.pan.y()*factor],
                                  'rotation':ref.view_rotation})
        finally: self._syncing_views=False

    def set_linked(self,enabled):
        for slider in self.shared_sliders: slider.setVisible(enabled)
        self.sync_slices()

    def set_tool(self,tool):
        for name,b in self.tool_buttons.items(): b.setChecked(name==tool)
        for panes in self.owner.landmark_panes.values():
            for pane in panes: pane.canvas.set_tool(tool)


class LandmarkTools(QTabWidget):
    def __init__(self,owner):
        super().__init__()
        self.owner=owner
        flat_tabs(self)
        transform=QWidget()
        layout=QVBoxLayout(transform)
        form=QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        owner.fit_model=SelectedCombo()
        owner.fit_model.addItems(['Affine','Similarity','Rigid'])
        owner.fit_model.currentIndexChanged.connect(owner.update_landmarks)
        form.addRow('Model',owner.fit_model)
        layout.addLayout(form)
        self.count=QLabel('0 pairs')
        row=QHBoxLayout(); row.addWidget(self.count,1)
        self.clear_button=view_button('clear','Clear all pairs · Ctrl/⌘+Z to undo',owner.clear_pairs)
        row.addWidget(self.clear_button); layout.addLayout(row)
        self.fit_status=QLabel('Not applied')
        self.fit_status.setObjectName('muted')
        self.fit_status.setWordWrap(True)
        layout.addWidget(self.fit_status)
        search=QHBoxLayout()
        self.search=QLineEdit()
        self.search.setPlaceholderText('Point number')
        self.search.setAccessibleName('Find landmark pair by number')
        self.search.setValidator(QIntValidator(1,2147483647,self.search))
        self.search.returnPressed.connect(self.find)
        search.addWidget(self.search,1)
        search.addWidget(icon_button('search','Find landmark pair',self.find))
        layout.addLayout(search)
        owner.link_slices=QCheckBox('Shared slice controls')
        owner.link_slices.setChecked(True)
        owner.link_slices.toggled.connect(owner.landmark_views.set_linked)
        layout.addWidget(owner.link_slices)
        layout.addStretch()
        visualization=QWidget()
        layout=QVBoxLayout(visualization)
        navigation=QFormLayout()
        navigation.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        navigation.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.adjust_range=QSpinBox()
        self.adjust_range.setRange(1,10000)
        self.adjust_range.setValue(15)
        self.adjust_range.setPrefix('± ')
        self.adjust_range.setSuffix(' slices')
        self.adjust_range.setKeyboardTracking(False)
        self.adjust_range.setAccessibleName('Individual slice adjustment range')
        self.adjust_range.valueChanged.connect(owner.landmark_views.set_adjust_range)
        self.adjust_range.valueChanged.connect(owner.schedule_save)
        self.adjust_range.installEventFilter(self)
        navigation.addRow('Adjust range',self.adjust_range)
        layout.addLayout(navigation)
        self.display=SliceDisplayControls(channels=('Atlas','Brain scan'),minimal=True)
        self.display.initialize(((0,1),(0,1)))
        self.display.changed.connect(owner.draw_landmarks)
        self.display.autoRequested.connect(self.auto_contrast)
        layout.addWidget(self.display)
        layout.addStretch()
        for title,panel in (('Transformation',transform),('Visualization',visualization)):
            scroll=QScrollArea(); scroll.setWidgetResizable(True)
            scroll.setFrameShape(QScrollArea.Shape.NoFrame)
            scroll.setWidget(panel)
            self.addTab(scroll,title)

    def eventFilter(self,watched,event):
        if watched is self.adjust_range and event.type()==QEvent.Type.Wheel:
            event.ignore()
            return True
        return super().eventFilter(watched,event)

    def find(self):
        if self.search.text(): self.owner.find_landmark(int(self.search.text())-1)

    def auto_contrast(self,index):
        volume=(self.owner.landmark_reference,self.owner.placed)[index]
        if volume is None: return
        data=volume.ravel()[::max(1,volume.size//500000)]
        data=data[np.isfinite(data)&(data>0)]
        low,high=np.percentile(data,(1,99.8)) if len(data) else (0,1)
        if high<=low: low,high=0,1
        self.display.records[index]['contrast']=[float(low),float(high)]
        self.display.sync()
        self.owner.draw_landmarks()

    def update_state(self):
        project=self.owner.project
        if project is None: return
        ref=sum(p.get('reference') is not None for p in project.pairs)
        mov=sum(p.get('moving') is not None for p in project.pairs)
        complete=sum(p.get('reference') is not None and p.get('moving') is not None for p in project.pairs)
        self.count.setText(f'{complete} pairs'+(f' · Atlas {ref} / Scan {mov}' if complete!=ref or complete!=mov else ''))
        self.clear_button.setEnabled(bool(ref or mov))
        current=project.fit_is_current and project.fit_result['model']==self.owner.fit_model.currentText().lower()
        self.fit_status.setText(f'Applied · RMS {project.fit_result["rms_um"]:.2f} µm' if current else 'Changes not applied' if project.fit_result else 'Not applied')
        if hasattr(self.owner,'landmark_apply'):
            minimum=4 if self.owner.fit_model.currentText()=='Affine' else 3
            self.owner.landmark_apply.setEnabled(complete>=minimum and complete==ref==mov)
            self.owner.landmark_apply.setToolTip('Match all atlas and scan points before applying' if complete!=ref or complete!=mov
                else f'At least {minimum} matching pairs are required' if complete<minimum else 'Apply landmark transformation')
            if self.owner.step==2: self.owner.next_button.setEnabled(current)

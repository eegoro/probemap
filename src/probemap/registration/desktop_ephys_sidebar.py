"""Task-based sidebar sections and a shared time navigator."""
from qtpy.QtCore import Qt,QTimer
from qtpy.QtWidgets import QWidget,QVBoxLayout,QHBoxLayout,QFormLayout,QTabWidget,QLabel,QSlider,QDoubleSpinBox,QPushButton
from .desktop_ephys_settings import Fold,number


def organize_sidebar(page,side,probe_box):
    items=[]
    while side.count(): items.append(side.takeAt(0))
    tabs=page.settings_tabs=QTabWidget(); side.addWidget(tabs)
    layouts={}
    for name in ('Recordings','Plots','Anatomy','Marks','Units'):
        host=QWidget(); layout=QVBoxLayout(host); layout.setContentsMargins(0,10,0,0); layout.setSpacing(10)
        tabs.addTab(host,name); layouts[name]=layout
    data,plots,anatomy,marks=(layouts[n] for n in ('Recordings','Plots','Anatomy','Marks'))
    unit_layout=layouts['Units']
    # Reuse the existing resizable library, anatomy editor and measurement tools.
    for item in items:
        widget=item.widget(); layout=item.layout()
        if layout is not None: layout.setParent(None)
        if layout is page.plot_selector_layout: plots.addLayout(layout)
        elif widget is page.units:
            unit_layout.addWidget(QLabel('Raster · visible units')); unit_layout.addWidget(widget)
        elif widget is page.wave_units:
            unit_layout.addWidget(QLabel('Waveforms · independent selection')); unit_layout.addWidget(widget)
        elif widget is page.levels: marks.addWidget(widget)
        elif widget in (page.controls,page.view_heading):
            if widget is page.view_heading: widget.hide()
        elif widget in (page.boundary_row,): marks.addWidget(widget)
        elif widget in (probe_box,page.regions_toggle,page.region_editor): anatomy.addWidget(widget)
        elif widget is page.message: pass
        elif widget is not None: data.addWidget(widget)
        elif layout is not None: data.addLayout(layout)
    # Move fields into the section where they make sense; labels follow visibility.
    form=page.controls.layout(); page._field_labels={}
    anatomy_form=QFormLayout(); anatomy.insertLayout(0,anatomy_form)
    data_form=QFormLayout(); data.addLayout(data_form)
    raster_button=QPushButton('Show raster'); raster_button.clicked.connect(page._show_unit_raster)
    unit_layout.insertWidget(0,raster_button)
    unit_form=QFormLayout(); unit_layout.insertLayout(1,unit_form)
    actions=QHBoxLayout()
    for text,callback in [('Show all',lambda:page._set_units_visible(True)),('Hide all',lambda:page._set_units_visible(False))]:
        button=QPushButton(text); button.clicked.connect(callback); actions.addWidget(button)
    unit_layout.insertLayout(2,actions)
    wave_button=QPushButton('Show waveforms'); wave_button.setToolTip('Kilosort templates on the peak channel; independent of raster visibility. Amplitudes are template units, not calibrated µV.'); wave_button.clicked.connect(lambda:page.plot_board.toggle_plot('waveform',True))
    unit_layout.addWidget(wave_button)
    for widget,label,target in [(page.source,'Image',anatomy_form),(page.labels,'',anatomy_form),
        (page.shank,'Shank',data_form),(page.good,'Show',unit_form),(page.span,'Spike time range',unit_form)]:
        row=form.takeRow(widget)
        if row.labelItem and row.labelItem.widget(): row.labelItem.widget().deleteLater()
        target.addRow(label,widget); page._field_labels[widget]=target.labelForField(widget)
    row=form.takeRow(page.time)
    if row.labelItem and row.labelItem.widget(): row.labelItem.widget().deleteLater()
    page.controls.hide()
    details=Fold('Analysis details'); details.content.addWidget(page.message); data.addWidget(details)
    for layout in layouts.values(): layout.addStretch()
    page.settings_tabs.setCurrentIndex(0)
    page.time_bar=QWidget(); time_layout=QHBoxLayout(page.time_bar); time_layout.setContentsMargins(8,8,8,4)
    page.window_controls=QHBoxLayout(); time_layout.addLayout(page.window_controls)
    time_layout.addWidget(page.time)
    page.time_slider=QSlider(Qt.Orientation.Horizontal); page.time_slider.setTracking(False); page.time_slider.setRange(0,10000); time_layout.addWidget(page.time_slider,1)
    page.window_duration=number(2,.5,60,1); page.window_duration.setSuffix(' s'); page.window_duration.setMaximumWidth(100)
    time_layout.addWidget(QLabel('Window')); time_layout.addWidget(page.window_duration)
    page.panel.layout().addWidget(page.time_bar)
    page.time_slider.sliderMoved.connect(lambda value:page.time.setToolTip(f'{value/100:g} s · release to update all time-based views'))
    page.time_slider.valueChanged.connect(lambda value:_seek(page,value/100))
    page.time.valueChanged.connect(lambda value:_sync_slider(page,value))
    page.window_duration.valueChanged.connect(lambda value:page._analysis_changed())
    page.window_timer=QTimer(page); page.window_timer.setSingleShot(True); page.window_timer.timeout.connect(lambda:page._load_notebook())


def _sync_slider(page,value):
    page.time_slider.blockSignals(True); page.time_slider.setValue(round(value*100)); page.time_slider.blockSignals(False)


def _seek(page,value):
    if hasattr(page,'plot_board') and not page._updating: page.plot_board.notebook_mode.setChecked(True)
    page.time.setValue(value)

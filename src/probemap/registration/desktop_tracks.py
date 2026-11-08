"""Track-plane selection and linked atlas annotation pages for ProbeMap."""
from __future__ import annotations

import copy
import tempfile
from pathlib import Path
from uuid import uuid4

import numpy as np
from qtpy.QtCore import QPointF, QThreadPool, QTimer, Qt
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (QCheckBox, QColorDialog, QComboBox, QDoubleSpinBox, QFormLayout,
                           QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
                           QListWidget, QListWidgetItem, QPushButton, QScrollArea, QSlider,
                           QSplitter, QStackedWidget, QTableWidget, QTableWidgetItem,
                           QTabWidget, QVBoxLayout, QWidget)

from .desktop_track_views import TrackSliceCanvas, TrackVolumeView
from .landmarks import transform_points
from .track_model import (active_session, export_track_analysis, load_label_atlas, load_track_data,
                          plane_geometry, plane_world, raw_volume, region_name, sample_world, track_report)


def note(text):
    widget=QLabel(text)
    widget.setWordWrap(True)
    widget.setObjectName("muted")
    return widget


def action(text,callback):
    widget=QPushButton(text)
    widget.clicked.connect(lambda checked=False:callback())
    return widget


class TrackWorkspace(QWidget):
    def __init__(self,owner):
        super().__init__()
        self.owner=owner
        self.project=self.session=None
        self.track_native=self.atlas_native=None
        self.track_gpu=None
        self.labels=None
        self._labels_path=None
        self._context=None
        self._updating=False
        self._closing=False
        self._token=0
        self._running=self._pending=False
        self._jobs=set()
        self._cache=None
        self._undo=[]
        self.last_export=None
        self.selected_id=None
        self.render_timer=QTimer(self)
        self.render_timer.setSingleShot(True)
        self.render_timer.timeout.connect(self._render)
        self.report_timer=QTimer(self)
        self.report_timer.setSingleShot(True)
        self.report_timer.timeout.connect(self._update_report)
        layout=QVBoxLayout(self)
        self.title=QLabel("Track plane")
        self.title.setObjectName("heading")
        layout.addWidget(self.title)
        self.info=note("Add an electrodes scan on Project and fit the brain registration.")
        layout.addWidget(self.info)
        split=QSplitter(Qt.Orientation.Horizontal)
        self.splitter=split
        self.splitter.setHandleWidth(7)
        self.splitter.setStyleSheet('QSplitter::handle{background:#354354;} QSplitter::handle:hover{background:#7ac7e5;}')
        self.controls=QStackedWidget()
        self.controls.setMinimumWidth(280)
        self.controls.setMaximumWidth(16777215)
        for panel in (self._plane_controls(),self._annotation_controls()):
            scroll=QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(panel)
            self.controls.addWidget(scroll)
        split.addWidget(self.controls)
        self.views=QStackedWidget()
        plane_page=QSplitter(Qt.Orientation.Horizontal)
        self.volume_view=TrackVolumeView()
        self.volume_view.opacityChanged.connect(self._volume_opacity)
        plane_page.addWidget(self.volume_view)
        right=QWidget()
        column=QVBoxLayout(right)
        column.addWidget(QLabel("Track scan · selected oblique plane"))
        self.plane_slice=TrackSliceCanvas()
        self.plane_slice.scrolled.connect(self._scroll_plane)
        column.addWidget(self.plane_slice,1)
        self.plane_info=note("Plane position and pixel size will appear here.")
        column.addWidget(self.plane_info)
        plane_page.addWidget(right)
        plane_page.setSizes([500,500])
        self.views.addWidget(plane_page)
        annotation=QWidget()
        grid=QGridLayout(annotation)
        self.track_slice=TrackSliceCanvas(drawable=True)
        self.atlas_slice=TrackSliceCanvas()
        self.track_slice.annotation_mode=self.atlas_slice.annotation_mode=True
        self.track_slice.vertexAdded.connect(self.add_vertex)
        self.track_slice.finished.connect(self.finish_track)
        for index,(title,canvas) in enumerate((("Track scan · draw electrode tracks",self.track_slice),
                                              ("Atlas · same physical plane",self.atlas_slice))):
            grid.addWidget(QLabel(title),0,index)
            grid.addWidget(canvas,1,index)
            grid.setColumnStretch(index,1)
            canvas.viewChanged.connect(lambda canvas=canvas:self._sync_views(canvas))
            canvas.hovered.connect(self._hover)
        grid.setRowStretch(1,1)
        self.views.addWidget(annotation)
        split.addWidget(self.views)
        split.setStretchFactor(1,1)
        split.setSizes([310,1000])
        layout.addWidget(split,1)
        self.hover_info=note("Track points are stored in physical 3D atlas coordinates.")
        layout.addWidget(self.hover_info)
        row=QHBoxLayout()
        self.back_button=action("← Review",lambda:owner.go_step(3 if owner.step==4 else 4))
        row.addWidget(self.back_button)
        row.addWidget(action("Save project",owner.save_project))
        row.addStretch()
        self.next_button=action("Use plane → Track annotation",lambda:owner.go_step(5))
        row.addWidget(self.next_button)
        self.export_button=action("Save tracks && region report",self.export_analysis)
        row.addWidget(self.export_button)
        layout.addLayout(row)

    def _plane_controls(self,include_visibility=True,include_display=True):
        panel=QWidget()
        layout=QVBoxLayout(panel)
        layout.addWidget(note("The track scan already uses the fitted brain transform. These controls move the viewing plane only."))
        self.preset=QComboBox()
        self.preset.addItems(["XY","XZ","YZ","Oblique"])
        self.preset.currentIndexChanged.connect(self._preset_changed)
        layout.addWidget(self.preset)
        self.pose={}
        for name,title,minimum,maximum,decimals,factor in (("center","Plane center in atlas axes (µm)",-1e6,1e6,2,1),
                                                         ("rotation","Plane rotation X / Y / Z (degrees)",-180,180,1,10)):
            box=QGroupBox(title)
            grid=QGridLayout(box)
            for axis in range(3):
                grid.addWidget(QLabel("XYZ"[axis]),axis,0)
                slider=QSlider(Qt.Orientation.Horizontal)
                slider.setRange(-100000,100000) if name=="center" else slider.setRange(-1800,1800)
                spin=QDoubleSpinBox()
                spin.setRange(minimum,maximum)
                spin.setDecimals(decimals)
                spin.setSingleStep(25 if name=="center" else 1)
                spin.setKeyboardTracking(False)
                slider.valueChanged.connect(lambda value,spin=spin,factor=factor:spin.setValue(value/factor) if not self._updating else None)
                spin.valueChanged.connect(lambda value,name=name,axis=axis:self._pose_changed(name,axis,value))
                grid.addWidget(slider,axis,1)
                grid.addWidget(spin,axis,2)
                self.pose[name,axis]=(slider,spin,factor)
            layout.addWidget(box)
        box=QGroupBox("Move along plane normal (µm)")
        row=QHBoxLayout(box)
        self.normal_slider=QSlider(Qt.Orientation.Horizontal)
        self.normal_slider.setRange(-50000,50000)
        self.normal_spin=QDoubleSpinBox()
        self.normal_spin.setRange(-1e6,1e6)
        self.normal_spin.setDecimals(2)
        self.normal_spin.setSingleStep(25)
        self.normal_spin.setKeyboardTracking(False)
        self.normal_slider.valueChanged.connect(self.normal_spin.setValue)
        self.normal_spin.valueChanged.connect(self._normal_changed)
        self._last_normal=0.
        row.addWidget(self.normal_slider,1)
        row.addWidget(self.normal_spin)
        layout.addWidget(box)
        box=QGroupBox("Plane field of view")
        form=QFormLayout(box)
        self.extents=[]
        for axis in ("Width","Height"):
            spin=QDoubleSpinBox()
            spin.setRange(1,1e6)
            spin.setDecimals(1)
            spin.setSuffix(" µm")
            spin.setSingleStep(250)
            spin.setKeyboardTracking(False)
            spin.valueChanged.connect(self._field_changed)
            self.extents.append(spin)
            form.addRow(axis,spin)
        self.quality=QComboBox()
        for text,size in (("Fast · 384",384),("Balanced · 512",512),("Detailed · 768",768),("Fine · 1024",1024)):
            self.quality.addItem(text,size)
        self.quality.currentIndexChanged.connect(self._field_changed)
        form.addRow("Plane pixels",self.quality)
        layout.addWidget(box)
        if include_visibility:
            box=QGroupBox("3D visibility / opacity")
            form=QFormLayout(box)
            self.opacity=[]
            for title,value in (("Atlas context",0),("Track scan",100)):
                slider=QSlider(Qt.Orientation.Horizontal)
                slider.setRange(0,100)
                slider.setValue(value)
                slider.valueChanged.connect(self._update_volume)
                self.opacity.append(slider)
                form.addRow(title,slider)
            layout.addWidget(box)
        self.track_gamma=QDoubleSpinBox()
        self.track_gamma.setRange(.1,5)
        self.track_gamma.setSingleStep(.1)
        self.track_gamma.setValue(1)
        self.track_gamma.valueChanged.connect(self.draw_cached)
        form=QFormLayout()
        if include_display:
            form.addRow("Track display gamma",self.track_gamma)
            layout.addLayout(form)
        else: self.track_gamma.hide()
        layout.addWidget(note("Drag 3D to inspect the plane. Wheel on the slice moves along its normal; Ctrl+wheel zooms. No additional scan transform is created."))
        layout.addStretch()
        return panel

    def _annotation_controls(self):
        tabs=QTabWidget()
        tracks=QWidget()
        layout=QVBoxLayout(tracks)
        layout.addWidget(note("New track → click vertices along the electrode on the left image → Finish. The identical 3D line appears on the atlas."))
        form=QFormLayout()
        self.annotation_track_gamma=QDoubleSpinBox()
        self.annotation_track_gamma.setRange(.1,5)
        self.annotation_track_gamma.setSingleStep(.1)
        self.annotation_track_gamma.setValue(1)
        self.annotation_track_gamma.valueChanged.connect(self.track_gamma.setValue)
        self.track_gamma.valueChanged.connect(self.annotation_track_gamma.setValue)
        form.addRow("Track display gamma",self.annotation_track_gamma)
        layout.addLayout(form)
        self.track_list=QListWidget()
        self.track_list.currentRowChanged.connect(self._select_track)
        layout.addWidget(self.track_list)
        self.track_name=QLineEdit()
        self.track_name.setPlaceholderText("Track name")
        self.track_name.editingFinished.connect(self._rename_track)
        layout.addWidget(self.track_name)
        layout.addWidget(action("Track color…",self._track_color))
        for text,callback in (("New track",self.new_track),("Finish track",self.finish_track),
                              ("Edit selected track",self.edit_track),("Show selected track plane",self.show_track_plane),
                              ("Undo",self.undo),("Delete selected track",self.delete_track)):
            layout.addWidget(action(text,callback))
        layout.addWidget(note("Wheel zooms both images; middle-drag pans both. Lines remain in 3D when you switch planes. Region distances start at the first vertex you draw."))
        tabs.addTab(tracks,"Tracks")
        tabs.addTab(self._region_controls(),"Regions")
        tabs.addTab(self._report_controls(),"Report")
        return tabs

    def _region_controls(self,include_display=True):
        regions=QWidget()
        layout=QVBoxLayout(regions)
        layout.addWidget(note("Hide an area, show its name, edit the name, or double-click its color. Source atlas masks remain unchanged."))
        row=QHBoxLayout()
        row.addWidget(action("Show all",lambda:self._set_all_regions(True)))
        row.addWidget(action("Hide all",lambda:self._set_all_regions(False)))
        layout.addLayout(row)
        self.region_table=QTableWidget(0,4)
        self.region_table.verticalHeader().setVisible(False)
        self.region_table.setHorizontalHeaderLabels(["Show","Label","Region name","Color"])
        self.region_table.horizontalHeader().setSectionResizeMode(2,QHeaderView.ResizeMode.Stretch)
        for index in (0,1,3): self.region_table.setColumnWidth(index,45)
        self.region_table.itemChanged.connect(self._region_changed)
        self.region_table.cellDoubleClicked.connect(self._region_color)
        layout.addWidget(self.region_table,1)
        form=QFormLayout()
        self.region_opacity=QSlider(Qt.Orientation.Horizontal)
        self.region_opacity.setRange(0,100)
        self.region_opacity.setValue(45)
        self.region_opacity.valueChanged.connect(self.draw_cached)
        form.addRow("Region opacity",self.region_opacity)
        self.atlas_gamma=QDoubleSpinBox()
        self.atlas_gamma.setRange(.1,5)
        self.atlas_gamma.setValue(1)
        self.atlas_gamma.setSingleStep(.1)
        self.atlas_gamma.valueChanged.connect(self.draw_cached)
        if include_display: form.addRow("Atlas gamma",self.atlas_gamma)
        else: self.atlas_gamma.hide()
        layout.addLayout(form)
        return regions

    def _report_controls(self):
        report=QWidget()
        layout=QVBoxLayout(report)
        self.report_info=note("Draw a track to identify the atlas regions along it.")
        layout.addWidget(self.report_info)
        self.report_table=QTableWidget(0,4)
        self.report_table.verticalHeader().setVisible(False)
        self.report_table.setHorizontalHeaderLabels(["Track / region","Entry µm","Exit µm","Length µm"])
        self.report_table.horizontalHeader().setSectionResizeMode(0,QHeaderView.ResizeMode.Stretch)
        for column in (1,2,3): self.report_table.setColumnWidth(column,70)
        self.report_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.report_table)
        return report

    def reset(self):
        self.capture_state()
        self._token+=1
        self.render_timer.stop()
        self.report_timer.stop()
        self.project=self.session=None
        self.track_native=self.atlas_native=self.track_gpu=None
        self.labels=self._labels_path=None
        self._context=self._cache=None
        self._undo.clear()

    def open(self,project,reference_preview,mode):
        project.validate_track()
        session=active_session(project)
        context=(id(project),session["id"])
        self._mode=mode
        if mode==0: self.track_slice.drawing=False
        self.controls.setMaximumWidth(16777215)
        def resize_panels():
            width=420 if self._mode else 310
            self.splitter.setSizes([width,max(360,self.splitter.width()-width-self.splitter.handleWidth())])
        QTimer.singleShot(0,resize_panels)
        self.controls.setCurrentIndex(mode)
        self.views.setCurrentIndex(mode)
        self.title.setText("Select the track plane" if mode==0 else "Trace electrodes on the atlas")
        self.back_button.setText("← Review" if mode==0 else "← Track plane")
        self.next_button.setVisible(mode==0)
        self.export_button.setVisible(mode==1)
        if context==self._context and self.track_native is not None:
            if mode==1: self._ensure_labels()
            else: self.request_render()
            return
        self.capture_state()
        self.project,self.session=project,session
        self._context=context
        self._token+=1
        self._undo.clear()
        self.labels=self._labels_path=None
        self._cache=None
        def load(progress):
            progress(10,"Loading the native electrode channel")
            native,gpu,limits=load_track_data(project.track,project.preview_dimension)
            progress(75,"Loading native atlas intensities for matched oblique slices")
            atlas=raw_volume(project.reference)
            sample=np.asarray(atlas.data).ravel()[::max(1,atlas.data.size//500000)]
            finite=sample[np.isfinite(sample)]
            low,high=np.percentile(finite,(1,99.8))
            if high<=low: low,high=float(finite.min()),max(float(finite.max()),float(low)+1)
            return native,gpu,limits,atlas,[float(low),float(high)]
        def ready(result):
            self.track_native,self.track_gpu,self.track_limits,self.atlas_native,self.atlas_limits=result
            self.volume_view.load(reference_preview,self.track_gpu)
            self._restore_state()
            self._update_volume()
            self._populate_plane()
            self._refresh_tracks()
            self.info.setText(f"Track: {project.track.path.name} · {project.fit_result['model'].title()} from brain scan\n"
                              "Native intensities for slices; peak-preserving 3D overview. The brain fit stays unchanged.")
            if mode==1: self._ensure_labels()
            else:
                self.request_render()
                self.owner.statusBar().showMessage("Track scan ready. Adjust the plane, then continue to Track annotation.")
        self.owner._start_job(load,ready,"Preparing track views…")

    def _ensure_labels(self):
        directory=self.project.labels_dir
        if directory is None:
            self.owner.statusBar().showMessage("Add a labels folder on Project.")
            self.request_render()
            return
        if self.labels is not None and self._labels_path==str(Path(directory).resolve()):
            self.request_render()
            return
        project=self.project
        def load(progress):
            cache=Path(tempfile.gettempdir())/"probemap"/"native-labels"
            return load_label_atlas(directory,project.reference,cache,progress,spacing_um_zyx=project.label_spacing)
        def ready(labels):
            self.labels=labels
            self._labels_path=str(Path(directory).resolve())
            self._populate_regions()
            self.request_render()
            self.report_timer.start(0)
            self.owner.statusBar().showMessage(f"{len(labels.regions)} atlas regions ready. Choose New track to draw an electrode.")
        self.owner._start_job(load,ready,"Loading atlas regions…")

    def _populate_plane(self):
        if self.session is None: return
        plane=self.session["plane"]
        self._updating=True
        for name,values in (("center",plane["center_um_zyx"][::-1]),("rotation",plane["rotation_xyz_deg"])):
            for axis,value in enumerate(values):
                slider,spin,factor=self.pose[name,axis]
                spin.setValue(value)
                slider.setRange(min(slider.minimum(),round(value*factor)),max(slider.maximum(),round(value*factor)))
                slider.setValue(round(value*factor))
        for spin,key in zip(self.extents,("width_um","height_um")): spin.setValue(plane[key])
        self.quality.setCurrentIndex(max(0,self.quality.findData(plane.get("max_dimension",512))))
        self.normal_slider.setValue(0)
        self.normal_spin.setValue(0)
        self._last_normal=0.
        presets=((0,0,0),(90,0,0),(90,0,90))
        self.preset.setCurrentIndex(next((index for index,angles in enumerate(presets)
                                          if np.allclose(plane["rotation_xyz_deg"],angles,atol=1e-8)),3))
        self._updating=False

    def _pose_changed(self,name,axis,value):
        if self._updating or self.session is None: return
        self.track_slice.drawing=False
        slider,_,factor=self.pose[name,axis]
        slider.blockSignals(True)
        slider.setRange(min(slider.minimum(),round(value*factor)),max(slider.maximum(),round(value*factor)))
        slider.setValue(round(value*factor))
        slider.blockSignals(False)
        self.session["plane"]["center_um_zyx" if name=="center" else "rotation_xyz_deg"][2-axis if name=="center" else axis]=float(value)
        self._updating=True
        if name=="rotation":
            presets=((0,0,0),(90,0,0),(90,0,90))
            self.preset.setCurrentIndex(next((index for index,angles in enumerate(presets)
                                             if np.allclose(self.session["plane"]["rotation_xyz_deg"],angles,atol=1e-8)),3))
        self.normal_slider.setValue(0)
        self.normal_spin.setValue(0)
        self._last_normal=0.
        self._updating=False
        self.request_render()

    def _normal_changed(self,value):
        if self._updating or self.session is None: return
        self.track_slice.drawing=False
        geometry=plane_geometry(self.session["plane"])
        center=np.asarray(self.session["plane"]["center_um_zyx"])+(value-self._last_normal)*np.asarray(geometry["normal_zyx"])
        self.session["plane"]["center_um_zyx"]=center.tolist()
        self._last_normal=value
        self._updating=True
        self.normal_slider.blockSignals(True)
        self.normal_slider.setRange(min(self.normal_slider.minimum(),round(value)),max(self.normal_slider.maximum(),round(value)))
        self.normal_slider.setValue(round(value))
        self.normal_slider.blockSignals(False)
        for axis,position in enumerate(center[::-1]):
            slider,spin,factor=self.pose["center",axis]
            spin.setValue(position)
            slider.setRange(min(slider.minimum(),round(position*factor)),max(slider.maximum(),round(position*factor)))
            slider.setValue(round(position*factor))
        self._updating=False
        self.request_render()

    def _scroll_plane(self,delta):
        if self.session is not None:
            self.normal_spin.setValue(self.normal_spin.value()+delta*min(self.project.track.spacing_um_zyx))

    def _preset_changed(self,index):
        if self._updating or self.session is None or index==3: return
        self.track_slice.drawing=False
        self.session["plane"]["rotation_xyz_deg"]=list(((0,0,0),(90,0,0),(90,0,90))[index])
        self._populate_plane()
        self._updating=True
        self.preset.setCurrentIndex(index)
        self._updating=False
        self.request_render()

    def _field_changed(self):
        if self._updating or self.session is None: return
        self.track_slice.drawing=False
        self.session["plane"]["width_um"]=self.extents[0].value()
        self.session["plane"]["height_um"]=self.extents[1].value()
        self.session["plane"]["max_dimension"]=self.quality.currentData()
        self.request_render()

    def _volume_opacity(self,index,value):
        self.opacity[index].setValue(value)

    def _update_volume(self):
        if self.project is not None and self.track_native is not None:
            self.volume_view.update_alignment(self.project.total,*[s.value()/100 for s in self.opacity])

    def request_render(self,*args):
        if self._closing or self.session is None or self.track_native is None: return
        self._token+=1
        self.volume_view.set_plane(plane_geometry(self.session["plane"]))
        self.render_timer.start(35)

    def _render(self):
        if self._closing or self.session is None or self.owner.step not in (4,5): return
        if self._running:
            self._pending=True
            return
        from .desktop_app import Job
        token=self._token
        geometry=plane_geometry(copy.deepcopy(self.session["plane"]))
        track,reference,labels=self.track_native,self.atlas_native,self.labels
        matrix=self.project.total.copy()
        def render(progress):
            world=plane_world(geometry)
            shape=geometry["shape_hw"]
            scan=sample_world(track,matrix,world).reshape(shape)
            atlas=sample_world(reference,np.eye(4),world).reshape(shape)
            ids=labels.ids_at(world).reshape(shape) if labels is not None else None
            return scan,atlas,ids,geometry
        job=Job(render)
        self._jobs.add(job)
        self._running,self._pending=True,False
        def finish(result=None,error=None):
            self._jobs.discard(job)
            self._running=False
            if self._closing: return
            if token==self._token and self.owner.step in (4,5):
                if error: self.owner.statusBar().showMessage(error)
                else:
                    self._cache=result
                    self.draw_cached()
            if self._pending or token!=self._token: self.render_timer.start(0)
        job.signals.done.connect(lambda value:finish(result=value))
        job.signals.failed.connect(lambda error:finish(error=error))
        QThreadPool.globalInstance().start(job)

    def draw_cached(self,*args):
        if self._cache is None or self.session is None: return
        scan,atlas,ids,geometry=self._cache
        normalize=lambda data,limits,gamma:np.power(np.nan_to_num(np.clip((data-limits[0])/max(limits[1]-limits[0],1e-9),0,1),nan=0),gamma)
        grey_scan=normalize(scan,self.track_limits,self.track_gamma.value())
        grey_atlas=normalize(atlas,self.atlas_limits,self.atlas_gamma.value())
        rgb=np.repeat(grey_atlas[...,None],3,axis=2)
        text=[]
        opacity=self.region_opacity.value()/100
        if ids is not None and self.labels is not None:
            for region in self.labels.regions:
                style=self.session["regions"].get(str(region["id"]),{})
                if not style.get("visible",True): continue
                mask=ids==region["id"]
                if not np.any(mask): continue
                color=style.get("color_rgb",region["color_rgb"])
                rgb[mask]=(1-opacity)*rgb[mask]+opacity*np.asarray(color)/255
                if style.get("label",False):
                    row,col=np.nonzero(mask)
                    nearest=int(np.argmin((row-row.mean())**2+(col-col.mean())**2))
                    text.append((style.get("name",region["name"]),[int(col[nearest]),int(row[nearest])],color))
        for canvas,image in ((self.plane_slice,grey_scan),(self.track_slice,grey_scan),(self.atlas_slice,rgb)):
            canvas.track_lines=self.session["tracks"]
            canvas.selected_track=self.selected_id
            canvas.set_plane_image(image,geometry)
        self.atlas_slice.region_labels=text
        self.volume_view.set_tracks(self.session["tracks"])
        self.plane_info.setText(f"Pixel size U/V: {geometry['pixel_size_um']:.2f} µm · "
                                f"{geometry['shape_hw'][1]} × {geometry['shape_hw'][0]} pixels\n"
                                "The right slice is the exact plane outlined in 3D.")

    def _sync_views(self,source):
        other=self.atlas_slice if source is self.track_slice else self.track_slice
        other.zoom,other.pan=source.zoom,QPointF(source.pan)
        other.update()

    def _hover(self,world):
        text="XYZ "+" / ".join(f"{value:.1f}" for value in world[::-1])+" µm"
        if self.labels is not None:
            identifier=int(self.labels.ids_at([world])[0])
            text+=" · "+region_name(identifier,self.labels.regions,self.session["regions"])
        self.hover_info.setText(text)

    def _remember(self):
        self._undo.append(copy.deepcopy(self.session["tracks"]))
        self._undo=self._undo[-100:]

    def current_track(self):
        return next((track for track in self.session["tracks"] if track["id"]==self.selected_id),None) if self.session else None

    def new_track(self):
        if self.session is None or self._cache is None: return
        self._remember()
        identifier=uuid4().hex[:12]
        color=("#ffc966","#5bdcf3","#ff82b9","#a8f08b")[len(self.session["tracks"])%4]
        self.session["tracks"].append({"id":identifier,"name":f"Track {len(self.session['tracks'])+1}","color":color,
                                        "plane":copy.deepcopy(self.session["plane"]),"points_atlas_zyx_um":[],
                                        "points_source_track_zyx_voxel":[]})
        self.selected_id=identifier
        self.track_slice.drawing=True
        self._refresh_tracks()
        self.owner.statusBar().showMessage("Click vertices along the electrode on the left track scan, then Finish track.")

    def add_vertex(self,world):
        track=self.current_track()
        if track is None or not self.track_slice.drawing or self._cache is None: return
        if self._cache[3]!=plane_geometry(self.session["plane"]):
            self.owner.statusBar().showMessage("Wait for the current plane slice before adding a vertex.")
            return
        world=np.asarray(world,dtype=float)
        if world.shape!=(3,) or not np.isfinite(world).all(): return
        original_plane=plane_geometry(track["plane"])
        normal=np.asarray(original_plane["normal_zyx"])
        if abs(np.dot(world-np.asarray(original_plane["center_um_zyx"]),normal))>1e-4:
            self.track_slice.drawing=False
            self.owner.statusBar().showMessage("This track belongs to another plane. Edit selected track to restore it, or create a new track.")
            return
        if track["points_atlas_zyx_um"] and np.linalg.norm(world-np.asarray(track["points_atlas_zyx_um"][-1]))<1e-6: return
        self._remember()
        raw=transform_points(world[None],np.linalg.inv(self.project.total))[0]/self.project.track.spacing_um_zyx
        track["points_atlas_zyx_um"].append(world.tolist())
        track["points_source_track_zyx_voxel"].append(raw.tolist())
        self._refresh_tracks()
        self.report_timer.start(150)

    def finish_track(self):
        self.track_slice.drawing=False
        self.report_timer.start(0)
        self.owner.statusBar().showMessage("Track saved in 3D. Open Report to inspect regions, or add another track.")

    def show_track_plane(self):
        track=self.current_track()
        if track is not None:
            self.session["plane"]=copy.deepcopy(track["plane"])
            self._populate_plane()
            self.request_render()

    def edit_track(self):
        if self.current_track() is not None:
            self.show_track_plane()
            self.track_slice.drawing=True

    def delete_track(self):
        if self.current_track() is None: return
        self._remember()
        self.session["tracks"]=[track for track in self.session["tracks"] if track["id"]!=self.selected_id]
        self.selected_id=None
        self.track_slice.drawing=False
        self._refresh_tracks()
        self.report_timer.start(0)

    def undo(self):
        if self.session is not None and self._undo:
            self.session["tracks"]=self._undo.pop()
            self._refresh_tracks()
            self.report_timer.start(0)

    def _refresh_tracks(self):
        if self.session is None: return
        self.track_list.blockSignals(True)
        self.track_list.clear()
        selected=-1
        for index,track in enumerate(self.session["tracks"]):
            item=QListWidgetItem(f"{track['name']} · {len(track['points_atlas_zyx_um'])} points")
            item.setData(Qt.ItemDataRole.UserRole,track["id"])
            self.track_list.addItem(item)
            if track["id"]==self.selected_id: selected=index
        self.track_list.setCurrentRow(selected)
        self.track_list.blockSignals(False)
        self.track_name.setText(self.current_track()["name"] if self.current_track() else "")
        self.draw_cached()

    def _select_track(self,row):
        item=self.track_list.item(row)
        self.selected_id=item.data(Qt.ItemDataRole.UserRole) if item is not None else None
        self.track_slice.drawing=False
        self.track_name.setText(self.current_track()["name"] if self.current_track() else "")
        self.draw_cached()

    def _rename_track(self):
        track=self.current_track()
        if track is not None and self.track_name.text().strip():
            track["name"]=self.track_name.text().strip()
            self._refresh_tracks()
            self.report_timer.start(0)

    def _track_color(self):
        track=self.current_track()
        if track is None: return
        color=QColorDialog.getColor(QColor(track["color"]),self,"Track color")
        if color.isValid():
            track["color"]=color.name()
            self.draw_cached()

    def _populate_regions(self):
        self.region_table.blockSignals(True)
        self.region_table.setRowCount(len(self.labels.regions))
        for row,region in enumerate(self.labels.regions):
            style=self.session["regions"].get(str(region["id"]),{})
            for column,key,default in ((0,"visible",True),(1,"label",False)):
                item=QTableWidgetItem()
                item.setFlags(Qt.ItemFlag.ItemIsEnabled|Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Checked if style.get(key,default) else Qt.CheckState.Unchecked)
                self.region_table.setItem(row,column,item)
            name=QTableWidgetItem(style.get("name",region["name"]))
            name.setToolTip(f"Region ID: {region['id']}\n{Path(region['mask_path']).name}")
            self.region_table.setItem(row,2,name)
            color=QTableWidgetItem()
            color.setFlags(Qt.ItemFlag.ItemIsEnabled|Qt.ItemFlag.ItemIsSelectable)
            color.setBackground(QColor(*style.get("color_rgb",region["color_rgb"])))
            self.region_table.setItem(row,3,color)
        self.region_table.blockSignals(False)

    def _region_changed(self,item):
        if self.labels is None or self.session is None: return
        row=item.row()
        region=self.labels.regions[row]
        style=self.session["regions"].setdefault(str(region["id"]),{})
        if item.column()==0: style["visible"]=item.checkState()==Qt.CheckState.Checked
        elif item.column()==1: style["label"]=item.checkState()==Qt.CheckState.Checked
        elif item.column()==2:
            style["name"]=item.text().strip() or region["name"]
            self.report_timer.start(150)
        self.draw_cached()

    def _region_color(self,row,column):
        if column!=3 or self.labels is None: return
        region=self.labels.regions[row]
        style=self.session["regions"].setdefault(str(region["id"]),{})
        color=QColorDialog.getColor(QColor(*style.get("color_rgb",region["color_rgb"])),self,"Region color")
        if color.isValid():
            style["color_rgb"]=[color.red(),color.green(),color.blue()]
            self._populate_regions()
            self.draw_cached()

    def _set_all_regions(self,visible):
        if self.labels is None or self.session is None: return
        for region in self.labels.regions:
            self.session["regions"].setdefault(str(region["id"]),{})["visible"]=visible
        self._populate_regions()
        self.draw_cached()

    def _update_report(self):
        if self.labels is None or self.session is None: return
        report=track_report(self.session,self.labels)
        self.report=report
        self.report_info.setText(f"Native atlas labels · sampling step {report['sampling_step_um']:.2f} µm\n"
                                 "Distances follow the line from its first vertex. Hidden regions remain in the report.")
        self.report_table.setRowCount(len(report["segments"]))
        for row,segment in enumerate(report["segments"]):
            values=[f"{segment['track_name']} · {segment['region_name']}",
                    *[f"{segment[key]:.1f}" for key in ("entry_um","exit_um","length_um")]]
            for column,value in enumerate(values):
                item=QTableWidgetItem(value)
                item.setToolTip(value)
                self.report_table.setItem(row,column,item)

    def capture_state(self):
        if self.session is None: return
        self.session["view"]={"camera":self.volume_view.camera_record(),"rendering":self.volume_view.render_settings_record(),
                              "opacity":[s.value() for s in self.opacity],"track_gamma":self.track_gamma.value(),
                              "atlas_gamma":self.atlas_gamma.value(),"region_opacity":self.region_opacity.value(),
                              "selected_track_id":self.selected_id}

    def _restore_state(self):
        view=self.session.get("view",{})
        if "rendering" in view: self.volume_view.restore_render_settings(view["rendering"])
        for spin,value in zip(self.opacity,view.get("opacity",[0,100])): spin.setValue(value)
        self.track_gamma.setValue(view.get("track_gamma",1))
        self.atlas_gamma.setValue(view.get("atlas_gamma",1))
        self.region_opacity.setValue(view.get("region_opacity",45))
        self.selected_id=view.get("selected_track_id")
        self.volume_view.restore_camera(view.get("camera",{}))
        self.track_slice.drawing=False

    def export_analysis(self):
        if self.project is None or self.owner.busy: return
        if self.labels is None:
            self.owner._error("Choose an atlas labels folder before saving a region report.")
            return
        path=self.owner.save_project()
        if path is None: return
        from .desktop_model import AlignmentProject
        frozen=AlignmentProject.load(path)
        labels=self.labels
        def ready(path):
            self.last_export=path
            self.owner.statusBar().showMessage(f"Tracks, native slices and region report saved: {path.parent}")
            self.info.setText(f"Track analysis saved\n{path.parent}")
        self.owner._start_job(lambda progress:export_track_analysis(frozen,labels,progress),ready,"Saving track analysis…")

    def stop(self):
        self._closing=True
        self._token+=1
        self.render_timer.stop()
        self.report_timer.stop()

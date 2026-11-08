"""Five insertion/probe workflow pages sharing one GL canvas and native slices."""
from __future__ import annotations

import copy
import tempfile
import time
from pathlib import Path
from uuid import uuid4

import numpy as np
from qtpy.QtCore import QPointF, QThreadPool, QTimer, Qt
from qtpy.QtGui import QAction, QColor, QKeySequence
from qtpy.QtWidgets import (QCheckBox, QColorDialog, QComboBox, QDoubleSpinBox,
    QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QInputDialog, QListWidget, QListWidgetItem, QScrollArea, QSizePolicy, QSlider, QSplitter,
    QTableWidget, QTableWidgetItem, QTabWidget, QToolButton, QVBoxLayout, QWidget)

from .desktop_tracks import TrackWorkspace, action, note
from .desktop_insertion_views import InsertionSliceCanvas, InsertionVolumeView
from .desktop_model import AlignmentProject
from .insertion_model import (add_insertion_point, brain_support, directed_track, export_insertions,
    fit_insertion_plane, inside_brain, insertion_session, move_points, new_insertion,
    plane_is_stale, points_signature, save_manual_plane, sync_tracks, move_tracks, add_measurement)
from .track_model import (load_label_atlas, load_track_data, plane_geometry, plane_world,
                          plane_uv, raw_volume, sample_world)
from .coarse import transformed_corners
from .desktop_tracing import ProbeTracingMixin, WorkflowControls
from .desktop_measurements import MeasurementControlsMixin
from .desktop_regions import RegionControlsMixin
from .desktop_widgets import flat_tabs, view_button
from .landmarks import transform_points


from .desktop_neuropixels import NeuropixelsWorkspaceMixin
from .desktop_probe_placement import ProbePlacementMixin
from .desktop_probe_placement_views import ProbePlacementSlice


class InsertionWorkspace(ProbePlacementMixin,ProbeTracingMixin,MeasurementControlsMixin,RegionControlsMixin,NeuropixelsWorkspaceMixin,TrackWorkspace):
    def __init__(self,owner):
        QWidget.__init__(self)
        self.owner=owner
        self.project=self.session=None
        self.track_native=self.atlas_native=self.brain_native=self.track_gpu=None
        self.brain_preview=self.brain_mask=None
        self.labels=self._labels_path=self._context=self._cache=None
        self._updating=self._closing=self._running=self._pending=False
        self._token=0
        self._jobs=set()
        self._undo=[]
        self._mode=0
        self.last_export=None
        self.selected_id=self.insertion_id=None
        self.selected_points=set()
        self._draft_start=None
        self._preview_start=self._draft_direction=None
        self._pending_draw=False
        self._drawing_geometry=None
        self._label_loading=False
        self._view_states={}
        self._ortho_positions=None
        self._tracing_positions=None
        self._track_move_preview=None
        self._tool_modes={}
        self._canvas_tool_modes={}
        self._canvas_tools=['pan']*3
        self._xyz_initialized=False
        self._sidebar_sizes=[340,1000]
        self._last_plane_edit=0.
        self.render_timer=QTimer(self)
        self.render_timer.setSingleShot(True)
        self.render_timer.timeout.connect(self._render)
        self.report_timer=QTimer(self)
        self.report_timer.setSingleShot(True)
        self.report_timer.timeout.connect(self._update_report)
        layout=QVBoxLayout(self)
        self.title=QLabel("Insertion points")
        self.title.setObjectName("heading")
        header=QHBoxLayout()
        self.workspace_header=header
        from .desktop_slice_tools import tool_icon
        self.sidebar_toggle=QToolButton()
        self.sidebar_toggle.setIcon(tool_icon('sidebar'))
        self.sidebar_toggle.setCheckable(True); self.sidebar_toggle.setChecked(True)
        self.sidebar_toggle.setToolTip('Show / hide settings panel · Ctrl/⌘+\\')
        self.sidebar_toggle.setAccessibleName('Show / hide settings panel')
        self.sidebar_toggle.toggled.connect(self._toggle_sidebar)
        header.addWidget(self.sidebar_toggle); header.addWidget(self.title,1)
        layout.addLayout(header)
        self.info=note("Add an electrodes scan on Project to start tracing.")
        layout.addWidget(self.info)
        self.splitter=QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setHandleWidth(7)
        self.splitter.setStyleSheet('QSplitter::handle{background:#354354;} QSplitter::handle:hover{background:#7ac7e5;}')
        self.control_content=QWidget()
        sidebar=QVBoxLayout(self.control_content)
        sidebar.setContentsMargins(0,0,0,0)
        insertion_box=QGroupBox('Insertion')
        self.insertion_box=insertion_box
        insertion_layout=QVBoxLayout(insertion_box)
        self.insertion_combo=QComboBox()
        self.insertion_combo.currentIndexChanged.connect(self._choose_insertion)
        insertion_layout.addWidget(self.insertion_combo)
        row=QHBoxLayout()
        self.new_insertion_button=action("New…",self.create_insertion)
        row.addWidget(self.new_insertion_button)
        self.rename_insertion_button=action('Rename…',self.rename_insertion)
        row.addWidget(self.rename_insertion_button)
        row.addWidget(action("Color…",self.color_insertion))
        insertion_layout.addLayout(row)
        sidebar.addWidget(insertion_box)
        self.sidebar_tabs=QTabWidget()
        work,display=QWidget(),QWidget()
        controls,display_layout=QVBoxLayout(work),QVBoxLayout(display)
        self.work_layout=controls
        self.display_content=display
        for title,page in (('Points',work),('3D',display)):
            scroll=QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(page)
            self.sidebar_tabs.addTab(scroll,title)
        sidebar.addWidget(self.sidebar_tabs,1)
        # Legacy controller remains off-screen for old saved tool indices.
        self.point_tool=QComboBox(self.control_content)
        self.point_tool.addItems(['Add points','Select points','Pan image','Select / move track','Measure distance'])
        self.point_tool.currentIndexChanged.connect(self._slice_tool_changed)
        self.point_tool.hide()
        self.controls=WorkflowControls()
        # Create the canvases first: inherited plane callbacks refer to track_slice.
        self.view_container=QWidget()
        self.view_layout=QVBoxLayout(self.view_container)
        self.view_layout.setContentsMargins(0,0,0,0)
        self._build_tracing_toolbar()
        self.workspace_header.addWidget(self.tracing_toolbar)
        self.view_panes=QWidget()
        self.view_layout.addWidget(self.view_panes,1)
        self.view_grid=QGridLayout(self.view_panes)
        self.view_grid.setContentsMargins(0,0,0,0)
        self.volume_view=InsertionVolumeView()
        self.volume_view.opacityChanged.connect(self._volume_opacity)
        self.volume_view.pointsSelected.connect(self.select_points)
        self.volume_view.trackSelected.connect(self.select_track_by_id)
        self.volume_view.annotationContext.connect(self._annotation_context)
        self.view_grid.addWidget(self.volume_view,0,0)
        self.slice_panels=[]
        self.slices=[]
        self.slice_titles=[]
        self.slice_sliders=[]
        for index,(row,column) in enumerate(((0,1),(1,0),(1,1))):
            panel=QWidget()
            column_layout=QVBoxLayout(panel)
            column_layout.setContentsMargins(3,3,3,3)
            title=QLabel()
            heading=QHBoxLayout(); heading.addWidget(title,1)
            column_layout.addLayout(heading)
            canvas=ProbePlacementSlice() if index==0 else InsertionSliceCanvas()
            canvas.background_color='#000000'
            canvas.show_axes=False
            canvas.setSizePolicy(QSizePolicy.Policy.Ignored,QSizePolicy.Policy.Ignored)
            heading.addWidget(view_button('fit','Reset slice view',canvas.fit_image))
            heading.addWidget(view_button('rotate','Rotate slice 90°',canvas.rotate_view))
            from .desktop_figures import figure_button
            heading.addWidget(figure_button(canvas))
            canvas.tools.toolSelected.connect(lambda tool,i=index:self._canvas_tool_selected(i,tool))
            canvas.trackPicked.connect(self.select_track_by_id)
            canvas.measurementAdded.connect(lambda start,end,i=index:self.measure_distance(start,end,self._measurement_anchor(i)))
            canvas.pointAdded.connect(self.add_point)
            canvas.pointsSelected.connect(self.select_points)
            canvas.pointsDeleted.connect(self.delete_points)
            canvas.annotationContext.connect(self._annotation_context)
            canvas.vertexAdded.connect(self.add_vertex)
            if index==0:
                self.track_draw_button=action("Draw new track",self.new_track)
                self.track_draw_button.setObjectName("primary")
                self.drawing_notice=note("Choose Draw new track, then drag from the deep tip toward the brain surface.")
                # Changing feedback text must never resize the image underneath
                # the cursor while a track is being clicked or dragged.
                self.drawing_notice.setFixedHeight(54)
                canvas.trackPreview.connect(self.preview_track)
                canvas.trackDragged.connect(self.draw_track_gesture)
                canvas.trackShifted.connect(self.shift_track)
                canvas.drawingCancelled.connect(self.finish_track)
            canvas.hovered.connect(self._hover)
            canvas.viewChanged.connect(lambda c=canvas:self._sync_views(c))
            column_layout.addWidget(canvas,1)
            slider=QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0,1000)
            slider.valueChanged.connect(lambda value,i=index:self._slice_position(i,value))
            column_layout.addWidget(slider)
            self.view_grid.addWidget(panel,row,column)
            self.slice_panels.append(panel)
            self.slices.append(canvas)
            self.slice_titles.append(title)
            self.slice_sliders.append(slider)
        self.track_slice,self.brain_slice,self.atlas_slice=self.slices
        volume_box=QGroupBox('3D layers / opacity')
        self.volume_box=volume_box
        volume_box.setSizePolicy(QSizePolicy.Policy.Preferred,QSizePolicy.Policy.Maximum)
        volume_layout=QVBoxLayout(volume_box)
        self.volume_view.layout.removeWidget(self.volume_view.layers)
        volume_layout.addWidget(self.volume_view.layers)
        self.volume_view.layout.removeWidget(self.volume_view.zoom_controls)
        volume_layout.addWidget(self.volume_view.zoom_controls)
        self.volume_view.layout.removeWidget(self.volume_view.plane_controls)
        volume_layout.addWidget(self.volume_view.plane_controls)
        self.volume_view.show_slice_planes.setChecked(True)
        self.volume_footer=QWidget()
        self.volume_footer.setFixedHeight(self.slice_sliders[0].sizeHint().height())
        self.volume_view.layout.addWidget(self.volume_footer)
        self.volume_view.settings_button.hide()
        display_layout.addWidget(volume_box)
        from .desktop_rendering import RenderSettingsPanel
        self.rendering_box=QGroupBox('3D rendering settings')
        self.rendering_box.setCheckable(True)
        rendering_layout=QVBoxLayout(self.rendering_box)
        self.rendering_panel=RenderSettingsPanel(self.volume_view,include_opacity=False,compact=True,minimal=True)
        self.volume_view.settings_dialog=self.rendering_panel
        self.volume_view.inline_settings_handler=self._show_rendering
        rendering_layout.addWidget(self.rendering_panel)
        self.rendering_box.toggled.connect(self.rendering_panel.setVisible)
        self.rendering_box.setChecked(False)
        display_layout.addWidget(self.rendering_box)
        from .desktop_slice_display import SliceDisplayControls
        self.slice_display=SliceDisplayControls(channels=('Electrodes scan','Brain scan','Atlas'),minimal=True)
        self.slice_display.changed.connect(self._slice_appearance_changed)
        self.slice_display.autoRequested.connect(self._auto_slice_contrast)
        self.slice_settings_page=QWidget()
        slice_settings=QVBoxLayout(self.slice_settings_page)
        self.slice_settings_layout=slice_settings
        slice_settings.addWidget(self.slice_display)
        slice_settings.addStretch()
        slice_scroll=QScrollArea(); slice_scroll.setWidgetResizable(True); slice_scroll.setWidget(self.slice_settings_page)
        self.sidebar_tabs.addTab(slice_scroll,'Slices')
        self.plane_slice=self.track_slice
        self.plane_info=note("")
        self.controls.addWidget(self._point_controls())
        self.plane_panel=self._plane_controls(include_visibility=False,include_display=False)
        self.opacity=self.volume_view.layers.sliders
        for slider,value in zip(self.opacity,[0,100,25]): slider.setValue(value)
        self.volume_view.layers.checks[0].setChecked(False)
        self.volume_view.layers.checks[2].setChecked(True)
        self.volume_view.layersRequested.connect(self._ensure_3d_labels)
        self.volume_view.viewSettingsChanged.connect(lambda:self.owner.schedule_save() if not self._updating else None)
        self.plane_tabs=QTabWidget()
        self.plane_tabs.addTab(self._track_controls(),"Tracks")
        self.plane_tabs.addTab(self.plane_panel,"Plane")
        regions=self._region_controls(include_display=False)
        region_scroll=QScrollArea(); region_scroll.setWidgetResizable(True); region_scroll.setWidget(regions)
        self.region_scroll=region_scroll
        self.sidebar_tabs.addTab(region_scroll,'Regions')
        self.region_group=QGroupBox('Regions')
        self.region_group.setCheckable(True)
        QVBoxLayout(self.region_group)
        self.region_group.toggled.connect(self._tracing_regions_toggled)
        self.region_group.setChecked(False)
        self.region_group.hide()
        display_layout.insertWidget(2,self.region_group)
        self.sidebar_tabs.currentChanged.connect(self._sidebar_tab_changed)
        self.plane_tabs.addTab(self._report_controls(),"Report")
        self.controls.addWidget(self.plane_tabs)
        self.controls.addWidget(self._review_controls())
        self.controls.addWidget(self._xyz_controls())
        self.controls.addWidget(self._probe_controls())
        controls.addWidget(self.controls,1)
        self.select_3d=QCheckBox("Select points in 3D")
        self.select_3d.setToolTip("Click a point, or drag a rectangle to select through the volume. Ctrl/⌘ extends selection. Turn off to rotate.")
        self.select_3d.toggled.connect(self.volume_view.set_selection_enabled)
        controls.insertWidget(1,self.select_3d)
        self.show_all=QCheckBox("Show all insertions")
        self.show_all.setChecked(True)
        self.show_all.toggled.connect(self.draw_cached)
        self.show_points_check=QCheckBox("Show insertion points")
        self.show_points_check.setChecked(True)
        self.show_points_check.toggled.connect(self.draw_cached)
        visibility=QGroupBox("Insertion visibility")
        self.visibility_box=visibility
        visibility.setSizePolicy(QSizePolicy.Policy.Preferred,QSizePolicy.Policy.Maximum)
        visibility_layout=QVBoxLayout(visibility)
        self.visibility_table=QTableWidget(0,3)
        self.visibility_table.setHorizontalHeaderLabels(["Insertion","Points","Tracks"])
        self.visibility_table.verticalHeader().hide()
        self.visibility_table.verticalHeader().setDefaultSectionSize(25)
        self.visibility_table.setMinimumHeight(75)
        self.visibility_table.setMaximumHeight(135)
        self.visibility_table.horizontalHeader().setSectionResizeMode(0,QHeaderView.ResizeMode.Stretch)
        for col in (1,2): self.visibility_table.setColumnWidth(col,65)
        self.visibility_table.itemChanged.connect(self._annotation_visibility_changed)
        visibility_layout.addWidget(self.visibility_table)
        visibility_layout.addWidget(self.show_all)
        visibility_layout.addWidget(self.show_points_check)
        display_layout.addWidget(visibility)
        display_layout.addStretch()
        form=QFormLayout()
        self.slab=QDoubleSpinBox()
        self.slab.setRange(1,2000)
        self.slab.setValue(50)
        self.slab.setSuffix(" µm")
        self.slab.valueChanged.connect(self.draw_cached)
        form.addRow("Track slab thickness",self.slab)
        self.slab_label=form.labelForField(self.slab)
        controls.addLayout(form)
        self.control_content.setMinimumWidth(330)
        self.control_content.setMaximumWidth(16777215)
        self.splitter.addWidget(self.control_content)
        self.splitter.addWidget(self.view_container)
        self.splitter.setStretchFactor(1,1)
        self.splitter.setSizes([340,1000])
        layout.addWidget(self.splitter,1)
        self.hover_info=note("Coordinates: physical atlas axes · Z/Y/X µm")
        layout.addWidget(self.hover_info)
        row=QHBoxLayout()
        self.back_button=action("Back",lambda:owner.go_step(max(3,owner.step-1)))
        row.addWidget(self.back_button)
        row.addStretch()
        self.apply_button=action('Apply',lambda:self.apply_placement() if self._mode==1 else self.apply_tracing())
        self.save_button=action('Save',owner.save_project)
        row.addWidget(self.apply_button); row.addWidget(self.save_button)
        self.next_button=action("Compute planes →",self.next_stage)
        self.next_button.setObjectName('primary')
        row.addWidget(self.next_button)
        self.export_button=action("Export insertions && tracks",self.export_analysis)
        row.addWidget(self.export_button)
        layout.addLayout(row)
        toggle=QAction('Toggle settings panel',self)
        toggle.triggered.connect(self.sidebar_toggle.toggle)
        self.addAction(toggle)
        self._build_placement_workspace(display_layout)
        flat_tabs(self.sidebar_tabs)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground,True)
        self.setObjectName('insertionWorkspace')
        self.setStyleSheet('''QWidget#insertionWorkspace {background:#242b36;}
            QWidget#insertionWorkspace QLabel, QWidget#insertionWorkspace QSlider {background:transparent;}
            QScrollArea {border:0;}
            QTableWidget {background:transparent;border:0;}
            QHeaderView::section {background:transparent;border:0;color:#9aabc0;padding:5px;}
            QTableCornerButton::section {background:transparent;border:0;}''')

    def _show_rendering(self):
        self.sidebar_tabs.setCurrentIndex(1)
        self.rendering_box.setChecked(True)
        self.rendering_panel.sync()

    def _toggle_sidebar(self,visible):
        if not hasattr(self,'control_content'): return
        self._stabilize_layout()
        if not visible: self._sidebar_sizes=self.splitter.sizes()
        self.control_content.setVisible(visible)
        if visible: self.splitter.setSizes(self._sidebar_sizes)
        self.owner.schedule_save()

    def _stabilize_layout(self):
        """Finish grid/visibility changes before Cocoa paints the GL framebuffer."""
        self.view_container.setUpdatesEnabled(False)
        if self.volume_view.canvas is not None:
            self.volume_view.gl_guard.suspended=True
            self.volume_view.canvas.native.setUpdatesEnabled(False)
        def ready():
            if self._closing: return
            self.view_grid.activate()
            self.view_container.setUpdatesEnabled(True)
            if self.volume_view.canvas is not None:
                self.volume_view.gl_guard.suspended=False
                self.volume_view.canvas.native.setUpdatesEnabled(True)
                self.volume_view.canvas.update()
        QTimer.singleShot(0,ready)

    def _sidebar_tab_changed(self,index):
        if index==3 and self.project is not None: self._ensure_labels()

    def _tracing_regions_toggled(self,visible):
        self.region_scroll.setVisible(visible)
        if visible and self.project is not None: self._ensure_labels()

    def _slice_appearance_changed(self):
        self.draw_cached()
        self.owner.schedule_save()

    def _auto_slice_contrast(self,channel):
        if self._cache is None: return
        images=self._cache.get('images',[])
        values=[np.asarray(image[0]).ravel() for index,image in enumerate(images) if
                (index if self._mode in (1,4) else min((self.slice_source if self._mode==0 else self.xyz_source).currentIndex(),2))==channel]
        if values: values=np.concatenate(values)
        else:
            volume=(self.track_native,self.brain_native,self.atlas_native)[channel]
            if volume is None: return
            values=volume.data.ravel()[::max(1,volume.data.size//500000)]
        values=values[np.isfinite(values)&(values>0)]
        if not len(values): return
        low,high=np.percentile(values,(1,99.8))
        if high>low:
            self.slice_display.records[channel]['contrast']=[float(low),float(high)]
            self.slice_display.sync(); self._slice_appearance_changed()

    def _configure_tools(self):
        allowed=(0,1,2,4) if self._mode==0 else ((2,4) if self._mode==4 else (2,3,4))
        for index in range(self.point_tool.count()): self.point_tool.model().item(index).setEnabled(index in allowed)
        index=self._tool_modes.get(str(self._mode),0 if self._mode==0 else 2)
        if index not in allowed: index=allowed[0]
        self.point_tool.blockSignals(True)
        self.point_tool.setCurrentIndex(index)
        self.point_tool.blockSignals(False)
        fallback={0:'add',1:'select',2:'pan',3:'track',4:'measure'}[index]
        self._canvas_tools=list(self._canvas_tool_modes.get(str(self._mode),[fallback]*3))
        if self._mode==0: self._canvas_tools=[self._canvas_tools[0]]*3
        for i,canvas in enumerate(self.slices):
            canvas.tools.configure(self._mode,i)
            allowed={'pan','measure'}|({'add','select'} if self._mode==0 else (set() if self._mode==4 else {'track'}))
            if self._mode==1 and i==0: allowed.add('draw')
            if self._canvas_tools[i] not in allowed: self._canvas_tools[i]='pan'
        self._apply_slice_tools()

    def _apply_slice_tools(self):
        if self._mode==1:
            for canvas in self.slices: canvas.tools.hide()
            return
        for index,canvas in enumerate(self.slices):
            selected='draw' if canvas.drawing else self._canvas_tools[index]
            if selected=='draw' and not canvas.drawing: selected='pan'
            tool='annotate' if selected in ('add','select','draw') else selected
            canvas.set_interaction_tool(tool)
            canvas.point_mode=selected if selected in ('add','select') and self._mode==0 else 'view'
            canvas.tools.set_active(selected)
            canvas.tools.setVisible(self._mode!=0)
            hint=('Click: mark insertion points' if self._mode==0 else
                  ('Draw: deep tip first, then toward the brain surface' if self._mode==1 else 'Inspect scan and reconstruction'))
            canvas.setToolTip(hint+'\nWheel: zoom · Shift+drag: pan · Right-drag: zoom'+
                              (' · Slider: slice' if self._mode in (0,3) else ''))
        if self._mode==0:
            selected=self._canvas_tools[0]
            for name,button in self.tracing_tools.items(): button.setChecked(name==selected)
            self.select_3d.setChecked(selected=='select')

    def _canvas_tool_selected(self,index,tool):
        allowed={'pan','measure'}|({'add','select'} if self._mode==0 else (set() if self._mode==4 else {'track'}))
        if self._mode==1 and index==0: allowed.add('draw')
        if tool not in allowed: return
        if index==0 and tool!='draw': self._cancel_drawing()
        self._canvas_tools[index]=tool
        if tool=='draw': self.new_track()
        self._apply_slice_tools()
        self._canvas_tool_modes[str(self._mode)]=list(self._canvas_tools)
        self.draw_cached(annotations_only=True)
        self.owner.schedule_save()

    def _slice_tool_changed(self):
        if self._updating: return
        self._cancel_drawing()
        self._track_move_preview=None
        self._tool_modes[str(self._mode)]=self.point_tool.currentIndex()
        self._canvas_tools=[{0:'add',1:'select',2:'pan',3:'track',4:'measure'}[self.point_tool.currentIndex()]]*3
        self._apply_slice_tools()
        self.draw_cached(annotations_only=True)
        self.owner.schedule_save()

    def _measurement_anchor(self,index):
        if self._mode in (1,4): return ('track','moving','reference')[index]
        source=(self.slice_source if self._mode==0 else self.xyz_source).currentIndex()
        return ('track','moving','reference','reference')[source]

    def measure_distance(self,start,end,anchor='track'):
        if self.session is None: return
        if self._mode in (0,1): start,end=transform_points([start,end],self._tracing_to_atlas())
        self._remember()
        try: record=add_measurement(self.session,start,end,anchor=anchor)
        except ValueError as error:
            self.owner.statusBar().showMessage(str(error))
            return
        self._refresh_measurements()
        self.draw_cached(annotations_only=True)
        self.owner.schedule_save()
        self.owner.statusBar().showMessage(f"Distance in source scan space: {record['length_um']:.1f} µm ({record['length_um']/1000:.3f} mm)")

    def _point_controls(self):
        panel=QWidget()
        layout=QVBoxLayout(panel)
        layout.setContentsMargins(4,4,4,4)
        self._tracing_controls(layout)
        self.slice_source=QComboBox()
        self.slice_source.addItems(["Electrodes scan","Brain scan"])
        self.slice_source.currentIndexChanged.connect(self._source_changed)
        self.slice_settings_layout.insertWidget(0,self.slice_source)
        self.point_gamma=QDoubleSpinBox()
        self.point_gamma.setRange(.1,5)
        self.point_gamma.setValue(1)
        self.point_gamma.setSingleStep(.1)
        self.point_gamma.valueChanged.connect(self.draw_cached)
        self.point_gamma.hide()
        self.fit_info=note("Plane fits are computed when you continue, not after every click.")
        self.fit_info.hide()
        return panel

    def _track_controls(self):
        panel=QWidget()
        layout=QVBoxLayout(panel)
        layout.addWidget(self.track_draw_button)
        layout.addWidget(self.drawing_notice)
        self.plane_status=note("")
        plane_layout=self.plane_panel.layout()
        plane_layout.insertWidget(0,self.plane_status)
        plane_layout.insertWidget(1,action("Save plane for insertion",self.save_plane))
        plane_layout.insertWidget(2,action("Recompute plane from points",self.refit_plane))
        self.track_editor=QGroupBox('Tracks in active insertion')
        self.track_editor_layout=QVBoxLayout(self.track_editor)
        editor=self.track_editor_layout
        self.track_list=QListWidget()
        self.track_list.setMaximumHeight(160)
        self.track_list.currentRowChanged.connect(self._select_track)
        editor.addWidget(self.track_list)
        self.track_name=QLineEdit()
        self.track_name.setPlaceholderText("Track name")
        self.track_name.editingFinished.connect(self._rename_track)
        editor.addWidget(self.track_name)
        self.track_move_target=QComboBox()
        editor.addWidget(QLabel('Transfer selected track to insertion'))
        editor.addWidget(self.track_move_target)
        editor.addWidget(action('Move track to insertion',self.move_selected_track))
        self.edit_track_button=action('Redraw selected track',self.edit_track)
        editor.addWidget(self.edit_track_button)
        editor.addWidget(action('Delete selected track',self.delete_track))
        editor.addWidget(action('Undo edit',self.undo))
        self.track_editor_holders={1:layout}
        layout.addWidget(self.track_editor)
        layout.addWidget(note('Select / move track: click a line to select; drag it on the electrode slice to translate the whole 4 mm track.'))
        layout.addWidget(action('Cancel drawing',self.finish_track))
        box=QGroupBox("Start inside brain · estimated foreground mask")
        box.setCheckable(True)
        mask_layout=QVBoxLayout(box)
        mask_controls=QWidget()
        form=QFormLayout(mask_controls)
        mask_layout.addWidget(mask_controls)
        box.toggled.connect(mask_controls.setVisible)
        box.setChecked(False)
        self.mask_threshold=QDoubleSpinBox()
        self.mask_threshold.setRange(0,.999)
        self.mask_threshold.setDecimals(4)
        self.mask_threshold.setSingleStep(.01)
        self.mask_threshold.setKeyboardTracking(False)
        self.mask_threshold.valueChanged.connect(self._mask_changed)
        form.addRow("Brain threshold (0–1)",self.mask_threshold)
        form.addRow(action("Auto threshold",lambda:self._rebuild_mask(None)))
        self.show_mask=QCheckBox("Show mask on brain slice")
        self.show_mask.toggled.connect(self.request_render)
        form.addRow(self.show_mask)
        form.addRow(action("Recheck all track starts",self.validate_starts))
        layout.addWidget(box)
        layout.addWidget(note("The mask is an editable intensity-based estimate, not an anatomical ground truth. Check it on the brain slice before accepting the start. Points outside it cannot start a track."))
        layout.addWidget(self.plane_info)
        layout.addStretch()
        return panel

    def _review_controls(self):
        panel=QWidget()
        layout=QVBoxLayout(panel)
        layout.addWidget(note("Directed tracks are 4000 µm long in source scan space, from the marked start. All tracks and points in one insertion share its color."))
        self.track_editor_holders[2]=layout
        self.track_summary=note("")
        layout.addWidget(self.track_summary)
        layout.addStretch()
        return panel

    def _xyz_controls(self):
        panel=QWidget()
        layout=QVBoxLayout(panel)
        layout.addWidget(QLabel('Slice background'))
        self.xyz_source=QComboBox()
        self.xyz_source.addItems(["Track scan","Brain scan","Atlas","Atlas + labels"])
        self.xyz_source.currentIndexChanged.connect(self._source_changed)
        layout.addWidget(self.xyz_source)
        self.xyz_gamma=QDoubleSpinBox()
        self.xyz_gamma.setRange(.1,5)
        self.xyz_gamma.setValue(1)
        self.xyz_gamma.setSingleStep(.1)
        self.xyz_gamma.valueChanged.connect(self.draw_cached)
        form=QFormLayout()
        self.xyz_gamma.hide()
        layout.addWidget(action('Center slices on selected track',self.center_track))
        layout.addWidget(note('Use the sliders for depth. Dots mark physical intersections with the saved 4 mm tracks.'))
        self.track_editor_holders[3]=layout
        layout.addStretch()
        return panel

    def stop(self):
        super().stop()
        self._reset_surface_entries()

    def reset(self):
        self._reset_surface_entries()
        self._hide_placement_recordings(dispose=True)
        for mesh in [*getattr(self,'_placement_visuals',{}).values(),*getattr(self,'_placement_contact_visuals',{}).values()]: mesh.parent=None
        self._placement_contact_visuals={}
        self._placement_visuals={}
        self._dispose_probe_visuals()
        super().reset()
        self._cancel_drawing()
        self._label_loading=False
        self.brain_native=self.brain_preview=self.brain_mask=None
        self.insertion_id=None
        self.selected_points.clear()
        self._ortho_positions=None
        self._tracing_positions=None
        self._track_move_preview=None
        self._tool_modes={}

    def current_insertion(self):
        if self.session is None: return None
        return next((item for item in self.session["insertions"] if item["id"]==self.insertion_id),None)

    def open(self,project,reference_preview,mode):
        project.validate_track()
        self._stabilize_layout()
        session=insertion_session(project)
        context=(id(project),session["id"])
        self.capture_state()
        self._track_move_preview=None
        if mode!=self._mode: self.sidebar_tabs.setCurrentIndex(0)
        self._mode=mode
        if mode>0 and session['insertions'] and not any(i['id']==self.insertion_id for i in session['insertions']):
            self.insertion_id=session['insertions'][0]['id']
        self.control_content.setMinimumWidth(400 if mode==4 else 330)
        self.probe_display_box.setVisible(mode==4)
        self.controls.setCurrentIndex(mode)
        if mode==0:
            self.annotation_splitter.addWidget(self.measurement_panel)
        else:
            self.work_layout.addWidget(self.measurement_panel)
        self.measurement_panel.setVisible(mode!=2)
        if mode in (1,2,3):
            for holder in self.track_editor_holders.values(): holder.removeWidget(self.track_editor)
            holder=self.track_editor_holders[mode]
            holder.insertWidget(2 if mode==1 else (4 if mode==3 else 1),self.track_editor)
            self.track_editor.show()
        self.track_editor.setVisible(mode in (2,3))
        self.edit_track_button.setVisible(False)
        self.title.setText(("Tracks","Probes","3D tracks · 4 mm in scan space","Tracks on XYZ slices","Neuropixels reconstruction")[mode])
        self.next_button.setText(("Next","Next","Inspect XYZ slices →","Reconstruct Neuropixels →","")[mode])
        self.next_button.setVisible(mode<4)
        self.new_insertion_button.hide()
        self.insertion_box.setVisible(mode not in (0,1))
        self.visibility_box.setVisible(mode!=0)
        self.visibility_table.horizontalHeaderItem(2).setText('Probe' if mode==1 else 'Tracks')
        self.show_all.setVisible(mode!=1)
        self.show_points_check.setVisible(mode!=1)
        self.tracing_toolbar.setVisible(mode==0)
        self.apply_button.setVisible(mode in (0,1))
        self.info.setVisible(mode not in (0,1))
        self.hover_info.setVisible(mode not in (0,1))
        self.sidebar_tabs.setTabText(0,'Points' if mode==0 else ('Placement' if mode==1 else 'Work'))
        region_index=self.sidebar_tabs.indexOf(self.region_scroll)
        if mode in (0,1):
            if region_index>=0: self.sidebar_tabs.removeTab(region_index)
            self.region_group.layout().addWidget(self.region_scroll)
            self.region_group.show()
            self.region_scroll.setVisible(self.region_group.isChecked())
        else:
            self.region_group.layout().removeWidget(self.region_scroll)
            if region_index<0: self.sidebar_tabs.addTab(self.region_scroll,'Regions')
            self.region_group.hide()
        self.slice_source.setVisible(mode==0)
        self.region_opacity.setVisible(mode!=0)
        self.region_opacity_label.setVisible(mode!=0)
        self.volume_view.zoom_controls.setVisible(mode not in (0,1))
        self.volume_footer.setVisible(mode in (0,3))
        self.volume_view.plane_controls.show()
        self.volume_view.show_slice_planes.setVisible(mode==3)
        self.volume_view.fill_slice_planes.setVisible(mode==3)
        self.volume_view.heading.setText("3D" if mode in (0,1) else "3D · drag to rotate · wheel to zoom")
        self.volume_view.reset_button.setText("Fit visible content")
        self.export_button.setVisible(mode>1)
        self.visibility_table.setColumnHidden(2,mode==0)
        self.slab.setVisible(False)
        self.slab_label.setVisible(False)
        self.select_3d.setText('Select points in 3D' if mode==0 else 'Select tracks in 3D')
        self.select_3d.setToolTip('Click to select · Drag background: rotate · Ctrl/⌘+drag: select a rectangle · Shift+drag: pan')
        self.volume_view.selection_mode='points' if mode==0 else 'tracks'
        self.select_3d.setChecked(False)
        self.select_3d.setVisible(mode not in (0,1,4))
        self._cancel_drawing(discard=False)
        self._hide_probe_visuals()
        self.export_button.setText("Export channels && probe" if mode==4 else "Export insertions && tracks")
        self.track_draw_button.setVisible(False)
        self.drawing_notice.setVisible(False)
        if self.volume_view.plane_mesh is not None:
            self.volume_view.plane_mesh.visible=self.volume_view.plane_outline.visible=False
        for mesh in [*getattr(self,'_placement_visuals',{}).values(),*getattr(self,'_placement_contact_visuals',{}).values()]: mesh.visible=False
        self.view_grid.removeWidget(self.volume_view)
        self.view_grid.addWidget(self.volume_view,0,0,2 if mode==2 else 1,2 if mode==2 else 1)
        for index,(panel,slider) in enumerate(zip(self.slice_panels,self.slice_sliders)):
            self.view_grid.removeWidget(panel)
            self.view_grid.addWidget(panel,*((0,1),(1,0),(1,1))[index])
            panel.setVisible(mode!=2 and (mode!=1 or index==0))
            slider.setVisible(mode in (0,3))
        for canvas in self.slices:
            canvas.click_navigation=mode==0
        self.view_grid.setRowStretch(0,1)
        self.view_grid.setRowStretch(1,0 if mode in (1,2) else 1)
        self.view_grid.setColumnStretch(0,1)
        self.view_grid.setColumnStretch(1,0 if mode==2 else 1)
        self._configure_placement_page()
        if context==self._context and self.track_native is not None:
            if self.labels is not None: self._populate_regions()
            self._set_bounds()
            self._configure_tools()
            self._restore_slice_views()
            if mode==1:
                self.compute_planes()
                self._restore_insertion_plane()
                self._placement_refresh_controls()
            self._update_volume()
            self._refresh_insertions()
            self._refresh_tracks()
            if mode==4:
                self._refresh_probe_controls()
                self._ensure_labels()
            self.request_render()
            if mode==1: self._placement_restore_camera()
            if mode!=1 and not self.session.get("insertion_view",{}).get("camera"):
                self.volume_view.reset_view()
            return
        self.project,self.session=project,session
        self._context=context
        self._token+=1
        self.labels=self._labels_path=self._cache=None
        self._undo=copy.deepcopy(session.get('edit_history',[]))
        display_brain=self.owner.moving
        def load(progress):
            progress(5,"Loading electrode and brain channels")
            native,gpu,limits=load_track_data(project.track,project.preview_dimension)
            brain=raw_volume(project.moving)
            atlas=raw_volume(project.reference)
            # Display quality must not change the mask used for entry surfaces
            # and start validation. Reconstruct its saved sampling lattice.
            from .desktop_model import load_preview
            from .volumes import read_volume
            saved_spacing=session.get('brain_mask',{}).get('preview_spacing_um_zyx')
            if saved_spacing is not None:
                factors=np.maximum(1,np.rint(np.asarray(saved_spacing)/project.moving.spacing_um_zyx).astype(int))
                brain_preview=read_volume(project.moving.path,project.moving.spacing_um_zyx,factors,normalize=True)
            else:
                brain_preview=load_preview(project.moving,256)
            def limits_for(volume):
                data=np.asarray(volume.data).ravel()[::max(1,volume.data.size//500000)]
                data=data[np.isfinite(data)]
                low,high=np.percentile(data,(1,99.8))
                return [float(low),float(max(high,low+1))]
            threshold=session.get("brain_mask",{}).get("threshold")
            mask,value=brain_support(brain_preview,threshold)
            return native,gpu,limits,brain,atlas,limits_for(brain),limits_for(atlas),mask,value,brain_preview
        def ready(result):
            (self.track_native,self.track_gpu,self.track_limits,self.brain_native,self.atlas_native,
             self.brain_limits,self.atlas_limits,self.brain_mask,threshold,self.brain_preview)=result
            self.volume_view.load_channels(reference_preview,self.track_gpu,display_brain)
            self._restore_state()
            self._configure_tools()
            self._configure_placement_page()
            self.mask_threshold.blockSignals(True)
            self.mask_threshold.setValue(threshold)
            self.mask_threshold.blockSignals(False)
            self.session["brain_mask"]={"threshold":threshold,"method":"Smoothed Otsu / largest foreground / filled envelope",
                                         "preview_spacing_um_zyx":list(self.brain_preview.spacing_um_zyx)}
            self._set_bounds()
            if not self.session["insertions"]: new_insertion(self.session)
            saved_view=self.session.get('insertion_view',{})
            if self.current_insertion() is None and (self._mode>0 or 'selected_insertion_id' not in saved_view or saved_view['selected_insertion_id'] is not None):
                self.insertion_id=self.session['insertions'][0]['id']
            self._refresh_insertions()
            if mode==1:
                self.compute_planes()
                self._restore_insertion_plane()
                self._placement_refresh_controls()
            self._update_volume()
            self.validate_starts()
            if self._mode==4: self._refresh_probe_controls()
            self.info.setText(f"Brain: {project.moving.path.name} · Electrodes: {project.track.path.name}")
            self.request_render()
            if mode==1: self._placement_restore_camera()
            if mode!=1 and not self.session.get("insertion_view",{}).get("camera"):
                self.volume_view.reset_view()
            if mode==4: self._ensure_labels()
            else: self.owner.statusBar().showMessage("Insertion workspace ready. Select an insertion and mark points on the slices.")
            if self.volume_view.layers.labels_check.isChecked(): self._ensure_3d_labels()
        self.owner._start_job(load,ready,"Preparing insertion workspace…")

    def _set_bounds(self):
        project=self.project
        self._volume_corner_sets=[transformed_corners(spec.shape_zyx,spec.spacing_um_zyx,matrix)
            for spec,matrix in ((project.reference,np.eye(4)),(project.moving,project.total),(project.track,project.total))]
        if self._mode==0:
            placement=self._tracing_matrix()
            corners=np.vstack([transformed_corners(spec.shape_zyx,spec.spacing_um_zyx,placement)
                               for spec in (project.moving,project.track)])
            if self._tracing_positions is None:
                source=(np.asarray(project.track.shape_zyx)-1)*project.track.spacing_um_zyx/2
                self._tracing_positions=transform_points(source[None],placement)[0]
        else: corners=np.vstack(self._volume_corner_sets)
        self.lower,self.upper=corners.min(0),corners.max(0)
        if self._ortho_positions is None:
            source=(np.asarray(project.moving.shape_zyx)-1)*project.moving.spacing_um_zyx/2
            self._ortho_positions=transform_points(source[None],project.total)[0]
        if self._mode==3 and not self._xyz_initialized and self.session['tracks']:
            track=self.current_track() or self.session['tracks'][0]
            self._ortho_positions=np.mean(track['points_atlas_zyx_um'],axis=0)
            self._xyz_initialized=True
        self.depth_spacing=(np.full(3,min(project.track.spacing_um_zyx)) if self._mode==0 else
                            np.asarray(project.reference.spacing_um_zyx,float))
        self.depth_shape=np.ceil((self.upper-self.lower)/self.depth_spacing).astype(int)+1
        for axis,slider in enumerate(self.slice_sliders):
            slider.blockSignals(True)
            slider.setRange(0,int(self.depth_shape[axis]-1))
            slider.setValue(round((self._slice_positions()[axis]-self.lower[axis])/self.depth_spacing[axis]))
            slider.blockSignals(False)
            self._slice_positions()[axis]=self.lower[axis]+slider.value()*self.depth_spacing[axis]

    def center_track(self):
        if self.session is None or not self.session['tracks']: return
        track=self.current_track() or self.session['tracks'][0]
        self._ortho_positions=np.mean(track['points_atlas_zyx_um'],axis=0)
        self._xyz_initialized=True
        for canvas in self.slices: canvas.fit_image()
        self._set_bounds(); self.request_render()

    def _source_changed(self):
        if self._updating: return
        source=(self.slice_source if self._mode==0 else self.xyz_source).currentIndex()
        if source==3: self._ensure_labels()
        else: self.request_render()

    def _ensure_labels(self):
        if self.project is None: return
        if self.project.labels_dir is None:
            self.owner.statusBar().showMessage("Add a labels folder on Project for region overlays and reports.")
            self.request_render()
            return
        if self.owner.busy:
            QTimer.singleShot(150,self._ensure_labels)
            return
        super()._ensure_labels()

    def _populate_regions(self):
        super()._populate_regions()
        self._filter_regions(self.region_search.text())
        if self.volume_view.layers.labels_check.isChecked(): self._ensure_3d_labels()

    def _ensure_3d_labels(self):
        if self._closing or self.project is None or self.track_native is None or self._label_loading: return
        if not self.volume_view.layers.labels_check.isChecked() or self.volume_view.label_visuals: return
        if self.project.labels_dir is None:
            self.volume_view.layers.labels_check.setToolTip("Add a labels folder on Project first.")
            return
        if self.owner.busy:
            QTimer.singleShot(150,self._ensure_3d_labels)
            return
        project=self.project
        self._label_loading=True
        def load(progress):
            from .desktop_layers import label_surfaces
            labels=self.labels or load_label_atlas(project.labels_dir,project.reference,
                    Path(tempfile.gettempdir())/"probemap"/"native-labels",progress,spacing_um_zyx=project.label_spacing)
            return labels,label_surfaces(labels)
        def ready(result):
            self._label_loading=False
            if self.project is not project: return
            self.labels,meshes=result
            self._labels_path=str(project.labels_dir.resolve())
            super(InsertionWorkspace,self)._populate_regions()
            self._filter_regions(self.region_search.text())
            self.volume_view.set_label_meshes(meshes,self.session['regions'])
            self.request_render()
            self.owner.statusBar().showMessage(f"{len(meshes)} atlas region surfaces ready.")
        def failed(error):
            self._label_loading=False
            self.volume_view.layers.labels_check.setChecked(False)
        self.owner._start_job(load,ready,"Preparing atlas labels in 3D…",on_error=failed)

    def _refresh_insertions(self):
        if self.session is None: return
        self._updating=True
        for combo in (self.insertion_combo,self.move_target,self.track_move_target):
            previous=combo.currentData()
            combo.blockSignals(True)
            combo.clear()
            for item in self.session["insertions"]:
                combo.addItem(item["name"],item["id"])
            combo.setCurrentIndex(max(0,combo.findData(self.insertion_id if combo is self.insertion_combo else previous)))
            combo.blockSignals(False)
        self._populate_visibility()
        self._refresh_insertion_table()
        self._refresh_recording_library()
        self._updating=False
        if self._mode==0:
            self.show_all.setChecked(True)
            self.show_points_check.setChecked(True)
        self._update_plane_status()
        self._refresh_measurements()
        self.draw_cached()
        self.owner.schedule_save()

    def _choose_insertion(self,index):
        if self._updating or self.session is None: return
        self.capture_state()
        self._cancel_drawing(discard=False)
        self.insertion_id=self.insertion_combo.currentData()
        self.selected_id=None
        item=self.current_insertion()
        if self._mode==1:
            self._restore_insertion_plane()
            self._placement_refresh_controls()
            self.request_render()
        self._refresh_tracks()
        if self._mode==4:
            self._refresh_probe_controls()
            self.request_render()
        self._update_plane_status()
        self.draw_cached()

    def _ask_insertion_name(self,current=''):
        name,ok=QInputDialog.getText(self,'Insertion name','Name (any descriptive label):',text=current)
        return name.strip() if ok and name.strip() else None

    def _valid_insertion_name(self,name,excluding=None):
        if not name or not name.strip(): return False
        if any(item['id']!=excluding and item['name'].casefold()==name.strip().casefold() for item in self.session['insertions']):
            self.owner._error('An insertion with this name already exists. Choose a different name.')
            return False
        return True

    def create_insertion(self,name=None):
        if self.session is None: return
        if not isinstance(name,str): name=None
        if name is not None and not self._valid_insertion_name(name): return
        self.capture_state()
        self._cancel_drawing()
        self._remember()
        item=new_insertion(self.session,name.strip() if name else None)
        self.insertion_id=item["id"]
        self.selected_id=None
        if self._mode==1:
            self._restore_insertion_plane()
            self._placement_refresh_controls()
        self._refresh_insertions()
        self._refresh_tracks()

        if self._mode==0: self._tracing_tool_selected('add')

    def rename_insertion(self,name=None):
        item=self.current_insertion()
        if item is None: return
        if not isinstance(name,str): name=self._ask_insertion_name(item['name'])
        if not self._valid_insertion_name(name,item['id']) or name.strip()==item['name']: return
        self._remember()
        item["name"]=name.strip()
        sync_tracks(self.session,self.project)
        self._refresh_insertions()
        self._refresh_tracks()

    def color_insertion(self):
        item=self.current_insertion()
        if item is None: return
        color=QColorDialog.getColor(QColor(item["color"]),self,"Insertion color")
        if color.isValid():
            self._remember()
            item["color"]=color.name()
            sync_tracks(self.session,self.project)
            self._refresh_insertions()
            self._refresh_tracks()

    def delete_insertion(self):
        item=self.current_insertion()
        if item is None: return
        if item["points"] or item["tracks"]:
            self.owner.statusBar().showMessage("Move or delete this insertion's points and tracks first.")
            return
        self._remember()
        self.session.setdefault('unassigned_recordings',[]).extend(item.get('recordings',[]))
        self.session["insertions"].remove(item)
        if not self.session["insertions"]: new_insertion(self.session)
        self.insertion_id=self.session["insertions"][0]["id"]
        self._refresh_insertions()

    def add_point(self,world):
        item=self.current_insertion()
        if self._mode!=0 or item is None or not self._cache_current(): return
        self._remember()
        p=add_insertion_point(self.project,item,self._to_atlas([world])[0],anchor=self._measurement_anchor(0))
        self.selected_points={p["id"]}
        self._refresh_insertions()
        self.owner.statusBar().showMessage(f"Point added to {item['name']}. Plane will be computed when you continue.")

    def select_points(self,identifiers,additive=False):
        if self.session is None: return
        if additive:
            self.selected_points.symmetric_difference_update(identifiers)
        else: self.selected_points=set(identifiers)
        self.draw_cached()
        self.owner.statusBar().showMessage(f"{len(self.selected_points)} points selected. Choose a target insertion or create one from selection.")

    def go_to_point(self):
        if not self.selected_points: return
        point=next((p for item in self.session["insertions"] for p in item["points"] if p["id"] in self.selected_points),None)
        if point:
            if self._mode==0:
                self._tracing_positions=transform_points([point['atlas_zyx_um']],np.linalg.inv(self._tracing_to_atlas()))[0]
            else: self._ortho_positions=np.array(point["atlas_zyx_um"])
            self._set_bounds()
            self.request_render()

    def move_selected(self):
        if not self.selected_points or self.move_target.currentData() is None: return
        self._remember()
        move_points(self.session,self.selected_points,self.move_target.currentData())
        self._refresh_insertions()

    def group_selected(self,name=None,ask_name=False):
        if not self.selected_points: return
        if ask_name:
            name=self._ask_insertion_name()
            if not self._valid_insertion_name(name): return
        self._remember()
        item=new_insertion(self.session,name)
        move_points(self.session,self.selected_points,item["id"])
        self.insertion_id=item["id"]
        self._refresh_insertions()

    def delete_points(self):
        if self.session is None or not self.selected_points: return
        self._remember()
        for item in self.session["insertions"]:
            item["points"]=[p for p in item["points"] if p["id"] not in self.selected_points]
        self.selected_points.clear()
        self._refresh_insertions()

    def _visibility_changed(self,entry):
        if self._updating or self.session is None: return
        item=next(i for i in self.session["insertions"] if i["id"]==entry.data(Qt.ItemDataRole.UserRole))
        item["visible"]=entry.checkState()==Qt.CheckState.Checked
        item["points_visible"]=item["tracks_visible"]=item["visible"]
        self._populate_visibility()
        self.draw_cached()

    def _populate_visibility(self):
        if self.session is None: return
        self.visibility_table.blockSignals(True)
        self.visibility_table.setRowCount(len(self.session["insertions"]))
        for row,item in enumerate(self.session["insertions"]):
            name=QTableWidgetItem(item["name"])
            name.setFlags(Qt.ItemFlag.ItemIsEnabled|Qt.ItemFlag.ItemIsSelectable)
            name.setForeground(QColor(item["color"]))
            name.setToolTip(item["name"])
            self.visibility_table.setItem(row,0,name)
            for col,key in ((1,"points_visible"),(2,"probe_visible" if self._mode==1 else "tracks_visible")):
                check=QTableWidgetItem()
                check.setFlags(Qt.ItemFlag.ItemIsEnabled|Qt.ItemFlag.ItemIsUserCheckable)
                check.setData(Qt.ItemDataRole.UserRole,(item["id"],key))
                visible=item.get(key,True if key=='probe_visible' else item.get('visible',True))
                check.setCheckState(Qt.CheckState.Checked if visible else Qt.CheckState.Unchecked)
                check.setToolTip('Show probe' if key=='probe_visible' else 'Show points' if key=='points_visible' else 'Show tracks')
                self.visibility_table.setItem(row,col,check)
        self.visibility_table.blockSignals(False)

    def _annotation_visibility_changed(self,cell):
        if self.session is None or cell.column()==0: return
        identifier,key=cell.data(Qt.ItemDataRole.UserRole)
        item=next(item for item in self.session["insertions"] if item["id"]==identifier)
        # Materialize the other flag before updating the legacy master value.
        for flag in ("points_visible","tracks_visible"):
            item.setdefault(flag,item.get("visible",True))
        item[key]=cell.checkState()==Qt.CheckState.Checked
        if key=="tracks_visible" and not item[key] and identifier==self.insertion_id:
            self._cancel_drawing()
        item["visible"]=item["points_visible"] or item["tracks_visible"]
        if self._mode==1: self._refresh_placement_table()
        self.draw_cached(annotations_only=True)
        self.owner.schedule_save()

    def compute_planes(self):
        messages=[]
        for item in self.session["insertions"]:
            native_fit=(item.get('plane_fit') or {}).get('measurement_frame')=='source scan physical micrometres'
            if not plane_is_stale(item) and (native_fit or (item.get('plane_fit') or {}).get('manual')):
                from .probe_placement import source_plane
                source_plane(item,self.session['plane'],self.project.total)
                continue
            if (item.get("plane_fit") or {}).get("manual"):
                messages.append(f"{item['name']}: manual plane retained; points changed")
                continue
            try: fit_insertion_plane(item,self.session["plane"],self.project.total)
            except ValueError as error: messages.append(f"{item['name']}: {error}")
        self.fit_info.setText("\n".join(messages) if messages else "Insertion planes computed and cached. Inspect and adjust on the next step.")
        if messages: self.owner.statusBar().showMessage("Some insertions need a manual plane or additional noncollinear points.")
        self._update_plane_status()

    def _restore_insertion_plane(self):
        item=self.current_insertion()
        if item is None: return
        if self._mode==1:
            from .probe_placement import source_plane
            source_plane(item,self.session["plane"],self.project.total)
        if item.get("plane") is not None:
            self.session["plane"]=copy.deepcopy(item.get("plane_draft",item["plane"]))
        else:
            # An editable draft, explicitly not a mathematically determined fit.
            plane=copy.deepcopy(self.session["plane"])
            if item["points"]:
                plane["center_um_zyx"]=np.mean([p["atlas_zyx_um"] for p in item["points"]],axis=0).tolist()
            self.session["plane"]=plane
        self._populate_plane()
        self._update_plane_status()

    def _update_plane_status(self):
        item=self.current_insertion()
        if item is None: return
        fit=item.get("plane_fit") or {}
        text="No saved plane: add noncollinear points or adjust Plane controls and Save plane for insertion."
        if item.get("plane") is not None:
            text=f"{fit.get('method','Saved plane')}"
            if "rms_um" in fit: text+=f" · RMS {fit['rms_um']:.1f} µm · max {fit['max_error_um']:.1f} µm"
            if plane_is_stale(item): text+="\nPoints changed. Saved plane retained; recompute or save manually."
            if self._mode==1 and item["plane"]!=self.session["plane"]: text+="\nAdjusted plane: saved automatically when you start drawing a track."
        self.plane_status.setText(text)

    def refit_plane(self):
        item=self.current_insertion()
        if item is None: return
        try:
            self._remember()
            fit_insertion_plane(item,self.session["plane"],self.project.total)
            self._restore_insertion_plane()
            self._placement_refresh_controls()
            self.request_render()
        except ValueError as error: self.owner._error(str(error))

    def save_plane(self):
        item=self.current_insertion()
        if item is None: return
        self._remember()
        save_manual_plane(item,self.session["plane"])
        self._update_plane_status()
        self.owner.save_project()
        self.owner.statusBar().showMessage(f"Plane saved for {item['name']}. Existing track coordinates stay fixed.")

    def next_stage(self):
        if self._mode==0: self.compute_planes()
        if self._mode==1: self.apply_placement()
        self.owner.go_step(min(6,self.owner.step+1))

    def _slice_position(self,axis,value):
        if self._updating or self.project is None or self._slice_positions() is None: return
        self._slice_positions()[axis]=self.lower[axis]+value*self.depth_spacing[axis]
        self.request_render()

    def _slice_scroll(self,axis,delta):
        # Canvas navigation never changes a physical slice or insertion plane.
        return

    def _hover(self,world):
        if self._mode!=0: super()._hover(world)

    def _layers_changed(self,index):
        for slider,value in zip(self.opacity,((0,100,25),(0,100,0),(0,0,75),(30,80,30),(0,0,0))[index]):
            slider.setValue(value)
        self._update_volume()

    def _update_volume(self):
        if self._mode==1 and self.project is not None and self.track_native is not None:
            self._update_placement_volume()
            return
        if self.project is not None and self.track_native is not None:
            self.volume_view.update_alignment(self.project.total,*[slider.value()/100 for slider in self.opacity])

    def _mask_changed(self,value):
        if self._updating: return
        self._rebuild_mask(value)

    def _rebuild_mask(self,threshold):
        if self.brain_preview is None or self.owner.busy: return
        preview=self.brain_preview
        def ready(result):
            self.brain_mask,value=result
            self.mask_threshold.blockSignals(True)
            self.mask_threshold.setValue(value)
            self.mask_threshold.blockSignals(False)
            self.session["brain_mask"]["threshold"]=value
            self.validate_starts()
            self.request_render()
        self.owner._start_job(lambda progress:brain_support(preview,threshold),ready,"Updating the brain foreground mask…")

    def validate_starts(self):
        if self.session is None or self.brain_mask is None: return
        for item in self.session["insertions"]:
            for track in item["tracks"]:
                start=track["control_points_atlas_zyx_um"][0]
                valid=bool(inside_brain([start],self.brain_mask,self.brain_preview,self.project.total)[0])
                track["start_validation"]={"inside":valid,"method":"Editable brain foreground envelope",
                    "threshold":self.mask_threshold.value(),"preview_spacing_um_zyx":list(self.brain_preview.spacing_um_zyx)}
        sync_tracks(self.session,self.project)
        self._refresh_tracks()

    def _remember(self):
        if self.session is None: return
        self._last_plane_edit=0.
        self._undo.append(copy.deepcopy({key:self.session.get(key,[] if key in ('insertions','measurements','unassigned_recordings','recording_order') else {})
                                        for key in ('insertions','measurements','plane','regions','unassigned_recordings','recording_order')}))
        self._undo=self._undo[-50:]
        self.owner.schedule_save()

    def _remember_plane(self):
        if self._updating or self.session is None: return
        now=time.monotonic()
        if now-self._last_plane_edit>.35: self._remember()
        self._last_plane_edit=now

    def _pose_changed(self,name,axis,value):
        self._remember_plane()
        super()._pose_changed(name,axis,value)

    def _normal_changed(self,value):
        self._remember_plane()
        super()._normal_changed(value)

    def _preset_changed(self,index):
        if index!=3: self._remember_plane()
        super()._preset_changed(index)

    def _field_changed(self):
        self._remember_plane()
        super()._field_changed()

    def _region_changed(self,item):
        if self.labels is not None and self.session is not None: self._remember()
        super()._region_changed(item)
        self.owner.schedule_save()

    def _set_all_regions(self,visible):
        if self.labels is not None and self.session is not None: self._remember()
        super()._set_all_regions(visible)
        self.owner.schedule_save()

    def undo(self):
        if self.session is None or not self._undo: return
        if self._mode==1:
            self._placement_ghost=self._placement_move_base=self._placement_rotate_base=None
            self.track_slice._probe_rotating=False
            self.track_slice._probe_anchor=None
            self.track_slice._probe_moving=False
            self.track_slice.placement_selected=False
            self._placement_tool('select')
        self._last_plane_edit=0.
        self._cancel_drawing()
        previous=self._undo.pop()
        if isinstance(previous,list): self.session['insertions']=previous
        else:
            for key,value in previous.items(): self.session[key]=value
        if self.current_insertion() is None and self.session["insertions"]:
            self.insertion_id=self.session["insertions"][0]["id"]
        self.selected_id=None
        sync_tracks(self.session,self.project)
        self._refresh_insertions()
        self._refresh_tracks()
        self._refresh_measurements()
        if self._mode==4: self._refresh_probe_controls()
        if self.labels is not None: self._populate_regions()
        if self._mode==1:
            self._restore_insertion_plane()
            self._placement_refresh_controls()
        self.request_render()

    def current_track(self):
        item=self.current_insertion()
        return next((t for t in item["tracks"] if t["id"]==self.selected_id),None) if item else None

    def _drawing_message(self,text,error=False):
        self.drawing_notice.setText(text)
        self.drawing_notice.setToolTip(text)
        self.drawing_notice.setStyleSheet("color:#ffca99; background:#4a3030; padding:6px;" if error else "color:#b4e1f5; padding:3px;")
        self.owner.statusBar().showMessage(text)

    def _cancel_drawing(self,discard=True):
        if discard and self._mode==1 and self.session is not None and self.current_insertion() is not None:
            self.current_insertion().pop('drawing_draft',None)
        self._pending_draw=False
        self._drawing_geometry=None
        self._draft_start=self._preview_start=self._draft_direction=None
        self.track_slice.drawing=False
        self.track_slice.unsetCursor()
        self.track_draw_button.setText("Draw new track")
        self.drawing_notice.setText("Choose Draw new track. Start at the deep end, then drag or click toward the brain surface. The current plane is saved automatically.")
        self.drawing_notice.setToolTip(self.drawing_notice.text())
        self.drawing_notice.setStyleSheet("")

    def new_track(self):
        if self._mode!=1 or self.current_insertion() is None: return
        self._cancel_drawing()
        self.selected_id=None
        self.track_list.blockSignals(True)
        self.track_list.setCurrentRow(-1)
        self.track_list.blockSignals(False)
        self.track_name.clear()
        self._pending_draw=True
        if self._cache_current(): self._activate_drawing()
        else:
            self._drawing_message("Updating the plane slice… drawing will start as soon as it is ready.")

    def _activate_drawing(self):
        if not self._pending_draw or not self._cache_current() or self._mode!=1 or self.owner.step!=5: return
        item=self.current_insertion()
        if item is None: return
        # A normal slice scroll creates a new plane draft. Save it here instead
        # of silently refusing the user's drawing action until a separate Save.
        if item.get("plane")!=self.session["plane"]:
            self._remember()
            save_manual_plane(item,self.session["plane"])
        item.setdefault("points_visible",item.get("visible",True))
        item["tracks_visible"]=True
        item["visible"]=True
        self._populate_visibility()
        self._update_plane_status()
        self._pending_draw=False
        self._drawing_geometry=plane_geometry(self.session["plane"])
        self.track_slice.drawing=True
        self.track_slice.setCursor(Qt.CursorShape.CrossCursor)
        self.track_draw_button.setText("Drawing · restart track")
        self._drawing_message("Start at the deep tip inside the brain, then drag or click toward the surface. Live preview: all slices + 3D. Esc cancels.")
        self.track_slice.setFocus()
        self.owner.save_project()
        self.draw_cached(annotations_only=True)

    def edit_track(self):
        if self.current_track() is None or self._mode!=1: return
        self._cancel_drawing()
        self._pending_draw=True
        if self._cache_current(): self._activate_drawing()
        else: self._drawing_message("Updating the slice… the existing track is retained until the replacement is completed.")

    def _select_track(self,row):
        self._cancel_drawing()
        super()._select_track(row)

    def select_track_by_id(self,identifier):
        if self.session is None or self._mode==0: return
        item=next((item for item in self.session['insertions'] if any(t['id']==identifier for t in item['tracks'])),None)
        if item is None: return
        self.capture_state()
        self._cancel_drawing()
        self.insertion_id,self.selected_id=item['id'],identifier
        self._refresh_insertions()
        self._refresh_tracks()

    def move_selected_track(self):
        track=self.current_track()
        target=self.track_move_target.currentData()
        if track is None or target is None or target==self.insertion_id: return
        self.capture_state()
        self._cancel_drawing()
        self._remember()
        move_tracks(self.session,{track['id']},target)
        self.insertion_id=target
        sync_tracks(self.session,self.project)
        self._refresh_insertions()
        self._refresh_tracks()
        self.report_timer.start(0)
        self.owner.save_project()

    def shift_track(self,identifier,delta,finished):
        if self._mode!=1 or not self._cache_current(): return
        track=self.current_track()
        if track is None or track['id']!=identifier: return
        delta=np.asarray(delta,float)
        if delta.shape!=(3,) or not np.isfinite(delta).all(): return
        points=np.asarray(track['control_points_atlas_zyx_um'],float)+delta
        self._track_move_preview=dict(track,points_atlas_zyx_um=directed_track(points,matrix=self.project.total).tolist())
        if finished:
            self._track_move_preview=None
            if np.linalg.norm(delta)>1e-6:
                if inside_brain(points[:1],self.brain_mask,self.brain_preview,self.project.total)[0]:
                    self._remember()
                    track['control_points_atlas_zyx_um']=points.tolist()
                    if track.get('plane'):
                        track['plane']['center_um_zyx']=(np.asarray(track['plane']['center_um_zyx'])+delta).tolist()
                    # Move the track's source plane with it, without changing the insertion's viewing plane.
                    sync_tracks(self.session,self.project)
                    self.validate_starts()
                    self.report_timer.start(0)
                    self.owner.save_project()
                else: self._drawing_message('Track was not moved: its start must stay inside the brain mask.',True)
        self.draw_cached(annotations_only=True)

    def preview_track(self,start,end):
        if self._mode!=1 or not self.track_slice.drawing or not self._cache_current(): return
        origin=self._draft_start if self._draft_start is not None else start
        if origin is None or end is None: return
        if not inside_brain([origin],self.brain_mask,self.brain_preview,self.project.total)[0]: return
        try: directed_track([origin,end])
        except ValueError: return
        self._preview_start=np.asarray(origin,float).tolist()
        self._draft_direction=np.asarray(end,float).tolist()
        self.draw_cached(annotations_only=True)

    def draw_track_gesture(self,start,end):
        if not self.track_slice.drawing: return
        if self._draft_start is None: self.add_vertex(start)
        if self._draft_start is not None: self.add_vertex(end)

    def add_vertex(self,world):
        if self._mode!=1 or not self.track_slice.drawing: return
        if not self._cache_current():
            self._drawing_message("Wait for the current slice to finish updating, then place the track.")
            return
        world=np.asarray(world,float)
        if world.shape!=(3,) or not np.isfinite(world).all(): return
        if abs(plane_uv(plane_geometry(self.session["plane"]),[world])[1][0])>1e-4:
            self._drawing_message("Choose both track control points on the current electrode slice.",True)
            return
        if self._draft_start is None:
            if not inside_brain([world],self.brain_mask,self.brain_preview,self.project.total)[0]:
                self._preview_start=self._draft_direction=None
                self.draw_cached(annotations_only=True)
                self._drawing_message("Start outside brain mask. Check Show mask on brain slice or adjust Brain threshold, then choose a start inside brain tissue.",True)
                return
            self._draft_start=world.tolist()
            self._preview_start=self._draft_start
            self._draft_direction=None
            self.draw_cached(annotations_only=True)
            self._drawing_message("Start accepted. Move the pointer to preview on all slices + 3D; click the direction point to save the 4 mm track.")
            return
        if not inside_brain([self._draft_start],self.brain_mask,self.brain_preview,self.project.total)[0]:
            self._draft_start=self._preview_start=self._draft_direction=None
            self.draw_cached(annotations_only=True)
            self._drawing_message("The start is outside the updated brain mask. Choose a new start inside brain tissue.",True)
            return
        try: directed_track([self._draft_start,world.tolist()])
        except ValueError as error:
            self._drawing_message(str(error),True)
            return
        self._remember()
        item=self.current_insertion()
        track=self.current_track()
        if track is None:
            track={"id":uuid4().hex[:12],"name":f"Track {len(item['tracks'])+1}"}
            item["tracks"].append(track)
        track["control_points_atlas_zyx_um"]=[self._draft_start,world.tolist()]
        track["plane"]=copy.deepcopy(self.session["plane"])
        self.selected_id=track["id"]
        self._cancel_drawing()
        self.validate_starts()
        self._refresh_insertions()
        self.report_timer.start(0)
        self.owner.save_project()
        self._drawing_message(f"{item['name']} · {track['name']}: 4000 µm track saved on all three slices and in 3D. Choose Draw new track to add another.")

    def finish_track(self):
        self._cancel_drawing()
        self.draw_cached(annotations_only=True)
        self._drawing_message("Drawing cancelled. Existing tracks are retained. Choose Draw new track to start again.")

    def delete_track(self):
        item=self.current_insertion()
        track=self.current_track()
        if track is None: return
        self._cancel_drawing()
        self._remember()
        item["tracks"].remove(track)
        self.selected_id=None
        sync_tracks(self.session,self.project)
        self._refresh_insertions()
        self._refresh_tracks()
        self.report_timer.start(0)

    def _refresh_tracks(self):
        if self.session is None: return
        sync_tracks(self.session,self.project)
        self.track_list.blockSignals(True)
        self.track_list.clear()
        item=self.current_insertion()
        selected=-1
        for index,t in enumerate(item["tracks"] if item else []):
            ok=t.get("start_validation",{}).get("inside",False)
            entry=QListWidgetItem(f"{t['name']} · 4 mm"+("" if ok else " · START OUTSIDE MASK"))
            entry.setData(Qt.ItemDataRole.UserRole,t["id"])
            entry.setForeground(QColor(item["color"]))
            self.track_list.addItem(entry)
            if t["id"]==self.selected_id: selected=index
        self.track_list.setCurrentRow(selected)
        self.track_list.blockSignals(False)
        track=self.current_track()
        self.track_name.setText(track["name"] if track else "")
        total=len(self.session["tracks"])
        invalid=sum(not t.get("start_validation",{}).get("inside",False) for t in self.session["tracks"])
        self.track_summary.setText(f"{len(self.session['insertions'])} insertions · {total} tracks\n"
                                   f"Length: 4.000 mm each · starts outside mask: {invalid}")
        self.draw_cached()

    def _rename_track(self):
        track=self.current_track()
        if track is not None and self.track_name.text().strip():
            track["name"]=self.track_name.text().strip()
            self._refresh_tracks()
            self.report_timer.start(0)

    def _update_report(self):
        if self.labels is None or self.session is None: return
        super()._update_report()
        self.report_table.setHorizontalHeaderLabels(["Insertion / track / region","Entry µm","Exit µm","Length µm"])
        tracks={t["id"]:t for t in self.session["tracks"]}
        for row,segment in enumerate(self.report["segments"]):
            insertion=tracks[segment["track_id"]].get("insertion_name","")
            text=f"{insertion} · {segment['track_name']} · {segment['region_name']}"
            self.report_table.item(row,0).setText(text)
            self.report_table.item(row,0).setToolTip(text)

    def _geometries(self):
        if self._mode==4: return [self._probe_geometry()]*3
        if self._mode==1: return [self._placement_geometry()]
        span=self.upper-self.lower
        centers=np.tile((self.lower+self.upper)/2,(3,1))
        for axis in range(3): centers[axis,axis]=self._slice_positions()[axis]
        return [plane_geometry({"center_um_zyx":center.tolist(),"rotation_xyz_deg":angles,
                 "width_um":float(span[a]),"height_um":float(span[b]),"max_dimension":512})
                for center,(angles,a,b) in zip(centers,(((0,0,0),2,1),((90,0,0),2,0),((90,0,90),1,0)))]

    def request_render(self,*args):
        if self._closing or self.session is None or self.track_native is None: return
        self._token+=1
        if self._mode==1:
            self.volume_view.orthogonal_planes_enabled=False
            self.volume_view._slice_visibility_changed()
            self._placement_show_plane()
            self.render_timer.start(35)
            return
        self.volume_view.orthogonal_planes_enabled=self._mode==3
        if self._mode==3:
            grid={'origin_um_zyx':self.lower.tolist(),'spacing_um_zyx':self.depth_spacing.tolist(),'shape_zyx':self.depth_shape.tolist()}
            self.volume_view.update_slice_planes(grid,self._ortho_positions)
        self.volume_view._slice_visibility_changed()
        if self._mode==4:
            self.volume_view.set_plane(self._probe_geometry())
            if self.volume_view.plane_mesh is not None:
                self.volume_view.plane_mesh.visible=self.volume_view.plane_outline.visible=True
            self.render_timer.start(35)
            return
        if self._mode==1:
            if self._drawing_geometry is not None and self._drawing_geometry!=plane_geometry(self.session["plane"]):
                self._cancel_drawing()
                self._drawing_message("Plane changed. Choose Draw new track on the updated slice.")
            item=self.current_insertion()
            if item is not None:
                if item.get("plane")!=self.session["plane"]:
                    item["plane_draft"]=copy.deepcopy(self.session["plane"])
                else: item.pop("plane_draft",None)
            self.volume_view.set_plane(plane_geometry(self.session["plane"]))
            if self.volume_view.plane_mesh is not None:
                self.volume_view.plane_mesh.visible=self.volume_view.plane_outline.visible=True
            self._update_plane_status()
        elif self.volume_view.plane_mesh is not None:
            self.volume_view.plane_mesh.visible=self.volume_view.plane_outline.visible=False
        self.render_timer.start(35)

    def _cache_current(self):
        return self._cache is not None and self._cache["token"]==self._token

    def _render(self):
        if self._closing or self.session is None or self.owner.step<4: return
        if self._mode==1:
            self._render_placement()
            return
        if self._mode==2:
            self._cache={"token":self._token,"mode":2,"images":[]}
            self.draw_cached()
            return
        if self._running:
            self._pending=True
            return
        from .desktop_app import Job
        token,mode=self._token,self._mode
        geometry=self._geometries()
        matrix=self.project.total.copy()
        if mode in (1,4):
            sources=[self.track_native,self.brain_native,self.atlas_native]
            matrices=[matrix,matrix,np.eye(4)]
        else:
            source=min((self.slice_source if mode==0 else self.xyz_source).currentIndex(),2)
            sources=[(self.track_native,self.brain_native,self.atlas_native)[source]]*3
            matrices=[(self._tracing_matrix() if mode==0 else (np.eye(4) if source==2 else matrix))]*3
        labels=self.labels
        mask=self.brain_mask
        brain_preview=self.brain_preview
        show_mask=self.show_mask.isChecked() and mode==1
        show_labels=(mode in (1,4) or (self.slice_source if mode==0 else self.xyz_source).currentIndex()==3)
        def render(progress):
            images=[]
            for index,(volume,forward,g) in enumerate(zip(sources,matrices,geometry)):
                world=plane_world(g)
                pixels=sample_world(volume,forward,world).reshape(g["shape_hw"])
                atlas_plane=(mode in (1,4) and index==2) or (mode not in (1,4) and source==2)
                ids=labels.ids_at(world).reshape(g["shape_hw"]) if labels is not None and show_labels and atlas_plane else None
                support=inside_brain(world,mask,brain_preview,matrix).reshape(g["shape_hw"]) if show_mask and index==1 else None
                images.append((pixels,ids,g,support))
            return {"token":token,"mode":mode,"images":images}
        job=Job(render)
        self._jobs.add(job)
        self._running,self._pending=True,False
        def finish(result=None,error=None):
            self._jobs.discard(job)
            self._running=False
            if self._closing: return
            if token==self._token and mode==self._mode:
                if error:
                    if self._pending_draw:
                        self._cancel_drawing()
                        self._drawing_message(error,True)
                    else: self.owner.statusBar().showMessage(error)
                else:
                    self._cache=result
                    self.draw_cached()
                    if self._pending_draw: self._activate_drawing()
                    else: self._restore_drawing_draft()
            if self._pending or token!=self._token: self.render_timer.start(0)
        job.signals.done.connect(lambda value:finish(result=value))
        job.signals.failed.connect(lambda error:finish(error=error))
        QThreadPool.globalInstance().start(job)

    def draw_cached(self,*args,annotations_only=False):
        if self._updating or self.session is None or self.track_native is None: return
        if not annotations_only: self.volume_view.set_label_styles(self.session['regions'])
        if self._mode==1:
            self._draw_placement(annotations_only)
            return
        visible=[item for item in self.session["insertions"]
                 if self.show_all.isChecked() or item["id"]==self.insertion_id]
        points=[dict(p,color=item["color"],label=f"{item['name']}:{index+1}")
                for item in visible if item.get("points_visible",item.get("visible",True))
                for index,p in enumerate(item["points"])]
        tracks=[t for item in visible if item.get("tracks_visible",item.get("visible",True)) for t in item["tracks"]] if self._mode!=0 else []
        if self._track_move_preview is not None:
            tracks=[self._track_move_preview if track['id']==self._track_move_preview['id'] else track for track in tracks]
        self.volume_view.selected_track_id=self.selected_id
        start=self._draft_start if self._draft_start is not None else self._preview_start
        if start is not None:
            item=self.current_insertion()
            points.append({"id":"draft","atlas_zyx_um":start,"color":item["color"],"label":"START"})
            if self._draft_direction is not None:
                tracks=tracks+[{"id":"draft_track","name":"Preview · 4 mm","insertion_id":item["id"],
                    "color":item["color"],"points_atlas_zyx_um":directed_track([start,self._draft_direction],matrix=self.project.total).tolist()}]
        self.volume_view.point_size=self.point_size.value()
        self.volume_view.set_points(points if self._mode!=4 else [],self.selected_points,self.show_points_check.isChecked())
        self.volume_view.set_tracks(tracks if self._mode!=4 else [])
        if self._mode==4:
            self._draw_probes()
        for index,canvas in enumerate(self.slices):
            canvas.track_lines=[]
            canvas.selected_track=self.selected_id
            canvas.measurements=self.session.get('measurements',[])
            canvas.measurement_unit=self.measurement_unit.currentText()
            canvas.measurement_inverse=np.linalg.inv(self.project.total)[:3,:3]
            canvas.insertion_points=points if self._mode!=4 else []
            canvas.point_size=self.point_size.value()
            canvas.slice_thickness_um=float(self.depth_spacing[index]) if self._mode==0 else None
            if self._mode==0:
                inverse=np.linalg.inv(self._tracing_to_atlas())
                canvas.insertion_points=[dict(p,atlas_zyx_um=transform_points([p['atlas_zyx_um']],inverse)[0].tolist()) for p in points]
                canvas.measurements=[dict(m,points_atlas_zyx_um=transform_points(m['points_atlas_zyx_um'],inverse).tolist())
                                     for m in canvas.measurements]
                canvas.measurement_inverse=np.linalg.inv(self._tracing_matrix())[:3,:3]
            canvas.insertion_tracks=tracks if self._mode!=4 else []
            canvas.selected_points=self.selected_points
            canvas.slab_um=self.slab.value()
            canvas.show_points=self.show_points_check.isChecked() and self._mode in (0,1,2)
            canvas.show_tracks=self._mode==1
            canvas.track_hits=self._mode==3
            canvas.project_tracks=self._mode==1
            canvas.annotation_mode=False
            canvas.axis_label="Plane U →   V ↓" if self._mode in (1,4) else ("X →   Y ↓","X →   Z ↓","Y →   Z ↓")[index]
            canvas.update()
        self._apply_slice_tools()
        if annotations_only: return
        if self._cache is None or self._cache["mode"]!=self._mode: return
        for index,(canvas,cached) in enumerate(zip(self.slices,self._cache["images"])):
            pixels,ids,geometry,support=cached
            channel=index if self._mode in (1,4) else min((self.slice_source if self._mode==0 else self.xyz_source).currentIndex(),2)
            from .desktop_slice_display import slice_color
            settings=self.slice_display.records[channel]
            self.slice_display.observe(channel,pixels)
            grey=slice_color(pixels,settings)
            canvas.display_interpolation=settings['interpolation']
            canvas.smooth_interpolation=settings['interpolation']=='linear'
            image=grey
            canvas.region_labels=[]
            if support is not None:
                image=grey.copy()
                image[support]=.7*image[support]+.3*np.array([.2,1,.4])
            if ids is not None and self.labels is not None:
                image=grey.copy()
                alpha=self.region_opacity.value()/100
                for region in self.labels.regions:
                    style=self.session["regions"].get(str(region["id"]),{})
                    selected=ids==region["id"]
                    if not style.get("visible",True) or not np.any(selected): continue
                    color=style.get("color_rgb",region["color_rgb"])
                    image[selected]=(1-alpha)*image[selected]+alpha*np.asarray(color)/255
                    if style.get("label",False):
                        row,col=np.nonzero(selected)
                        nearest=int(np.argmin((row-row.mean())**2+(col-col.mean())**2))
                        canvas.region_labels.append((style.get("name",region["name"]),[int(col[nearest]),int(row[nearest])],color))
            canvas.set_plane_image(image,geometry)
            from .desktop_figures import slice_sampler
            source=(self.track_native,self.brain_native,self.atlas_native)[channel]
            forward=self._tracing_matrix() if self._mode==0 else (np.eye(4) if channel==2 else self.project.total)
            canvas.figure_sampler=slice_sampler([source],[forward],[settings],visible_pixels=[pixels],labels=self.labels,
                label_matrix=np.eye(4),label_mode='overlay' if ids is not None else 'atlas',
                styles=self.session['regions'],label_opacity=self.region_opacity.value()/100)
            if self._mode in (1,4):
                self.slice_titles[index].setText(("Electrodes · probe plane" if self._mode==4 else "Electrodes · draw tracks here","Brain scan · same plane","Atlas · same plane")[index])
            else:
                source=(self.slice_source if self._mode==0 else self.xyz_source).currentIndex()
                self.slice_titles[index].setText(('XY','XZ','YZ')[index] if self._mode==0 else
                    f"{('XY','XZ','YZ')[index]} · {'ZYX'[index]} {self.slice_sliders[index].value()} / {self.slice_sliders[index].maximum()} · {self._ortho_positions[index]:.1f} µm · {('Electrodes','Brain','Atlas','Atlas + labels')[source]}")
        if self._mode==1:
            g=self._cache["images"][0][2]
            self.plane_info.setText(f"Matched plane · pixel size {g['pixel_size_um']:.2f} µm\n"
                                   "Only points in the current slice are shown. Tracks and their live previews share one 3D geometry on all views.")

    def _sync_views(self,source):
        if self._mode not in (1,4): return
        for canvas in self.slices:
            if canvas is not source:
                canvas.zoom,canvas.pan=source.zoom,QPointF(source.pan)
                canvas.view_rotation=source.view_rotation
                canvas.update()

    def capture_state(self):
        if self.session is None: return
        track=self.current_track()
        if track is not None and self.track_name.text().strip(): track['name']=self.track_name.text().strip()
        sync_tracks(self.session,self.project)
        self.session['edit_history']=copy.deepcopy(self._undo)
        self._view_states[str(self._mode)]=[canvas.view_record() for canvas in self.slices]
        self._tool_modes[str(self._mode)]=self.point_tool.currentIndex()
        self._canvas_tool_modes[str(self._mode)]=list(self._canvas_tools)
        if self.control_content.isVisible(): self._sidebar_sizes=self.splitter.sizes()
        item=self.current_insertion()
        start=self._draft_start if self._draft_start is not None else self._preview_start
        if self._mode==1 and item is not None and start is not None:
            item['drawing_draft']={'start':start,'direction':self._draft_direction,'track_id':self.selected_id,
                                  'plane':copy.deepcopy(self.session['plane'])}
        if self._mode==1: self._placement_cameras[self._placement_scene]=self.volume_view.camera_record()
        self.session["insertion_view"]={"ui_version":4,"placement_cameras":copy.deepcopy(self._placement_cameras),"selected_insertion_id":self.insertion_id,"selected_point_ids":sorted(self.selected_points),
            "selected_track_id":self.selected_id,"camera":self.volume_view.camera_record(),
            "rendering":self.volume_view.render_settings_record(),"opacity":[s.value() for s in self.opacity],
            "orthogonal_positions_zyx_um":self._ortho_positions.tolist() if self._ortho_positions is not None else None,
            "tracing_positions_zyx_um":self._tracing_positions.tolist() if self._tracing_positions is not None else None,
            "point_size_px":self.point_size.value(),
            "placement_scene":self._placement_scene,
            'placement_view':self.placement_view.currentData(),
            "placement_columns":[self.placement_table.columnWidth(i) for i in range(4)],
            "placement_table_height":self.placement_table.height(),
            "recording_columns":[self.placement_recordings_table.columnWidth(i) for i in range(5)],
            "recording_table_height":self.placement_recordings_table.height(),
            'show_volume_axes':self.volume_view.show_axes.isChecked(),
            'flip_xyz':self.volume_view.flip_record(),
            'insertion_columns':[self.insertion_table.columnWidth(i) for i in range(3)],
            'measurement_unit':self.measurement_unit.currentText(),
            'region_table_height':self.region_table.height(),
            'region_columns':[self.region_table.columnWidth(i) for i in range(4)],
            'region_side_filter':self.region_side_filter.currentData(),
            'measurement_columns':[self.measurement_table.columnWidth(i) for i in range(4)],
            'annotation_panel_sizes':(self.annotation_splitter.sizes() if self._mode==0 else
                self.session.get('insertion_view',{}).get('annotation_panel_sizes',[260,180])),
            "slice_source":self.slice_source.currentIndex(),"xyz_source":self.xyz_source.currentIndex(),
            "slab_um":self.slab.value(),"show_all":self.show_all.isChecked(),"show_points":self.show_points_check.isChecked(),
            "track_gamma":self.track_gamma.value(),
            "atlas_gamma":self.atlas_gamma.value(),"point_gamma":self.point_gamma.value(),
            "xyz_gamma":self.xyz_gamma.value(),"region_opacity":self.region_opacity.value(),
            'layers':self.volume_view.layers.record(),'slice_views':copy.deepcopy(self._view_states),
            'point_tool':self.point_tool.currentIndex(),'tool_modes':dict(self._tool_modes),
            'canvas_tool_modes':copy.deepcopy(self._canvas_tool_modes),'xyz_initialized':self._xyz_initialized,
            'sidebar_visible':self.sidebar_toggle.isChecked(),'sidebar_sizes':self._sidebar_sizes,
            'slice_display':copy.deepcopy(self.slice_display.records),
            'show_slice_planes':self.volume_view.show_slice_planes.isChecked(),'fill_slice_planes':self.volume_view.fill_slice_planes.isChecked(),
            'sidebar_tab':self.sidebar_tabs.currentIndex(),'plane_tab':self.plane_tabs.currentIndex()}

    def _restore_state(self):
        view=self.session.get("insertion_view",{})
        self._view_states=copy.deepcopy(view.get('slice_views',{}))
        if view.get('ui_version',1)<3: self._view_states.pop('0',None)
        if view.get('ui_version',1)<4: self._view_states.pop('1',None)
        self._tool_modes=dict(view.get('tool_modes',{}))
        self._canvas_tool_modes=copy.deepcopy(view.get('canvas_tool_modes',{}))
        self._xyz_initialized=bool(view.get('xyz_initialized',False))
        self.slice_display.initialize((self.track_limits,self.brain_limits,self.atlas_limits),view.get('slice_display'))
        self._updating=True
        self.placement_view.setCurrentIndex(max(0,self.placement_view.findData(view.get('placement_view','probes'))))
        self.insertion_id=view.get("selected_insertion_id")
        self.selected_points=set(view.get("selected_point_ids",[]))
        self.selected_id=view.get("selected_track_id")
        positions=view.get("orthogonal_positions_zyx_um")
        self._ortho_positions=np.asarray(positions,float) if positions is not None else None
        positions=view.get('tracing_positions_zyx_um')
        self._tracing_positions=np.asarray(positions,float) if positions is not None else None
        for i,width in enumerate(view.get('placement_columns',[])):
            if i<4: self.placement_table.setColumnWidth(i,max(30,int(width)))
        self.placement_table.setFixedHeight(max(80,int(view.get('placement_table_height',155))))
        for i,width in enumerate(view.get('recording_columns',[])):
            if i<5: self.placement_recordings_table.setColumnWidth(i,max(25,int(width)))
        self.placement_recordings_table.setFixedHeight(max(80,int(view.get('recording_table_height',124))))
        self._placement_cameras=copy.deepcopy(view.get('placement_cameras',{}))
        # Legacy projects may have selected an unregistered 3D view. The 3D
        # workspace now always uses atlas space; native data stay in the slice.
        self._placement_scene='atlas'
        self.point_size.setValue(view.get('point_size_px',8))
        self.measurement_unit.setCurrentText(view.get('measurement_unit','µm'))
        self.region_table.setFixedHeight(max(80,int(view.get('region_table_height',250))))
        for i,width in enumerate(view.get('region_columns',[])):
            if i<4: self.region_table.setColumnWidth(i,max(25,int(width)))
        self.region_side_filter.setCurrentIndex(max(0,self.region_side_filter.findData(view.get('region_side_filter','all'))))
        for i,width in enumerate(view.get('measurement_columns',[])):
            if i<4: self.measurement_table.setColumnWidth(i,max(30,int(width)))
        if self._mode==0: self.annotation_splitter.setSizes(view.get('annotation_panel_sizes',[260,180]))
        if "rendering" in view: self.volume_view.restore_render_settings(view["rendering"])
        for slider,value in zip(self.opacity,view.get("opacity",[0,100,25])): slider.setValue(value)
        if 'layers' in view: self.volume_view.layers.restore(view['layers'])
        for widget,key,default in ((self.slab,"slab_um",50),(self.track_gamma,"track_gamma",1),
            (self.atlas_gamma,"atlas_gamma",1),(self.point_gamma,"point_gamma",1),
            (self.xyz_gamma,"xyz_gamma",1),(self.region_opacity,"region_opacity",45)):
            widget.setValue(view.get(key,default))
        for widget,key,default in ((self.show_all,"show_all",True),(self.show_points_check,"show_points",True),
                                   ):
            widget.setChecked(view.get(key,default))
        self.slice_source.setCurrentIndex(max(0,min(1,view.get("slice_source",0))))
        self.xyz_source.setCurrentIndex(view.get("xyz_source",0))
        self.point_tool.setCurrentIndex(view.get('point_tool',0))
        plane_tab,sidebar_tab=view.get('plane_tab',0),view.get('sidebar_tab',0)
        if view.get('ui_version',1)<2:
            if plane_tab==2: plane_tab,sidebar_tab=0,2
            elif plane_tab==3: plane_tab=2
        if view.get('ui_version',1)<3 and sidebar_tab==2:
            sidebar_tab=1 if self._mode==0 else 3
        self.plane_tabs.setCurrentIndex(plane_tab)
        self.sidebar_tabs.setCurrentIndex(sidebar_tab)
        self._sidebar_sizes=view.get('sidebar_sizes',[340,1000])
        self.sidebar_toggle.setChecked(view.get('sidebar_visible',True))
        self.volume_view.show_slice_planes.setChecked(view.get('show_slice_planes',True))
        self.volume_view.fill_slice_planes.setChecked(view.get('fill_slice_planes',False))
        self._restore_slice_views()
        self.volume_view.show_axes.setChecked(view.get('show_volume_axes',True))
        self.volume_view.restore_flip(view.get('flip_xyz'))
        for i,width in enumerate(view.get('insertion_columns',[])):
            if i<3: self.insertion_table.setColumnWidth(i,max(35,int(width)))
        self.volume_view.restore_camera(view.get("camera",{}))
        self._updating=False

    def _restore_slice_views(self):
        records=self._view_states.get(str(self._mode),[])
        for index,canvas in enumerate(self.slices): canvas.restore_view(records[index] if index<len(records) else {})

    def _restore_drawing_draft(self):
        if self._mode!=1 or self.track_slice.drawing or not self._cache_current(): return
        item=self.current_insertion()
        draft=copy.deepcopy(item.get('drawing_draft')) if item else None
        if not draft or draft['plane']!=self.session['plane']: return
        self.selected_id=draft.get('track_id')
        self._pending_draw=True
        self._activate_drawing()
        self._draft_start=draft['start']
        self._preview_start=draft['start']
        self._draft_direction=draft.get('direction')
        self.draw_cached(annotations_only=True)

    def export_analysis(self):
        if self._mode==4:
            self.export_probe_reports()
            return
        if self.project is None or self.owner.busy: return
        self.validate_starts()
        path=self.owner.save_project()
        if path is None: return
        frozen=AlignmentProject.load(path)
        labels=self.labels
        def ready(path):
            self.last_export=path
            self.owner.statusBar().showMessage(f"Insertions, 4 mm tracks and matched native planes saved: {path.parent}")
            self.info.setText(f"Insertion analysis saved\n{path.parent}")
        def export(progress):
            atlas=labels
            if atlas is None and frozen.labels_dir is not None:
                cache=Path(tempfile.gettempdir())/"probemap"/"native-labels"
                atlas=load_label_atlas(frozen.labels_dir,frozen.reference,cache,progress,spacing_um_zyx=frozen.label_spacing)
            return export_insertions(frozen,atlas,progress)
        self.owner._start_job(export,ready,"Saving insertion analysis…")

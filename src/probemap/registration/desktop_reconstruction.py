"""Final reconstruction: fitted probes, native measurements and linked views."""
from __future__ import annotations

import copy
from pathlib import Path
import tempfile

import numpy as np
from qtpy.QtCore import QPointF, QRectF, QThreadPool, QTimer, Qt
from qtpy.QtGui import QColor, QPainter, QPen
from qtpy.QtWidgets import (QAbstractItemView,QCheckBox,QFormLayout,QGridLayout,QHBoxLayout,
    QGroupBox,QHeaderView,QLabel,QPushButton,QScrollArea,QSlider,QSpinBox,QSplitter,QStackedWidget,
    QTableWidget,QTableWidgetItem,QTabWidget,QVBoxLayout,QWidget)

from .desktop_tables import CheckHeader,TableHeightGrip
from .desktop_insertion_views import InsertionVolumeView
from .desktop_track_views import TrackSliceCanvas
from .desktop_slice_display import SliceDisplayControls,slice_grayscale,slice_color
from .desktop_recordings import recording_rgba
from .desktop_widgets import PopupComboBox,flat_tabs,view_button
from .insertion_model import insertion_session
from .landmarks import transform_points
from .neuropixels_model import contact_mesh,recording_sites,shank_mesh
from .reconstruction import (fitted_probes,orthogonal_geometry,project_reconstruction_report,
    shanks,slice_reconstruction,write_reconstruction_report,
    probe_comparison_geometries,label_slice_image,saved_probe_plane_geometries)
from .review import placement_without_scale
from .surface_entries import build_atlas_surface,probe_surface_entries,probe_shank_sections
from .track_model import load_label_atlas,load_track_data,plane_world,raw_volume,sample_world


class ReconstructionSlice(TrackSliceCanvas):
    def __init__(self):
        super().__init__(drawable=False)
        self.background_color='#000000'; self.show_axes=False; self.pan_mode=True
        self.setToolTip('Pinch or wheel: zoom · Drag: pan')
        self.reconstruction=dict(lines=[],sites=[],bands=[])

    def paintEvent(self,event):
        names=self.region_labels; self.region_labels=[]
        try: super().paintEvent(event)
        finally: self.region_labels=names
        if self.image is None: return
        painter=QPainter(self); painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setClipRect(self.rect())
        scale=self.fit_scale()*self.zoom
        for key,width in (('lines',2),('bands',7)):
            for row in self.reconstruction[key]:
                width=2 if key=='lines' else min(7,max(2,row.get('width_um',70)*scale*1.4))
                a,b=[self._pixel(p) for p in row['points']]
                color=(QColor(row['color']) if key=='lines' else
                    QColor.fromRgbF(*recording_rgba(row['color'],row.get('opacity',1))))
                backing=QColor(0,0,0,round(210*color.alphaF()))
                painter.setPen(QPen(backing,width+2)); painter.drawLine(a,b)
                painter.setPen(QPen(color,width)); painter.drawLine(a,b)
                if (a-b).manhattanLength()<4:
                    painter.setBrush(color); painter.drawEllipse(a,width/2+1,width/2+1)
        for row in self.reconstruction['sites']:
            # Dense contacts are represented by their recording bands until
            # zoom resolves individual sites; fixed white rings hide the scan.
            if row.get('pitch_um',15)*scale<4 and not getattr(self,'_vector_export',False): continue
            radius=min(3,max(.8,row.get('size_um',12)*scale/2))
            point=self._pixel(row['point']); color=QColor.fromRgbF(*recording_rgba(row['color'],row['opacity']))
            painter.setPen(QPen(color,1)); painter.setBrush(color); painter.drawEllipse(point,radius,radius)
        self._paint_region_names(painter)

    def _paint_region_names(self,painter):
        from .track_model import plane_world
        occupied=[]; metrics=painter.fontMetrics()
        for name,uv,rgb in self.region_labels:
            anchor=self._pixel(plane_world(self.geometry_record,[uv])[0])
            if not self.rect().contains(anchor.toPoint()): continue
            text=metrics.elidedText(name,Qt.TextElideMode.ElideRight,max(50,min(210,int(self.width()*.7))))
            width=metrics.horizontalAdvance(text)+8; height=metrics.height()+4
            candidates=[(8,0),(-width-8,0)]
            candidates.extend((x,sign*y) for y in range(height,7*height,height) for sign in (-1,1) for x in (8,-width-8))
            chosen=None
            for dx,dy in candidates:
                x=float(np.clip(anchor.x()+dx,4,max(4,self.width()-width-4)))
                y=float(np.clip(anchor.y()+dy-height/2,4,max(4,self.height()-height-4)))
                box=QRectF(x,y,width,height)
                if not any(box.adjusted(-3,-3,3,3).intersects(other) for other in occupied):
                    chosen=box; break
            if chosen is None: continue
            occupied.append(chosen)
            end=QPointF(float(np.clip(anchor.x(),chosen.left(),chosen.right())),chosen.center().y())
            color=QColor(*rgb)
            if (end-anchor).manhattanLength()>12:
                leader=QColor(color); leader.setAlpha(155); painter.setPen(QPen(leader,1)); painter.drawLine(anchor,end)
            painter.fillRect(chosen,QColor(0,0,0,190)); painter.setPen(color)
            painter.drawText(chosen.adjusted(4,0,-4,0),Qt.AlignmentFlag.AlignVCenter,text)


class ReconstructionWorkspace(QWidget):
    def __init__(self,owner):
        super().__init__(); self.owner=owner; self.project=self.session=None
        self.setObjectName('reconstructionWorkspace'); self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground,True)
        self.setStyleSheet('QWidget#reconstructionWorkspace{background:#242b36;} QStackedWidget{background:#242b36;border:0;} QLabel{background:transparent;} QHeaderView::section{background:transparent;color:#9aafc1;border:0;}')
        self._updating=False; self._closing=False; self._token=0; self._running=False; self._pending=False
        self._jobs=set(); self._visuals=[]; self.surface_visuals=None; self.report=None
        self._cache=[]; self._context=None; self.native={}; self.surface=self.labels=None
        self._syncing=False; self._probe_offsets={}; self._shank_sections={}
        root=QVBoxLayout(self); root.setContentsMargins(12,12,12,8)
        heading=QHBoxLayout(); self.collapse=view_button('sidebar','Show or hide settings',self.toggle_sidebar)
        heading.addWidget(self.collapse); title=QLabel('Reconstruction'); title.setObjectName('pageTitle')
        title.setStyleSheet('font-size:22px;font-weight:600'); heading.addWidget(title); heading.addStretch()
        self.view_choice=PopupComboBox(); self.view_choice.addItems(['3D','Slices','Comparison','Probe plane'])
        self.view_choice.currentIndexChanged.connect(self._change_view); heading.addWidget(self.view_choice)
        root.addLayout(heading)
        self.splitter=QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setHandleWidth(7)
        self.splitter.setStyleSheet('QSplitter::handle{background:#354354;} QSplitter::handle:hover{background:#7ac7e5;}'); root.addWidget(self.splitter,1)
        self.sidebar=QTabWidget(); flat_tabs(self.sidebar); self.sidebar.setMinimumWidth(290); self.sidebar.setMaximumWidth(16777215)
        self.splitter.addWidget(self.sidebar)
        report_panel=QWidget(); report_layout=QVBoxLayout(report_panel); report_layout.setContentsMargins(6,10,6,6)
        row=QHBoxLayout(); row.addWidget(QLabel('Probes'),1); self.unit=PopupComboBox(); self.unit.addItems(['µm','mm'])
        self.unit.currentIndexChanged.connect(self.refresh_report); row.addWidget(self.unit); report_layout.addLayout(row)
        self.probe_table=self._table(['Probe','Mean depth','Shanks','Tilt Y · °'],selection=True)
        self.probe_table.horizontalHeaderItem(3).setToolTip('Inclination to the atlas Y axis after registration · 0° parallel, 90° perpendicular')
        self.probe_table.itemSelectionChanged.connect(self._select_report_probe); report_layout.addWidget(self.probe_table); report_layout.addWidget(TableHeightGrip(self.probe_table))
        self.probe_table.horizontalHeaderItem(1).setToolTip('Mean entry-to-tip distance along the shanks, in source scan units')
        self.report_note=QLabel(); self.report_note.setWordWrap(True); self.report_note.setStyleSheet('color:#9aafc1;')
        report_layout.addWidget(self.report_note)
        self.dimension_label=QLabel(); self.dimension_label.setWordWrap(True); report_layout.addWidget(self.dimension_label)
        self.shank_table=self._table(['Shank','Depth','Length']); self.shank_table.horizontalHeaderItem(1).setToolTip('Atlas entry mapped back to the source scan; physical distance to the probe tip'); report_layout.addWidget(self.shank_table); report_layout.addWidget(TableHeightGrip(self.shank_table))
        self.region_heading=QLabel('Regions'); report_layout.addWidget(self.region_heading)
        self.region_table=self._table(['Region','Shank','Length','Contacts']); report_layout.addWidget(self.region_table,1); report_layout.addWidget(TableHeightGrip(self.region_table))
        self.recording_summary=QLabel(); self.recording_summary.setWordWrap(True); report_layout.addWidget(self.recording_summary)
        report_layout.addStretch(); self.sidebar.addTab(self._scroll(report_panel),'Report')
        self.volume_view=InsertionVolumeView()
        self.volume_view.set_layer_titles(('Atlas','Electrodes','Brain'))
        self.volume_view.heading.setText('3D'); self.volume_view.layers.hide()
        self.volume_view.zoom_controls.hide(); self.volume_view.settings_button.hide(); self.volume_view.plane_controls.hide()
        visual_panel=QWidget(); visual_layout=QVBoxLayout(visual_panel); visual_layout.setContentsMargins(6,10,6,6)
        form=QFormLayout(); self.scene_mode=PopupComboBox(); self.scene_mode.addItems(['Anatomy','Recordings','Surface entries','Regions'])
        self.scene_mode.currentIndexChanged.connect(self._scene_changed); form.addRow('View',self.scene_mode)
        self.space=PopupComboBox(); self.space.addItems(['Atlas','Source scan'])
        self.space.currentIndexChanged.connect(self._space_changed); form.addRow('Space',self.space)
        visual_layout.addLayout(form)
        self.probe_filter=self._table(['Probe','Visible'],selection=False)
        self.probe_filter.itemChanged.connect(self._filter_changed); self._visibility_header(self.probe_filter,1); visual_layout.addWidget(self.probe_filter); visual_layout.addWidget(TableHeightGrip(self.probe_filter))
        self.recording_filter=self._table(['Recording','Probe','Visible'],selection=False)
        self.recording_filter.itemChanged.connect(self._filter_changed); self._visibility_header(self.recording_filter,2); visual_layout.addWidget(self.recording_filter); visual_layout.addWidget(TableHeightGrip(self.recording_filter))
        self.surface_entries_check=QCheckBox('Surface entries')
        self.surface_entries_check.setToolTip('Show fitted probe entry points together with the tracks')
        self.surface_entries_check.toggled.connect(self._draw_3d); visual_layout.addWidget(self.surface_entries_check)
        self.track_opacity_controls=QWidget(); opacity_form=QFormLayout(self.track_opacity_controls); opacity_form.setContentsMargins(0,0,0,0)
        opacity_form.addRow(QLabel('Track opacity'))
        for name,title,value in (('probe_opacity_inside','Inside brain',50),('probe_opacity_outside','Outside brain',100)):
            slider=QSlider(Qt.Orientation.Horizontal); slider.setRange(0,100); slider.setValue(value); setattr(self,name,slider)
            row=QWidget(); line=QHBoxLayout(row); line.setContentsMargins(0,0,0,0); amount=QLabel(f'{value}%'); amount.setMinimumWidth(35)
            line.addWidget(slider,1); line.addWidget(amount)
            slider.valueChanged.connect(lambda value,label=amount:label.setText(f'{value}%'))
            slider.valueChanged.connect(self._draw_3d); opacity_form.addRow(title,row)
        visual_layout.addWidget(self.track_opacity_controls)
        visual_layout.addWidget(self.volume_view.layers); self.volume_view.layers.show()
        visual_layout.addWidget(self.volume_view.plane_controls); self.volume_view.plane_controls.show()
        self.volume_view.show_slice_planes.hide(); self.volume_view.fill_slice_planes.hide()
        self.volume_view.show_axes.setChecked(False)
        self.volume_view.layers.visibilityChanged.connect(self.owner.schedule_save)
        self.volume_view.layers.labels_check.toggled.connect(self._labels_toggled)
        self.volume_view.opacityChanged.connect(lambda *args:self.owner.schedule_save())
        self.volume_view.viewSettingsChanged.connect(self.owner.schedule_save)
        from .desktop_rendering import RenderSettingsPanel
        self.render_group=QGroupBox('Rendering'); self.render_group.setCheckable(True); self.render_group.setChecked(False)
        render_layout=QVBoxLayout(self.render_group)
        self.render_panel=RenderSettingsPanel(self.volume_view,include_opacity=False,compact=True,minimal=True)
        render_layout.addWidget(self.render_panel); self.render_panel.hide()
        self.render_group.toggled.connect(self.render_panel.setVisible); visual_layout.addWidget(self.render_group)
        self.volume_view.inline_settings_handler=self.open_rendering
        self.volume_view.renderSettingsChanged.connect(self.owner.schedule_save)
        visual_layout.addStretch(); self.sidebar.addTab(self._scroll(visual_panel),'3D')
        slices_panel=QWidget(); slice_layout=QVBoxLayout(slices_panel); slice_layout.setContentsMargins(6,10,6,6)
        self.appearance=SliceDisplayControls(('Atlas','Brain','Electrodes'),minimal=True)
        self.appearance.changed.connect(self._appearance_changed); self.appearance.autoRequested.connect(self.auto_contrast)
        slice_layout.addWidget(self.appearance)
        form=QFormLayout(); self.slab=QSpinBox(); self.slab.setRange(1,1000); self.slab.setValue(25); self.slab.setSuffix(' µm')
        self.slab.setToolTip('Physical thickness used for probe and recording intersections')
        self.slab.valueChanged.connect(self.draw_cached); form.addRow('Slice thickness',self.slab)
        self.region_overlay=QCheckBox('Labels on scans'); self.region_overlay.toggled.connect(self.draw_cached); form.addRow(self.region_overlay)
        slice_layout.addLayout(form); slice_layout.addStretch(); self.sidebar.addTab(self._scroll(slices_panel),'Slices')
        from .desktop_reconstruction_regions import ReconstructionRegions
        self.regions=ReconstructionRegions(); self.regions.changed.connect(self._regions_changed)
        self.sidebar.addTab(self._scroll(self.regions),'Regions')
        self.body=QWidget(); body=QVBoxLayout(self.body); body.setContentsMargins(0,0,0,0); self.splitter.addWidget(self.body)
        self.splitter.setStretchFactor(1,1); self.splitter.setSizes([340,1100])
        row=QHBoxLayout(); self.source=PopupComboBox()
        for title,key in [('Atlas','atlas'),('Brain · before','brain_before'),('Brain · after','brain_after'),('Electrodes · before','electrodes_before'),('Electrodes · after','electrodes_after')]: self.source.addItem(title,key)
        self.source.currentIndexChanged.connect(self._source_changed); row.addWidget(self.source)
        self.compare_channel=PopupComboBox(); self.compare_channel.addItems(['Brain','Electrodes']); self.compare_channel.currentIndexChanged.connect(self.request_render); row.addWidget(self.compare_channel)
        self.compare_axis=PopupComboBox()
        for index,name in enumerate(('XY','XZ','YZ')): self.compare_axis.addItem(name,index)
        self.compare_axis.setToolTip('Orthogonal slice or a fitted probe plane')
        self.compare_axis.currentIndexChanged.connect(self._axis_changed); row.addWidget(self.compare_axis)
        self.atlas_display=PopupComboBox()
        for title,key in (('Atlas','atlas'),('Atlas + labels','overlay'),('Labels','labels')): self.atlas_display.addItem(title,key)
        self.atlas_display.currentIndexChanged.connect(self.draw_cached); row.addWidget(self.atlas_display)
        self.plane_probe=PopupComboBox(); self.plane_probe.setToolTip('Saved probe plane')
        self.plane_probe.currentIndexChanged.connect(self._plane_probe_changed); row.addWidget(self.plane_probe)
        self.plane_stage=PopupComboBox(); self.plane_stage.addItems(['Before registration','After registration']); self.plane_stage.setCurrentIndex(1)
        self.plane_stage.currentIndexChanged.connect(self.request_render); row.addWidget(self.plane_stage)
        self.notice=QLabel(); row.addWidget(self.notice,1); self.source_row=row; body.addLayout(row)
        self.views=QStackedWidget(); body.addWidget(self.views,1)
        self.view_3d=QWidget(); self.layout_3d=QVBoxLayout(self.view_3d); self.layout_3d.setContentsMargins(0,0,0,0); self.layout_3d.addWidget(self.volume_view)
        self.views.addWidget(self.view_3d)
        ortho=QWidget(); self.ortho_grid=QGridLayout(ortho); self.ortho_grid.setContentsMargins(0,0,0,0); self.ortho_grid.setSpacing(10)
        self.canvases=[]; self.sliders=[]; self.slice_panels=[]
        for axis,name in enumerate(('XY','XZ','YZ')):
            panel,canvas,slider=self._slice_panel(name,axis,True)
            self.slice_panels.append(panel); self.canvases.append(canvas); self.sliders.append(slider)
            self.ortho_grid.addWidget(panel,*((0,0),(0,1),(1,0))[axis])
        for index in (0,1):
            self.ortho_grid.setColumnStretch(index,1); self.ortho_grid.setRowStretch(index,1)
        self.views.addWidget(ortho)
        comparison=QWidget(); comparison_layout=QVBoxLayout(comparison); comparison_layout.setContentsMargins(0,0,0,0)
        grid=QHBoxLayout(); self.compare_canvases=[]
        for name in ('Atlas','Before','After'):
            panel,canvas,_=self._slice_panel(name,0,False); grid.addWidget(panel,1); self.compare_canvases.append(canvas)
        comparison_layout.addLayout(grid,1)
        self.compare_slider=QSlider(Qt.Orientation.Horizontal); self.compare_slider.valueChanged.connect(self._comparison_depth_changed); comparison_layout.addWidget(self.compare_slider)
        self.views.addWidget(comparison)
        for canvas in self.compare_canvases: canvas.viewChanged.connect(lambda c=canvas:self._sync_views(c))
        plane_page=QWidget(); plane_layout=QHBoxLayout(plane_page); plane_layout.setContentsMargins(0,0,0,0); plane_layout.setSpacing(10)
        self.probe_plane_canvases=[]
        for name in ('Atlas','Brain','Electrodes'):
            panel,canvas=self._probe_plane_panel(name); plane_layout.addWidget(panel,1); self.probe_plane_canvases.append(canvas)
            canvas.viewChanged.connect(lambda c=canvas:self._sync_probe_plane_views(c))
        self.views.addWidget(plane_page)
        footer=QHBoxLayout(); back=QPushButton('Back'); back.clicked.connect(lambda:owner.go_step(5)); footer.addWidget(back); footer.addStretch()
        next_step=QPushButton('Electrophysiology'); next_step.clicked.connect(lambda:owner.go_step(7)); footer.addWidget(next_step)
        save=QPushButton('Save'); save.clicked.connect(owner.save_project); footer.addWidget(save)
        root.addLayout(footer)
        self.timer=QTimer(self); self.timer.setSingleShot(True); self.timer.timeout.connect(self._render)
        self._change_view()

    @staticmethod
    def _scroll(panel):
        scroll=QScrollArea(); scroll.setFrameShape(QScrollArea.Shape.NoFrame); scroll.setWidgetResizable(True); scroll.setWidget(panel)
        scroll.setStyleSheet('QScrollArea,QScrollArea>QWidget>QWidget{background:transparent;border:0;}'); return scroll

    @staticmethod
    def _table(headers,selection=False):
        table=QTableWidget(0,len(headers)); table.setHorizontalHeaderLabels(headers); table.verticalHeader().hide()
        table.verticalHeader().setDefaultSectionSize(29); table.setShowGrid(False); table.setWordWrap(False)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection if selection else QAbstractItemView.SelectionMode.NoSelection)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive); table.horizontalHeader().setStretchLastSection(True)
        for i in range(len(headers)): table.setColumnWidth(i,90 if i==0 else 80)
        table.setMinimumHeight(90); table.setMaximumHeight(220); table.setStyleSheet('QTableWidget{background:transparent;border:0;}'); return table

    def _slice_panel(self,title,axis,with_slider):
        panel=QWidget(); layout=QVBoxLayout(panel); layout.setContentsMargins(0,0,0,0); layout.setSpacing(4)
        header=QHBoxLayout(); header.addWidget(QLabel(title),1)
        canvas=ReconstructionSlice(); canvas.setMinimumSize(160,160)
        header.addWidget(view_button('fit','Reset view',lambda:self._reset_slice(canvas)))
        header.addWidget(view_button('rotate','Rotate view',canvas.rotate_view))
        from .desktop_figures import figure_button
        header.addWidget(figure_button(canvas)); layout.addLayout(header)
        layout.addWidget(canvas,1); slider=None
        if with_slider:
            slider=QSlider(Qt.Orientation.Horizontal); slider.valueChanged.connect(lambda value,a=axis:self._depth_changed(a,value)); layout.addWidget(slider)
        return panel,canvas,slider

    def _probe_plane_panel(self,title):
        from .desktop_probe_placement_views import ProbePlacementSlice
        panel=QWidget(); layout=QVBoxLayout(panel); layout.setContentsMargins(0,0,0,0); layout.setSpacing(4)
        header=QHBoxLayout(); header.addWidget(QLabel(title),1)
        canvas=ProbePlacementSlice(); canvas.setMinimumSize(160,160)
        canvas.placement_enabled=True; canvas.placement_read_only=True; canvas.placement_selected=False
        canvas.background_color='#000000'; canvas.show_axes=False; canvas.show_points=False
        canvas.drawing=False; canvas.point_mode='view'; canvas.set_interaction_tool('pan'); canvas.tools.hide()
        canvas.setToolTip('Pinch or wheel: zoom · Drag: pan')
        header.addWidget(view_button('fit','Reset view',lambda:self._reset_slice(canvas)))
        header.addWidget(view_button('rotate','Rotate view',canvas.rotate_view))
        from .desktop_figures import figure_button
        header.addWidget(figure_button(canvas)); layout.addLayout(header); layout.addWidget(canvas,1)
        return panel,canvas

    def _plane_item(self):
        if self.project is None: return None
        return next((item for item in self.items if item['id']==self.plane_probe.currentData()),None)

    def _plane_probe_changed(self,*args):
        if self._updating: return
        item=self._plane_item()
        if item is not None and self.report is not None:
            row=next((i for i,p in enumerate(self.report['probes']) if p['id']==item['id']),None)
            if row is not None: self.probe_table.selectRow(row)
        self.request_render()

    def _sync_probe_plane_views(self,source):
        if self._syncing: return
        self._syncing=True
        for canvas in self.probe_plane_canvases:
            if canvas is not source: canvas.restore_view(source.view_record())
        self._syncing=False

    def toggle_sidebar(self): self.sidebar.setVisible(self.sidebar.isHidden()); self.owner.schedule_save()

    def _change_view(self,*args):
        mode=self.view_choice.currentIndex(); self.views.setCurrentIndex(mode)
        if mode==1: self.ortho_grid.addWidget(self.volume_view,1,1); self.volume_view.show()
        else: self.layout_3d.addWidget(self.volume_view); self.volume_view.setVisible(mode==0)
        self.source.setVisible(mode==1); self.compare_axis.setVisible(mode==2); self.compare_channel.setVisible(mode==2)
        self.atlas_display.setVisible(mode in (2,3) or (mode==1 and self.source.currentData()=='atlas'))
        self.plane_probe.setVisible(mode==3); self.plane_stage.setVisible(mode==3)
        if mode==3: self._plane_probe_changed()
        self.request_render(); self.owner.schedule_save() if self.project else None

    def _reset_slice(self,canvas): canvas.restore_view({})

    def _sync_views(self,source):
        if self._syncing: return
        self._syncing=True
        for canvas in self.compare_canvases:
            if canvas is not source: canvas.restore_view(source.view_record())
        self._syncing=False

    def open(self,project,reference):
        self._closing=False; self._token+=1; self._cache=[]
        self.project=project; self.session=insertion_session(project)
        self.items=fitted_probes(self.session); self.rigid=placement_without_scale(project)
        context=(id(project),self.session['id'])
        if context==self._context and self.native:
            self._ready_views(reload=False); return
        self._context=context; t=self.owner.track_workspace
        def load(progress):
            progress(5,'Loading reconstruction')
            same=t.project is project and t.track_native is not None
            atlas=t.atlas_native if same else raw_volume(project.reference)
            brain=t.brain_native if same else raw_volume(project.moving)
            if same: electrode,gpu,limits=t.track_native,t.track_gpu,t.track_limits
            else: electrode,gpu,limits=load_track_data(project.track,project.preview_dimension)
            progress(35,'Finding atlas surface')
            surface=t._entry_surface if same and t._entry_surface is not None else build_atlas_surface(atlas)
            labels=t.labels if same else None
            if labels is None and project.labels_dir:
                progress(60,'Reading atlas regions')
                labels=load_label_atlas(project.labels_dir,project.reference,Path(tempfile.gettempdir())/'probemap'/'native-labels',progress,spacing_um_zyx=project.label_spacing)
            native=dict(atlas=atlas,brain=brain,electrodes=electrode)
            ranges=[]
            for volume in native.values():
                data=np.asarray(volume.data).ravel()[::max(1,volume.data.size//300000)]; data=data[np.isfinite(data)]
                low,high=np.percentile(data,(1,99.9)); ranges.append([float(low),float(max(low+1,high))])
            return native,gpu,ranges,surface,labels
        def ready(result):
            self.native,gpu,ranges,self.surface,self.labels=result
            self.volume_view.load_channels(reference,gpu,self.owner.moving)
            self.appearance.initialize(ranges,self.session.get('reconstruction_view',{}).get('appearance'))
            self._ready_views(reload=True)
        self.owner._start_job(load,ready,'Loading reconstruction')

    def _ready_views(self,reload):
        from .coarse import transformed_corners
        p=self.project
        corners=[transformed_corners(p.reference.shape_zyx,p.reference.spacing_um_zyx,np.eye(4)),
            transformed_corners(p.moving.shape_zyx,p.moving.spacing_um_zyx,self.rigid),
            transformed_corners(p.moving.shape_zyx,p.moving.spacing_um_zyx,p.total)]
        corners.extend(transformed_corners(p.track.shape_zyx,p.track.spacing_um_zyx,m) for m in (self.rigid,p.total))
        all_corners=np.vstack(corners); self.bounds=np.array([all_corners.min(0),all_corners.max(0)])
        self.depth_step=float(min(*p.reference.spacing_um_zyx,*p.moving.spacing_um_zyx))
        saved=self.session.get('reconstruction_view',{}); self._updating=True
        selected_plane=saved.get('comparison_plane',saved.get('axis',0)) if reload else self.compare_axis.currentData()
        self.compare_axis.blockSignals(True)
        while self.compare_axis.count()>3: self.compare_axis.removeItem(3)
        for item in self.items: self.compare_axis.addItem('Probe · '+item['name'],item['id'])
        self.compare_axis.setCurrentIndex(max(0,self.compare_axis.findData(selected_plane)))
        self.compare_axis.blockSignals(False)
        selected_probe=saved.get('probe_plane_id') if reload else self.plane_probe.currentData()
        self.plane_probe.blockSignals(True); self.plane_probe.clear()
        for item in self.items: self.plane_probe.addItem(item['name'],item['id'])
        self.plane_probe.setCurrentIndex(max(0,self.plane_probe.findData(selected_probe))); self.plane_probe.blockSignals(False)
        self._probe_offsets=copy.deepcopy(saved.get('probe_offsets_um',{}))
        depths=saved.get('depths_zyx_um',self.bounds.mean(0).tolist())
        for axis,slider in enumerate(self.sliders):
            slider.setRange(0,int(np.ceil(np.ptp(self.bounds,axis=0)[axis]/self.depth_step)))
            slider.setValue(int((depths[axis]-self.bounds[0,axis])/self.depth_step))
        if reload:
            self.plane_stage.setCurrentIndex(saved.get('probe_plane_stage',1))
            self.source.setCurrentIndex(max(0,self.source.findData(saved.get('source','atlas'))))
            self.compare_channel.setCurrentIndex(saved.get('comparison_channel',0))
            self.atlas_display.setCurrentIndex(max(0,self.atlas_display.findData(saved.get('atlas_display','overlay' if saved.get('regions') else 'atlas') if self.labels else 'atlas')))
            self.view_choice.setCurrentIndex(saved.get('view',0)); self.scene_mode.setCurrentIndex(saved.get('scene',0)); self.space.setCurrentIndex(saved.get('space',0))
            self.slab.setValue(saved.get('thickness_um',25)); self.unit.setCurrentIndex(saved.get('unit',0))
            self.probe_opacity_inside.setValue(saved.get('probe_opacity_inside',50))
            self.probe_opacity_outside.setValue(saved.get('probe_opacity_outside',100))
            self.surface_entries_check.setChecked(saved.get('surface_entries',False))
            self.region_overlay.setChecked(saved.get('regions',False)); self.volume_view.layers.restore(saved.get('layers',{}))
            if saved.get('rendering'): self.volume_view.restore_render_settings(saved['rendering'])
            else:
                records=self.volume_view.render_settings_record()
                records[0].update(colormap='gray',rendering='iso',blending='translucent_no_depth',iso_threshold=.1)
                self.volume_view.restore_render_settings(records)
            self.volume_view.show_axes.setChecked(saved.get('axes',False))
            for check,value in zip(self.volume_view.flip_checks,saved.get('flip',[False]*3)): check.setChecked(value)
            if 'layers' not in saved:
                for i,check in enumerate(self.volume_view.layers.checks): check.setChecked(i==0)
                for i,slider in enumerate(self.volume_view.layers.sliders): slider.setValue(25 if i==0 else 35)
            self.sidebar.setVisible(saved.get('sidebar',True)); self.sidebar.setCurrentIndex(saved.get('tab',0))
            self.render_group.setChecked(saved.get('rendering_expanded',False))
            self.render_panel.sync()
        self.regions.initialize(self.labels.regions if self.labels else [],self.session.get('regions',{}),saved.get('region_display',{}))
        self.sidebar.setTabEnabled(3,self.labels is not None)
        for index in (1,2): self.atlas_display.model().item(index).setEnabled(self.labels is not None)
        self._refresh_filters(); self._updating=False
        self.report=project_reconstruction_report(p,self.surface,self.labels); self.refresh_report()
        for index,key in enumerate(('atlas','brain','brain','electrodes','electrodes')):
            self.source.setItemData(index,str(self.native[key].path),Qt.ItemDataRole.ToolTipRole)
        self.notice.setText('' if self.items else 'Place a probe in Probes to see its reconstruction.')
        self._axis_changed(); self._change_view(); self._draw_3d()
        if reload:
            self.volume_view.reset_view()
            self.volume_view.restore_camera(saved.get('camera',{}))
            for canvas,record in zip(self.canvases+self.compare_canvases,saved.get('slice_views',[])): canvas.restore_view(record)
            for canvas,record in zip(self.probe_plane_canvases,saved.get('probe_plane_views',[])): canvas.restore_view(record)
        self.owner.schedule_save()

    def _refresh_filters(self):
        saved=self.session.get('reconstruction_view',{})
        hidden=set(saved.get('hidden_probes',[]))
        for table in (self.probe_filter,self.recording_filter): table.blockSignals(True); table.setRowCount(0)
        for item in self.items:
            row=self.probe_filter.rowCount(); self.probe_filter.insertRow(row)
            self.probe_filter.setItem(row,0,QTableWidgetItem(item['name']))
            check=QTableWidgetItem(); check.setData(Qt.ItemDataRole.UserRole,item['id']); check.setCheckState(Qt.CheckState.Unchecked if item['id'] in hidden else Qt.CheckState.Checked); self.probe_filter.setItem(row,1,check)
            for rec in item.get('recordings',[]):
                row=self.recording_filter.rowCount(); self.recording_filter.insertRow(row)
                self.recording_filter.setItem(row,0,QTableWidgetItem(rec['name'])); self.recording_filter.setItem(row,1,QTableWidgetItem(item['name']))
                check=QTableWidgetItem(); check.setData(Qt.ItemDataRole.UserRole,rec['id']); check.setCheckState(Qt.CheckState.Checked if rec.get('visible',True) else Qt.CheckState.Unchecked); self.recording_filter.setItem(row,2,check)
        for table in (self.probe_filter,self.recording_filter): table.blockSignals(False)

    def _visible_items(self):
        hidden={self.probe_filter.item(r,1).data(Qt.ItemDataRole.UserRole) for r in range(self.probe_filter.rowCount()) if self.probe_filter.item(r,1).checkState()!=Qt.CheckState.Checked}
        result=copy.deepcopy([i for i in self.items if i['id'] not in hidden])
        for item in result:
            item['probe_visible']=True
        return result

    def _visibility_header(self,table,column):
        header=CheckHeader(table,{column}); header.setStretchLastSection(True); table.setHorizontalHeader(header)
        header.toggled.connect(lambda col,visible:self._filter_all(table,col,visible))

    def _filter_all(self,table,column,visible):
        table.blockSignals(True)
        for row in range(table.rowCount()): table.item(row,column).setCheckState(Qt.CheckState.Checked if visible else Qt.CheckState.Unchecked)
        table.blockSignals(False)
        if table is self.recording_filter: self._sync_recording_visibility()
        self._annotations_changed()

    def _filter_changed(self,*args):
        if self._updating: return
        if args and args[0].tableWidget() is self.recording_filter: self._sync_recording_visibility()
        self._annotations_changed()

    def _sync_recording_visibility(self):
        states={self.recording_filter.item(r,2).data(Qt.ItemDataRole.UserRole):
            self.recording_filter.item(r,2).checkState()==Qt.CheckState.Checked
            for r in range(self.recording_filter.rowCount())}
        for item in self.items:
            for rec in item.get('recordings',[]): rec['visible']=states.get(rec['id'],rec.get('visible',True))

    def _annotations_changed(self): self._draw_3d(); self.draw_cached(); self.owner.schedule_save() if self.project else None

    def refresh_report(self,*args):
        if self.report is None: return
        selected=self.probe_table.currentRow()
        plane_id=self.plane_probe.currentData() if self.view_choice.currentIndex()==3 else self.compare_axis.currentData()
        selected=next((i for i,p in enumerate(self.report['probes']) if p['id']==plane_id),selected)
        self.probe_table.blockSignals(True)
        self.probe_table.setRowCount(len(self.report['probes']))
        for row,probe in enumerate(self.report['probes']):
            for col,value in enumerate((probe['name'],self._distance(probe['mean_depth_source_um']),f"{probe['valid_depth_shanks']}/4",f"{probe['tilt_atlas_y_deg']:.1f}")):
                cell=QTableWidgetItem(value)
                if col==0: cell.setForeground(QColor(probe['color']))
                self.probe_table.setItem(row,col,cell)
        self.probe_table.setFixedHeight(50+29*min(6,max(1,len(self.report['probes']))))
        self.probe_table.blockSignals(False)
        if self.report['probes']: self.probe_table.selectRow(max(0,min(selected,len(self.report['probes'])-1)))
        self.report_note.setText('Estimated atlas surface · lengths in source scan '+self.unit.currentText())
        if not self.report['probes']:
            self.shank_table.setRowCount(0); self.region_table.setRowCount(0)
            self.dimension_label.clear(); self.recording_summary.clear()
            self.region_heading.hide(); self.region_table.hide()
        self._select_report_probe()

    def _distance(self,value):
        return '—' if value is None else (f'{value/1000:.3f}' if self.unit.currentIndex() else f'{value:.0f}')

    def _select_report_probe(self):
        if self.report is None: return
        row=self.probe_table.currentRow()
        if not 0<=row<len(self.report['probes']): return
        probe=self.report['probes'][row]; g=probe['shanks'][0]
        if self.view_choice.currentIndex()==2 and self.compare_axis.currentIndex()>=3:
            choice=self.compare_axis.findData(probe['id'])
            if choice>=0 and choice!=self.compare_axis.currentIndex(): self.compare_axis.setCurrentIndex(choice)
        if self.view_choice.currentIndex()==3:
            choice=self.plane_probe.findData(probe['id'])
            if choice>=0 and choice!=self.plane_probe.currentIndex(): self.plane_probe.setCurrentIndex(choice)
        self.dimension_label.setText(f"{probe['model']} · {self._distance(g['width_source_um'])} × {self._distance(g['thickness_source_um'])} {self.unit.currentText()}\nPitch · {self._distance(probe['shank_pitch_source_um'])} {self.unit.currentText()}")
        self.shank_table.setRowCount(4); self.shank_table.setFixedHeight(172)
        for row,s in enumerate(probe['shanks']):
            for col,value in enumerate((str(s['shank']),self._distance(s['depth_source_um']),self._distance(s['length_source_um']))): self.shank_table.setItem(row,col,QTableWidgetItem(value))
        rows=[]
        for s in probe['shanks']:
            for region in s['regions']:
                count=sum(c['shank']==s['shank'] and c['region_id']==region['id'] for c in probe['channels'])
                rows.append((region['name'],str(s['shank']),self._distance(region['length_source_um']),str(count)))
        self.region_table.setRowCount(len(rows))
        self.region_table.setVisible(bool(rows)); self.region_heading.setVisible(bool(rows))
        for row,values in enumerate(rows):
            for col,value in enumerate(values): self.region_table.setItem(row,col,QTableWidgetItem(value))
        counts={}
        for c in probe['channels']: counts[c['recording']]=counts.get(c['recording'],0)+1
        self.recording_summary.setText('\n'.join(f'{name} · {count} contacts' for name,count in counts.items())+('' if self.labels else '\nAtlas regions unavailable'))

    def _dispose_visuals(self):
        for visual in self._visuals: visual.parent=None
        self._visuals=[]
        if self.surface_visuals is not None: self.surface_visuals.dispose(); self.surface_visuals=None

    def _scene_changed(self,*args):
        if self._updating: return
        if self.scene_mode.currentIndex()==3:
            check=self.volume_view.layers.labels_check
            check.blockSignals(True); check.setChecked(True); check.blockSignals(False)
        self._draw_3d()

    def _labels_toggled(self,*args):
        if not self._updating: self._draw_3d()

    def _space_changed(self,*args):
        if self._updating: return
        before=self.space.currentIndex()==1
        for i,value in ((0,not before),(2,before)):
            check=self.volume_view.layers.checks[i]; check.blockSignals(True); check.setChecked(value); check.blockSignals(False)
        self._draw_3d()

    def _draw_3d(self,*args):
        if self._updating or not self.native or self.volume_view.canvas is None: return
        from vispy import scene
        self._dispose_visuals(); v=self.volume_view; mode=self.scene_mode.currentIndex(); before=self.space.currentIndex()==1
        if mode in (2,3) and before:
            self.space.blockSignals(True); self.space.setCurrentIndex(0); self.space.blockSignals(False); before=False
        matrix=self.rigid if before else self.project.total
        opacity=[s.value()/100 for s in v.layers.sliders]
        v.surface_entries_active=mode==2
        v.update_alignment(matrix,*opacity)
        v._apply_layer_visibility()
        if before: v.visuals[0].visible=False
        items=self._visible_items(); v.set_points([]); v.set_tracks([])
        v.annotation_bounds_zyx=[]
        self.surface_entries_check.setVisible(mode!=2)
        self.track_opacity_controls.setVisible(mode!=2)
        if mode==2 or self.surface_entries_check.isChecked():
            from .desktop_surface_entries import SurfaceEntryVisuals
            self.surface_visuals=SurfaceEntryVisuals(v,self.surface,show_surface=mode==2,
                matrix=matrix@np.linalg.inv(self.project.total))
            entries=[e for item in items for e in probe_surface_entries(item,self.project.total,self.surface)]
            self.surface_visuals.set_entries(entries)
        if mode==2:
            v.surface_entry_bounds=np.array([self.surface.vertices_zyx_um.min(0),self.surface.vertices_zyx_um.max(0)])
            v.canvas.update(); self.owner.schedule_save()
            return
        v.surface_entry_bounds=None
        for item in items:
            color=QColor(item['color']).getRgbF()
            key=(id(self.surface),self.project.total.tobytes(),repr((item['probe']['pose'],item['probe']['geometry'])))
            cached=self._shank_sections.get(item['id'])
            if cached is None or cached[0]!=key:
                cached=(key,probe_shank_sections(item,self.project.total,self.surface)); self._shank_sections[item['id']]=cached
            for inside,slider in ((True,self.probe_opacity_inside),(False,self.probe_opacity_outside)):
                sections=cached[1][inside]
                if not sections or slider.value()==0: continue
                vertices,faces=shank_mesh(item,matrix,sections)
                mesh=scene.visuals.Mesh(vertices=vertices,faces=faces,color=(*color[:3],slider.value()/100),shading=None,parent=v.view.scene)
                mesh.set_gl_state('translucent',depth_test=False,cull_face=False); mesh.order=20; self._visuals.append(mesh)
            v.annotation_bounds_zyx.extend(shanks(item,matrix).reshape(-1,3).tolist())
            for rec_index,rec in enumerate(item.get('recordings',[])):
                if not rec.get('visible',True) or rec.get('opacity',1)<=0: continue
                rgba=recording_rgba(rec.get('color') or item['color'],rec.get('opacity',1))
                channels,_,sources=recording_sites(item,rec)
                if mode==1 and channels:
                    center=transform_points(np.mean(sources,axis=0)[None],matrix)[0][::-1]
                    center[2]+=rec_index*min(self.project.reference.spacing_um_zyx)*3
                    label=scene.visuals.Text(f"{item['name']} · {rec['name']}",pos=center,color='white',font_size=9,anchor_x='left',anchor_y='bottom',parent=v.view.scene)
                    label.set_gl_state('translucent',depth_test=False); label.order=35; self._visuals.append(label)
                if channels:
                    vertices,faces=contact_mesh(item,sources,matrix)
                    mesh=scene.visuals.Mesh(vertices=vertices,faces=faces,color=rgba,shading=None,parent=v.view.scene)
                    mesh.set_gl_state('translucent',depth_test=False,cull_face=False); mesh.order=28; self._visuals.append(mesh)
        if not before and self.labels and v.layers.labels_check.isChecked():
            self._ensure_region_surfaces()
        else:
            for visual in v.label_visuals.values(): visual.visible=False
        v.canvas.update(); self.owner.schedule_save()

    def _ensure_region_surfaces(self):
        v=self.volume_view
        if getattr(self,'_region_meshes',None) is not None:
            if not v.label_visuals: v.set_label_meshes(self._region_meshes)
            v.set_label_styles(self.regions.styles)
            v._apply_layer_visibility(); return
        if getattr(self,'_regions_loading',False): return
        self._regions_loading=True; context=self._context; self._regions_context=context
        from .desktop_app import Job
        from .desktop_layers import label_surfaces
        job=Job(lambda progress:label_surfaces(self.labels,max_dimension=80)); self._jobs.add(job)
        def done(meshes=None,error=None):
            self._jobs.discard(job)
            if getattr(self,'_regions_context',None)==context: self._regions_loading=False
            if self._closing or context!=self._context: return
            if error: self.owner.statusBar().showMessage(error)
            else: self._region_meshes=meshes; self._draw_3d()
        job.signals.done.connect(lambda value:done(value)); job.signals.failed.connect(lambda error:done(error=error)); QThreadPool.globalInstance().start(job)

    def _regions_changed(self):
        if self._updating: return
        self.draw_cached()
        if self.volume_view.canvas is not None:
            self.volume_view.set_label_styles(self.regions.styles)
            if self.space.currentIndex()==1 or self.scene_mode.currentIndex()==2:
                for visual in self.volume_view.label_visuals.values(): visual.visible=False
        self.owner.schedule_save() if self.project else None

    def _source_changed(self,*args):
        self.atlas_display.setVisible(self.view_choice.currentIndex() in (2,3) or (self.view_choice.currentIndex()==1 and self.source.currentData()=='atlas'))
        self.request_render()

    def _probe_plane_item(self):
        key=self.compare_axis.currentData()
        return next((item for item in self.items if item['id']==key),None) if self.project else None

    def _depth_changed(self,axis,value):
        if self._updating or not self.native: return
        if axis==self.compare_axis.currentIndex():
            self.compare_slider.blockSignals(True); self.compare_slider.setValue(value); self.compare_slider.blockSignals(False)
        self.request_render()

    def _comparison_depth_changed(self,value):
        if self._updating: return
        item=self._probe_plane_item()
        if item is not None:
            self._probe_offsets[item['id']]=float(value*self._probe_step)
            self.request_render()
        elif self.compare_axis.currentIndex()<3:
            self.sliders[self.compare_axis.currentIndex()].setValue(value)

    def _axis_changed(self,*args):
        self.compare_slider.blockSignals(True)
        item=self._probe_plane_item()
        if item is not None and self.native:
            if self.report is not None and self.view_choice.currentIndex()==2:
                row=next((i for i,p in enumerate(self.report['probes']) if p['id']==item['id']),None)
                if row is not None: self.probe_table.selectRow(row)
            self._probe_step=float(min(self.project.track.spacing_um_zyx))
            limit=int(np.ceil(np.linalg.norm(np.asarray(self.project.track.shape_zyx)*self.project.track.spacing_um_zyx)/self._probe_step))
            self.compare_slider.setRange(-limit,limit)
            self.compare_slider.setValue(int(round(self._probe_offsets.get(item['id'],0.)/self._probe_step)))
            self.compare_slider.setToolTip('Move through parallel probe planes · source scan µm')
        else:
            axis=min(2,self.compare_axis.currentIndex())
            slider=self.sliders[axis]
            self.compare_slider.setRange(slider.minimum(),slider.maximum()); self.compare_slider.setValue(slider.value())
            self.compare_slider.setToolTip('Slice position')
        self.compare_slider.blockSignals(False)
        self.request_render()

    def request_render(self,*args):
        if self._updating or not self.native or self._closing: return
        self._token+=1; self._pending=True; self.timer.start(35)

    def _render(self):
        if self._running or not self.native or self.owner.step!=6: return
        token=self._token; mode=self.view_choice.currentIndex()
        if mode==0: self._pending=False; return
        if mode==3:
            self._render_probe_plane(token); return
        from .desktop_app import Job
        requests=[]; probe_planes=None
        item=self._probe_plane_item() if mode==2 else None
        if item is not None:
            from .coarse import transformed_corners
            corners=lambda spec:transformed_corners(spec.shape_zyx,spec.spacing_um_zyx,np.eye(4))
            probe_planes=probe_comparison_geometries(item,self.rigid,self.project.total,
                [corners(self.project.moving),corners(self.project.track)],corners(self.project.reference),
                self._probe_offsets.get(item['id'],0.))
        if mode==1:
            keys=[self.source.currentData()]*3; axes=range(3); canvases=self.canvases
        else:
            channel='brain' if self.compare_channel.currentIndex()==0 else 'electrodes'
            keys=['atlas',channel+'_before',channel+'_after']; axes=[min(2,self.compare_axis.currentIndex())]*3; canvases=self.compare_canvases
        for key,axis,canvas in zip(keys,axes,canvases):
            if probe_planes is not None:
                geometry=probe_planes['before' if key.endswith('before') else 'after']
            else:
                depth=self.bounds[0,axis]+self.sliders[axis].value()*self.depth_step
                geometry=orthogonal_geometry(self.bounds,axis,depth)
            channel=key.split('_')[0]; matrix=np.eye(4) if channel=='atlas' else (self.rigid if key.endswith('before') else self.project.total)
            probe_matrix=self.project.total if channel=='atlas' else matrix
            order=0 if self.appearance.records[('atlas','brain','electrodes').index(channel)]['interpolation']=='nearest' else 1
            requests.append((canvas,channel,matrix.copy(),probe_matrix.copy(),geometry,order))
        native=self.native; labels=self.labels; total=self.project.total.copy()
        def render(progress):
            result=[]
            for canvas,channel,matrix,probe_matrix,geometry,order in requests:
                world=plane_world(geometry)
                pixels=sample_world(native[channel],matrix,world,order=order).reshape(geometry['shape_hw'])
                region_world=world if channel=='atlas' else transform_points(world,total@np.linalg.inv(matrix))
                ids=labels.ids_at(region_world).reshape(geometry['shape_hw']) if labels is not None else None
                result.append((canvas,channel,pixels,ids,probe_matrix,geometry))
            return result
        job=Job(render); self._jobs.add(job); self._running=True; self._pending=False
        def done(result=None,error=None):
            self._jobs.discard(job); self._running=False
            if self._closing: return
            if token==self._token and self.owner.step==6:
                if error: self.owner.statusBar().showMessage(error)
                else: self._cache=result; self.draw_cached()
            if self._pending or token!=self._token: self.timer.start(0)
        job.signals.done.connect(lambda value:done(value)); job.signals.failed.connect(lambda error:done(error=error)); QThreadPool.globalInstance().start(job)

    def _render_probe_plane(self,token):
        from .coarse import transformed_corners
        from .desktop_app import Job
        item=self._plane_item()
        if item is None:
            self._cache=[]; self._pending=False
            for canvas in self.probe_plane_canvases:
                canvas.image=None; canvas.probe_polygons=[]; canvas.placement_contacts=[]
                canvas.probe_sites=[]; canvas.region_labels=[]; canvas.update()
            return
        p=self.project
        corners=transformed_corners(p.track.shape_zyx,p.track.spacing_um_zyx,np.eye(4))
        geometries=saved_probe_plane_geometries(item,self.rigid,p.total,corners)
        before=self.plane_stage.currentIndex()==0
        stage='before' if before else 'after'; display=self.rigid.copy() if before else p.total.copy()
        native=self.native; labels=self.labels; total=p.total.copy()
        requests=[(self.probe_plane_canvases[0],'atlas',np.eye(4),total,geometries['atlas']),
            (self.probe_plane_canvases[1],'brain',display,display,geometries[stage]),
            (self.probe_plane_canvases[2],'electrodes',display,display,geometries[stage])]
        orders={key:0 if settings['interpolation']=='nearest' else 1 for key,settings in zip(('atlas','brain','electrodes'),self.appearance.records)}
        native_world=plane_world(geometries['native'])
        def render(progress):
            result=[]
            for canvas,channel,matrix,probe_matrix,geometry in requests:
                world=plane_world(geometry)
                # Before is the identical original sampling lattice used by Probes.
                points=native_world if before and channel!='atlas' else world
                pull=np.eye(4) if before and channel!='atlas' else matrix
                pixels=sample_world(native[channel],pull,points,order=orders[channel]).reshape(geometry['shape_hw'])
                region_world=world if channel=='atlas' else transform_points(world,total@np.linalg.inv(matrix))
                ids=labels.ids_at(region_world).reshape(geometry['shape_hw']) if labels is not None else None
                result.append((canvas,channel,pixels,ids,probe_matrix,geometry))
            return result
        job=Job(render); self._jobs.add(job); self._running=True; self._pending=False
        def done(result=None,error=None):
            self._jobs.discard(job); self._running=False
            if self._closing: return
            if token==self._token and self.owner.step==6:
                if error: self.owner.statusBar().showMessage(error)
                else: self._cache=result; self.draw_cached()
            if self._pending or token!=self._token: self.timer.start(0)
        job.signals.done.connect(lambda value:done(value)); job.signals.failed.connect(lambda error:done(error=error)); QThreadPool.globalInstance().start(job)

    def _draw_saved_probe(self,canvas,items,matrix):
        from .probe_placement import probe_contours,contact_preview,contact_side
        selected=self.plane_probe.currentData()
        item=next((i for i in items if i['id']==selected),None)
        canvas.probe_polygons=[]; canvas.placement_contacts=[]; canvas.probe_sites=[]
        canvas.insertion_points=[]; canvas.insertion_tracks=[]; canvas.track_lines=[]
        if item is None: return
        rgb=QColor(item['color']).getRgbF()[:3]
        canvas.probe_polygons=[(poly.tolist(),(*rgb,.4)) for poly in probe_contours(item,matrix)]
        canvas.placement_contacts=contact_preview(item,matrix).tolist()
        if item.get('source_plane'):
            canvas.placement_contact_front=contact_side(item,item['source_plane'])>0
        else: canvas.placement_contact_front=True
        for rec in item.get('recordings',[]):
            if not rec.get('visible',True) or rec.get('opacity',1)<=0: continue
            channels,_,sources=recording_sites(item,rec)
            rgba=recording_rgba(rec.get('color') or item['color'],rec.get('opacity',1))
            canvas.probe_sites.extend(dict(atlas_zyx_um=point,rgba=rgba,
                label=f"{rec['name']} · ch {channel['channel']}",depth=None,recording_id=rec['id'])
                for point,channel in zip(transform_points(sources,matrix),channels))
        canvas.update()

    def draw_cached(self,*args):
        if self._updating or not self.native or not self.appearance.records: return
        plane_view=self.view_choice.currentIndex()==3 or (self.view_choice.currentIndex()==2 and self._probe_plane_item() is not None)
        atlas_ids=next((entry[3] for entry in self._cache if entry[1]=='atlas'),None)
        if plane_view and atlas_ids is not None: self.regions.prioritize(np.unique(atlas_ids))
        items=self._visible_items()
        for canvas,channel,pixels,ids,matrix,geometry in self._cache:
            index=('atlas','brain','electrodes').index(channel); settings=self.appearance.records[index]
            self.appearance.observe(index,pixels)
            gray=slice_color(pixels,settings)
            label_mode=self.atlas_display.currentData() if channel=='atlas' else ('overlay' if self.region_overlay.isChecked() else 'atlas')
            rgb,names=label_slice_image(gray,ids,self.labels.regions if self.labels else [],self.regions.styles,
                mode=label_mode,opacity=self.regions.opacity.value()/100,names=self.regions.names.isChecked())
            canvas.region_labels=names
            canvas.smooth_interpolation=settings['interpolation']=='linear'
            canvas.display_interpolation=settings['interpolation']
            canvas.set_plane_image(rgb,geometry)
            from .desktop_figures import slice_sampler
            source_matrix=np.eye(4) if channel=='atlas' else matrix
            canvas.figure_sampler=slice_sampler([self.native[channel]],[source_matrix],[settings],visible_pixels=[pixels],labels=self.labels,
                label_matrix=np.eye(4) if channel=='atlas' else self.project.total@np.linalg.inv(matrix),
                label_mode=label_mode,styles=self.regions.styles,label_opacity=self.regions.opacity.value()/100)
            if canvas in self.probe_plane_canvases:
                self._draw_saved_probe(canvas,items,matrix)
            else:
                canvas.reconstruction=slice_reconstruction(items,matrix,geometry,self.slab.value()*geometry.get('source_normal_scale',1.))
        self.owner.schedule_save()

    def _appearance_changed(self):
        orders=[r['interpolation'] for r in self.appearance.records]
        if orders!=getattr(self,'_sampling_orders',None):
            self._sampling_orders=orders; self.request_render()
        self.draw_cached()

    def auto_contrast(self,index):
        if not self.native: return
        volume=self.native[('atlas','brain','electrodes')[index]]
        data=np.asarray(volume.data).ravel()[::max(1,volume.data.size//300000)]
        low,high=np.percentile(data,(1,99.9)); self.appearance.records[index]['contrast']=[float(low),float(max(low+1,high))]
        self.appearance.sync(); self.draw_cached()

    def capture_state(self):
        if self.project is None or not self.native or self._updating: return
        self.session['reconstruction_view']=dict(view=self.view_choice.currentIndex(),scene=self.scene_mode.currentIndex(),space=self.space.currentIndex(),
            probe_plane_id=self.plane_probe.currentData(),probe_plane_stage=self.plane_stage.currentIndex(),
            probe_plane_views=[c.view_record() for c in self.probe_plane_canvases],
            source=self.source.currentData(),axis=min(2,self.compare_axis.currentIndex()),comparison_plane=self.compare_axis.currentData(),
            probe_offsets_um=copy.deepcopy(self._probe_offsets),atlas_display=self.atlas_display.currentData(),region_display=self.regions.state_record(),comparison_channel=self.compare_channel.currentIndex(),
            depths_zyx_um=[float(self.bounds[0,a]+slider.value()*self.depth_step) for a,slider in enumerate(self.sliders)],
            appearance=copy.deepcopy(self.appearance.records),thickness_um=self.slab.value(),unit=self.unit.currentIndex(),
            probe_opacity_inside=self.probe_opacity_inside.value(),probe_opacity_outside=self.probe_opacity_outside.value(),
            surface_entries=self.surface_entries_check.isChecked(),regions=self.region_overlay.isChecked(),
            sidebar=not self.sidebar.isHidden(),tab=self.sidebar.currentIndex(),layers=self.volume_view.layers.record(),rendering_expanded=self.render_group.isChecked(),
            slice_views=[c.view_record() for c in self.canvases+self.compare_canvases],
            rendering=self.volume_view.render_settings_record(),camera=self.volume_view.camera_record(),
            axes=self.volume_view.show_axes.isChecked(),flip=self.volume_view.flip_record(),
            hidden_probes=[self.probe_filter.item(r,1).data(Qt.ItemDataRole.UserRole) for r in range(self.probe_filter.rowCount()) if self.probe_filter.item(r,1).checkState()!=Qt.CheckState.Checked])

    def open_rendering(self):
        self.sidebar.show(); self.sidebar.setCurrentIndex(1); self.render_group.setChecked(True)

    def open_export(self):
        self.owner.open_export()
        dialog=getattr(self.owner,'export_dialog',None)
        if dialog is not None: dialog.kind.setCurrentIndex(dialog.kind.findData('reconstruction'))

    def write_report(self,folder):
        if self.report is not None and self.project is self.owner.project and self.project.fit_is_current:
            self.report=project_reconstruction_report(self.project,self.surface,self.labels)
            write_reconstruction_report(self.report,folder)

    def reset(self):
        self._token+=1; self.timer.stop(); self._dispose_visuals(); self._cache=[]; self.native={}; self._context=None
        self.project=self.session=self.report=self.surface=self.labels=None; self._region_meshes=None; self._regions_loading=False; self._regions_context=None; self._shank_sections={}
        for canvas in self.canvases+self.compare_canvases: canvas.image=None; canvas.reconstruction=dict(lines=[],bands=[],sites=[]); canvas.update()
        for canvas in self.probe_plane_canvases: canvas.image=None; canvas.probe_polygons=[]; canvas.placement_contacts=[]; canvas.probe_sites=[]; canvas.region_labels=[]; canvas.update()

    def stop(self):
        self._closing=True; self._token+=1; self.timer.stop(); self._dispose_visuals()

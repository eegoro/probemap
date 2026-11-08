"""Quiet electrophysiology workspace. All sources are local to this page."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from threading import Event

import numpy as np
from qtpy.QtCore import QObject, Signal, QThreadPool, QTimer, Qt
from qtpy.QtWidgets import (QAbstractItemView, QCheckBox, QDoubleSpinBox,
    QFileDialog, QHBoxLayout, QLabel, QPushButton,
    QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
    QFormLayout, QHeaderView, QMenu, QScrollArea, QSizePolicy, QStackedWidget, QSlider, QGroupBox, QColorDialog, QMessageBox)

from .desktop_widgets import PopupComboBox, view_button
from .desktop_tables import TableHeightGrip, ColorDotDelegate
from .desktop_ephys_regions import EphysRegions
from .desktop_ephys_levels import DepthLevels
from .desktop_ephys_board import PlotBoard
from .desktop_ephys_probe import draw_probe, hover_contact, depth_limits
from .desktop_task_progress import TaskProgress
from .desktop_ephys_table import EphysInsertionDelegate, SortingCombo, TimestampDelegate, recording_time
from .ephys import discover, import_folders, sorting_source
from .ephys_analysis import (BANDS, boundary_candidates, load_sorting, load_summary,
                            prepare_recording, prepare_sorting, fingerprint)
from .ephys_anatomy import prepare_anatomy
from .ephys_cache import recording_root, migrate_recording_cache
from .ephys_portable import sync_exports, merge_scanned_exports
from .insertion_model import insertion_session


class EphysJobSignals(QObject):
    done = Signal(object)
    failed = Signal(str)
    progress = Signal(float, str)


class UnitTable(QTableWidget):
    """A visibility checkbox must not also select/load a different unit."""
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            cell = self.itemAt(event.position().toPoint())
            if cell is not None and cell.column() == 0 and event.position().x()-self.visualItemRect(cell).left() < 24:
                cell.setCheckState(Qt.CheckState.Unchecked if cell.checkState() == Qt.CheckState.Checked else Qt.CheckState.Checked)
                event.accept(); return
        super().mousePressEvent(event)


class ElectrophysiologyWorkspace(QWidget):
    def __init__(self, owner):
        super().__init__(); self.owner = owner; self.project = None
        self._jobs = set(); self._cancel = Event(); self._busy = False; self._token = 0
        self._updating = False; self._closing = False; self.summary = self.sorting = self.anatomy = None
        self.notebook=None; self._notebook_attempted=None; self.profile_fits=None; self._overview_attempted=None
        self._window_auto=False
        self._depth_axes=[]; self._depth_callbacks=[]
        self._anatomy_key = None; self._anatomy_resources = None; self.comparisons = []; self.unit_view = None; self._canvas = None; self._active_id = None; self._signal_valid = {}
        self.setObjectName('ephysWorkspace')
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        from qtpy.QtGui import QColor, QPalette
        palette = self.palette(); palette.setColor(QPalette.ColorRole.Window, QColor('#242b36'))
        for role in (QPalette.ColorRole.Base, QPalette.ColorRole.AlternateBase, QPalette.ColorRole.Button):
            palette.setColor(role, QColor('#242b36'))
        for role in (QPalette.ColorRole.WindowText, QPalette.ColorRole.Text, QPalette.ColorRole.ButtonText):
            palette.setColor(role, QColor('#e5eaf2'))
        self.setPalette(palette)
        self.setStyleSheet('QWidget#ephysWorkspace{background:#242b36;} QLabel{background:transparent;} QTableWidget{background:transparent;border:0;} QHeaderView::section{background:transparent;color:#9aafc1;border:0;} QScrollBar:horizontal{background:transparent;height:6px;margin:0;} QScrollBar:vertical{background:transparent;width:6px;margin:0;} QScrollBar::handle{background:#45556a;border-radius:3px;min-width:24px;min-height:24px;} QScrollBar::add-line,QScrollBar::sub-line{width:0;height:0;} QScrollBar::add-page,QScrollBar::sub-page{background:transparent;}')
        root = QVBoxLayout(self); root.setContentsMargins(12, 12, 12, 8)
        heading = QHBoxLayout()
        self.collapse = view_button('sidebar', 'Show or hide settings', self.toggle_sidebar); heading.addWidget(self.collapse)
        title = QLabel('Electrophysiology'); title.setStyleSheet('font-size:22px;font-weight:600;')
        heading.addWidget(title); heading.addStretch()
        root.addLayout(heading)
        self.splitter = splitter = QSplitter(Qt.Orientation.Horizontal); root.addWidget(splitter, 1)
        self.sidebar = QScrollArea(); self.sidebar.setWidgetResizable(True)
        self.sidebar.setFrameShape(QScrollArea.Shape.NoFrame)
        self.sidebar.setMinimumWidth(300); self.sidebar.setMaximumWidth(16777215)
        self.sidebar.setStyleSheet('QScrollArea,QScrollArea>QWidget>QWidget{background:transparent;border:0;}')
        splitter.setChildrenCollapsible(False); splitter.setHandleWidth(7)
        splitter.setStyleSheet('QSplitter::handle{background:#354354;} QSplitter::handle:hover{background:#7ac7e5;}')
        settings = QWidget(); side = QVBoxLayout(settings); side.setContentsMargins(6, 8, 12, 0); side.setSpacing(10)
        self.sidebar.setWidget(settings); splitter.addWidget(self.sidebar)
        library_heading = QHBoxLayout(); self.library_caption = QLabel('Recordings'); library_heading.addWidget(self.library_caption, 1)
        self.add = view_button('add', 'Add recording folders', self.add_folders); library_heading.addWidget(self.add)
        side.addLayout(library_heading)
        self.table = QTableWidget(0, 6); self.table.setProperty('persistent_selection', True); self.table.setHorizontalHeaderLabels(['Name', 'Insertion', 'Time', 'Kilosort', 'Signal', 'Color'])
        self.table.verticalHeader().hide(); self.table.verticalHeader().setDefaultSectionSize(42)
        self.table.setShowGrid(False); self.table.setWordWrap(False); self.table.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked | QAbstractItemView.EditTrigger.EditKeyPressed)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.horizontalHeader().moveSection(5,2)
        for column, width in enumerate((95, 72, 85, 80, 52, 45)): self.table.setColumnWidth(column, width)
        self.table.setItemDelegateForColumn(0, TimestampDelegate(self.table))
        self.table.setItemDelegateForColumn(2, TimestampDelegate(self.table))
        self.table.setItemDelegateForColumn(1, EphysInsertionDelegate(self, self.table))
        self.table.setItemDelegateForColumn(5, ColorDotDelegate(self.table))
        self.table.setFixedHeight(132)
        self.table.itemSelectionChanged.connect(self._selection_changed)
        self.table.itemChanged.connect(self._recording_edited)
        self.table.cellClicked.connect(self._recording_clicked)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._recording_menu)
        self.table.horizontalHeader().sectionResized.connect(self._table_resized)
        side.addWidget(self.table); side.addWidget(TableHeightGrip(self.table))
        self.note = QLabel('Add recording folders.'); self.note.setWordWrap(True)
        self.note.setStyleSheet('color:#9aafc1;'); side.addWidget(self.note)
        self.units = UnitTable(0, 5); self.units.setProperty('persistent_selection', True); self.units.setHorizontalHeaderLabels(['Show · Unit', 'Depth · µm', 'Raster Y · µm', 'Hz', 'Label'])
        self.units.verticalHeader().hide(); self.units.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.units.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.units.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.units.horizontalHeader().setStretchLastSection(True); self.units.setMaximumHeight(250)
        self.units.itemChanged.connect(self._unit_option_changed)
        self.units.itemSelectionChanged.connect(self._unit_changed); self.units.hide(); side.addWidget(self.units)
        self.wave_units = UnitTable(0, 4)
        self.wave_units.setHorizontalHeaderLabels(['Show · Unit', 'Depth · µm', 'Hz', 'Label'])
        self.wave_units.verticalHeader().hide()
        self.wave_units.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.wave_units.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.wave_units.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.wave_units.horizontalHeader().setStretchLastSection(True)
        self.wave_units.setMaximumHeight(250)
        self.wave_units.itemChanged.connect(self._wave_option_changed)
        side.addWidget(self.wave_units)
        self.plot_selector_layout = QVBoxLayout(); side.addLayout(self.plot_selector_layout)
        self.levels = DepthLevels(self); side.addWidget(self.levels)
        self.panel = panel = QWidget(); content = QVBoxLayout(panel); content.setContentsMargins(0, 0, 0, 0); splitter.addWidget(panel)
        plot_heading = QHBoxLayout(); plot_heading.addStretch()
        self.plot_tools = QWidget(); self.plot_tools_layout = QHBoxLayout(self.plot_tools); self.plot_tools_layout.setContentsMargins(0,0,0,0); self.plot_tools_layout.setSpacing(3)
        plot_heading.addWidget(self.plot_tools); content.addLayout(plot_heading)
        controls = QFormLayout(); self.controls = QWidget(); self.controls.setLayout(controls); controls.setContentsMargins(0, 0, 0, 0)
        controls.setFormAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        controls.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        controls.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        controls.setHorizontalSpacing(14); controls.setVerticalSpacing(10)
        self.source = PopupComboBox(); self.source.addItems(['Atlas', 'Scan']); self.source.currentIndexChanged.connect(self.draw); controls.addRow('Image', self.source)
        self.labels = QCheckBox('Labels'); self.labels.toggled.connect(self.draw); controls.addRow('', self.labels)
        self.shank = PopupComboBox(); self.shank.currentIndexChanged.connect(self._shank_changed); controls.addRow('Shank', self.shank)
        self.span = PopupComboBox(); self.span.addItem('Whole recording', 0); self.span.addItem('10 s window', 10); self.span.addItem('60 s window', 60); self.span.currentIndexChanged.connect(self._unit_changed); controls.addRow('Time window', self.span)
        self.good = PopupComboBox(); self.good.addItems(['Good units', 'All units']); self.good.currentIndexChanged.connect(self._fill_units); controls.addRow('Units', self.good)
        self.time = QDoubleSpinBox(); self.time.setDecimals(1); self.time.setSuffix(' s'); self.time.setSingleStep(2); self.time.setToolTip('Start of the displayed time window')
        self.time.valueChanged.connect(self._request_trace); controls.addRow('Start', self.time)
        self.controls.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        self.view_heading = view_heading = QLabel('View'); view_heading.setStyleSheet('color:#9aafc1;font-weight:600;'); side.addWidget(view_heading)
        side.addWidget(self.controls)
        self.visual_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.visual_splitter.setChildrenCollapsible(False); self.visual_splitter.setHandleWidth(7); splitter.setHandleWidth(7)
        splitter.setStyleSheet('QSplitter::handle{background:#354354;} QSplitter::handle:hover{background:#7ac7e5;}'); content.addWidget(self.visual_splitter, 1)
        self.anatomy_host = QWidget(); self.anatomy_layout = QVBoxLayout(self.anatomy_host); self.anatomy_layout.setContentsMargins(0,0,0,0)
        self.anatomy_caption = QLabel('Probe'); self.anatomy_caption.setFixedHeight(26); self.anatomy_caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.anatomy_caption.hide()
        self.anatomy_host.setMinimumWidth(230); self.visual_splitter.addWidget(self.anatomy_host)
        self.plot_stack = QStackedWidget(); self.visual_splitter.addWidget(self.plot_stack)
        self.visual_splitter.setSizes([260, 800])
        self.empty = QWidget(); empty_layout = QVBoxLayout(self.empty); empty_layout.setSpacing(10); empty_layout.addStretch()
        self.empty_title = QLabel('Add a recording'); self.empty_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_title.setStyleSheet('font-size:18px;font-weight:500;color:#dce6f0;'); empty_layout.addWidget(self.empty_title)
        self.empty_description = QLabel('Select recording folders to begin.'); self.empty_description.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_description.setWordWrap(True); self.empty_description.setStyleSheet('color:#9aafc1;'); empty_layout.addWidget(self.empty_description)
        empty_action = QHBoxLayout(); empty_action.addStretch(); self.empty_button = QPushButton('Add folders…')
        self.empty_button.setMaximumWidth(180); self.empty_button.clicked.connect(self._empty_action); empty_action.addWidget(self.empty_button); empty_action.addStretch()
        empty_layout.addLayout(empty_action); empty_layout.addStretch(); self.plot_stack.addWidget(self.empty)
        self.plot_host = QWidget(); self.plot_layout = QVBoxLayout(self.plot_host); self.plot_layout.setContentsMargins(0, 0, 0, 0)
        self.plot_stack.addWidget(self.plot_host)
        self.message = QLabel(''); self.message.setWordWrap(True); self.message.setStyleSheet('color:#9aafc1;'); side.addWidget(self.message)
        self.boundary_row = QWidget(); br = QFormLayout(self.boundary_row); br.setContentsMargins(0, 4, 0, 0)
        br.setFormAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        br.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        br.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.boundary = QCheckBox('Brain boundary'); br.addRow(self.boundary)
        boundary_field = QWidget(); boundary_line = QHBoxLayout(boundary_field); boundary_line.setContentsMargins(0,0,0,0)
        self.boundary_value = QDoubleSpinBox(); self.boundary_value.setRange(0, 12000); self.boundary_value.setSuffix(' µm'); self.boundary_value.setDecimals(0)
        self.boundary_value.setSingleStep(15); self.boundary_value.setMaximumWidth(140); self.boundary_value.setToolTip('Distance from probe tip')
        boundary_line.addWidget(self.boundary_value)
        self.use_candidate = view_button('fit', 'Use estimated transition as brain boundary', self._use_candidate); boundary_line.addWidget(self.use_candidate); boundary_line.addStretch()
        br.addRow('From tip', boundary_field)
        self.boundary.toggled.connect(self._boundary_changed); self.boundary_value.valueChanged.connect(self._boundary_changed)
        self.boundary_row.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        side.addWidget(self.boundary_row)
        probe_box = QGroupBox('Probe display'); probe_form = QFormLayout(probe_box)
        self.probe_body = QCheckBox('Shank bodies'); self.probe_body.setChecked(True); self.probe_body.toggled.connect(self._probe_display_changed); probe_form.addRow(self.probe_body)
        self.inactive_contacts = QCheckBox('Inactive contacts'); self.inactive_contacts.setChecked(True); self.inactive_contacts.toggled.connect(self._probe_display_changed); probe_form.addRow(self.inactive_contacts)
        self.probe_opacity = QSlider(Qt.Orientation.Horizontal); self.probe_opacity.setRange(0,100); self.probe_opacity.setValue(18)
        self.probe_opacity.valueChanged.connect(self._probe_display_changed); probe_form.addRow('Shank opacity',self.probe_opacity)
        self.contact_size = QSlider(Qt.Orientation.Horizontal); self.contact_size.setRange(1,20); self.contact_size.setValue(6)
        self.contact_size.valueChanged.connect(self._probe_display_changed); probe_form.addRow('Contact size',self.contact_size)
        side.addWidget(probe_box)
        self.regions_toggle = QPushButton('Regions'); self.regions_toggle.setCheckable(True); side.addWidget(self.regions_toggle)
        self.region_editor = EphysRegions(self); self.region_editor.hide(); side.addWidget(self.region_editor)
        self.regions_toggle.toggled.connect(self.region_editor.setVisible)
        side.addStretch()
        self.activity = TaskProgress(); self.progress = self.activity.bar; self.cancel = self.activity.cancel
        self.cancel.clicked.connect(lambda: self._cancel.set())
        self.cancel.clicked.connect(lambda: self._unit_cancel.set() if getattr(self,'_unit_cancel',None) else None); root.addWidget(self.activity)
        footer = QHBoxLayout(); self.back = back = QPushButton('Back'); back.clicked.connect(lambda: owner.go_step(6)); footer.addWidget(back)
        footer.addStretch(); self.save = save = QPushButton('Save'); save.clicked.connect(owner.save_project); footer.addWidget(save); root.addLayout(footer)
        self._sidebar_sizes = [480, 900]
        splitter.setSizes(self._sidebar_sizes)
        self.selection_timer = QTimer(self); self.selection_timer.setSingleShot(True); self.selection_timer.timeout.connect(self._load_selection)
        self.export_timer = QTimer(self); self.export_timer.setInterval(4000)
        self.export_timer.timeout.connect(self._sync_exports); self.export_timer.start()
        self.trace_timer = QTimer(self); self.trace_timer.setSingleShot(True); self.trace_timer.timeout.connect(self._load_trace)
        self.trace = None
        self.profile_timer=QTimer(self); self.profile_timer.setSingleShot(True); self.profile_timer.timeout.connect(self._load_overview_fits)
        self.unit_timer = QTimer(self); self.unit_timer.setSingleShot(True); self.unit_timer.timeout.connect(self._load_unit_view)
        from .desktop_ephys_sidebar import organize_sidebar
        organize_sidebar(self,side,probe_box)
        self._mode_changed()

    def _ensure_canvas(self):
        if self._canvas is not None: return
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg,NavigationToolbar2QT
        from matplotlib.figure import Figure
        self.anatomy_figure=Figure(facecolor='#242b36')
        self.anatomy_canvas=FigureCanvasQTAgg(self.anatomy_figure); self.anatomy_canvas.setMinimumSize(200,220)
        self.anatomy_layout.addWidget(self.anatomy_canvas,1); self.anatomy_ax=self.anatomy_figure.add_axes([.2,.1,.75,.8])
        self._anatomy_toolbar=NavigationToolbar2QT(self.anatomy_canvas,self,coordinates=False); self._anatomy_toolbar.hide()
        self.figure=self.anatomy_figure; self._canvas=self.anatomy_canvas; self._toolbar=self._anatomy_toolbar
        self.plot_board=PlotBoard(self,self.plot_selector_layout,self.plot_layout)
        self.fit_plot=view_button('fit','Fit depth range',self._fit_view)
        self.pan_plot=view_button('pan','Pan',lambda:self._plot_tool('pan')); self.pan_plot.setCheckable(True)
        self.zoom_plot=view_button('search','Zoom to selection',lambda:self._plot_tool('zoom')); self.zoom_plot.setCheckable(True)
        self.save_plot=view_button('save','Save anatomy and selected plots as PDF',self.plot_board.export_pdf)
        self.level_tool=view_button('level','Place or move a depth level',lambda:self.levels.tool.toggle()); self.level_tool.setCheckable(True)
        self.levels.tool.toggled.connect(self.level_tool.setChecked)
        self.levels.tool.toggled.connect(self._level_mode)
        for button in (self.level_tool,self.fit_plot,self.pan_plot,self.zoom_plot,self.save_plot): self.plot_tools_layout.addWidget(button)
        self._hover_text=''; self._hover_sites=None; self._hover_tree=None; self._hover_key=None
        self.anatomy_canvas.mpl_connect('motion_notify_event',lambda event:hover_contact(self,event) if self.levels.drag is None else None)
        self.anatomy_canvas.mpl_connect('figure_leave_event',lambda event:hover_contact(self,event))

    def _fit_view(self):
        self.plot_board._ylim=None; self.plot_board._probe_draw_key=None; self.draw()

    def _level_mode(self,active):
        if not active: return
        for toolbar in [self._anatomy_toolbar,*[p['toolbar'] for p in self.plot_board.panels.values()]]:
            if toolbar.mode.name=='PAN': toolbar.pan()
            elif toolbar.mode.name=='ZOOM': toolbar.zoom()
        self.pan_plot.setChecked(False); self.zoom_plot.setChecked(False)

    def _plot_tool(self,tool):
        self.levels.tool.setChecked(False)
        for toolbar in [self._anatomy_toolbar,*[p['toolbar'] for p in self.plot_board.panels.values()]]: getattr(toolbar,tool)()
        self.pan_plot.setChecked(self._toolbar.mode.name=='PAN'); self.zoom_plot.setChecked(self._toolbar.mode.name=='ZOOM')

    def _plot_scroll(self, event):
        if self._busy or event.inaxes is None or event.xdata is None or event.ydata is None: return
        ax = event.inaxes; factor = .8 if event.button == 'up' else 1.25
        x, y = event.xdata, event.ydata
        lo, hi = ax.get_xlim(); ax.set_xlim(x+(lo-x)*factor, x+(hi-x)*factor)
        lo, hi = ax.get_ylim(); ax.set_ylim(y+(lo-y)*factor, y+(hi-y)*factor)
        if ax is self.anatomy_ax or ax in getattr(self,'_depth_axes',[]):
            self._sync_depth(ax)
        event.canvas.draw_idle()

    def _empty_action(self):
        rec = self.current()
        if not rec: self.add_folders()
        elif self.plot_board.has_units() and self.sorting is None: self.attach_sorting()
        else: self.prepare_signal(rec['id'])

    def _show_empty(self, title, description, action):
        if self.current() and self._canvas is not None:
            self.figure.clear(); ax = self.figure.add_subplot(111); ax.axis('off')
            ax.text(.5,.55,title,ha='center',color='#dce6f0',fontsize=13,transform=ax.transAxes)
            ax.text(.5,.4,description,ha='center',color='#9aafc1',fontsize=9,wrap=True,transform=ax.transAxes)
            self.plot_stack.setCurrentWidget(self.plot_host); self.plot_tools.show(); self._canvas.draw_idle(); return
        self.empty_title.setText(title); self.empty_description.setText(description); self.empty_button.setText(action)
        self.plot_stack.setCurrentWidget(self.empty); self.plot_tools.hide()

    def open(self, project, reference=None):
        self._closing = False
        if self.project is project and self._busy: return
        if self.project is not project:
            self.reset(); self.project = project
        QTimer.singleShot(0,lambda:self._sync_exports(force=True))
        self._ensure_canvas()
        self._updating = True
        state = project.ephys.get('view', {})
        self.settings_tabs.setCurrentIndex(state.get('sidebar_tab',0))
        self.source.setCurrentText(state.get('source', 'Atlas'))
        self.labels.setChecked(state.get('labels', False))
        self.good.setCurrentIndex(state.get('good', 0))
        self.plot_board.restore(state.get('plot_board',{}))
        display = state.get('probe_display',{})
        self.probe_body.setChecked(display.get('body',True)); self.inactive_contacts.setChecked(display.get('inactive',True))
        self.probe_opacity.setValue(display.get('opacity',18)); self.contact_size.setValue(display.get('size',6))
        self.regions_toggle.setChecked(state.get('regions_open',False))
        self.region_editor.set_anatomy(self.anatomy,project)
        self.sidebar.setVisible(state.get('sidebar_visible', True))
        self._sidebar_sizes = state.get('splitter_sizes', [480, 900]) if state.get('layout_version') == 2 else [480, 900]
        if not self._sidebar_sizes[0]: self._sidebar_sizes = [480, 900]
        self.splitter.setSizes(self._sidebar_sizes)
        self.visual_splitter.setSizes(state.get('anatomy_sizes',[300,800]))
        widths = state.get('column_widths', []) if state.get('layout_version') == 2 else (95,72,85,80,52,45)
        for col, width in enumerate(widths):
            if col < self.table.columnCount(): self.table.setColumnWidth(col, max(30, int(width)))
        self.table.setFixedHeight(max(90, int(state.get('table_height', 132) if state.get('layout_version') == 2 else 132)))
        self._updating = False
        self.refresh(state.get('selected_ids', [None])[0] if state.get('selected_ids') else None, reload=False)
        from qtpy.QtCore import QItemSelectionModel
        self._updating = True
        for row, rec in enumerate(self._records()):
            if rec['id'] in state.get('selected_ids', []):
                self.table.selectionModel().select(self.table.model().index(row, 0), QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows)
        self._updating = False
        if not self._busy: self._load_selection()

    def reset(self):
        if getattr(self,'_unit_cancel',None): self._unit_cancel.set()
        self._candidate_memo=None
        if hasattr(self,'plot_board'):
            self.plot_board.timer.stop(); self.plot_board.settings_timer.stop(); self.plot_board._pending_panels.clear(); self.plot_board._pending_analysis=False
        self._cancel.set(); self._token += 1; self.selection_timer.stop(); self.trace_timer.stop(); self.unit_timer.stop()
        if hasattr(self,'window_timer'): self.window_timer.stop()
        if hasattr(self,'profile_timer'): self.profile_timer.stop()
        self.notebook=None; self._notebook_attempted=None; self.profile_fits=None; self._overview_attempted=None
        self.project = None; self.summary = self.sorting = self.anatomy = self.trace = None
        self._active_id = self._anatomy_key = self._anatomy_resources = None
        if hasattr(self,'plot_board'): self.plot_board._ylim=None
        self._signal_valid = {}
        self._updating = True; self.table.setRowCount(0); self._updating = False
        if self._canvas is not None: self._canvas.draw_idle()

    def stop(self):
        if hasattr(self,'plot_board'):
            self.plot_board.timer.stop(); self.plot_board.settings_timer.stop(); self.plot_board._pending_panels.clear()
        if getattr(self,'_unit_cancel',None): self._unit_cancel.set()
        self._closing = True; self._cancel.set(); self._token += 1
        self.selection_timer.stop(); self.trace_timer.stop(); self.unit_timer.stop()
        if hasattr(self,'window_timer'): self.window_timer.stop()
        if hasattr(self,'profile_timer'): self.profile_timer.stop()

    def capture_state(self):
        if not self.project or self._updating: return
        self.project.ephys['view'] = dict(layout_version=2, selected_ids=[r['id'] for r in self.selected()],
            source=self.source.currentText(), labels=self.labels.isChecked(),
            shank=self.shank.currentData(), good=self.good.currentIndex(),
            sidebar_tab=self.settings_tabs.currentIndex(),sidebar_visible=not self.sidebar.isHidden(), splitter_sizes=self._sidebar_sizes if self.sidebar.isHidden() else self.splitter.sizes(),
            column_widths=[self.table.columnWidth(i) for i in range(self.table.columnCount())], table_height=self.table.height(),
            plot_board=self.plot_board.state(),
            anatomy_sizes=self.visual_splitter.sizes(), regions_open=self.regions_toggle.isChecked(),
            probe_display=dict(body=self.probe_body.isChecked(),inactive=self.inactive_contacts.isChecked(),opacity=self.probe_opacity.value(),size=self.contact_size.value()))

    def _records(self): return self.project.ephys.setdefault('recordings', []) if self.project else []

    def _sync_exports(self,force=False):
        if not self.project or self._busy or self._closing or (not force and not self.isVisible()) or getattr(self,'_export_job',None): return
        from .desktop_app import Job
        project=self.project; output=project.output_dir
        # Copy only small editable structures here; metadata copying and IO run off-thread.
        baseline=[dict(r,sortings=[dict(s) for s in r.get('sortings',[])]) for r in self._records()]
        show_loading=not baseline
        if show_loading: self.activity.start('Finding prepared recordings')
        def scan(progress):
            snapshot=copy.deepcopy(baseline)
            return sync_exports(output,snapshot),snapshot
        job=Job(scan,signals=EphysJobSignals()); self._export_job=job; self._jobs.add(job)
        def finish(result=None,error=None):
            self._jobs.discard(job); self._export_job=None
            if self._closing or self.project is not project: return
            if show_loading and not self._busy: self.activity.finish(error=bool(error))
            if error:
                import logging
                logging.getLogger(__name__).warning('Background ephys discovery failed: %s',error)
                return
            changed,scanned=result
            if not changed: return
            ids=merge_scanned_exports(self._records(),baseline,scanned)
            if ids:
                self.owner.schedule_save()
                self.refresh(self._active_id,reload=self._active_id in ids)
        job.signals.done.connect(lambda value:finish(result=value))
        job.signals.failed.connect(lambda error:finish(error=error))
        QThreadPool.globalInstance().start(job)

    def selected(self):
        rows = [self._records()[index.row()] for index in self.table.selectionModel().selectedRows() if index.row() < len(self._records())]
        return rows or ([self.current()] if self.current() else [])

    def current(self):
        row = self.table.currentRow()
        return self._records()[row] if 0 <= row < len(self._records()) else next((r for r in self._records() if r['id'] == self._active_id), None)

    def _items(self): return insertion_session(self.project)['insertions'] if self.project else []

    def toggle_sidebar(self):
        if self._busy: return
        if self.sidebar.isHidden():
            self.sidebar.show(); self.splitter.setSizes(self._sidebar_sizes)
        else:
            self._sidebar_sizes = self.splitter.sizes(); self.sidebar.hide()
        if self._canvas is not None: self._canvas.draw_idle()
        self.owner.schedule_save()

    def _table_resized(self, *args):
        if self.project and not self._updating: self.owner.schedule_save()

    def record_by_id(self, identifier):
        return next((r for r in self._records() if r['id'] == identifier), None)

    def _select_record(self, identifier):
        self.selection_timer.stop()
        row = next((i for i, r in enumerate(self._records()) if r['id'] == identifier), None)
        if row is None: return
        self._updating = True; self.table.setCurrentCell(row, 0); self.table.selectRow(row); self._updating = False

    def refresh(self, select_id=None, reload=True):
        if not self.project: return
        previous = select_id or (self.current() or {}).get('id')
        chosen = {r['id'] for r in self.selected()}
        if select_id: chosen = {select_id}
        self._updating = True
        names = {i['id']: i['name'] for i in self._items()}
        self.table.setRowCount(0); self.table.setRowCount(len(self._records()))
        self.library_caption.setText(f'Recordings · {len(self._records())}')
        self.table.setToolTip('Scroll inside the table for more recordings. Drag the handle below to resize.')
        for row, rec in enumerate(self._records()):
            values = (rec['name'], names.get(rec['insertion_id'], '—'), recording_time(rec), '', '', '')
            for col, value in enumerate(values):
                cell = QTableWidgetItem(value)
                cell.setData(Qt.ItemDataRole.UserRole, rec['insertion_id'] if col == 1 else rec['id'])
                if col >= 2: cell.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                cell.setToolTip('Choose an insertion' if col == 1 else rec['meta_path'] if col == 0 else rec.get('recorded_at', '') if col == 2 else '')
                self.table.setItem(row, col, cell)
            self.table.item(row,5).setData(Qt.ItemDataRole.UserRole,self.record_color(rec))
            self.table.item(row,5).setToolTip('Recording color · inherits insertion color unless changed')
            self.table.setCellWidget(row, 3, SortingCombo(self, rec, self.table))
            self._signal_cell(row, rec)
        if self._records():
            from qtpy.QtCore import QItemSelectionModel
            index = next((i for i, r in enumerate(self._records()) if r['id'] == previous), 0)
            self.table.setCurrentCell(index, 0)
            self.table.clearSelection()
            for row, rec in enumerate(self._records()):
                if rec['id'] in chosen or (not chosen and row == index):
                    self.table.selectionModel().select(self.table.model().index(row, 0), QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows)
        self._updating = False
        if reload and not self._busy: self._load_selection()

    def _signal_cell(self, row, rec):
        self.table.removeCellWidget(row, 4)
        cell = self.table.item(row, 4)
        if self._signal_valid.get(rec['id'], bool(rec.get('cache_path'))):
            from qtpy.QtGui import QColor
            cell.setText('✓'); cell.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            cell.setForeground(QColor('#84ce9d')); cell.setToolTip('Signal prepared')
        else:
            cell.setText(''); cell.setToolTip('Prepare signal')
            button = view_button('run', 'Prepare signal', lambda identifier=rec['id']: self.prepare_signal(identifier))
            holder = QWidget(); layout = QHBoxLayout(holder); layout.setContentsMargins(0, 0, 0, 0)
            layout.addStretch(); layout.addWidget(button); layout.addStretch()
            self.table.setCellWidget(row, 4, holder)

    def _recording_clicked(self, row, column):
        if not self._busy and column == 1: self.table.editItem(self.table.item(row, column))
        elif not self._busy and column == 5:
            from qtpy.QtGui import QColor
            rec = self._records()[row]; color = QColorDialog.getColor(QColor(self.record_color(rec)),self,'Recording color')
            if color.isValid():
                rec['color'] = color.name(); self.refresh(reload=False); self.draw(); self.owner.schedule_save()

    def _recording_edited(self, cell):
        if self._updating or self._busy or not self.project: return
        record = self._records()[cell.row()]
        if cell.column() == 0:
            name = cell.text().strip()
            if not name: self.refresh(reload=False); return
            record['name'] = name
        elif cell.column() == 1:
            record['insertion_id'] = cell.data(Qt.ItemDataRole.UserRole); self._anatomy_key = None
        else: return
        self.owner.schedule_save()
        if cell.column() == 1:
            self._updating = True; self.table.item(cell.row(),5).setData(Qt.ItemDataRole.UserRole,self.record_color(record)); self._updating = False
            self._select_record(record['id']); self._load_selection()

    def _recording_menu(self, position):
        if self._busy: return
        row = self.table.rowAt(position.y())
        if not 0 <= row < len(self._records()): return
        rec = self._records()[row]; self._select_record(rec['id'])
        menu = QMenu(self)
        menu.addAction('Rename', lambda: self.table.editItem(self.table.item(row, 0)))
        if rec.get('color'): menu.addAction('Use insertion color',lambda:self._reset_record_color(rec))
        menu.addAction('Connect Kilosort…', lambda: self.attach_sorting(record_id=rec['id']))
        menu.addAction('Prepare signal' if not rec.get('cache_path') else 'Check / prepare signal', lambda: self.prepare_signal(rec['id']))
        menu.addSeparator(); menu.addAction('Remove recording', self.remove_selected)
        menu.exec(self.table.viewport().mapToGlobal(position))
        if not self._busy: self._load_selection()

    def prepare_signal(self, identifier):
        if self._busy: return
        rec = self.record_by_id(identifier)
        if rec is None: return
        self._select_record(identifier)
        self.analyze(records=[rec])

    def switch_sorting(self, record_id, sorting_id):
        if self._busy: return
        rec = self.record_by_id(record_id)
        if rec is None: return
        rec['selected_sorting'] = sorting_id
        self._select_record(record_id); self.sorting = self.unit_view = None
        self.owner.schedule_save(); self._load_selection()

    def remove_sorting(self, record_id, sorting_id):
        if self._busy: return
        rec = self.record_by_id(record_id)
        if rec is None: return
        active = rec.get('selected_sorting') == sorting_id
        rec['sortings'] = [s for s in rec['sortings'] if s['id'] != sorting_id]
        if active: rec['selected_sorting'] = rec['sortings'][0]['id'] if rec['sortings'] else None
        current_id = (self.current() or {}).get('id')
        self.owner.schedule_save()
        self.refresh(select_id=current_id, reload=active and current_id == record_id)

    def _task(self, function, complete, title='Opening recording'):
        if self._busy or self._closing: return
        from .desktop_app import Job
        if getattr(self,'_unit_cancel',None): self._unit_cancel.set()
        self._busy = True; self._cancel = Event(); token = self._token; cancel = self._cancel
        self.selection_timer.stop(); self.trace_timer.stop(); self.unit_timer.stop()
        if hasattr(self,'window_timer'): self.window_timer.stop()
        if hasattr(self,'profile_timer'): self.profile_timer.stop()
        self.splitter.setEnabled(False); self.collapse.setEnabled(False); self.back.setEnabled(False); self.save.setEnabled(False)
        self.activity.start(title); self.message.clear()
        job = Job(lambda progress: function(progress, cancel.is_set), signals=EphysJobSignals()); self._jobs.add(job)
        def progress(value, message):
            if token != self._token or self._closing: return
            self.activity.report(value, message)
        def finish(value=None, error=None):
            self._jobs.discard(job); self._busy = False
            if self._closing: return
            self.splitter.setEnabled(True); self.collapse.setEnabled(True); self.back.setEnabled(True); self.save.setEnabled(True)

            if token != self._token:
                self.activity.finish(cancelled=True)
                if self.project: self.selection_timer.start(0)
                return
            if error:
                self.message.setText('Cancelled. Completed caches are kept.' if cancel.is_set() else error)
            else: complete(value)
            if not self._busy:
                self.activity.finish(cancelled=cancel.is_set(),error=bool(error))
                if error and not cancel.is_set(): self.activity.stage.setText(error); self.activity.stage.setWordWrap(True)
        job.signals.progress.connect(progress)
        job.signals.done.connect(lambda value: finish(value=value)); job.signals.failed.connect(lambda error: finish(error=error))
        QThreadPool.globalInstance().start(job)

    def add_folders(self, checked=False, paths=None):
        if not self.project or self._busy: return
        if paths is None:
            from .desktop_folder_picker import recording_folders
            paths = recording_folders(self, self.project.ephys.get('last_import_folder'))
            if not paths: return
            self.project.ephys['last_import_folder'] = str(Path(paths[0]).parent)
        existing = copy.deepcopy(self._records())
        def ready(result):
            added, errors = result
            self._records().extend(added); self.owner.schedule_save()
            self.refresh(added[0]['id'] if added else None,reload=False)
            self.message.setText('\n'.join(errors) if errors else '')
            folders={str(Path(p).resolve()) for p in paths}
            repeated=[r for r in existing if str(Path(r['folder']).resolve()) in folders]
            cached=[r for r in [*repeated,*added] if (recording_root(r,self.project.output_dir)/'signal').exists() or r.get('cache_path') and Path(r['cache_path']).exists()]
            if cached:
                # analyze() presents the overwrite/reuse decision for these entries.
                from qtpy.QtCore import QTimer
                QTimer.singleShot(0,lambda:self.analyze(records=[self.record_by_id(r['id']) for r in cached]))
            else: self._load_selection()
        self._task(lambda p, c: import_folders(paths, existing, p, c), ready, 'Reading recording folders')

    def remove_selected(self):
        ids = {r['id'] for r in self.selected()}
        self.project.ephys['recordings'] = [r for r in self._records() if r['id'] not in ids]
        self.owner.schedule_save(); self.refresh()

    def _reset_record_color(self, record):
        record.pop('color',None); self.refresh(reload=False); self.draw(); self.owner.schedule_save()

    def _selection_changed(self):
        if not self._updating: self.selection_timer.start(100)

    def _load_selection(self):
        if self._busy or not self.project: return
        self.notebook=None; self._notebook_attempted=None; self.profile_fits=None; self._overview_attempted=None
        rec = self.current(); self.summary = self.sorting = self.trace = self.unit_view = None
        self.comparisons = []
        if rec and rec['id'] != self._active_id:
            self.anatomy = None; self.region_editor.set_anatomy(None,self.project)
        self._updating = True
        self.shank.clear()
        if rec:
            self._active_id = rec['id']
            shanks = sorted({c['shank'] for c in rec['channels']})
            for shank in shanks: self.shank.addItem(f'Shank {shank}', shank)
            self._control_visible(self.shank, len(shanks) > 1)
            saved_shank = self.project.ephys.get('view', {}).get('shank')
            self.shank.setCurrentIndex(max(0, self.shank.findData(saved_shank)))
            self.window_duration.setMaximum(min(60,rec['duration_s']))
            self.time.setRange(0,max(0,rec['duration_s']-self.window_duration.value()))
            self.time_slider.blockSignals(True); self.time_slider.setRange(0,round(self.time.maximum()*100)); self.time_slider.setValue(round(self.time.value()*100)); self.time_slider.blockSignals(False)
            self.note.setText(f"{rec['duration_s']/60:.1f} min · {len(rec['channels'])} contacts · " + ', '.join(f'shank {s}' for s in shanks))
        self._updating = False
        self._restore_boundary(); self._mode_changed(redraw=False)
        if not rec:
            self.anatomy = None; self.comparisons = []; self.draw(); return
        self.draw()
        record = copy.deepcopy(rec); output = Path(self.project.output_dir)
        selected_records = copy.deepcopy(self.selected())
        need_sorting = self.plot_board.has_units()
        def load(p, cancel):
            p(0, 'Checking prepared files')
            migrated=[]
            for selected_record in selected_records:
                migrate_recording_cache(selected_record,output); migrated.append(selected_record)
            record.update(next((r for r in migrated if r['id']==record['id']),{}))
            result = dict(signal=None, sorting=None, comparisons=[], sorting_path=None, migrated=migrated)
            for selected_record in selected_records:
                if selected_record.get('cache_path'):
                    try: result['comparisons'].append((selected_record, load_summary(selected_record['cache_path'])))
                    except (OSError, ValueError, KeyError): pass
            p(20, 'Reading signal profiles')
            if record.get('cache_path'):
                try:
                    info, data = next((value for selected_record, value in result['comparisons'] if selected_record['id'] == record['id']), (None,None))
                    if info is None: info, data = load_summary(record['cache_path'])
                    p(35, 'Checking recording source')
                    source_paths = [Path(record['meta_path']), Path(record['bin_path'])]
                    if not all(path.exists() for path in source_paths) or fingerprint(source_paths) == info['key']:
                        result['signal'] = (info, data)
                except (OSError, ValueError, KeyError): pass
            p(45, 'Reading sorting results')
            selected = next((s for s in record['sortings'] if s['id'] == record.get('selected_sorting')), None)
            if selected and selected.get('cache_path'):
                try: result['sorting'] = load_sorting(selected['cache_path'],fields=('waveforms',))
                except (OSError, ValueError, KeyError): pass
            if selected and result['sorting'] is None and need_sorting:
                cache = prepare_sorting(record, selected, output, lambda percent, message: p(45+round(percent*.5), message), cancel)
                result['sorting'] = load_sorting(cache,fields=('waveforms',)); result['sorting_path'] = cache
            p(100, 'Recording ready')
            return result
        def ready(result):
            for updated in result.get('migrated',[]):
                actual=self.record_by_id(updated['id'])
                if actual is None: continue
                for key in ('cache_path','cache_root'):
                    if updated.get(key): actual[key]=updated[key]
                sources={s['id']:s for s in updated.get('sortings',[])}
                for source in actual.get('sortings',[]):
                    if sources.get(source['id'],{}).get('cache_path'): source['cache_path']=sources[source['id']]['cache_path']
            if result.get('migrated'): self.owner.schedule_save()
            self._signal_valid[rec['id']] = result['signal'] is not None
            self._updating = True
            row = next((i for i,r in enumerate(self._records()) if r['id'] == rec['id']), None)
            if row is not None: self._signal_cell(row, rec)
            self._updating = False
            if result['sorting_path']:
                source = next((s for s in rec['sortings'] if s['id'] == rec.get('selected_sorting')), None)
                if source: source['cache_path'] = result['sorting_path']; self.owner.schedule_save()
            self.summary = result['signal']; self.sorting = result['sorting']; self.comparisons = result['comparisons']
            self._fill_units(); self._load_anatomy()
        self._task(load, ready)

    def _shank_changed(self):
        if self._updating: return
        self.notebook=None; self._notebook_attempted=None; self.profile_fits=None; self._overview_attempted=None
        self._restore_boundary(); self._fill_units(); self._load_anatomy()

    def _load_anatomy(self):
        rec = self.current()
        if not rec or not self.project or self._busy: return
        shank = self.shank.currentData()
        insertion = next((i for i in self._items() if i['id'] == rec['insertion_id']), None)
        key = json.dumps([insertion, self.project.total.tolist(), shank, rec['id']], sort_keys=True)
        if key == self._anatomy_key: self.draw(); return
        self.anatomy = None
        if not insertion or not insertion.get('probe', {}).get('pose'):
            self._anatomy_key = key; self.region_editor.set_anatomy(None,self.project); self.draw(); return
        project = copy.deepcopy(self.project); item = copy.deepcopy(insertion); record = copy.deepcopy(rec)
        resources = self._anatomy_resources
        def ready(result):
            self.anatomy = result; self._anatomy_resources = result['resources'] if result else None
            self._anatomy_key = key; self.region_editor.set_anatomy(result,self.project); self.draw()
        self._task(lambda p, c: prepare_anatomy(project, item, record, shank, project.output_dir, p, resources, c), ready, 'Loading anatomy')

    def analyze(self, checked=False, records=None):
        if self._busy or not self.project: return
        records = copy.deepcopy(records if records is not None else self.selected() or ([self.current()] if self.current() else []))
        project = self.project; output = Path(project.output_dir)
        overwrite_ids=set()
        for record in records:
            target=recording_root(record,output)/'signal'
            if target.exists() or record.get('cache_path') and Path(record['cache_path']).exists():
                answer=QMessageBox.question(self,'Prepared files already exist',
                    f"Replace prepared signal files for {Path(record['folder']).name}?\nThe current complete result is kept until the new preparation succeeds.",
                    QMessageBox.StandardButton.Yes|QMessageBox.StandardButton.No|QMessageBox.StandardButton.Cancel,QMessageBox.StandardButton.No)
                if answer==QMessageBox.StandardButton.Cancel: return
                if answer==QMessageBox.StandardButton.Yes: overwrite_ids.add(record['id'])
        def work(progress, cancel):
            results = []
            for index, record in enumerate(records):
                def update(percent, message): progress((index + percent/100)*100/max(1, len(records)), message)
                migrate_recording_cache(record,output)
                cache = prepare_recording(record, output, update, cancel, overwrite=record['id'] in overwrite_ids)
                results.append((record['id'], cache))
            return results
        def ready(results):
            for identifier, path in results:
                rec = next(r for r in self._records() if r['id'] == identifier); rec['cache_path'] = path; rec['cache_root']=str(Path(path).parent)
                self._signal_valid[identifier] = True
            self.owner.schedule_save(); self.refresh(reload=False); self._load_selection()
        self._task(work, ready, 'Preparing signal')

    def attach_sorting(self, checked=False, path=None, record_id=None):
        rec = self.record_by_id(record_id) if record_id else self.current()
        if not rec or self._busy: return
        if path is None: path = QFileDialog.getExistingDirectory(self, 'Connect sorting results', rec.get('folder', ''))
        if not path: return
        identifier = rec['id']
        def work(p, cancel):
            p(5, 'Finding sorting results'); _, folders = discover(path, cancel)
            if not folders: raise ValueError('No spike_times.npy and spike_clusters.npy found in this folder.')
            return [sorting_source(folder) for folder in folders]
        def ready(found):
            target = next(r for r in self._records() if r['id'] == identifier)
            known = {s['path'] for s in target['sortings']}
            target['sortings'].extend(s for s in found if s['path'] not in known)
            first_new = next((s for s in target['sortings'] if s['path'] not in known), None)
            if first_new: target['selected_sorting'] = first_new['id']
            self.refresh(select_id=identifier, reload=False)
            self.switch_sorting(identifier, target.get('selected_sorting'))
        self._task(work, ready, 'Finding Kilosort results')

    def _prepare_sorting(self):
        rec = self.current()
        if not rec or self._busy: return
        sorting = next((s for s in rec['sortings'] if s['id'] == rec.get('selected_sorting')), None)
        if not sorting: return
        record = copy.deepcopy(rec); source = copy.deepcopy(sorting); output = Path(self.project.output_dir)
        def ready(path):
            sorting['cache_path'] = path; self.owner.schedule_save(); self._load_selection()
        self._task(lambda p, c: prepare_sorting(record, source, output, p, c), ready, 'Reading sorted units')

    def _control_visible(self, widget, visible):
        widget.setVisible(visible)
        label = getattr(self,'_field_labels',{}).get(widget)
        if label is None and not hasattr(self,'_field_labels'): label=self.controls.layout().labelForField(widget)
        if label: label.setVisible(visible)

    def _mode_changed(self,*args,redraw=True):
        if self._updating: return
        if redraw: self.draw()

    def _fill_units(self, *args):
        if self._updating: return
        preferences = self._unit_preferences()
        previous = self.units.item(self.units.currentRow(),0)
        previous_id = previous.text() if previous else None
        self._updating = True; self.units.setRowCount(0); self.wave_units.setRowCount(0)
        if self.sorting:
            info, _ = self.sorting
            rows = [(i, u) for i, u in enumerate(info['units']) if u['shank'] == self.shank.currentData() and
                    (self.good.currentIndex() == 1 or u['label'] == 'good')]
            positions = self._raster_positions()
            rows.sort(key=lambda pair: -positions[pair[1]['id']])
            self.units.setRowCount(len(rows))
            self.wave_units.setRowCount(len(rows))
            positions = self._raster_positions()
            for row, (index, unit) in enumerate(rows):
                texts = (str(unit['id']), f"{unit['height_um']:.1f}",
                         f"{positions[unit['id']]:.2f}", f"{unit['rate_hz']:.2f}", unit['label'])
                for col, text in enumerate(texts):
                    cell = QTableWidgetItem(text); cell.setData(Qt.ItemDataRole.UserRole, index)
                    self.units.setItem(row, col, cell)
                    if col == 0:
                        cell.setFlags(cell.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                        cell.setCheckState(Qt.CheckState.Unchecked if unit['id'] in preferences['hidden'] else Qt.CheckState.Checked)
                    cell.setToolTip(f"Unit {unit['id']} · depth {unit['height_um']:.1f} µm · raster Y {positions[unit['id']]:.2f} µm")
                for col,text in enumerate((texts[0],texts[1],texts[3],texts[4])):
                    cell = QTableWidgetItem(text); cell.setData(Qt.ItemDataRole.UserRole,index)
                    self.wave_units.setItem(row,col,cell)
                    if col == 0:
                        cell.setFlags(cell.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                        cell.setCheckState(Qt.CheckState.Checked if unit['id'] in preferences['waveforms'] else Qt.CheckState.Unchecked)
            if rows: self.units.selectRow(next((r for r,(_,u) in enumerate(rows) if str(u['id']) == previous_id),0))
        self.units.setFixedHeight(max(65,min(220,32+self.units.rowCount()*28)))
        self.wave_units.setFixedHeight(max(65,min(220,32+self.wave_units.rowCount()*28)))
        self._updating = False; self._unit_changed()
        board=getattr(self,'plot_board',None)
        if board:
            from .desktop_ephys_plots import SPECS
            for key in board.keys():
                if SPECS[key].source=='units': self._redraw_unit_panel(key)

    def _unit_preferences(self):
        rec = self.current()
        if not rec: return dict(hidden=[], waveforms=[])
        key = str(rec.get('selected_sorting'))
        return rec.setdefault('unit_preferences', {}).setdefault(key, dict(hidden=[], waveforms=[]))

    def _visible_units(self):
        if not self.sorting: return []
        hidden = set(self._unit_preferences()['hidden'])
        return [(i,u) for i,u in enumerate(self.sorting[0]['units'])
                if u['shank'] == self.shank.currentData()
                and (self.good.currentIndex() == 1 or u['label'] == 'good')
                and u['id'] not in hidden]

    def _waveform_units(self):
        chosen = set(self._unit_preferences()['waveforms'])
        return [(i,u) for i,u in self._filtered_units() if u['id'] in chosen]

    def _filtered_units(self):
        if not self.sorting: return []
        return [(i,u) for i,u in enumerate(self.sorting[0]['units'])
                if u['shank'] == self.shank.currentData()
                and (self.good.currentIndex() == 1 or u['label'] == 'good')]

    def _raster_positions(self):
        from .ephys_units import raster_positions
        return raster_positions([u for u in self.sorting[0]['units']
                                 if u['shank'] == self.shank.currentData()]) if self.sorting else {}

    def _unit_option_changed(self, cell):
        if self._updating or not self.sorting or cell.column() != 0: return
        unit = self.sorting[0]['units'][cell.data(Qt.ItemDataRole.UserRole)]
        hidden = set(self._unit_preferences()['hidden'])
        if cell.checkState() == Qt.CheckState.Checked: hidden.discard(unit['id'])
        else: hidden.add(unit['id'])
        self._unit_preferences()['hidden'] = sorted(hidden)
        self.owner.schedule_save(); self._refresh_unit_visibility()

    def _wave_option_changed(self, cell):
        if self._updating or not self.sorting or cell.column() != 0: return
        unit = self.sorting[0]['units'][cell.data(Qt.ItemDataRole.UserRole)]
        chosen = set(self._unit_preferences()['waveforms'])
        if cell.checkState() == Qt.CheckState.Checked: chosen.add(unit['id'])
        else: chosen.discard(unit['id'])
        self._unit_preferences()['waveforms'] = sorted(chosen)
        self.owner.schedule_save(); self._redraw_unit_panel('waveform')

    def _redraw_unit_panel(self, key):
        board = getattr(self,'plot_board',None)
        if board and key in board.keys(): board.queue_panel(key)

    def _refresh_unit_visibility(self):
        board=getattr(self,'plot_board',None)
        if not board: return
        visible={u['id'] for _,u in self._visible_units()}
        panel=board.panels.get('raster')
        if panel:
            for ax in panel['axes']:
                for uid,artist in getattr(ax,'_unit_artists',{}).items():
                    artist.set_visible(uid in visible)
            self._paint_raster(panel['canvas'])
        for key in ('density','region_units'):
            self._redraw_unit_panel(key)

    def _paint_raster(self, canvas):
        if not self.current(): return
        axes=[ax for ax in canvas.figure.axes if hasattr(ax,'_unit_artists')]
        if not axes: return
        if not canvas.supports_blit or any(not hasattr(ax,'_raster_background') for ax in axes):
            canvas.draw_idle(); return
        for ax in axes:
            canvas.restore_region(ax._raster_background)
            for artist in ax._unit_artists.values():
                if artist.get_visible(): ax.draw_artist(artist)
            canvas.blit(ax.bbox)

    def _raster_draw_event(self, event):
        canvas=event.canvas
        for ax in canvas.figure.axes:
            if hasattr(ax,'_unit_artists') and canvas.supports_blit:
                ax._raster_background=canvas.copy_from_bbox(ax.bbox)
        if canvas.supports_blit: self._paint_raster(canvas)

    def _hover_unit(self, event):
        from qtpy.QtWidgets import QToolTip
        from qtpy.QtGui import QCursor
        ax=event.inaxes
        artists=getattr(ax,'_unit_artists',{}) if ax is not None else {}
        if not artists:
            QToolTip.hideText(); return
        for uid,artist in artists.items():
            if not artist.get_visible(): continue
            tolerance=abs(ax.get_ylim()[1]-ax.get_ylim()[0])*5/max(1,ax.bbox.height)
            if event.ydata is None or abs(artist._raster_y-event.ydata)>tolerance: continue
            hit,detail=artist.contains(event)
            if hit and len(detail.get('ind',[])):
                depth=artist._unit_depth; y=float(artist.get_offsets()[0,1])
                QToolTip.showText(QCursor.pos(),f'Unit {uid} · depth {depth:.1f} µm · raster Y {y:.2f} µm')
                return
        QToolTip.hideText()

    def _set_units_visible(self, visible):
        self._updating = True
        for row in range(self.units.rowCount()):
            self.units.item(row,0).setCheckState(Qt.CheckState.Checked if visible else Qt.CheckState.Unchecked)
        self._updating = False
        hidden = set(self._unit_preferences()['hidden'])
        for row in range(self.units.rowCount()):
            index = self.units.item(row,0).data(Qt.ItemDataRole.UserRole)
            uid = self.sorting[0]['units'][index]['id']
            if visible: hidden.discard(uid)
            else: hidden.add(uid)
        self._unit_preferences()['hidden'] = sorted(hidden)
        self.owner.schedule_save(); self._refresh_unit_visibility()

    def _show_unit_raster(self):
        self.plot_board.toggle_plot('raster', True)
        self.settings_tabs.setCurrentIndex(4)

    def _selected_waveforms(self):
        for index in self.units.selectionModel().selectedRows():
            self.wave_units.item(index.row(),0).setCheckState(Qt.CheckState.Checked)
        self.plot_board.toggle_plot('waveform', True)
        self.settings_tabs.setCurrentIndex(4)

    def _pick_unit(self, event):
        ids = getattr(event.artist, '_unit_ids', None)
        if ids is None or not len(event.ind) or not self.sorting: return
        unit_id = int(ids[event.ind[0]])
        for row in range(self.units.rowCount()):
            index = self.units.item(row,0).data(Qt.ItemDataRole.UserRole)
            if self.sorting[0]['units'][index]['id'] == unit_id:
                self.units.selectRow(row)
                self.wave_units.item(row,0).setCheckState(Qt.CheckState.Checked)
                self.plot_board.toggle_plot('waveform', True)
                break

    def _restore_boundary(self):
        self._updating = True
        rec = self.current(); value = (rec or {}).get('boundaries', {}).get(str(self.shank.currentData()))
        self.boundary.setChecked(value is not None); self.boundary_value.setValue(value or 0)
        self._updating = False

    def _boundary_changed(self, *args):
        if self._updating or not self.current(): return
        self.current().setdefault('boundaries', {})[str(self.shank.currentData())] = self.boundary_value.value() if self.boundary.isChecked() else None
        self.owner.schedule_save(); self.draw()

    def _indices_heights(self, record=None):
        rec = record or self.current()
        indices = [i for i, c in enumerate(rec['channels']) if c['shank'] == self.shank.currentData()]
        first = self.anatomy['first_site_um'] if self.anatomy else rec.get('first_site_um', 217.)
        pitch = self.anatomy['pitch_um'] if self.anatomy else 15.
        heights = np.array([first + rec['channels'][i]['row']*pitch for i in indices])
        order = np.argsort(heights, kind='stable')
        return np.asarray(indices,dtype=int)[order], heights[order]

    @staticmethod
    def _depth_grid(heights, values):
        """Average the two columns at a depth; leave unrecorded rows masked."""
        levels = np.arange(heights.min(), heights.max()+7.5, 15.)
        result = np.full((len(levels), *values.shape[1:]), np.nan)
        for index, height in enumerate(levels):
            mask = np.isclose(heights, height, atol=.1)
            if mask.any(): result[index] = np.mean(values[mask], axis=0)
        return levels, np.ma.masked_invalid(result)

    def _profile_line(self, ax, values, heights, indices, **kwargs):
        record = self.current(); columns = np.array([record['channels'][i]['column'] for i in indices])
        for number, column in enumerate(np.unique(columns)):
            mask = columns == column; h = heights[mask]; v = values[mask]
            breaks = np.flatnonzero(np.diff(h) > 22.5)+1
            h = np.insert(h, breaks, np.nan); v = np.insert(v, breaks, np.nan)
            options = dict(kwargs)
            if number: options.pop('label', None)
            ax.plot(v, h, **options)

    def _candidate(self):
        if not self.summary or not self.current(): return np.array([])
        if self.plot_board.notebook_mode.isChecked():
            if self.notebook is None: return np.array([])
            values=np.concatenate([self.notebook[k+'_boundaries'] for k in self.plot_board.column_choices() if k+'_boundaries' in self.notebook] or [np.array([])])
            return values[np.isfinite(values)]
        geometry = (self.shank.currentData(), (self.current() or {}).get('first_site_um'),
                    (self.anatomy or {}).get('first_site_um'), (self.anatomy or {}).get('pitch_um'))
        saved = getattr(self,'_candidate_memo',None)
        if saved is not None and saved[0] is self.summary and saved[1] == geometry: return saved[2]
        idx, heights = self._indices_heights(); data = self.summary[1]
        result = boundary_candidates(heights, data['band_windows'][:, :, idx], data['bad_channels'][idx])
        self._candidate_memo = (self.summary,geometry,result)
        return result

    def _use_candidate(self):
        values = self._candidate()
        if len(values): self.boundary_value.setValue(float(np.mean(values) if self.plot_board.notebook_mode.isChecked() else np.median(values))); self.boundary.setChecked(True)

    def _overview_key(self):
        rec=self.current()
        return ((rec or {}).get('id'),self.shank.currentData(),(self.summary or [{}])[0].get('key'),
            self.plot_board.analysis_fields['segments'].value(),json.dumps([[b['low'],b['high']] for b in self.plot_board.plot_options('lfp')['bands']]))

    def _load_overview_fits(self):
        from .ephys_profiles import fit_overview
        if self._busy or not self.summary or not self.current() or self.plot_board.notebook_mode.isChecked(): return
        key=self._overview_key(); self._overview_attempted=key
        record=copy.deepcopy(self.current()); indices,heights=self._indices_heights(); data=self.summary[1]
        bands=copy.deepcopy(self.plot_board.plot_options('lfp')['bands']); segments=self.plot_board.analysis_fields['segments'].value()
        def ready(value): self.profile_fits=value; self.draw()
        self._task(lambda p,c:fit_overview(record,indices,heights,data,bands,segments,p,c),ready,'Fitting depth profiles')

    def _load_notebook(self,cache_only=False):
        from .ephys_notebook import prepare_window
        rec=self.current()
        if not rec or not self.summary or self._busy: return
        record=copy.deepcopy(rec); indices,heights=self._indices_heights(); shank=self.shank.currentData()
        start=self.time.value(); source_key=self.summary[0]['key']; mode=self.plot_board.profile_source.currentData()
        lfp_fs=self.summary[0]['sampling_rate']; options=self.plot_board.analysis_options()
        self._notebook_attempted=(rec['id'],shank,start,mode)
        def ready(value):
            self.notebook=value; self.draw()
            if cache_only and value is None and self._window_auto and mode=='cached': self.window_timer.start(0)
            if not cache_only: self._window_auto=False
        self._task(lambda p,c:prepare_window(record,shank,indices,heights,start,source_key,p,c,allow_source=not cache_only,mode=mode,lfp_fs=lfp_fs,options=options),
            ready,'Checking saved profile' if cache_only else 'Analyzing time window')

    def _analysis_changed(self):
        if self._updating or not hasattr(self,'plot_board') or self.plot_board._restoring: return
        rec=self.current()
        if rec:
            self.time.setMaximum(max(0,rec['duration_s']-self.window_duration.value())); self.time_slider.setMaximum(round(self.time.maximum()*100))
        self._request_trace(); self.owner.schedule_save()

    def _request_trace(self):
        if self._updating or not hasattr(self,'plot_board'): return
        self.notebook=None; self._notebook_attempted=None
        self._window_auto=self.plot_board.needs_notebook() and self.plot_board.profile_source.currentData()=='cached'
        if self.plot_board.needs_notebook():
            self.draw()
            if self.plot_board.profile_source.currentData()=='cached': self.window_timer.start(350)
        if self.plot_board.has_units(): self._unit_changed()
        if self.plot_board.needs_traces(): self.trace_timer.start(180)

    def _trace_ready(self):
        value=self.trace
        if value is None or len(value)<5: return False
        return (abs(value[0]-self.time.value())<.01 and value[3]==self.window_duration.value()
                and value[4]==self.shank.currentData()
                and ('traces_clustered' not in self.plot_board.keys() or value[2] is not None))

    def _load_trace(self):
        rec = self.current()
        if not rec or not self.summary or self._busy: return
        start=self.time.value(); duration=self.window_duration.value(); shank=self.shank.currentData()
        cache=rec['cache_path']; fs=self.summary[0]['sampling_rate']
        record=copy.deepcopy(rec); indices,heights=self._indices_heights()
        clustered='traces_clustered' in self.plot_board.keys()
        old=self.trace
        reuse=old is not None and len(old)>=5 and old[0]==start and old[3]==duration and old[4]==shank
        def load(progress,cancel):
            if reuse: values=old[1]
            else:
                progress(5,'Reading prepared LFP interval')
                lfp=np.load(Path(cache)/'lfp.npy',mmap_mode='r',allow_pickle=False)
                i=round(start*fs); values=np.array(lfp[:,i:i+round(duration*fs)])
            labels=old[2] if reuse else None
            if clustered and labels is None:
                from .desktop_ephys_plots import column_series
                from .ephys_notebook import clusters
                labels={}
                groups=list(column_series(record,indices,heights,values[indices],['0','1','mean']))
                for number,(kind,_,wave) in enumerate(groups):
                    if cancel(): raise InterruptedError('Cancelled')
                    progress(20+70*(number+1)/max(1,len(groups)),'Clustering LFP traces')
                    with np.errstate(invalid='ignore',divide='ignore'): corr=np.corrcoef(wave)
                    labels[kind]=clusters(np.nan_to_num(np.atleast_2d(corr),nan=0,posinf=0,neginf=0))
            return start,values,labels,duration,shank
        def ready(value):
            self.trace=value
            for key in ('traces','traces_clustered'): self._redraw_unit_panel(key)
        self._task(load,ready,'Reading LFP traces')

    def record_color(self,record):
        insertion=next((i for i in self._items() if i['id']==record.get('insertion_id')),{})
        return record.get('color') or insertion.get('color','#7ac7e5')

    def _probe_display_changed(self,*args):
        if self._updating: return
        if hasattr(self,'plot_board'): self.plot_board.queue_panel('__anatomy__')
        self.owner.schedule_save()

    def draw(self,*args):
        if self._updating or self._canvas is None or self._closing: return
        self.plot_stack.setCurrentWidget(self.plot_host if self.current() else self.empty)
        self.plot_tools.setVisible(bool(self.current()))
        self.time_bar.setVisible(bool(self.current()))
        board=self.plot_board
        self._control_visible(self.span,board.has_units()); self._control_visible(self.good,board.has_units())
        self.units.setVisible(board.has_units())
        self.time.setVisible(True)
        self._control_visible(self.source,self.anatomy is not None)
        self._control_visible(self.labels,self.anatomy is not None and self.anatomy.get('overlay') is not None)
        self.regions_toggle.setEnabled(self.anatomy is not None and bool(self.anatomy.get('label_regions')))
        self.boundary_row.setVisible(self.summary is not None)
        self.boundary_value.setEnabled(self.boundary.isChecked()); self.use_candidate.setEnabled(bool(len(self._candidate())))
        board.render()
        if not board.has_units(): self.unit_timer.stop()
        if not board.needs_traces(): self.trace_timer.stop()
        info=[]
        if board.needs_notebook():
            info.append(f'Time window: {self.time.value():g}–{self.time.value()+self.window_duration.value():g} s')
            if board.profile_source.currentData()=='cached': info.append('Prepared signal · local cache. Processing settings are applied to the selected time window.')
            else: info.append('Original-source filtering reads large source blocks on first analysis.')
            if self.notebook is None: info.append('Analyze the selected interval to update the signal profiles.')
            else:
                info.append('Mean uses averaged voltage before C-factor. CSD is a spatial second difference in relative units. Boundaries are unconfirmed estimates.')
                if 'timings_s' in self.notebook:
                    timings=json.loads(str(self.notebook['timings_s'])); info.append(f'Last calculation: {timings["total"]:.1f} s · signal {timings["signal"]:.1f} s · analysis {timings["metrics_and_fits"]:.1f} s')
        else: info.append('Whole-record overview: 20 s windows · Welch 4096 · no C-factor.')
        if self.anatomy and self.anatomy['depth_um'] is not None: info.append(f"Anatomical entry: {self.anatomy['depth_um']:.0f} µm · dashed yellow")
        candidates=self._candidate()
        if len(candidates):
            estimate=np.mean(candidates) if board.notebook_mode.isChecked() else np.median(candidates)
            info.append(f'Candidate, unconfirmed: {estimate:.0f} µm · {min(candidates):.0f}–{max(candidates):.0f} µm')
            if board.notebook_mode.isChecked(): info.append(f'Band boundaries: median {np.median(candidates):.0f} µm · SD {np.std(candidates):.0f} µm')
        if self.boundary.isChecked(): info.append(f'Marked boundary: {self.boundary_value.value():.0f} µm · cyan')
        if self.unit_view and self.unit_view.get('sampled'): info.append('Spikes subsampled for display and export (maximum 200,000).')
        if board.has_units(): info.append('Template and exported amplitude units are not calibrated µV.')
        if not board.column_choices(): info.append('Select an electrode column or Mean to show signal profiles.')
        self.message.setText('\n'.join(info))

    def _sync_depth(self,changed):
        if getattr(self,'_syncing_depth',False): return
        self._syncing_depth=True
        if self.levels.backgrounds: self.levels.clear_preview()
        try:
            limits=changed.get_ylim(); self.plot_board._ylim=limits
            for ax in [self.anatomy_ax,*self._depth_axes]:
                if ax is not changed: ax.set_ylim(limits,emit=False)
            ticks=self.anatomy_ax.get_yticks()
            for ax in self._depth_axes: ax.set_yticks(ticks); ax.set_ylim(limits,emit=False)
            for canvas in self.plot_board.canvases(): canvas.draw_idle()
        finally: self._syncing_depth=False

    def _unit_changed(self, *args):
        if self._updating: return
        for key in ('isi','rate'): self._redraw_unit_panel(key)
        if self.unit_view is not None and self.unit_view.get('key') == self._unit_view_key(): return
        self.unit_view = None
        if not self._busy: self.unit_timer.start(100)

    def _spike_span(self):
        span=self.span.currentData()
        return self.window_duration.value() if not span and self.plot_board.notebook_mode.isChecked() else span

    def _unit_view_key(self):
        rec = self.current()
        return (rec['id'] if rec else None, (rec or {}).get('selected_sorting'),
                self.units.currentRow() if any(k in self.plot_board.keys() for k in ('acg','amplitude')) else None,
                self.shank.currentData(), self.good.currentIndex(), self.time.value(), self._spike_span(),
                tuple(u['id'] for _,u in self._filtered_units()),
                tuple(k for k in ('raster','acg','amplitude') if k in self.plot_board.keys()))

    def _load_unit_view(self):
        if self._busy or not self.sorting or not self.current(): return
        if not self.plot_board.needs_detail(): return
        if getattr(self,'_unit_job',None):
            self._unit_cancel.set(); return
        key = self._unit_view_key(); record = self.current()
        sorting = next((s for s in record['sortings'] if s['id'] == record.get('selected_sorting')), None)
        if not sorting or not sorting.get('cache_path'): return
        row = self.units.currentRow()
        if row < 0: return
        selected = copy.deepcopy(self.sorting[0]['units'][self.units.item(row, 0).data(Qt.ItemDataRole.UserRole)])
        units = [copy.deepcopy(u) for _,u in self._filtered_units()]
        need_acg = 'acg' in self.plot_board.keys()
        need_raster = 'raster' in self.plot_board.keys()
        need_amplitudes = 'amplitude' in self.plot_board.keys()
        start = self.time.value(); span = self._spike_span(); cache = sorting['cache_path']
        def work(progress, cancel):
            progress(5, 'Reading unit profile')
            times = np.load(Path(cache)/'times.npy', mmap_mode='r', allow_pickle=False)
            amplitudes = np.load(Path(cache)/'amplitudes.npy', mmap_mode='r', allow_pickle=False) if need_amplitudes else None
            unit_times = times[selected['start']:selected['start']+selected['count']]
            step = max(1, len(unit_times)//10000)
            acg = np.zeros(100)
            for reference_index in (range(0, len(unit_times), step) if need_acg else ()):
                t = unit_times[reference_index]
                if cancel(): raise InterruptedError('Cancelled')
                lo, hi = np.searchsorted(unit_times, [t-.05, t+.05])
                neighbors = (unit_times[lo:hi]-t)*1000
                neighbors = np.delete(neighbors, reference_index-lo)
                acg += np.histogram(neighbors, np.linspace(-50, 50, 101))[0]
            subset = slice(selected['start'], selected['start']+selected['count'], step)
            raster_times, raster_depths, raster_ids = [], [], []
            sampled = False
            budget = max(1, 200000//max(1,len(units)))
            for number,unit in enumerate(units if need_raster else []):
                if cancel(): raise InterruptedError('Cancelled')
                values = times[unit['start']:unit['start']+unit['count']]
                if span:
                    lo, hi = np.searchsorted(values, [start, start+span]); values = values[lo:hi]
                stride = max(1, int(np.ceil(len(values)/budget)))
                sampled |= stride > 1
                values = values[::stride]
                raster_times.append(np.array(values))
                raster_depths.append(np.full(len(values), unit['height_um']))
                raster_ids.append(np.full(len(values), unit['id'],dtype=np.int32))
                if number % max(1,len(units)//40)==0:
                    progress(10+85*(number+1)/len(units),f'Reading units · {number+1} / {len(units)}')
            return dict(key=key, times=np.array(times[subset]) if need_amplitudes else np.array([]),
                        amplitudes=np.array(amplitudes[subset]) if need_amplitudes else np.array([]), acg=acg,
                        raster_times=np.concatenate(raster_times) if raster_times else np.array([]),
                        raster_depths=np.concatenate(raster_depths) if raster_depths else np.array([]),
                        raster_ids=np.concatenate(raster_ids) if raster_ids else np.array([],dtype=np.int32), sampled=sampled)
        def ready(value):
            self.unit_view=value
            for plot in ('raster','acg','amplitude'):
                self._redraw_unit_panel(plot)
        self._run_unit_job(work, ready, key)

    def _run_unit_job(self,work,ready,key):
        """Read display spikes without disabling controls; discard obsolete results."""
        from .desktop_app import Job
        cancel=Event(); self._unit_cancel=cancel
        job=Job(lambda progress:work(progress,cancel.is_set),signals=EphysJobSignals())
        self._unit_job=job; self._jobs.add(job)
        self.activity.start('Reading unit profiles')
        job.signals.progress.connect(lambda value,text:self.activity.report(value,text) if not self._busy and not self._closing else None)
        def finish(value=None,error=None):
            self._jobs.discard(job); self._unit_job=None
            if self._closing: return
            stale=key!=self._unit_view_key()
            if not self._busy: self.activity.finish(cancelled=cancel.is_set(),error=bool(error))
            if cancel.is_set() or stale:
                if stale and not self._busy and self.current(): self.unit_timer.start(100)
                return
            if error:
                self.message.setText(error); return
            ready(value)
        job.signals.done.connect(lambda value:finish(value=value))
        job.signals.failed.connect(lambda error:finish(error=error))
        QThreadPool.globalInstance().start(job)


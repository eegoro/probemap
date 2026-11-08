"""ProbeMap: a standalone, English, Qt/VisPy landmark registration app."""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
import warnings
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from qtpy.QtCore import QObject, QRunnable, QThreadPool, QTimer, Qt, Signal
from qtpy.QtGui import QAction, QIcon, QKeySequence
from qtpy.QtWidgets import (
    QApplication, QCheckBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QMainWindow, QMessageBox, QProgressBar, QPushButton, QScrollArea, QSlider,
    QSplitter, QStackedWidget, QTabWidget, QVBoxLayout, QWidget,
)

from .coarse import pose_matrix
from .desktop_model import (AlignmentProject, load_preview,
                            orientation_linear, full_view_grid, prepare_landmark_views, sample_preview_grid_slice)
from .desktop_views import PlacementSlicePane, SliceComparison, VolumeView
from .desktop_project import CalibrationDialog, ImageInput, PreviewSettings, ProjectPage, icon_button
from .landmarks import transform_points, validate_affine
from .. import __version__
from ..branding import APP_NAME, RESOURCE_DIR
from ..project_paths import project_file_path
from ..recent_projects import RecentProjects


STYLE = """
QWidget {background:#242b36; color:#e5eaf2; font-size:12px;}
QMainWindow, QStackedWidget {background:#1c222c;}
QLabel#heading {font-size:22px; font-weight:600; color:#f5f8ff;}
QLabel#muted {color:#9aabc0;}
QGroupBox {border:1px solid #435064; border-radius:6px; margin-top:12px; padding:12px 8px 8px;}
QGroupBox::title {subcontrol-origin:margin; left:10px; color:#c9d9ed;}
QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox, QTableWidget, QListWidget {background:#18202b; border:1px solid #455369; border-radius:4px; padding:5px;}
QAbstractItemView#choicePopup {background:#18202b; border:1px solid #455369; selection-background-color:#30607d; selection-color:#f5f8ff; outline:0;}
QAbstractItemView#choicePopup::item {padding:5px 8px; min-height:18px;}
QPushButton {background:#344257; border:1px solid #50617b; border-radius:5px; padding:8px 10px;}
QPushButton:hover {background:#425774;}
QPushButton:disabled {color:#6d7b8e; background:#29313e; border-color:#354152;}
QPushButton#recentProject {background:transparent; border:1px solid #39485b; padding:0; text-align:left;}
QPushButton#recentProject:hover {background:#2d4153; border-color:#65a8bd;}
QPushButton#recentProject:focus {border-color:#65d0e0;}
QPushButton#recentProject QLabel {background:transparent;}
QPushButton#primary {background:#287da5; border:1px solid #55acd0; color:white; font-weight:600;}
QPushButton:checked {background:#285b7a; border-color:#70c2e7;}
QSlider::groove:horizontal {height:4px; background:#536277;}
QSlider::handle:horizontal {background:#7ac7e5; width:12px; margin:-5px 0; border-radius:5px;}
QHeaderView::section {background:#344257; padding:5px; border:0;}
QTableWidget::item:selected, QListWidget::item:selected {background:#30607d; color:#f5f8ff;}
QTabWidget::pane {border:1px solid #435064;}
QTabBar::tab {background:#293544; padding:8px 14px; color:#aebdd0;}
QTabBar::tab:selected {background:#344e65; color:#f5f8ff;}
QProgressBar {border:1px solid #455369; text-align:center;}
QProgressBar::chunk {background:#398cb1;}
QToolTip {color:#f5f8ff; background:#344257; border:1px solid #7e9aba;}
QWidget#sidebar {background:#202936; border-radius:8px;}
QWidget#sidebar QLabel {background:transparent;}
QPushButton#stepNav {text-align:left; background:transparent; border:0; border-left:2px solid transparent; padding:10px 12px; color:#b6c5d5;}
QPushButton#stepNav:hover {background:#2c394a; color:#f5f8ff;}
QPushButton#stepNav:checked {background:#2d4153; border-left:2px solid #65d0e0; color:#f5f8ff;}
QPushButton#stepNav:disabled {background:transparent; border-left:2px solid transparent; color:#5e6d7e;}
QToolButton {background:transparent; border:0; border-radius:5px; padding:4px;}
QToolButton:hover {background:#344257;}
QFrame#fileCard {background:#242e3b; border:1px solid #39485b; border-radius:8px;}
QFrame#fileCard QWidget {background:transparent;}
QFrame#fileCard QLineEdit, QFrame#fileCard QDoubleSpinBox {background:#1c2633;}
QLabel#fileTitle {font-weight:600; color:#edf3fa;}
QLabel#projectName {font-size:26px; font-weight:600; color:#edf3fa;}
QLineEdit#folderPath {border:0; background:transparent; color:#9aabc0;}
"""


def label(text, heading=False):
    widget = QLabel(text)
    widget.setWordWrap(True)
    widget.setObjectName("heading" if heading else "muted")
    return widget


def button(text, callback, primary=False):
    widget = QPushButton(text)
    if primary:
        widget.setObjectName("primary")
    widget.clicked.connect(lambda checked=False: callback())
    return widget


class JobSignals(QObject):
    done = Signal(object)
    failed = Signal(str)
    progress = Signal(int, str)


class Job(QRunnable):
    def __init__(self, function, signals=None):
        super().__init__()
        self.function, self.signals = function, signals if signals is not None else JobSignals()

    def run(self):
        try:
            self.signals.done.emit(self.function(self.signals.progress.emit))
        except BaseException as error:
            traceback.print_exc()
            self.signals.failed.emit(str(error))


class ProbeMapWindow(QMainWindow):
    def __init__(self, config=None, *, recent_projects=None):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.setWindowIcon(QIcon(str(RESOURCE_DIR/'icon.svg')))
        self.resize(1440, 940)
        self.setMinimumSize(1080, 720)
        self.setStyleSheet(STYLE)
        self.project = self.reference = self.moving = self.placed = None
        self.project_path=None
        self.recent_projects=recent_projects if recent_projects is not None else RecentProjects()
        self._form_loading=False
        self._save_pending=False
        self._review_aux_loading=False
        self.save_timer=QTimer(self)
        self.save_timer.setSingleShot(True)
        self.save_timer.timeout.connect(self.save_project)
        self.landmark_reference = self.landmark_grid = self.overlay_grid = None
        self._pending_landmark_positions = self._pending_overlay_position = None
        self.step = 0
        self._pending_track_step = None
        self.busy = False
        self._jobs = set()
        self._render_running, self._render_pending = False, False
        self._render_token = 0
        self._baseline = np.eye(3)
        self._updating = False
        self._undo = []
        self._selected_landmark=-1
        self._landmark_tool='add'
        self._cancelled = False
        self._closing = False
        self._pending_camera = None
        self._review_tab_index = 0
        self._workspace_panels={}
        self._panel_width=340
        self._review_find_pair=None
        self.last_export = None
        self.render_timer = QTimer(self)
        self.render_timer.setSingleShot(True)
        self.render_timer.timeout.connect(self._render)
        central = QWidget()
        root = QHBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        nav = QWidget()
        nav.setObjectName('sidebar')
        nav.setFixedWidth(175)
        nav_layout = QVBoxLayout(nav)
        brand=QHBoxLayout()
        mark=QLabel(); mark.setPixmap(QIcon(str(RESOURCE_DIR/'logo.svg')).pixmap(30,30))
        mark.setAccessibleName('ProbeMap logo')
        brand.addWidget(mark)
        brand.addWidget(label(APP_NAME, True),1)
        nav_layout.addLayout(brand)
        nav_layout.addSpacing(20)
        self.nav_buttons = []
        for i, text in enumerate(("Project", "Placement", "Registration", "Review",
                                  "Tracks", "Probes", "Reconstruction", "Electrophysiology")):
            item = button(text, lambda i=i: self.go_step(i))
            item.setObjectName('stepNav')
            item.setCheckable(True)
            item.setMinimumHeight(40)
            self.nav_buttons.append(item)
            nav_layout.addWidget(item)
        nav_layout.addStretch()
        self.settings_button=button('Settings',self.open_settings)
        from .desktop_slice_tools import tool_icon
        self.settings_button.setIcon(tool_icon('settings'))
        self.settings_button.hide()
        self.export_button=button('Export',self.open_export)
        self.export_button.setEnabled(False)
        self.export_button.hide()
        save_row=QHBoxLayout()
        self.save_button=icon_button('save','Save project · Ctrl/⌘+S',self.save_project)
        self.save_button.setEnabled(False)
        self.save_button.hide()
        self.save_state=label('')
        save_row.addWidget(self.save_button)
        save_row.addWidget(self.save_state,1)
        self.save_state.hide()
        root.addWidget(nav)
        self.pages = QStackedWidget()
        root.addWidget(self.pages, 1)
        self.settings_dialog=PreviewSettings(self)
        self.quality=self.settings_dialog.quality
        self.settings_dialog.apply_button.clicked.connect(self.apply_preview_quality)
        self._build_inputs()
        self._build_workspace()
        self.volume_view.layersRequested.connect(self._ensure_review_layers)
        self.volume_view.viewSettingsChanged.connect(self.schedule_save)
        from .desktop_insertions import InsertionWorkspace
        self.track_workspace=InsertionWorkspace(self)
        from .desktop_tables import TableSelectionGuard
        self.table_selection_guard=TableSelectionGuard(self)
        self.pages.addWidget(self.track_workspace)
        from .desktop_reconstruction import ReconstructionWorkspace
        self.reconstruction=ReconstructionWorkspace(self)
        self.pages.addWidget(self.reconstruction)
        from .desktop_ephys import ElectrophysiologyWorkspace
        self.electrophysiology=ElectrophysiologyWorkspace(self)
        self.pages.addWidget(self.electrophysiology)
        self.setCentralWidget(central)
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(240)
        self.progress.hide()
        self.statusBar().addPermanentWidget(self.progress)
        self.cancel_button = button("Cancel export", self.cancel_export)
        self.cancel_button.hide()
        self.statusBar().addPermanentWidget(self.cancel_button)
        self.statusBar().showMessage("")
        file_menu=self.menuBar().addMenu('File')
        file_menu.aboutToShow.connect(self._refresh_file_menu)
        for title,key,callback in (('New project…',QKeySequence.StandardKey.New,self.new_project),
                                   ('Open project…',QKeySequence.StandardKey.Open,self.open_project)):
            action=QAction(title,self)
            action.setShortcut(key)
            action.triggered.connect(callback)
            file_menu.addAction(action)
        self.recent_menu=file_menu.addMenu('Open recent')
        file_menu.addSeparator()
        save = QAction("Save project", self)
        save.setShortcut(QKeySequence.StandardKey.Save)
        save.triggered.connect(lambda:self.save_project() if self.project_page.mode!='setup' else None)
        self.addAction(save)
        file_menu.addAction(save)
        save_as=QAction('Save project as…',self)
        save_as.setShortcut(QKeySequence.StandardKey.SaveAs)
        save_as.triggered.connect(self.save_project_as)
        self.addAction(save_as)
        file_menu.addAction(save_as)
        export_action=QAction('Export…',self)
        export_action.triggered.connect(self.open_export)
        self.export_action=export_action
        file_menu.addAction(export_action)
        self.project_folder_action=file_menu.addAction('Show project folder',self.show_project_folder)
        file_menu.addSeparator()
        settings=QAction('Settings…',self)
        settings.setShortcut(QKeySequence.StandardKey.Preferences)
        settings.setMenuRole(QAction.MenuRole.PreferencesRole)
        settings.triggered.connect(self.open_settings)
        file_menu.addAction(settings)
        undo=QAction('Undo',self)
        keys=[QKeySequence(QKeySequence.StandardKey.Undo),QKeySequence('Ctrl+Z'),QKeySequence('Meta+Z')]
        unique={key.toString():key for key in keys}
        undo.setShortcuts(list(unique.values()))
        undo.triggered.connect(self.undo_current_step)
        self.addAction(undo)
        edit_menu=self.menuBar().addMenu('Edit'); edit_menu.addAction(undo)
        self.probes_library_action=QAction('Probes Library…',self)
        self.probes_library_action.triggered.connect(self.open_probes_library)
        self.probes_library_menu=self.menuBar().addMenu('Probes Library')
        self.probes_library_menu.addAction(self.probes_library_action)
        view_menu=self.menuBar().addMenu('View')
        sidebar_action=QAction('Show / hide settings panel',self)
        sidebar_action.setShortcut(QKeySequence('Ctrl+\\'))
        sidebar_action.triggered.connect(self.toggle_current_sidebar); view_menu.addAction(sidebar_action)
        self.go_step(0)
        if config:
            self.prefill(config)

    def open_probes_library(self):
        from .desktop_probe_library import ProbesLibraryDialog
        dialog=ProbesLibraryDialog(self)
        dialog.libraryChanged.connect(self.track_workspace._refresh_probe_models)
        dialog.exec()

    def _refresh_file_menu(self):
        self.export_action.setEnabled(self.export_button.isEnabled())
        self.project_folder_action.setEnabled(self.project is not None)
        self.recent_menu.clear()
        from .desktop_project import project_name
        for path in self.recent_projects.entries():
            action=self.recent_menu.addAction(project_name(path.parent))
            action.setToolTip(str(path.parent))
            action.triggered.connect(lambda checked=False,path=str(path):self.open_recent_project(path))
        self.recent_menu.setEnabled(bool(self.recent_menu.actions()))

    def show_project_folder(self):
        if self.project is None: return
        from qtpy.QtCore import QUrl
        from qtpy.QtGui import QDesktopServices
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.project.output_dir)))

    def save_figure(self,canvas):
        if self.project is None or self.busy: return
        from .desktop_figures import next_figure_path,export_slice,export_volume
        path=None
        try:
            path=next_figure_path(self.project.output_dir,self.settings_dialog.figure_format.currentData())
            if hasattr(canvas,'render_volumes'): export_volume(self,canvas,path)
            else: export_slice(self,canvas,path)
        except Exception as error:
            if path is not None: path.unlink(missing_ok=True)
            self._error(str(error))

    def toggle_current_sidebar(self):
        if self.step==7: self.electrophysiology.toggle_sidebar()
        elif self.step==6: self.reconstruction.toggle_sidebar()
        elif self.step>=4: self.track_workspace.sidebar_toggle.toggle()
        elif self.step: self.toggle_workspace_panel()

    def _build_inputs(self):
        self.project_page=ProjectPage()
        for name in ('atlas_input','scan_input','track_input','labels_path','output_path'):
            setattr(self,name,getattr(self.project_page,name))
        self.project_page.openRequested.connect(self.open_project)
        self.project_page.recentRequested.connect(self.open_recent_project)
        self.project_page.set_recent_projects(self.recent_projects.entries())
        self.project_page.newRequested.connect(self.new_project)
        self.project_page.cancelRequested.connect(self.cancel_new_project)
        self.project_page.folderRequested.connect(self.choose_output)
        self.project_page.labelsRequested.connect(self.choose_labels)
        self.project_page.electrodesRequested.connect(self.change_electrodes)
        self.project_page.calibrationRequested.connect(self.open_calibration)
        self.project_page.loadRequested.connect(self.load_images)
        self.atlas_input.changed.connect(self._inputs_changed)
        self.scan_input.changed.connect(self._inputs_changed)
        self.inputs_page=self.project_page
        self.pages.addWidget(self.project_page)

    def _build_workspace(self):
        workspace = QWidget()
        layout = QVBoxLayout(workspace)
        self.workspace_title = label("Placement", True)
        self.workspace_header=QHBoxLayout()
        self.panel_toggle=icon_button('sidebar','Hide settings panel',self.toggle_workspace_panel)
        self.workspace_header.addWidget(self.panel_toggle)
        self.workspace_header.addWidget(self.workspace_title,1)
        layout.addLayout(self.workspace_header)
        self.source_summary = label("")
        layout.addWidget(self.source_summary)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.workspace_splitter=splitter
        splitter.setHandleWidth(7)
        splitter.setStyleSheet('QSplitter::handle {background:#354354;} QSplitter::handle:hover {background:#7ac7e5;}')
        self.controls_stack = QStackedWidget()
        self.controls_stack.setMinimumWidth(300)
        self.controls_stack.setMaximumWidth(16777215)
        self.controls_stack.setStyleSheet('QStackedWidget {background:#242b36;border:0;}')
        splitter.addWidget(self.controls_stack)
        self.views_stack = QStackedWidget()
        self.views_stack.setStyleSheet('QStackedWidget {background:#242b36;border:0;}')
        overlay = QWidget()
        grid = QGridLayout(overlay)
        self.overlay_panes = []
        for axis in range(3):
            pane = PlacementSlicePane(axis)
            for canvas in pane.canvases: canvas.background_color='#000000'
            pane.changed.connect(self.request_render)
            for canvas in (pane.canvas,pane.scan_canvas):
                canvas.translated.connect(self._drag_translate)
                canvas.rotated.connect(lambda delta, axis=axis: self._drag_rotate(axis, delta))
            self.overlay_panes.append(pane)
            grid.addWidget(pane, axis//2, axis%2)
        self.volume_view = VolumeView(('Atlas','Brain scan'))
        self.volume_view.opacityChanged.connect(self._volume_opacity_changed)
        grid.addWidget(self.volume_view, 1, 1)
        for n in (0, 1):
            grid.setColumnStretch(n, 1)
            grid.setRowStretch(n, 1)
        self.preview_tabs = QTabWidget()
        from .desktop_widgets import flat_tabs
        flat_tabs(self.preview_tabs)
        self.preview_tabs.setElideMode(Qt.TextElideMode.ElideNone)
        self.preview_tabs.addTab(overlay, "Overlay && 3D")
        self.slice_comparison = SliceComparison()
        for canvas in self.slice_comparison.canvases: canvas.background_color='#000000'
        self.slice_comparison.changed.connect(self.request_render)
        self.preview_tabs.addTab(self.slice_comparison, "Slice comparison")
        self.preview_tabs.currentChanged.connect(self._preview_tab_changed)
        self.views_stack.addWidget(self.preview_tabs)
        from .desktop_landmarks import LandmarkViews
        self.landmark_views=LandmarkViews(self)
        self.workspace_header.addWidget(self.landmark_views.toolbar)
        self.landmark_views.set_tool(self._landmark_tool)
        self.views_stack.addWidget(self.landmark_views)
        self.controls_stack.addWidget(self._placement_controls())
        from .desktop_landmarks import LandmarkTools
        self.landmark_tools=LandmarkTools(self)
        self.controls_stack.addWidget(self.landmark_tools)
        from .desktop_review import ReviewTools
        self.review_tools=ReviewTools(self)
        self.controls_stack.addWidget(self.review_tools)
        splitter.addWidget(self.views_stack)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([340, 1000])
        layout.addWidget(splitter, 1)
        self.footer = QWidget()
        row = QHBoxLayout(self.footer)
        row.setContentsMargins(0,3,0,0)
        row.addWidget(button("Back",lambda:self.go_step(max(0,self.step-1))))
        row.addStretch()
        self.placement_apply=button('Apply',self.apply_workspace)
        self.landmark_apply=self.placement_apply
        self.placement_apply.setToolTip('Apply the current scan placement')
        row.addWidget(self.placement_apply)
        self.footer_placement_save=button('Save',self.save_project)
        self.footer_placement_save.setToolTip('Save project · Ctrl/⌘+S')
        row.addWidget(self.footer_placement_save)
        self.next_button = button("Next",self._next_step,True)
        row.addWidget(self.next_button)
        layout.addWidget(self.footer)
        self.pages.addWidget(workspace)

    def toggle_workspace_panel(self):
        self._set_workspace_panel(not self.controls_stack.isVisible())

    def _set_workspace_panel(self,visible,*,remember=True):
        if not visible and self.controls_stack.isVisible():
            self._panel_width=max(300,self.workspace_splitter.sizes()[0])
        self.controls_stack.setVisible(visible)
        if visible:
            self.workspace_splitter.setSizes([self._panel_width,max(1,self.workspace_splitter.width()-self._panel_width)])
        title='Hide settings panel' if visible else 'Show settings panel'
        self.panel_toggle.setToolTip(title); self.panel_toggle.setAccessibleName(title)
        if remember:
            self._workspace_panels[str(self.step)]=visible
            self.schedule_save()

    def _next_step(self):
        if self.step == 1:
            self.go_step(2)
        elif self.step == 2:
            if self.project.fit_is_current:
                self.go_step(3)
        elif self.step == 3:
            self.go_step(4)

    def apply_placement(self):
        if self.project is None or self.busy or self.step!=1: return
        # The preview already reslices in atlas axes. Applying must preserve
        # the user's placement, especially for scans missing part of the brain.
        self.request_render()
        self.save_project()
        self.statusBar().showMessage('Placement applied',2000)

    def _appearance(self, layout, prefix):
        group = QGroupBox('Opacity')
        form = QGridLayout(group)
        sliders = []
        for row, (name, value) in enumerate((("Atlas", 65), ("Scan", 90))):
            if prefix=='placement' and name=='Scan': name='Brain scan'
            form.addWidget(QLabel(name), row, 0)
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, 100)
            slider.setValue(value)
            amount = QLabel(f"{value}%")
            slider.valueChanged.connect(lambda v, amount=amount: amount.setText(f"{v}%"))
            slider.valueChanged.connect(self.request_render)
            form.addWidget(slider, row, 1)
            form.addWidget(amount, row, 2)
            sliders.append(slider)
        setattr(self, prefix + "_opacity", sliders)
        layout.addWidget(group)

    def _volume_opacity_changed(self,index,value):
        self.volume_view._apply_layer_visibility()

    def _placement_controls(self):
        from .desktop_placement import PlacementTools
        self.placement_tools=PlacementTools(self)
        return self.placement_tools

    def apply_workspace(self):
        if self.step==1: self.apply_placement()
        elif self.step==2: self.fit_points(advance=False)

    def prefill(self, config_path):
        from .volumes import load_project
        config = load_project(config_path)
        self.project_page.show_setup(config['output_dir'])
        self._form_loading=True
        for widget, name in ((self.atlas_input,"reference"),(self.scan_input,"moving")):
            widget.path.setText(str(config[name]["path"]))
            widget.read_metadata()
            for spin, value in zip(widget.spacing, config[name]["spacing_um_zyx"][::-1]):
                spin.setValue(value)
        self.output_path.setText(str(config["output_dir"]))
        self._suggest_auxiliary(config["reference"]["path"],config["moving"]["path"])
        self._form_loading=False
        self.project_page.update_ready()

    def new_project(self):
        if self.busy: return
        folder=QFileDialog.getExistingDirectory(self,'New project folder',
            self.output_path.text() or str(Path.home()))
        if not folder: return
        folder=Path(folder).resolve()
        if (folder/'project.json').is_file():
            self.load_project_path(folder)
            return
        if self.project is not None and self.save_project() is None: return
        self._setup_return_step=self.step
        self.go_step(0)
        self.project_page.show_setup(folder)
        for item in self.nav_buttons[1:]: item.setEnabled(False)
        self.save_button.setEnabled(False)
        self.export_button.setEnabled(False)
        self.statusBar().clearMessage()

    def cancel_new_project(self):
        if self.project is not None:
            self._sync_project_inputs(self.project)
            self.project_page.show_summary(self.project)
            self.export_button.setEnabled(True)
            self.go_step(getattr(self,'_setup_return_step',0))
        else:
            self.project_page.show_welcome()

    def _inputs_changed(self):
        if self._form_loading or self.project_page.mode!='setup': return
        reference,brain=self.atlas_input.path.text().strip(),self.scan_input.path.text().strip()
        if reference and Path(reference).is_file() and brain and Path(brain).is_file():
            self._suggest_auxiliary(reference,brain)
        self.project_page.update_ready()

    def _sync_project_inputs(self,project):
        self._form_loading=True
        try:
            self.atlas_input.set_spec(project.reference)
            self.scan_input.set_spec(project.moving)
            self.track_input.clear()
            if project.track is not None: self.track_input.set_spec(project.track)
            self.labels_path.setText(str(project.labels_dir) if project.labels_dir else '')
            self.output_path.setText(str(project.output_dir))
        finally:
            self._form_loading=False

    def open_settings(self):
        if self.busy: return
        if self.project is not None and self.project_page.mode!='setup':
            self.quality.setCurrentIndex(max(0,self.quality.findData(self.project.preview_dimension)))
        self.settings_dialog.show()
        self.settings_dialog.raise_()
        self.settings_dialog.activateWindow()

    def apply_preview_quality(self):
        if self.busy: return
        dimension=self.quality.currentData()
        self.settings_dialog.hide()
        if self.project is None or self.project_page.mode=='setup' or dimension==self.project.preview_dimension: return
        project=self.project
        project.display=self._display_record()
        previous=project.preview_dimension
        project.preview_dimension=dimension
        self._render_token+=1
        self.render_timer.stop()
        def failed(error):
            project.preview_dimension=previous
            self.quality.setCurrentIndex(max(0,self.quality.findData(previous)))
        self._load_project_volumes(project,self.step,self.project_path,
                                   preserve_workspace=True,on_error=failed)

    def open_calibration(self):
        if self.project is None or self.busy: return
        dialog=CalibrationDialog(self.project,self)
        if dialog.exec():
            values=dialog.values()
            self.apply_calibration(values['reference'],values['moving'],
                                   labels_spacing=values['labels'],track_spacing=values['track'])

    def apply_calibration(self,reference_spacing,scan_spacing,*,labels_spacing=None,track_spacing=None):
        if self.project is None or self.busy: return
        project=self.project
        if (np.array_equal(reference_spacing,project.reference.spacing_um_zyx)
                and np.array_equal(scan_spacing,project.moving.spacing_um_zyx)
                and (labels_spacing is None or np.array_equal(labels_spacing,project.label_spacing))
                and (track_spacing is None or project.track is None or np.array_equal(track_spacing,project.track.spacing_um_zyx))): return
        project.display=self._display_record()
        project_file=self.project_path or project.output_dir/'project.json'
        atlas_ratio=np.asarray(reference_spacing)/project.reference.spacing_um_zyx
        import copy
        snapshot=copy.deepcopy(project)
        landmark_undo=copy.deepcopy(self._undo)
        for pairs in landmark_undo:
            for pair in pairs:
                for key in ('reference','moving'):
                    if pair.get(key) is not None: pair[key]=(np.asarray(pair[key])*atlas_ratio).tolist()
        self._cancelled=False
        def recalculate(progress):
            from .calibration import prepare_calibration_update
            try:
                return prepare_calibration_update(snapshot,reference_spacing,scan_spacing,project_file,
                                                  progress,lambda:self._cancelled,
                                                  labels_spacing=labels_spacing,track_spacing=track_spacing)
            except InterruptedError: return None
        def ready(result):
            if result is None:
                self.statusBar().showMessage('Calibration cancelled. Project unchanged.')
                return
            updated,reference,moving=result
            self._load_project_volumes(updated,0,project_file,save_current=False,prepared=(reference,moving))
            self._undo=landmark_undo
            self.statusBar().showMessage('Calibration updated. Registration, annotations and results recalculated.')
        self._start_job(recalculate,ready,'Recalculating physical calibration…')
        self.cancel_button.setText('Cancel update')
        self.cancel_button.show()

    def _suggest_auxiliary(self,reference_path,brain_path):
        brain=Path(brain_path)
        candidate=brain.with_name(brain.name.replace("Ch0_brain","Ch1_electrodes"))
        if candidate!=brain and candidate.is_file() and not self.track_input.path.text().strip():
            self.track_input.path.setText(str(candidate))
            self.track_input.read_metadata()
        if not self.labels_path.text().strip():
            folders=sorted(p for p in Path(reference_path).parent.iterdir() if p.is_dir() and "region_masks" in p.name)
            if len(folders)==1: self.labels_path.setText(str(folders[0]))

    def choose_labels(self):
        if self.busy: return
        path=QFileDialog.getExistingDirectory(self,"Choose atlas region masks folder",self.labels_path.text() or str(Path.home()))
        if path:
            self.labels_path.setText(path)
            if self.project_page.mode=='summary': self._apply_auxiliary()

    def change_electrodes(self):
        if self.project is None or self.busy: return
        from qtpy.QtWidgets import QDialog,QDialogButtonBox
        field=ImageInput('Electrodes scan')
        field.browse()
        if not field.info: return
        if any(spin.value()<=0 for spin in field.spacing):
            dialog=QDialog(self)
            dialog.setWindowTitle('Electrodes scan')
            layout=QVBoxLayout(dialog)
            layout.addWidget(field)
            buttons=QDialogButtonBox(QDialogButtonBox.StandardButton.Open|QDialogButtonBox.StandardButton.Cancel)
            buttons.accepted.connect(dialog.accept)
            buttons.rejected.connect(dialog.reject)
            layout.addWidget(buttons)
            if not dialog.exec(): return
        try:
            self.track_input.set_spec(field.spec())
            self._apply_auxiliary()
        except (ValueError,OSError) as error:
            self._error(str(error))

    def _auxiliary_inputs(self,project):
        track=self.track_input.spec() if self.track_input.path.text().strip() else None
        directory=Path(self.labels_path.text().strip()).expanduser().resolve() if self.labels_path.text().strip() else None
        if directory is not None and not directory.is_dir():
            raise ValueError("Choose an existing atlas labels folder.")
        previous=project.track
        project.track=track
        try: project.validate_track()
        except ValueError:
            project.track=previous
            raise
        if directory!=project.labels_dir: project.labels_spacing_um_zyx=None
        project.labels_dir=directory

    def _apply_auxiliary(self):
        if self.project is None or self.busy: return
        try:
            for widget,spec in ((self.atlas_input,self.project.reference),(self.scan_input,self.project.moving)):
                if Path(widget.path.text().strip()).resolve()!=spec.path:
                    raise ValueError("Create a new project to register a different atlas or brain scan.")
            self.track_workspace.capture_state()
            self._auxiliary_inputs(self.project)
            self.volume_view.set_layer_titles(("Atlas","Brain scan","Track scan") if self.project.track else ("Atlas","Brain scan"))
            self.volume_view.set_label_meshes([])
            self.track_workspace.reset()
            self.reconstruction.reset()
            self.electrophysiology.reset()
            self.save_project()
            self._update_summary()
            self.project_page.show_summary(self.project)
            self.statusBar().showMessage("Project files updated.")
            if self.project.track is not None:
                target=self._pending_track_step
                self._pending_track_step=None
                if target is not None: self.go_step(target)
        except (ValueError,OSError,KeyError) as error:
            self._sync_project_inputs(self.project)
            self._error(str(error))

    def choose_output(self):
        path = QFileDialog.getExistingDirectory(self, "Choose results folder", self.output_path.text() or str(Path.home()))
        if path:
            self.output_path.setText(path)
            self._output_folder_edited()

    def _output_folder_edited(self):
        if not self.output_path.text().strip(): return
        path=Path(self.output_path.text().strip()).expanduser().resolve()
        self.output_path.setText(str(path))
        if self.project and self.project_page.mode!='setup':
            self.project.output_dir=path
            self.project_path=path/'project.json'
        self._update_summary()
        self.project_page.update_ready()

    def _error(self, message):
        self.statusBar().showMessage(message)
        QMessageBox.warning(self, "ProbeMap", message)

    def _start_job(self, function, complete, message,on_error=None):
        if self.busy:
            return
        self.busy = True
        self.cancel_button.setEnabled(True)
        self.progress.setRange(0,0)
        self.progress.show()
        self.statusBar().showMessage(message)
        self.inputs_page.setEnabled(False)
        self.controls_stack.setEnabled(False)
        self.footer.setEnabled(False)
        self.track_workspace.setEnabled(False)
        self.reconstruction.setEnabled(False)
        self.settings_button.setEnabled(False)
        self.export_button.setEnabled(False)
        self.settings_dialog.apply_button.setEnabled(False)
        self.save_button.setEnabled(False)
        for item in self.nav_buttons:
            item.setEnabled(False)
        job = Job(function)
        self._jobs.add(job)
        def finish(value=None, error=None):
            self._jobs.discard(job)
            self.busy = False
            self.progress.hide()
            self.cancel_button.hide()
            self.inputs_page.setEnabled(True)
            self.controls_stack.setEnabled(True)
            self.footer.setEnabled(True)
            self.track_workspace.setEnabled(True)
            self.reconstruction.setEnabled(True)
            self.settings_button.setEnabled(True)
            self.export_button.setEnabled(self.project is not None and self.project_page.mode!='setup')
            self.settings_dialog.apply_button.setEnabled(True)
            self.save_button.setEnabled(self.project is not None and self.project_page.mode!='setup')
            for index,item in enumerate(self.nav_buttons):
                item.setEnabled(index==0 or (self.project is not None and self.project_page.mode!='setup'))
            if error:
                if on_error is not None: on_error(error)
                self._error(error)
            else:
                complete(value)
            if not self.busy:
                if self._save_pending: QTimer.singleShot(0,self.save_project)
                QTimer.singleShot(0,self._ensure_review_layers)
        job.signals.done.connect(lambda value: finish(value=value))
        job.signals.failed.connect(lambda error: finish(error=error))
        job.signals.progress.connect(self._job_progress)
        QThreadPool.globalInstance().start(job)

    def _job_progress(self, percent, message):
        self.progress.setRange(0,100)
        self.progress.setValue(percent)
        self.statusBar().showMessage(message)

    def _ensure_review_layers(self):
        if self.project is None or self.busy or self._review_aux_loading or self.volume_view.canvas is None: return
        if self.step!=3: return
        project=self.project
        view=self.volume_view
        if view.label_visuals:
            active=next((s for s in project.track_analysis.get('sessions',[]) if s['id']==project.track_analysis.get('active_session')), {})
            view.set_label_styles(active.get('regions',{}))
        need_track=bool(project.track and len(view.visuals)<3 and view.layers.checks[2].isChecked())
        need_labels=view.layers.labels_check.isChecked() and not view.label_visuals
        if need_labels and project.labels_dir is None:
            view.layers.labels_check.setChecked(False)
            view.layers.labels_check.setToolTip('Add a labels folder on Project first.')
            need_labels=False
        if not need_track and not need_labels: return
        self._review_aux_loading=True
        def load(progress):
            from .track_model import load_track_data,load_label_atlas
            from .desktop_layers import label_surfaces
            import tempfile
            gpu=load_track_data(project.track,project.preview_dimension)[1] if need_track else None
            meshes=None
            if need_labels:
                atlas=load_label_atlas(project.labels_dir,project.reference,
                      Path(tempfile.gettempdir())/'probemap'/'native-labels',progress,spacing_um_zyx=project.label_spacing)
                meshes=label_surfaces(atlas)
            return gpu,meshes
        def ready(result):
            self._review_aux_loading=False
            if self.project is not project: return
            gpu,meshes=result
            if gpu is not None: view.set_extra_volume(gpu)
            if meshes is not None:
                sessions=project.track_analysis.get('sessions',[])
                active=next((s for s in sessions if s['id']==project.track_analysis.get('active_session')), {})
                view.set_label_meshes(meshes,active.get('regions',{}))
            self.statusBar().clearMessage()
            self.request_render()
        def failed(error):
            self._review_aux_loading=False
            if need_track: view.layers.checks[2].setChecked(False)
            if need_labels: view.layers.labels_check.setChecked(False)
        self._start_job(load,ready,'Preparing selected 3D layers…',on_error=failed)

    def load_images(self, skip_placement=False):
        if self.busy:
            return
        try:
            reference, moving = self.atlas_input.spec(), self.scan_input.spec()
            if not self.output_path.text().strip():
                raise ValueError("Choose a results folder.")
            output = Path(self.output_path.text()).expanduser().resolve()
            output.mkdir(parents=True, exist_ok=True)
            project = AlignmentProject(reference, moving, output, preview_dimension=self.quality.currentData())
            self._auxiliary_inputs(project)
            if skip_placement:
                project.coarse[:3,3] = np.array(moving.origin_um_zyx)-reference.origin_um_zyx
            else:
                project.coarse[:3,3] = ((np.array(reference.shape_zyx)-1)*reference.spacing_um_zyx
                                        -(np.array(moving.shape_zyx)-1)*moving.spacing_um_zyx)/2
        except (ValueError, OSError) as error:
            self._error(str(error))
            return
        self._load_project_volumes(project, 2 if skip_placement else 1)

    def _load_project_volumes(self, project, target,project_path=None,*,preserve_workspace=False,on_error=None,save_current=True,prepared=None):
        if save_current and self.project is not None and project is not self.project and not self.busy:
            if self.save_project() is None: return
        undo=list(self._undo) if preserve_workspace else []
        def load(progress):
            progress(5,"Loading atlas preview")
            reference = load_preview(project.reference, project.preview_dimension)
            progress(50,"Loading scan preview")
            moving = load_preview(project.moving, project.preview_dimension)
            return reference, moving
        def ready(volumes):
            self._render_token += 1
            self.track_workspace.reset()
            self.reconstruction.reset()
            self.electrophysiology.reset()
            self.project = project
            self.project_path=Path(project_path or project.output_dir/'project.json').resolve()
            self.reference, self.moving = volumes
            self.placed = None
            self.landmark_reference = self.landmark_grid = self.overlay_grid = None
            self._pending_landmark_positions = self._pending_overlay_position = None
            self._undo=undo
            self._set_pose_baseline()
            for pane in self.overlay_panes:
                pane.configure(self.reference.data.shape, self.reference.spacing_um_zyx)
            for panes in self.landmark_panes.values():
                for pane in panes:
                    pane.configure(self.reference.data.shape, self.reference.spacing_um_zyx)
            self.volume_view.set_layer_titles(("Atlas","Brain scan","Track scan") if project.track else ("Atlas","Brain scan"))
            self.volume_view.load(self.reference, self.moving)
            self.review_tools.refresh_rendering()
            self.export_button.setEnabled(True)
            self._restore_display()
            self._update_summary()
            self.update_landmarks()
            self._sync_project_inputs(project)
            self.project_page.show_summary(project)
            self.save_button.setEnabled(True)
            self.go_step(target)
            self._remember_project()
            self.statusBar().showMessage("Preview updated." if preserve_workspace else "Project ready.")
        def failed(error):
            if self.project is not None and project is not self.project and self.project_page.mode!='setup':
                self._sync_project_inputs(self.project)
                self.quality.setCurrentIndex(max(0,self.quality.findData(self.project.preview_dimension)))
            if on_error is not None: on_error(error)
        if prepared is not None: ready(prepared)
        else: self._start_job(load, ready,"Updating preview…" if preserve_workspace else "Opening project…",on_error=failed)

    def _set_pose_baseline(self):
        self._updating = True
        self._baseline = self.project.coarse[:3,:3].copy()
        centre = (np.array(self.project.moving.shape_zyx)-1)/2*self.project.moving.spacing_um_zyx
        shift = transform_points(centre[None], self.project.coarse)[0]-centre
        span = max(np.array(self.project.reference.shape_zyx)*self.project.reference.spacing_um_zyx)*3
        for (name,axis),(slider,spin,factor) in self.pose_controls.items():
            if name == "translation":
                slider.setRange(round(-span),round(span))
            value = shift[2-axis] if name == "translation" else (1 if name == "scale" else 0)
            self._set_control(slider,spin,factor,value)
        self.axis_order.setCurrentIndex(0)
        for check in self.flips:
            check.setChecked(False)
        self._updating = False

    @staticmethod
    def _set_control(slider,spin,factor,value):
        slider.blockSignals(True)
        spin.blockSignals(True)
        slider.setMinimum(min(slider.minimum(), round(value*factor)))
        slider.setMaximum(max(slider.maximum(), round(value*factor)))
        spin.setValue(float(value))
        slider.setValue(round(value*factor))
        slider.blockSignals(False)
        spin.blockSignals(False)

    def _pose_changed(self, name=None, axis=None, value=None):
        if self._updating or self.project is None:
            return
        if name is not None:
            self._set_control(*self.pose_controls[name,axis],value)
        values = {key: [self.pose_controls[key,a][1].value() for a in range(3)] for key in ("translation","rotation","scale")}
        spec = self.project.moving
        matrix = pose_matrix(spec.shape_zyx,spec.spacing_um_zyx,values["translation"][::-1],values["rotation"],values["scale"])
        matrix[:3,:3] = matrix[:3,:3] @ orientation_linear(self.axis_order.currentText(),[c.isChecked() for c in self.flips]) @ self._baseline
        centre = (np.array(spec.shape_zyx)-1)/2*spec.spacing_um_zyx
        matrix[:3,3] = centre+values["translation"][::-1]-matrix[:3,:3]@centre
        self._change_coarse(matrix)

    def _change_coarse(self, matrix):
        if np.allclose(matrix,self.project.coarse,rtol=0,atol=1e-10):
            return
        delta = matrix@np.linalg.inv(self.project.coarse)
        self._undo.clear()
        for pair in self.project.pairs:
            if pair.get("moving") is not None:
                pair["moving"] = transform_points(np.array(pair["moving"])[None],delta)[0].tolist()
        self.project.coarse = validate_affine(matrix)
        self.placed = None
        self.landmark_reference = self.landmark_grid = None
        self.request_render()
        if self.project.fit_result:
            self.statusBar().showMessage("Placement changed. Refit the landmark pairs before exporting an affine result.")

    def centre_scan(self):
        if not self.project:
            return
        ref, mov = self.project.reference,self.project.moving
        shift = ((np.array(ref.shape_zyx)-1)*ref.spacing_um_zyx-(np.array(mov.shape_zyx)-1)*mov.spacing_um_zyx)/2
        self._updating = True
        for axis in range(3):
            self._set_control(*self.pose_controls["translation",axis],shift[2-axis])
        self._updating = False
        self._pose_changed()

    def reset_pose(self):
        if not self.project:
            return
        self._updating = True
        self._baseline = np.eye(3)
        self.axis_order.setCurrentIndex(0)
        for check in self.flips:
            check.setChecked(False)
        for name in ("rotation","scale"):
            for axis in range(3):
                self._set_control(*self.pose_controls[name,axis],1 if name == "scale" else 0)
        self._updating = False
        self._pose_changed()

    def _drag_translate(self, delta):
        if self.step != 1 or self.busy:
            return
        self._updating = True
        for axis in range(3):
            slider,spin,factor = self.pose_controls["translation",axis]
            self._set_control(slider,spin,factor,spin.value()+delta[2-axis])
        self._updating = False
        self._pose_changed()

    def _drag_rotate(self, axis, delta):
        if self.step != 1 or self.busy:
            return
        angles = [self.pose_controls["rotation",i][1].value() for i in range(3)]
        rotation = Rotation.from_rotvec(np.eye(3)[2-axis]*np.radians(delta))*Rotation.from_euler("xyz",angles,degrees=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore",UserWarning)
            angles = rotation.as_euler("xyz",degrees=True)
        self._updating = True
        for i, value in enumerate(angles):
            self._set_control(*self.pose_controls["rotation",i],value)
        self._updating = False
        self._pose_changed()

    def import_placement(self):
        if not self.project:
            return
        path,_ = QFileDialog.getOpenFileName(self,"Import placement JSON",str(self.project.output_dir),"JSON (*.json)")
        if not path:
            return
        try:
            record = json.loads(Path(path).read_text(encoding='utf-8'))
            matrix = record.get("moving_to_reference_zyx_um",record.get("coarse"))
            if matrix is None:
                matrix = record.get("moving_to_atlas_zyx_um")
            if record.get("coordinate_order","ZYX") != "ZYX" or record.get("units","um") != "um":
                raise ValueError("Import requires a ZYX forward transform in micrometres.")
            for name,saved in record.get("sources",{}).items():
                spec = getattr(self.project,name)
                if Path(saved["path"]).resolve() != spec.path or ("spacing_um_zyx" in saved and not np.allclose(saved["spacing_um_zyx"],spec.spacing_um_zyx)):
                    raise ValueError("Transform source files or voxel sizes do not match this project.")
            self._change_coarse(validate_affine(matrix))
            self._set_pose_baseline()
        except (ValueError,OSError,KeyError,TypeError) as error:
            self._error(str(error))

    def go_step(self, step):
        step=max(0,min(7,int(step)))
        # A rejected click must not leave a second checkable nav button active.
        for i,item in enumerate(self.nav_buttons): item.setChecked(i==self.step)
        if self.busy:
            return
        if step and self.project_page.mode=='setup': return
        if step and self.project is None:
            self.statusBar().showMessage("Open or create a project first.")
            return
        if step >= 3 and not self.project.fit_is_current:
            self.statusBar().showMessage("Apply the current registration to continue.")
            return
        if step >= 4 and self.project.track is None:
            self._pending_track_step=step
            self.go_step(0)
            self.project_page.progress_summary.setText('Add an electrodes scan to continue.')
            self.project_page.summary_actions['track'].setFocus()
            self.statusBar().showMessage("Choose an electrodes scan on Project.")
            return
        self.track_workspace.capture_state()
        self.reconstruction.capture_state()
        self.electrophysiology.capture_state()
        self._render_token += 1
        self.step = step
        self.landmark_views.toolbar.setVisible(step==2)
        self.save_button.hide()
        self.save_state.hide()
        # Release both owners before attaching shared live volume controls.
        self.review_tools.attach_volume_controls(False)
        self.placement_tools.attach_volume_controls(step==1)
        self.review_tools.attach_volume_controls(step==3)
        for i,item in enumerate(self.nav_buttons):
            item.setChecked(i == step)
            item.setEnabled(i==0 or self.project is not None)
        self.pages.setCurrentIndex(0 if step == 0 else (4 if step==7 else (3 if step==6 else (2 if step>=4 else 1))))
        if self.project is not None:
            self.save_project()
        if not step:
            if self.project is not None and self.project_page.mode!='setup':
                self.project_page.show_summary(self.project)
            return
        if step==7:
            self.electrophysiology.open(self.project,self.reference)
            return
        if step==6:
            self.reconstruction.open(self.project,self.reference)
            return
        if step>=4:
            self.track_workspace.open(self.project,self.reference,step-4)
            return
        self.controls_stack.setCurrentIndex(step-1)
        self._set_workspace_panel(self._workspace_panels.get(str(step),True),remember=False)
        self.views_stack.setCurrentIndex(1 if step == 2 else 0)
        self.preview_tabs.blockSignals(True)
        self.preview_tabs.setCurrentIndex(self._review_tab_index if step == 3 else 0)
        self.preview_tabs.tabBar().setVisible(step == 3)
        self.preview_tabs.blockSignals(False)
        self.review_tools.set_comparison(step==3 and self._review_tab_index==1)
        self.workspace_title.setText(("", "Placement", "Registration", "Review")[step])
        self.source_summary.hide()
        self.next_button.setText(("", "Next", "Next", "Next")[step])
        self.next_button.setEnabled(True)
        self.footer_placement_save.setVisible(step in (1,2,3))
        self.placement_apply.setVisible(step in (1,2))
        self.placement_apply.setEnabled(True)
        self.placement_apply.setToolTip('Apply the current scan placement' if step==1 else 'Apply landmark transformation')
        for pane in self.overlay_panes:
            pane.set_placement_mode(step == 1)
        self._update_summary()
        if step == 2:
            self.update_landmarks()
            if self.placed is None:
                coarse = self.project.coarse.copy()
                self._start_job(lambda progress: prepare_landmark_views(self.reference,self.moving,
                                self.project.reference,self.project.moving,coarse),
                                self._placed_ready,"Preparing greyscale landmark views…")
            else:
                self.draw_landmarks()
        else:
            self.request_render()
            QTimer.singleShot(0,self._restore_or_fit_camera)
            if step==3: QTimer.singleShot(0,self._ensure_review_layers)

    def _restore_or_fit_camera(self):
        if self._pending_camera:
            self.volume_view.restore_camera(self._pending_camera)
            self._pending_camera = None
        else:
            self.volume_view.reset_view()

    def _placed_ready(self, data):
        self.landmark_reference, self.placed, self.landmark_grid = data
        grid = self.landmark_grid
        for name,panes in self.landmark_panes.items():
            positions = (self._pending_landmark_positions or {}).get(name)
            for axis,pane in enumerate(panes):
                pane.configure(grid["shape_zyx"],grid["spacing_um_zyx"],grid["origin_um_zyx"],preserve_world=True)
                if positions is not None:
                    pane.slider.blockSignals(True)
                    pane.set_world_depth(positions[axis])
                    pane.slider.blockSignals(False)
        self._pending_landmark_positions = None
        self.landmark_views.sync_slices()
        self.landmark_views.set_linked(self.link_slices.isChecked())
        self.update_landmarks()
        self.draw_landmarks()
        self.statusBar().clearMessage()
        if self._review_find_pair is not None:
            row=self._review_find_pair; self._review_find_pair=None
            self.find_landmark(row)

    def request_render(self, *args):
        if self._closing or self.project is None or self.step not in (1,3):
            return
        self._render_token += 1
        matrix = self.project.coarse if self.step == 1 else self.project.total
        opacity = self.volume_view.layers.sliders[:2]
        self.volume_view.update_alignment(matrix,*[s.value()/100 for s in opacity])
        if self.step == 1 or self.preview_tabs.currentIndex() == 0:
            self.volume_view.update_slice_planes(self.overlay_grid,[p.world_depth for p in self.overlay_panes])
        self.render_timer.start(35)

    def _preview_tab_changed(self, index):
        if self.step == 3:
            self._review_tab_index = index
            self.request_render()

    def _render(self):
        if self.project is None or self.step not in (1,3):
            return
        if self._render_running:
            self._render_pending = True
            return
        token = self._render_token
        matrix = (self.project.coarse if self.step == 1 else self.project.total).copy()
        comparing = self.step == 3 and self.preview_tabs.currentIndex() == 1
        from .review import placement_without_scale
        before = placement_without_scale(self.project)
        grid = full_view_grid(self.project.reference,self.project.moving,matrix,self.reference.spacing_um_zyx,
                              additional_matrices=(before,) if comparing else ())
        if comparing:
            self.slice_comparison.configure(grid)
        elif grid != self.overlay_grid:
            for axis,pane in enumerate(self.overlay_panes):
                pane.configure(grid["shape_zyx"],grid["spacing_um_zyx"],grid["origin_um_zyx"],preserve_world=True)
                if self._pending_overlay_position is not None:
                    pane.slider.blockSignals(True)
                    pane.set_world_depth(self._pending_overlay_position[axis])
                    pane.slider.blockSignals(False)
            self.overlay_grid = grid
            self._pending_overlay_position = None
        if not comparing and self._pending_overlay_position is not None:
            for axis,pane in enumerate(self.overlay_panes):
                pane.slider.blockSignals(True)
                pane.set_world_depth(self._pending_overlay_position[axis])
                pane.slider.blockSignals(False)
            self._pending_overlay_position=None
        indices = [pane.slider.value() for pane in self.overlay_panes]
        if not comparing:
            cross = np.array(grid["origin_um_zyx"])+np.array(indices)*grid["spacing_um_zyx"]
            self.volume_view.update_slice_planes(grid,cross)
        reference,moving = self.reference,self.moving
        opacity = self.placement_opacity if self.step == 1 else self.review_opacity
        alpha,beta = [s.value()/100 for s in opacity]
        comparison_axis = self.slice_comparison.axis
        comparison_index = self.slice_comparison.slider.value()
        placement=self.step==1
        appearance=[dict(record) for record in (self.placement_tools if placement else self.review_tools).slice_display.records]
        def render(progress):
            if comparing:
                # Reslice each stage directly from its source preview. Never
                # interpolate an already placed or previously exported scan.
                ref = sample_preview_grid_slice(reference,np.eye(4),grid,comparison_axis,comparison_index,
                    order=0 if appearance[0]['interpolation']=='nearest' else 1)
                original = sample_preview_grid_slice(moving,before,grid,comparison_axis,comparison_index,
                    order=0 if appearance[1]['interpolation']=='nearest' else 1)
                fitted = sample_preview_grid_slice(moving,matrix,grid,comparison_axis,comparison_index,
                    order=0 if appearance[1]['interpolation']=='nearest' else 1)
                from .desktop_slice_display import slice_color
                return [(slice_color(pixels,settings),pixels) for pixels,settings in
                        ((ref,appearance[0]),(original,appearance[1]),(fitted,appearance[1]))]
            images = []
            for axis,index in enumerate(indices):
                ref = sample_preview_grid_slice(reference,np.eye(4),grid,axis,index,
                    order=0 if appearance[0]['interpolation']=='nearest' else 1)
                mov = sample_preview_grid_slice(moving,matrix,grid,axis,index,
                    order=0 if appearance[1]['interpolation']=='nearest' else 1)
                from .desktop_slice_display import slice_grayscale,slice_rgb,slice_color,blend_rgb
                ref_raw,mov_raw=ref,mov
                ref=slice_grayscale(ref,appearance[0])
                mov=slice_grayscale(mov,appearance[1])
                image=np.zeros((*ref.shape,3),np.float32)
                for pixels,settings,opacity in ((ref,appearance[0],alpha),(mov,appearance[1],beta)):
                    image=blend_rgb(image,slice_rgb(pixels,settings['colormap'],settings.get('colormap_colors')),
                        opacity*settings.get('opacity',1),settings.get('blending','additive'))
                images.append((image,slice_color(ref_raw,appearance[0]),slice_color(mov_raw,appearance[1]),ref_raw,mov_raw))
            return images
        self._render_running, self._render_pending = True,False
        job = Job(render)
        self._jobs.add(job)
        def finish(images=None,error=None):
            self._jobs.discard(job)
            self._render_running = False
            if self._closing:
                return
            if token == self._render_token and self.step in (1,3):
                if error:
                    self.statusBar().showMessage(error)
                elif comparing:
                    self.slice_comparison.display([image for image,raw in images],self.project.fit_result["model"])
                    from .desktop_figures import slice_sampler
                    specs=(self.project.reference,self.project.moving,self.project.moving)
                    for canvas,settings,spec,forward,norm,(_,raw) in zip(self.slice_comparison.canvases,(appearance[0],appearance[1],appearance[1]),specs,(np.eye(4),before,matrix),(reference.intensity_limits,moving.intensity_limits,moving.intensity_limits),images):
                        canvas.smooth=settings['interpolation']=='linear'
                        canvas.display_interpolation=settings['interpolation']
                        canvas.figure_sampler=slice_sampler([spec],[forward],[settings],normalizations=[norm],visible_pixels=[raw])
                else:
                    centre = (np.array(self.project.moving.shape_zyx)-1)/2*self.project.moving.spacing_um_zyx
                    pivot = transform_points(centre[None],matrix)[0]
                    cross = np.array(grid["origin_um_zyx"])+np.array(indices)*grid["spacing_um_zyx"]
                    for pane,(image,ref,mov,ref_raw,mov_raw) in zip(self.overlay_panes,images):
                        controls=(self.placement_tools if placement else self.review_tools).slice_display
                        controls.observe(0,ref_raw); controls.observe(1,mov_raw)
                        for canvas in pane.canvases:
                            canvas.pivot, canvas.crosshair = pivot,cross
                        pane.reference_canvas.smooth=appearance[0]['interpolation']=='linear'
                        pane.scan_canvas.smooth=appearance[1]['interpolation']=='linear'
                        pane.canvas.smooth=pane.reference_canvas.smooth or pane.scan_canvas.smooth
                        pane.reference_canvas.display_interpolation=appearance[0]['interpolation']
                        pane.scan_canvas.display_interpolation=appearance[1]['interpolation']
                        pane.canvas.display_interpolation=appearance[1]['interpolation']
                        pane.display(image,ref,mov)
                        from .desktop_figures import slice_sampler
                        specs=[self.project.reference,self.project.moving]; matrices=[np.eye(4),matrix]
                        norms=[reference.intensity_limits,moving.intensity_limits]
                        pane.canvas.figure_sampler=slice_sampler(specs,matrices,appearance,normalizations=norms,opacities=[alpha,beta],visible_pixels=[ref_raw,mov_raw])
                        for i,canvas in enumerate((pane.reference_canvas,pane.scan_canvas)):
                            canvas.figure_sampler=slice_sampler([specs[i]],[matrices[i]],[appearance[i]],normalizations=[norms[i]],visible_pixels=[(ref_raw,mov_raw)[i]])
            if self._render_pending or token != self._render_token:
                self.render_timer.start(0)
        job.signals.done.connect(lambda images: finish(images=images))
        job.signals.failed.connect(lambda error: finish(error=error))
        QThreadPool.globalInstance().start(job)

    def _remember_points(self):
        self._undo.append(json.loads(json.dumps(self.project.pairs)))
        self._undo = self._undo[-100:]

    def set_landmark_tool(self,tool):
        self._landmark_tool=tool
        self.landmark_views.set_tool(tool)

    def select_landmark(self,row):
        self._selected_landmark=row if self.project is not None and 0<=row<len(self.project.pairs) else -1
        self.draw_landmarks()

    def find_landmark(self,row):
        if self.project is None or self.placed is None or self.busy: return
        if not 0<=row<len(self.project.pairs):
            self.statusBar().showMessage('Point number not found.',3000)
            return
        self._selected_landmark=row
        pair=self.project.pairs[row]
        for name,panes in self.landmark_panes.items():
            if pair.get(name) is None: continue
            for pane in panes:
                pane.slider.blockSignals(True)
                pane.set_world_depth(pair[name][pane.axis])
                pane.slider.blockSignals(False)
        self.landmark_views.sync_slices()
        self.draw_landmarks()
        self.landmark_views.center_pair(pair)
        self.set_landmark_tool('select')
        self.landmark_panes['reference' if pair.get('reference') is not None else 'moving'][0].canvas.setFocus()

    def delete_pair(self):
        row = self._selected_landmark
        if self.project is not None and self.step==2 and not self.busy and 0 <= row < len(self.project.pairs):
            self._remember_points()
            self.project.pairs.pop(row)
            self._selected_landmark=min(row,len(self.project.pairs)-1)
            self.update_landmarks()
            self.schedule_save()

    def clear_pairs(self):
        if self.project is None or self.busy or self.step!=2 or not self.project.pairs: return
        self._remember_points()
        self.project.pairs=[]; self._selected_landmark=-1
        self.update_landmarks(); self.schedule_save()

    def review_pair(self,row):
        self._review_find_pair=row
        self.go_step(2)
        if self.placed is not None:
            self._review_find_pair=None; self.find_landmark(row)

    def undo_points(self):
        if self.project is not None and self._undo:
            self.project.pairs = self._undo.pop()
            self._selected_landmark=min(self._selected_landmark,len(self.project.pairs)-1)
            self.update_landmarks()
            self.schedule_save()

    def add_point(self,name,point):
        if self.project is None or self.placed is None or self.busy or self.step!=2:
            return
        point=np.asarray(point,dtype=float)
        if name not in self.landmark_panes or point.shape!=(3,) or not np.isfinite(point).all(): return
        self._remember_points()
        # Each image has its own sequential queue. Filling the earliest missing
        # side supports Atlas 1,2,3 then Scan 1,2,3, or either interleaved order.
        row=next((i for i,pair in enumerate(self.project.pairs) if pair.get(name) is None),len(self.project.pairs))
        if row==len(self.project.pairs):
            self.project.pairs.append({"reference":None,"moving":None})
        self.project.pairs[row][name] = point.tolist()
        self._selected_landmark=row
        for pane in self.landmark_panes[name]:
            pane.slider.blockSignals(True)
            pane.set_world_depth(point[pane.axis])
            pane.slider.blockSignals(False)
        self.landmark_views.sync_slices()
        self.update_landmarks()
        self.schedule_save()

    def update_landmarks(self,*args):
        self.landmark_tools.update_state()
        self.draw_landmarks()

    def _landmark_slice_changed(self,name,axis):
        self.landmark_views.sync_slices()
        self.draw_landmarks()

    def draw_landmarks(self,*args):
        if self.project is None or self.placed is None:
            return
        from .desktop_slice_display import slice_color
        selected = self._selected_landmark
        for index,(name,volume) in enumerate((("reference",self.landmark_reference),("moving",self.placed))):
            panes = self.landmark_panes[name]
            cross = np.array([pane.world_depth for pane in panes])
            points = [(i+1,np.array(pair[name]),i == selected) for i,pair in enumerate(self.project.pairs) if pair.get(name) is not None]
            settings=self.landmark_tools.display.records[index]
            for pane in panes:
                pane.canvas.points, pane.canvas.crosshair = points,cross
                pane.canvas.smooth=settings['interpolation']=='linear'
                pixels=np.take(volume,pane.slider.value(),axis=pane.axis)
                self.landmark_tools.display.observe(index,pixels)
                pane.canvas.display_interpolation=settings['interpolation']
                pane.display(slice_color(pixels,settings))
                from .desktop_figures import slice_sampler
                spec=self.project.reference if index==0 else self.project.moving
                forward=np.eye(4) if index==0 else self.project.coarse
                norm=(self.reference if index==0 else self.moving).intensity_limits
                pane.canvas.figure_sampler=slice_sampler([spec],[forward],[settings],normalizations=[norm],visible_pixels=[pixels])

    def fit_points(self,*,advance=True):
        if not self.project or self.busy:
            return
        try:
            result = self.project.fit(self.fit_model.currentText().lower())
            self.update_landmarks()
            self.review_tools.update_report()
            if advance: self.go_step(3)
            else:
                self.schedule_save()
                self.statusBar().showMessage('Transformation applied.',2500)
        except ValueError as error:
            self._error(str(error))

    def _update_summary(self):
        if not self.project:
            return
        p = self.project
        xyz = lambda spec: " / ".join(f"{v:g}" for v in spec.spacing_um_zyx[::-1])
        self.source_summary.setText(f"Atlas: {p.reference.path.name} · XYZ {xyz(p.reference)} µm    |    "
                                    f"Scan: {p.moving.path.name} · XYZ {xyz(p.moving)} µm\n"
                                    f"Interactive preview Z/Y/X: atlas {self.reference.data.shape}, scan {self.moving.data.shape}")
        self.review_tools.update_report()

    def _display_record(self):
        self.track_workspace.capture_state()
        self.reconstruction.capture_state()
        self.electrophysiology.capture_state()
        return {"workflow_version":3,"step":self.step,"placement_opacity":[s.value() for s in self.placement_opacity],
                "placement_panel_tab":self.placement_tools.currentIndex(),
                "placement_slice_appearance":self.placement_tools.slice_display.records,
                "placement_slice_views":[[canvas.view_record() for canvas in pane.canvases] for pane in self.overlay_panes],
                "review_tab":self._review_tab_index, "slice_comparison":self.slice_comparison.state_record(),
                "review_opacity":[s.value() for s in self.review_opacity],
                "overlay_slices":[p.slider.value() for p in self.overlay_panes],
                "placement_slice_modes":[p.mode_selector.currentData() for p in self.overlay_panes],
                "landmark_slices":{k:[p.slider.value() for p in v] for k,v in self.landmark_panes.items()},
                "overlay_position_um_zyx":self._pending_overlay_position or [p.world_depth for p in self.overlay_panes],
                "landmark_positions_um_zyx":self._pending_landmark_positions or
                    {k:[p.world_depth for p in v] for k,v in self.landmark_panes.items()},
                "landmark_panel_tab":self.landmark_tools.currentIndex(),
                "landmark_tool":self._landmark_tool,"landmark_selected":self._selected_landmark,
                "landmark_model":self.fit_model.currentText(),"landmark_link_slices":self.link_slices.isChecked(),
                "landmark_appearance":self.landmark_tools.display.records,
                "landmark_adjust_range":self.landmark_tools.adjust_range.value(),
                "landmark_slice_views":{k:[p.canvas.view_record() for p in v] for k,v in self.landmark_panes.items()},
                "review_panel_tab":self.review_tools.currentIndex(),
                "workspace_panels":dict(self._workspace_panels),
                "review_slice_appearance":self.review_tools.slice_display.records,
                "camera_3d":self.volume_view.camera_record(),
                "rendering_3d":self.volume_view.render_settings_record(),
                "volume_layers":self.volume_view.layers.record(),
                "slice_planes_3d":self.volume_view.slice_settings_record(),
                "volume_axes_3d":self.volume_view.show_axes.isChecked(),
                "volume_flip_xyz":self.volume_view.flip_record(),
                "placement_controls":{name:[self.pose_controls[name,i][1].value() for i in range(3)]
                                      for name in ("translation","rotation","scale")},
                "placement_baseline_zyx":self._baseline.tolist(),
                "axis_order":self.axis_order.currentText(),"flips_xyz":[c.isChecked() for c in self.flips]}

    def _restore_display(self):
        d = self.project.display
        self._workspace_panels=dict(d.get('workspace_panels',{}))
        self.volume_view.show_axes.setChecked(bool(d.get('volume_axes_3d',True)))
        self.volume_view.restore_flip(d.get('volume_flip_xyz'))
        self.placement_tools.setCurrentIndex(max(0,min(2,d.get('placement_panel_tab',0))))
        self.placement_tools.slice_display.initialize(((0,1),(0,1)),d.get('placement_slice_appearance'))
        for pane,records in zip(self.overlay_panes,d.get('placement_slice_views',[])):
            for canvas,record in zip(pane.canvases,records): canvas.restore_view(record)
        self._review_tab_index = 1 if d.get("review_tab") == 1 else 0
        self.slice_comparison.restore_state(d.get("slice_comparison",{}))
        self.volume_view.restore_slice_settings(d.get("slice_planes_3d",{}))
        for pane,mode in zip(self.overlay_panes,d.get("placement_slice_modes",["overlay"]*3)):
            pane.mode_selector.blockSignals(True)
            pane.mode_selector.setCurrentIndex(max(0,pane.mode_selector.findData(mode)))
            pane.mode_selector.blockSignals(False)
        self._pending_camera = d.get("camera_3d")
        if "rendering_3d" in d:
            records=d["rendering_3d"]
            from .desktop_rendering import default_render_settings
            records=(records+default_render_settings(len(self.volume_view.layer_titles))[len(records):])[:len(self.volume_view.layer_titles)]
            self.volume_view.restore_render_settings(records)
        if 'volume_layers' in d: self.volume_view.layers.restore(d['volume_layers'])
        self._pending_overlay_position = d.get("overlay_position_um_zyx")
        self._pending_landmark_positions = d.get("landmark_positions_um_zyx")
        self.landmark_tools.setCurrentIndex(1 if d.get('landmark_panel_tab')==1 else 0)
        self.set_landmark_tool(d.get('landmark_tool') if d.get('landmark_tool') in ('pan','add','select') else 'add')
        self._selected_landmark=max(-1,min(int(d.get('landmark_selected',-1)),len(self.project.pairs)-1))
        self.link_slices.setChecked(bool(d.get('landmark_link_slices',True)))
        appearance=d.get('landmark_appearance')
        if appearance is None:
            appearance=[{'gamma':d.get('contrast',{}).get(name,1)} for name in ('reference','moving')]
        self.landmark_tools.display.initialize(((0,1),(0,1)),appearance)
        self.landmark_tools.adjust_range.blockSignals(True)
        self.landmark_tools.adjust_range.setValue(int(d.get('landmark_adjust_range',15)))
        self.landmark_tools.adjust_range.blockSignals(False)
        self.landmark_views.set_adjust_range(self.landmark_tools.adjust_range.value())
        self.landmark_views.restore_views(d.get('landmark_slice_views',{}))
        if "placement_controls" in d:
            self._updating = True
            base = np.eye(4)
            base[:3,:3] = d.get("placement_baseline_zyx",np.eye(3))
            self._baseline = validate_affine(base)[:3,:3].copy()
            for name,values in d["placement_controls"].items():
                for axis,value in enumerate(values):
                    self._set_control(*self.pose_controls[name,axis],value)
            self.axis_order.setCurrentText(d.get("axis_order","XYZ"))
            for check,value in zip(self.flips,d.get("flips_xyz",[])):
                check.setChecked(value)
            self._updating = False
        for key in ("placement_opacity","review_opacity"):
            for slider,value in zip(getattr(self,key),d.get(key,[])):
                slider.setValue(int(value))
        for pane,value in zip(self.overlay_panes,d.get("overlay_slices",[])):
            pane.slider.setValue(int(value))
        for key,values in d.get("landmark_slices",{}).items():
            for pane,value in zip(self.landmark_panes.get(key,[]),values):
                pane.slider.setValue(int(value))
        self.review_tools.setCurrentIndex(max(0,min(2,d.get('review_panel_tab',0))))
        appearance=d.get('review_slice_appearance')
        if appearance is None:
            appearance=[dict(gamma=v) for v in d.get('slice_comparison',{}).get('gamma',[1,1])]
        self.review_tools.slice_display.initialize(((0,1),(0,1)),appearance)
        if self.project.fit_result:
            r = self.project.fit_result
            self.fit_model.setCurrentText(r["model"].title())
            self.review_tools.update_report()
        model=d.get('landmark_model',self.project.fit_result['model'].title() if self.project.fit_result else 'Affine')
        self.fit_model.setCurrentText(model if model in ('Affine','Similarity','Rigid') else 'Affine')

    def save_project(self):
        if self.project is None: return
        if self.busy:
            self._save_pending=True
            self.statusBar().showMessage('Save queued until image preparation finishes.')
            return
        try:
            self._save_pending=False
            self.save_timer.stop()
            self.project.display = self._display_record()
            path = self.project.save(self.project_path or Path(self.project.output_dir)/"project.json")
            self.project_path=path
            self._remember_project()
            from .neuropixels_model import write_reports
            if any(i.get('probe',{}).get('pose') or s.get('measurements')
                   for s in self.project.track_analysis.get('sessions',[]) for i in s.get('insertions',[])):
                write_reports(self.project,self.project.output_dir,self.track_workspace.labels)
            self.reconstruction.write_report(self.project.output_dir)
            self.statusBar().showMessage('Project saved',2500)
            return path
        except (ValueError,OSError) as error:
            self._error(str(error))

    def schedule_save(self):
        if self.project is not None and not self._closing:
            self.save_timer.start(700)

    def save_project_as(self):
        if self.project is None or self.busy or self.project_page.mode=='setup': return
        path=QFileDialog.getExistingDirectory(self,'Save project in folder',str(self.project.output_dir))
        if path:
            self.output_path.setText(path)
            self._output_folder_edited()
            self.save_project()

    def open_project(self):
        if self.busy:
            return
        path = QFileDialog.getExistingDirectory(self,"Open project folder",self.output_path.text() or str(Path.home()))
        if not path:
            return
        self.load_project_path(path)

    def _remember_project(self):
        if self.project_path is not None:
            self.recent_projects.remember(self.project_path)
            self.project_page.set_recent_projects(self.recent_projects.entries())

    def open_recent_project(self,path):
        if self.busy: return
        if not Path(path).is_file():
            self.recent_projects.forget(path)
            self.project_page.set_recent_projects(self.recent_projects.entries())
            self.statusBar().showMessage('This project folder is no longer available.',4000)
            return
        self.load_project_path(path)

    def load_project_path(self,path):
        """Resume a project from either the dialog or the standalone CLI."""
        if self.busy:
            return
        try:
            # Save before reading: opening the current folder must not load
            # a stale on-disk snapshot over edits still present in memory.
            if self.project is not None and self.save_project() is None: return
            project = AlignmentProject.load(path)
            destination=project_file_path(path)
            if destination.parent.name=='.probemap_history':
                destination=project.output_dir/'project.json'
            self._sync_project_inputs(project)
            self._suggest_auxiliary(project.reference.path,project.moving.path)
            if project.track is None and self.track_input.info and all(spin.value()>0 for spin in self.track_input.spacing):
                try: self._auxiliary_inputs(project)
                except ValueError: self.track_input.clear()
            if project.labels_dir is None and self.labels_path.text():
                project.labels_dir=Path(self.labels_path.text()).resolve()
            index = self.quality.findData(project.preview_dimension)
            self.quality.setCurrentIndex(max(index,0))
            target = int(project.display.get("step",1))
            if target >= 3 and not project.fit_is_current:
                target = 2
            if target>=4 and project.track is None: target=3
            self._pending_track_step=None
            self._load_project_volumes(project,max(0,min(target if project.display.get('workflow_version',0)>=3 else min(target,6),7)),project_path=destination,save_current=False)
        except (ValueError,OSError,KeyError) as error:
            self._error(str(error))

    def open_export(self):
        if self.project is None or self.busy or self.project_page.mode=='setup': return
        from .desktop_export import ExportDialog
        self.export_dialog=ExportDialog(self.project,self)
        self.export_dialog.exportRequested.connect(self.export)
        self.export_dialog.open()

    def export(self,options):
        if self.project is None or self.busy: return
        from copy import deepcopy
        from .exports import export_selection
        if self.save_project() is None: return
        snapshot=deepcopy(self.project)
        self._cancelled=False
        self.cancel_button.setEnabled(True)
        self.cancel_button.setText('Cancel export')
        def ready(path):
            self.last_export=path
            record=dict(options,path=str(path))
            self.project.exports=[old for old in self.project.exports if Path(old['path']).resolve()!=path]+[record]
            self.save_project()
            self.statusBar().showMessage(f'Exported · {path.name}',8000)
        self._start_job(lambda progress:export_selection(snapshot,options,progress,lambda:self._cancelled),
                        ready,'Exporting…')
        self.cancel_button.show()

    def cancel_export(self):
        self._cancelled = True
        self.cancel_button.setEnabled(False)
        self.statusBar().showMessage("Stopping after the current output block…")


    def undo_current_step(self):
        if self.busy: return
        focus=self.focusWidget()
        if isinstance(focus,QLineEdit) and focus.isUndoAvailable():
            focus.undo()
        elif self.step in (4,5): self.track_workspace.undo()
        elif self.step==2: self.undo_points()

    def closeEvent(self,event):
        if self.busy:
            self.statusBar().showMessage("A background task is running. Wait for it, or cancel the export before closing.")
            event.ignore()
            return
        if self.project:
            try:
                self.project.display = self._display_record()
                self.project.save(self.project_path or Path(self.project.output_dir)/"project.json")
            except OSError as error:
                self.statusBar().showMessage(str(error))
                event.ignore()
                return
        self._render_token += 1
        self.settings_dialog.hide()
        self._closing = True
        self.render_timer.stop()
        self.save_timer.stop()
        self.track_workspace.stop()
        self.reconstruction.stop()
        self.electrophysiology.stop()
        # Finish any in-flight CPU slice request before Qt/GL resources and
        # Python callbacks are destroyed on application shutdown.
        QThreadPool.globalInstance().waitForDone(2000)
        self.volume_view.close_canvas()
        self.track_workspace.volume_view.close_canvas()
        self.reconstruction.volume_view.close_canvas()
        event.accept()


# Compatibility for existing analysis scripts and saved development sessions.
AlignWindow = ProbeMapWindow


def main(argv=None):
    parser = argparse.ArgumentParser(prog='probemap',description="ProbeMap · standalone TIFF landmark registration")
    parser.add_argument("--config",type=Path,help="Optional ProbeMap registration config to prefill image paths")
    parser.add_argument("--project",type=Path,help="Open a results folder or a saved ProbeMap project JSON")
    parser.add_argument("--version",action="version",version=f'{APP_NAME} {__version__}')
    args = parser.parse_args(argv)
    os.environ.setdefault("QT_API","pyqt6")
    os.environ.setdefault("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS","4")
    from .macos_qt import initialize_qt
    app = initialize_qt()
    app.setApplicationName(APP_NAME)
    app.setApplicationDisplayName(APP_NAME)
    app.setApplicationVersion(__version__)
    app.setWindowIcon(QIcon(str(RESOURCE_DIR/'icon.svg')))
    app.setOrganizationName("ProbeMap")
    from qtpy.QtCore import QSettings
    window = ProbeMapWindow(args.config,recent_projects=RecentProjects(QSettings('ProbeMap','ProbeMap')))
    def report_exception(kind,error,tb):
        traceback.print_exception(kind,error,tb)
        window._error(str(error))
    sys.excepthook = report_exception
    window.show()
    if args.project:
        QTimer.singleShot(0,lambda:window.load_project_path(args.project))
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())

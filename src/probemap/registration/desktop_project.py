"""Project entry, compact acquisition inputs and a read-only project overview."""
from pathlib import Path

from qtpy.QtCore import Qt, Signal, QSize
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox,
    QDoubleSpinBox, QFileDialog, QFormLayout, QFrame, QGridLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QScrollArea, QSizePolicy, QStackedWidget,
    QToolButton, QVBoxLayout, QWidget)

from ..branding import RESOURCE_DIR
from .desktop_model import ImageSpec, inspect_tiff
from .desktop_slice_tools import tool_icon


def text(value, name='muted'):
    widget=QLabel(value)
    widget.setObjectName(name)
    widget.setWordWrap(True)
    return widget


def icon_button(icon, title, callback):
    widget=QToolButton()
    widget.setIcon(tool_icon(icon))
    widget.setIconSize(QSize(20,20))
    widget.setToolTip(title)
    widget.setAccessibleName(title)
    widget.setFixedSize(32,32)
    widget.clicked.connect(lambda checked=False:callback())
    return widget


class PathLabel(QLabel):
    """Elide long acquisition paths without making the window wider."""
    def __init__(self):
        super().__init__()
        self.value=''
        self.setObjectName('muted')
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Policy.Ignored,QSizePolicy.Policy.Preferred)

    def set_path(self,value):
        self.value=str(value) if value else 'Not added'
        self.setToolTip(str(value) if value else '')
        self._elide()

    def _elide(self):
        self.setText(self.fontMetrics().elidedText(self.value,Qt.TextElideMode.ElideMiddle,max(1,self.width())))

    def resizeEvent(self,event):
        super().resizeEvent(event)
        self._elide()


class VoxelSizeSpin(QDoubleSpinBox):
    def textFromValue(self,value):
        return f'{value:.5f}'.rstrip('0').rstrip('.')


def project_name(folder):
    folder=Path(folder)
    return folder.parent.name if folder.name.lower() in ('outputs','results') else folder.name


class RecentProjectButton(QPushButton):
    def __init__(self,path):
        super().__init__()
        self.project_path=Path(path)
        folder=self.project_path.parent
        title=project_name(folder)
        self.setObjectName('recentProject')
        self.setAccessibleName('Open '+title)
        self.setAccessibleDescription(str(folder))
        self.setToolTip(str(folder))
        self.setMinimumHeight(54)
        row=QHBoxLayout(self)
        row.setContentsMargins(12,8,12,8)
        icon=QLabel()
        icon.setPixmap(tool_icon('folder').pixmap(22,22))
        row.addWidget(icon)
        labels=QVBoxLayout()
        labels.setSpacing(2)
        self.title_label=PathLabel()
        self.title_label.setObjectName('fileTitle')
        self.title_label.set_path(title)
        self.path_label=PathLabel()
        self.path_label.set_path(folder)
        labels.addWidget(self.title_label)
        labels.addWidget(self.path_label)
        row.addLayout(labels,1)
        for label in (icon,self.title_label,self.path_label):
            label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)


class ImageInput(QFrame):
    changed=Signal()

    def __init__(self,title,optional=False):
        super().__init__()
        self._title=title
        self.info=None
        self.initial_spacing=None
        self._setting=False
        self.setObjectName('fileCard')
        layout=QVBoxLayout(self)
        layout.setContentsMargins(16,12,16,12)
        layout.setSpacing(9)
        row=QHBoxLayout()
        row.addWidget(text(title,'fileTitle'))
        row.addStretch()
        if optional: row.addWidget(text('Optional'))
        layout.addLayout(row)
        row=QHBoxLayout()
        self.path=QLineEdit()
        self.path.setPlaceholderText('Choose TIFF…')
        self.path.setAccessibleName(title+' file')
        self.path.editingFinished.connect(self.read_metadata)
        row.addWidget(self.path,1)
        self.browse_button=icon_button('folder','Choose '+title,self.browse)
        row.addWidget(self.browse_button)
        layout.addLayout(row)
        row=QHBoxLayout()
        row.addWidget(text('Voxel size · µm'))
        self.spacing=[]
        for axis in 'XYZ':
            row.addSpacing(10)
            row.addWidget(QLabel(axis))
            spin=VoxelSizeSpin()
            spin.setRange(0,1e6)
            spin.setDecimals(5)
            spin.setSpecialValueText('Unknown')
            spin.setKeyboardTracking(False)
            spin.setAccessibleName(f'{title} voxel size {axis} in micrometres')
            spin.valueChanged.connect(self._changed)
            row.addWidget(spin,1)
            self.spacing.append(spin)
        layout.addLayout(row)
        self.metadata=text('')
        self.metadata.hide()
        layout.addWidget(self.metadata)

    def title(self): return self._title

    def _changed(self,*args):
        if not self._setting:
            if self.info:
                missing=any(spin.value()<=0 for spin in self.spacing)
                self.metadata.setText('Enter the missing voxel sizes.' if missing else '')
                self.metadata.setVisible(missing)
            self.changed.emit()

    def browse(self):
        path,_=QFileDialog.getOpenFileName(self,'Choose '+self._title,
            self.path.text() or str(Path.home()),'TIFF volumes (*.tif *.tiff)')
        if path:
            self.path.setText(path)
            self.read_metadata()

    def read_metadata(self):
        value=self.path.text().strip()
        if not value:
            self.info=None
            self.metadata.hide()
            self.changed.emit()
            return
        path=Path(value).expanduser().resolve()
        # Finishing an edit on the same file must preserve a manual calibration.
        if self.info and path==Path(self.info['path']): return
        self._setting=True
        try:
            self.info=inspect_tiff(path)
            self.initial_spacing=None
            self.path.setText(str(path))
            for spin,value in zip(self.spacing,self.info['spacing_um_zyx'][::-1]):
                spin.setValue(value if value is not None else 0)
            missing=any(value is None for value in self.info['spacing_um_zyx'])
            self.metadata.setText('Enter the missing voxel sizes.' if missing else '')
            self.metadata.setVisible(missing)
            self.path.setToolTip(str(path))
        except (OSError,ValueError,KeyError) as error:
            self.info=None
            self.metadata.setText(str(error))
            self.metadata.show()
        finally:
            self._setting=False
        self.changed.emit()

    def spec(self):
        self.read_metadata()
        if not self.info: raise ValueError(f'Choose a valid TIFF for {self._title}.')
        spacing=tuple(spin.value() for spin in self.spacing)[::-1]
        if min(spacing)<=0: raise ValueError(f'Enter the voxel size for each axis of {self._title}.')
        return ImageSpec(Path(self.info['path']),tuple(self.info['shape_zyx']),spacing,
                         tuple(self.info['origin_um_zyx']),self.initial_spacing)

    def set_spec(self,spec):
        self._setting=True
        try:
            self.info=inspect_tiff(spec.path)
            self.initial_spacing=spec.initial_spacing_um_zyx
            self.path.setText(str(spec.path))
            self.path.setToolTip(str(spec.path))
            for spin,value in zip(self.spacing,spec.spacing_um_zyx[::-1]): spin.setValue(value)
            self.metadata.hide()
        finally:
            self._setting=False

    def clear(self):
        self._setting=True
        self.path.clear()
        self.info=None
        self.initial_spacing=None
        for spin in self.spacing: spin.setValue(0)
        self.metadata.hide()
        self._setting=False


class ProjectPage(QStackedWidget):
    openRequested=Signal()
    recentRequested=Signal(str)
    newRequested=Signal()
    loadRequested=Signal(bool)
    cancelRequested=Signal()
    folderRequested=Signal()
    labelsRequested=Signal()
    electrodesRequested=Signal()
    calibrationRequested=Signal()

    def __init__(self):
        super().__init__()
        self.mode='welcome'
        self._build_welcome()
        self._build_setup()
        self._build_summary()

    def _button(self,title,signal,primary=False,icon=None):
        widget=QPushButton(title)
        if primary: widget.setObjectName('primary')
        if icon: widget.setIcon(tool_icon(icon))
        widget.clicked.connect(lambda checked=False:signal.emit())
        return widget

    def _build_welcome(self):
        page=QWidget()
        layout=QVBoxLayout(page)
        layout.setContentsMargins(32,28,32,28)
        layout.addWidget(text('Project','heading'))
        layout.addStretch(1)
        content=QWidget()
        content.setMaximumWidth(620)
        center=QVBoxLayout(content)
        center.setContentsMargins(0,0,0,0)
        mark=QLabel()
        mark.setPixmap(QIcon(str(RESOURCE_DIR/'logo.svg')).pixmap(80,80))
        mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        center.addWidget(mark)
        center.addSpacing(24)
        row=QHBoxLayout()
        row.addStretch()
        self.open_button=self._button('Open project',self.openRequested,icon='folder')
        self.new_button=self._button('New project',self.newRequested,True,icon='new')
        for widget in (self.open_button,self.new_button):
            widget.setMinimumSize(190,52)
            row.addWidget(widget)
        row.addStretch()
        center.addLayout(row)
        self.recent_section=QWidget()
        recent=QVBoxLayout(self.recent_section)
        recent.setContentsMargins(0,18,0,0)
        recent.addWidget(text('Recent projects'))
        self.recent_rows=QVBoxLayout()
        self.recent_rows.setSpacing(6)
        recent.addLayout(self.recent_rows)
        self.recent_buttons=[]
        self._recent_paths=[]
        self.recent_section.hide()
        center.addWidget(self.recent_section)
        content_row=QHBoxLayout()
        content_row.addStretch()
        content_row.addWidget(content,1)
        content_row.addStretch()
        layout.addLayout(content_row)
        layout.addStretch(1)
        self.addWidget(page)

    def set_recent_projects(self,paths):
        paths=list(paths)
        if paths==self._recent_paths: return
        self._recent_paths=paths
        while self.recent_rows.count():
            widget=self.recent_rows.takeAt(0).widget()
            widget.hide()
            widget.deleteLater()
        self.recent_buttons=[]
        for path in paths:
            button=RecentProjectButton(path)
            button.clicked.connect(lambda checked=False,path=str(path):self.recentRequested.emit(path))
            self.recent_rows.addWidget(button)
            self.recent_buttons.append(button)
        self.recent_section.setVisible(bool(paths))
        if hasattr(self,'summary_recent_rows'):
            while self.summary_recent_rows.count():
                item=self.summary_recent_rows.takeAt(0).widget(); item.hide(); item.deleteLater()
            self.summary_recent_buttons=[]
            for path in paths:
                button=RecentProjectButton(path)
                button.clicked.connect(lambda checked=False,path=str(path):self.recentRequested.emit(path))
                self.summary_recent_rows.addWidget(button); self.summary_recent_buttons.append(button)
            self.summary_recent_section.setVisible(bool(paths))

    def _build_setup(self):
        page=QWidget()
        layout=QVBoxLayout(page)
        layout.setContentsMargins(28,24,28,24)
        header=QHBoxLayout()
        header.addWidget(text('New project','heading'),1)
        header.addWidget(self._button('Cancel',self.cancelRequested))
        layout.addLayout(header)
        self.output_path=QLineEdit()
        self.output_path.setReadOnly(True)
        self.output_path.setAccessibleName('Project folder')
        self.output_path.setObjectName('folderPath')
        row=QHBoxLayout()
        row.addWidget(text('Project folder'))
        row.addWidget(self.output_path,1)
        row.addWidget(icon_button('folder','Change project folder',self.folderRequested.emit))
        layout.addLayout(row)
        layout.addSpacing(10)
        self.atlas_input=ImageInput('Atlas')
        self.scan_input=ImageInput('Brain scan')
        self.track_input=ImageInput('Electrodes scan',optional=True)
        for widget in (self.atlas_input,self.scan_input,self.track_input):
            layout.addWidget(widget)
            widget.changed.connect(self.update_ready)
        labels=QFrame()
        labels.setObjectName('fileCard')
        row=QHBoxLayout(labels)
        row.setContentsMargins(16,12,16,12)
        row.addWidget(text('Labels','fileTitle'))
        self.labels_path=QLineEdit()
        self.labels_path.setPlaceholderText('Choose labels folder…')
        self.labels_path.setAccessibleName('Atlas labels folder')
        row.addWidget(self.labels_path,1)
        row.addWidget(icon_button('folder','Choose labels folder',self.labelsRequested.emit))
        layout.insertWidget(4,labels)
        self.already_placed=QCheckBox('Scans are already placed in atlas axes')
        layout.addWidget(self.already_placed)
        layout.addStretch()
        row=QHBoxLayout()
        row.addStretch()
        self.continue_button=QPushButton('Continue to placement')
        self.continue_button.setObjectName('primary')
        self.continue_button.setMinimumWidth(200)
        self.continue_button.clicked.connect(lambda:self.loadRequested.emit(self.already_placed.isChecked()))
        self.already_placed.toggled.connect(lambda checked:self.continue_button.setText(
            'Continue to registration' if checked else 'Continue to placement'))
        row.addWidget(self.continue_button)
        layout.addLayout(row)
        scroll=QScrollArea()
        scroll.setWidgetResizable(True)
        page.setMaximumWidth(1040)
        scroll.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(page)
        self.setup_scroll=scroll
        self.addWidget(scroll)
        self.update_ready()

    def _build_summary(self):
        page=QWidget()
        layout=QVBoxLayout(page)
        layout.setContentsMargins(32,28,32,28)
        header=QHBoxLayout()
        header.addWidget(text('Project','heading'),1)
        header.addWidget(self._button('Open project',self.openRequested,icon='folder'))
        header.addWidget(self._button('New project',self.newRequested,icon='new'))
        layout.addLayout(header)
        layout.addSpacing(28)
        self.project_name=text('','projectName')
        layout.addWidget(self.project_name)
        self.summary_folder=PathLabel()
        layout.addWidget(self.summary_folder)
        layout.addSpacing(24)
        files=QFrame()
        files.setObjectName('fileCard')
        grid=QGridLayout(files)
        grid.setContentsMargins(20,8,12,8)
        grid.setVerticalSpacing(0)
        self.summary_paths={}
        self.summary_details={}
        self.summary_actions={}
        for row,(key,title) in enumerate((('reference','Atlas'),('labels','Labels'),('moving','Brain scan'),('track','Electrodes scan'))):
            grid.addWidget(text(title,'fileTitle'),row,0)
            container=QWidget()
            content=QVBoxLayout(container)
            content.setContentsMargins(14,14,14,14)
            content.setSpacing(5)
            path=PathLabel()
            detail=text('')
            content.addWidget(path)
            content.addWidget(detail)
            self.summary_paths[key],self.summary_details[key]=path,detail
            grid.addWidget(container,row,1)
            if key in ('labels','track'):
                signal=self.labelsRequested if key=='labels' else self.electrodesRequested
                edit=icon_button('folder','Change '+title,signal.emit)
                grid.addWidget(edit,row,2)
                self.summary_actions[key]=edit
        grid.setColumnStretch(1,1)
        layout.addWidget(files)
        layout.addSpacing(14)
        self.progress_summary=text('')
        footer=QHBoxLayout()
        footer.addWidget(self.progress_summary,1)
        self.calibration_button=self._button('Calibration…',self.calibrationRequested,icon='measure')
        footer.addWidget(self.calibration_button)
        layout.addLayout(footer)
        self.summary_recent_section=QWidget(); recents=QVBoxLayout(self.summary_recent_section)
        recents.setContentsMargins(0,24,0,0); recents.addWidget(text('Recent projects'))
        self.summary_recent_rows=QVBoxLayout(); self.summary_recent_rows.setSpacing(6)
        recents.addLayout(self.summary_recent_rows); self.summary_recent_buttons=[]
        self.summary_recent_section.hide(); layout.addWidget(self.summary_recent_section)
        layout.addStretch()
        scroll=QScrollArea(); scroll.setWidgetResizable(True); scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(page); self.addWidget(scroll)

    def update_ready(self):
        valid=all(widget.info and all(spin.value()>0 for spin in widget.spacing)
                  for widget in (self.atlas_input,self.scan_input))
        if self.track_input.path.text().strip():
            valid=valid and self.track_input.info and all(spin.value()>0 for spin in self.track_input.spacing)
        self.continue_button.setEnabled(bool(valid and self.output_path.text().strip()))

    def show_setup(self,folder):
        self.mode='setup'
        for widget in (self.atlas_input,self.scan_input,self.track_input): widget.clear()
        self.labels_path.clear()
        self.output_path.setText(str(folder))
        self.already_placed.setChecked(False)
        self.setCurrentIndex(1)
        self.update_ready()

    def show_summary(self,project):
        self.mode='summary'
        folder=Path(project.output_dir)
        self.project_name.setText(project_name(folder))
        self.summary_folder.set_path(folder)
        self.output_path.setText(str(folder))
        self.labels_path.setText(str(project.labels_dir) if project.labels_dir else '')
        for key,spec in (('reference',project.reference),('moving',project.moving),('track',project.track)):
            self.summary_paths[key].set_path(spec.path if spec else None)
            detail='XYZ · '+' / '.join(f'{v:g}' for v in spec.spacing_um_zyx[::-1])+' µm' if spec else ''
            self.summary_details[key].setText(detail)
            self.summary_details[key].setVisible(bool(detail))
        self.summary_paths['labels'].set_path(project.labels_dir)
        self.summary_details['labels'].setText('XYZ · '+' / '.join(f'{v:g}' for v in project.label_spacing[::-1])+' µm' if project.labels_dir else '')
        self.summary_details['labels'].setVisible(project.labels_dir is not None)
        sessions=project.track_analysis.get('sessions',[])
        active=next((s for s in sessions if s['id']==project.track_analysis.get('active_session')), {})
        items=active.get('insertions',[])
        parts=[f'{len(project.pairs)} landmark pairs',f'{len(items)} insertions',f"{sum(len(i.get('tracks',[])) for i in items)} tracks"]
        self.progress_summary.setText(' · '.join(parts))
        self.setCurrentIndex(2)

    def show_welcome(self):
        self.mode='welcome'
        self.setCurrentIndex(0)


class CalibrationDialog(QDialog):
    def __init__(self,project,parent=None):
        super().__init__(parent)
        self.setWindowTitle('Calibration')
        self.setMinimumWidth(510)
        self.original={'reference':tuple(project.reference.spacing_um_zyx),
                       'labels':tuple(project.label_spacing),
                       'moving':tuple(project.moving.spacing_um_zyx),
                       'track':tuple(project.track.spacing_um_zyx if project.track else project.moving.spacing_um_zyx)}
        self.available={'reference':True,'moving':True,'labels':project.labels_dir is not None,'track':project.track is not None}
        self.fields={}
        layout=QVBoxLayout(self)
        layout.setContentsMargins(24,22,24,22)
        layout.addWidget(text('Voxel size · µm','heading'))
        layout.addWidget(text('Keeps annotations on their scan voxels and recalculates registration, measurements and existing exports.'))
        grid=QGridLayout()
        for col,axis in enumerate('XYZ',1): grid.addWidget(QLabel(axis),0,col)
        for row,(key,title) in enumerate((('reference','Atlas'),('labels','Labels'),('moving','Brain scan'),('track','Electrodes scan')),1):
            grid.addWidget(text(title,'fileTitle'),row,0)
            self.fields[key]=[]
            for col,value in enumerate(self.original[key][::-1],1):
                spin=VoxelSizeSpin()
                spin.setRange(0,1e6)
                spin.setDecimals(5)
                spin.setKeyboardTracking(False)
                spin.setValue(value)
                spin.setEnabled(self.available[key])
                if not self.available[key]: spin.setToolTip('Add this volume on Project first.')
                spin.setAccessibleName(f'{title} voxel size {"XYZ"[col-1]}')
                spin.valueChanged.connect(self._validate)
                self.fields[key].append(spin)
                grid.addWidget(spin,row,col)
        # Metadata can contain more decimals than the editable precision.
        # Opening the dialog alone must not enable Apply through rounding.
        self.original=self.values()
        layout.addLayout(grid)
        layout.addSpacing(8)
        buttons=QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel|QDialogButtonBox.StandardButton.Apply)
        self.apply_button=buttons.button(QDialogButtonBox.StandardButton.Apply)
        self.apply_button.setObjectName('primary')
        self.apply_button.clicked.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._validate()

    def values(self):
        return {key:tuple(spin.value() for spin in fields)[::-1] for key,fields in self.fields.items()}

    def _validate(self):
        values=self.values()
        valid=all(min(value)>0 for value in values.values())
        self.apply_button.setEnabled(valid and values!=self.original)


class PreviewSettings(QDialog):
    def __init__(self,parent=None):
        super().__init__(parent)
        self.setWindowTitle('ProbeMap Settings')
        self.setMinimumWidth(400)
        self.setModal(False)
        layout=QVBoxLayout(self)
        layout.setContentsMargins(24,20,24,20)
        layout.addWidget(text('Display','heading'))
        form=QFormLayout()
        self.quality=QComboBox()
        for name,size in (('Fast',192),('Balanced',256),('Detailed',384),('Fine',512)):
            self.quality.addItem(name,size)
        self.quality.setCurrentIndex(1)
        form.addRow('Preview quality',self.quality)
        from qtpy.QtCore import QSettings
        self.figure_format=QComboBox()
        self.figure_format.addItem('PNG · image','.png')
        self.figure_format.addItem('SVG · editable regions and electrodes','.svg')
        preferences=QSettings('ProbeMap','ProbeMap')
        self.figure_format.setCurrentIndex(max(0,self.figure_format.findData(preferences.value('figure_format','.png'))))
        self.figure_format.currentIndexChanged.connect(lambda _:preferences.setValue('figure_format',self.figure_format.currentData()))
        form.addRow('Figure format',self.figure_format)
        layout.addLayout(form)
        layout.addWidget(text('Preview quality changes the display resolution.\nSVG keeps regions and electrodes editable; scan images stay raster.\nElectrophysiology figures are saved as PDF.'))
        self.apply_button=QPushButton('Apply')
        self.apply_button.setObjectName('primary')
        buttons=QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.hide)
        buttons.addButton(self.apply_button,QDialogButtonBox.ButtonRole.ApplyRole)
        layout.addSpacing(10)
        layout.addWidget(buttons)

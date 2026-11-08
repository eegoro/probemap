"""The global export sheet, separate from registration inspection."""
from pathlib import Path
from importlib.util import find_spec

from qtpy.QtCore import Signal
from qtpy.QtWidgets import (QCheckBox,QDialog,QDialogButtonBox,QFileDialog,QFormLayout,
                           QHBoxLayout,QLabel,QVBoxLayout,QWidget)

from .desktop_placement import EditedSpin
from .desktop_widgets import PopupComboBox


class ExportDialog(QDialog):
    exportRequested=Signal(dict)

    def __init__(self,project,parent=None):
        super().__init__(parent)
        self.project=project
        self.setWindowTitle('Export')
        self.setMinimumWidth(450)
        layout=QVBoxLayout(self)
        form=QFormLayout(); layout.addLayout(form)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.kind=PopupComboBox()
        for title,value in (('Brain scan','brain'),('Electrodes scan','electrodes'),
                            ('Transforms','transforms'),('Registration report','report'),('Analysis reports','analysis'),('Reconstruction report','reconstruction')):
            self.kind.addItem(title,value)
        if project.track is None: self.kind.model().item(1).setEnabled(False)
        form.addRow('Export',self.kind)
        self.stage=PopupComboBox()
        self.stage.addItem('After placement','placement')
        self.stage.addItem('After registration','registration')
        self.stage.setCurrentIndex(1 if project.fit_is_current else 0)
        if not project.fit_is_current: self.stage.model().item(1).setEnabled(False)
        form.addRow('Stage',self.stage)
        self.format=PopupComboBox(); form.addRow('Format',self.format)
        self.extent=PopupComboBox()
        self.extent.addItem('Full scan','full'); self.extent.addItem('Atlas bounds','atlas')
        self.extent.setToolTip('Full scan retains all transformed source voxels. Atlas bounds may crop the scan.')
        form.addRow('Bounds',self.extent)
        self.voxels=QWidget(); row=QHBoxLayout(self.voxels); row.setContentsMargins(0,0,0,0)
        self.spacing=[]
        for axis in 'XYZ':
            row.addWidget(QLabel(axis))
            spin=EditedSpin(); spin.setRange(.001,1e6); spin.setDecimals(5); spin.setKeyboardTracking(False)
            row.addWidget(spin,1); self.spacing.append(spin)
        form.addRow('Voxel · µm',self.voxels)
        self.atlas_spacing=QCheckBox('Use atlas voxel size')
        form.addRow(self.atlas_spacing)
        self.interpolation=PopupComboBox()
        self.interpolation.addItem('Linear','linear'); self.interpolation.addItem('Nearest','nearest')
        form.addRow('Interpolation',self.interpolation)
        self.form=form
        self.kind.currentIndexChanged.connect(self._update)
        self.stage.currentIndexChanged.connect(self._set_spacing)
        self.atlas_spacing.toggled.connect(self._set_spacing)
        self.buttons=QDialogButtonBox(QDialogButtonBox.StandardButton.Save|QDialogButtonBox.StandardButton.Cancel)
        self.buttons.button(QDialogButtonBox.StandardButton.Save).setText('Export…')
        self.buttons.accepted.connect(self.choose_destination)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self._update()

    def _update(self):
        kind=self.kind.currentData(); volume=kind in ('brain','electrodes')
        self.format.clear()
        formats=(('TIFF','tiff'),('OME-TIFF','ome')) if volume else (
            (('JSON','json'),('NumPy archive','npz'),*([('ANTs transform','ants')] if find_spec('ants') else []))
            if kind=='transforms' else ((('ZIP','zip'),) if kind=='analysis' else ((('JSON','json'),('ZIP · tables + JSON','zip')) if kind=='reconstruction' else (('JSON','json'),('CSV · per-pair errors','csv')))))
        for title,value in formats: self.format.addItem(title,value)
        for widget in (self.extent,self.voxels,self.atlas_spacing,self.interpolation):
            self.form.setRowVisible(widget,volume)
        self.form.setRowVisible(self.stage,kind not in ('report','analysis','reconstruction'))
        valid=kind not in ('report','analysis','reconstruction') or self.project.fit_is_current
        self.buttons.button(QDialogButtonBox.StandardButton.Save).setEnabled(valid)
        self._set_spacing(); self.adjustSize()

    def _set_spacing(self):
        source=self.project.track if self.kind.currentData()=='electrodes' else self.project.moving
        source=source or self.project.moving
        values=self.project.reference.spacing_um_zyx if self.atlas_spacing.isChecked() else source.spacing_um_zyx
        for spin,value in zip(self.spacing,values[::-1]):
            spin.setValue(value); spin.setEnabled(not self.atlas_spacing.isChecked())

    def _suffix(self):
        return {'tiff':'.tif','ome':'.ome.tif','json':'.json','npz':'.npz','ants':'.mat','csv':'.csv','zip':'.zip'}.get(self.format.currentData(),'.tif')

    def choose_destination(self):
        kind=self.kind.currentData(); stage=self.stage.currentData()
        if kind in ('report','analysis','reconstruction'): stage='registration'
        suffix=self._suffix()
        default=Path(self.project.output_dir)/f'{kind}_{stage}{suffix}'
        path,_=QFileDialog.getSaveFileName(self,'Export',str(default),f'{self.format.currentText()} (*{suffix})')
        if not path: return
        if not Path(path).suffix: path+=suffix
        spacing=[s.value() for s in self.spacing][::-1]
        source=self.project.track if kind=='electrodes' else self.project.moving
        native=all(abs(value-original)<1e-5 for value,original in zip(spacing,source.spacing_um_zyx))
        policy='atlas' if self.atlas_spacing.isChecked() else ('source' if native else 'custom')
        self.exportRequested.emit(dict(path=path,kind=kind,stage=stage,format=self.format.currentData(),
            extent=self.extent.currentData(),spacing_um_zyx=[s.value() for s in self.spacing][::-1],
            spacing_policy=policy,interpolation=self.interpolation.currentData()))
        self.accept()

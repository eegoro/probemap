"""Review report and display tools. No export controls belong in this workspace."""
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (QAbstractItemView,QDialog,QDialogButtonBox,QFormLayout,QGridLayout,
    QGroupBox,QHeaderView,QHBoxLayout,QLabel,QPushButton,QScrollArea,QTableWidget,QTableWidgetItem,
    QTabWidget,QVBoxLayout,QWidget)

from .desktop_rendering import RenderSettingsPanel
from .desktop_slice_display import SliceDisplayControls
from .review import registration_report
from .desktop_widgets import flat_tabs


class ResidualItem(QTableWidgetItem):
    def __init__(self,value):
        super().__init__(f'{value:.1f}')
        self.setData(Qt.ItemDataRole.UserRole,float(value))
        self.setTextAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignVCenter)

    def __lt__(self,other):
        return self.data(Qt.ItemDataRole.UserRole)<other.data(Qt.ItemDataRole.UserRole)


class ResidualDialog(QDialog):
    def __init__(self,report,parent=None):
        super().__init__(parent)
        self.setWindowTitle('Landmark residuals')
        self.resize(540,520)
        layout=QVBoxLayout(self)
        self.table=QTableWidget(0,3)
        self.table.setShowGrid(False)
        self.table.setAlternatingRowColors(True)
        self.table.setStyleSheet('''QTableWidget {border:0;background:#242b36;alternate-background-color:#202936;}
            QHeaderView::section {background:#242b36;color:#9aabc0;border:0;padding:8px;}''')
        self.table.setHorizontalHeaderLabels(['Pair','Before · µm','After · µm'])
        self.table.horizontalHeaderItem(2).setToolTip('Distance after the applied registration')
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setToolTip('Double-click a pair to inspect it in Registration')
        for values in report['pair_errors']:
            row=self.table.rowCount(); self.table.insertRow(row)
            item=QTableWidgetItem(); item.setData(Qt.ItemDataRole.DisplayRole,values['pair'])
            self.table.setItem(row,0,item)
            self.table.setItem(row,1,ResidualItem(values['before_um']))
            self.table.setItem(row,2,ResidualItem(values['after_um']))
        self.table.setSortingEnabled(True)
        self.table.sortItems(2,Qt.SortOrder.DescendingOrder)
        self.table.cellDoubleClicked.connect(self.open_pair)
        layout.addWidget(self.table)
        buttons=QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject); layout.addWidget(buttons)

    def open_pair(self,row,column):
        pair=int(self.table.item(row,0).text())-1
        owner=self.parent().owner
        self.accept(); owner.review_pair(pair)


class ReviewTools(QTabWidget):
    def __init__(self,owner):
        super().__init__()
        self.owner=owner
        flat_tabs(self)
        owner.preview_tabs.currentChanged.connect(lambda index:self.set_comparison(index==1))
        self._attached=False
        self.report_key=None
        for title,build in (('Report',self._report),('3D',self._volume),('Slices',self._slices)):
            page=QWidget(); layout=QVBoxLayout(page)
            layout.setContentsMargins(8,10,8,10)
            build(layout); layout.addStretch()
            scroll=QScrollArea(); scroll.setWidgetResizable(True)
            scroll.setFrameShape(QScrollArea.Shape.NoFrame); scroll.setWidget(page)
            self.addTab(scroll,title)

    def _report(self,layout):
        self.summary=QLabel(); self.summary.setWordWrap(True); layout.addWidget(self.summary)
        group=QGroupBox('Landmark error · atlas µm'); grid=QGridLayout(group)
        before=QLabel('Before'); before.setToolTip('Placement position, rotation and axes; scale removed')
        grid.addWidget(before,0,1); grid.addWidget(QLabel('After'),0,2)
        self.metrics={}
        for row,(key,title) in enumerate((('rms','RMS'),('median','Median'),('p95','95th percentile'),('maximum','Maximum')),1):
            grid.addWidget(QLabel(title),row,0)
            for col,stage in enumerate(('before','after'),1):
                value=QLabel('—'); value.setAlignment(Qt.AlignmentFlag.AlignRight)
                grid.addWidget(value,row,col); self.metrics[stage,key]=value
        layout.addWidget(group)
        self.validation=QLabel(); self.validation.setWordWrap(True)
        self.validation.setToolTip('Each pair is predicted by a fit made without that pair.\nThis is cross-validation, not an independent measurement of whole-brain accuracy.')
        layout.addWidget(self.validation)
        self.scale_context=QLabel(); self.scale_context.setWordWrap(True)
        layout.addWidget(self.scale_context)
        row=QHBoxLayout()
        self.largest=QLabel(); row.addWidget(self.largest,1)
        self.more_button=QPushButton('More…')
        self.more_button.setStyleSheet('QPushButton {background:transparent;border:0;padding:3px;color:#9fc9df;} QPushButton:hover {color:#e5f4ff;}')
        self.more_button.clicked.connect(self.show_residuals)
        row.addWidget(self.more_button); layout.addLayout(row)
        self.details=QLabel(); self.details.setWordWrap(True); layout.addWidget(self.details)
        self.calibration=QLabel(); self.calibration.setWordWrap(True); self.calibration.setObjectName('muted')
        layout.addWidget(self.calibration)

    def show_residuals(self):
        self.residual_dialog=ResidualDialog(self.report,self)
        self.residual_dialog.open()

    def _volume(self,layout):
        self.volume_layout=layout
        self.rendering=None
        self.refresh_rendering()

    def refresh_rendering(self):
        view=self.owner.volume_view
        position=self.volume_layout.count()
        if self.rendering is not None:
            if self.rendering.layer_count==len(view.layer_titles): return
            position=self.volume_layout.indexOf(self.rendering)
            self.volume_layout.removeWidget(self.rendering); self.rendering.deleteLater()
        self.rendering=RenderSettingsPanel(view,include_opacity=False,compact=True,minimal=True)
        for index,title in enumerate(('Atlas','Brain','Electrodes')[:len(view.layer_titles)]):
            self.rendering.tabs.setTabText(index,title)
        self.volume_layout.insertWidget(position,self.rendering)
        if not hasattr(self,'_render_connected'):
            view.renderSettingsChanged.connect(lambda:self.rendering.sync())
            self._render_connected=True

    def _slices(self,layout):
        group=QGroupBox('Comparison'); form=QFormLayout(group)
        self.comparison_group=group
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.addRow('Plane',self.owner.slice_comparison.plane)
        layout.addWidget(group)
        self.owner._appearance(layout,'review')
        self.opacity_group=layout.itemAt(layout.count()-1).widget()
        self.slice_display=SliceDisplayControls(channels=('Atlas','Brain scan'),colors=('green','magenta'),minimal=True)
        self.slice_display.initialize(((0,1),(0,1)))
        self.slice_display.changed.connect(self.owner.request_render)
        self.slice_display.autoRequested.connect(self.auto_contrast)
        layout.addWidget(self.slice_display)
        self.set_comparison(False)

    def set_comparison(self,enabled):
        self.comparison_group.setVisible(enabled)
        self.opacity_group.setVisible(not enabled)
        # Every image keeps its own napari colormap in both comparison modes.

    def auto_contrast(self,index):
        import numpy as np
        volume=(self.owner.reference,self.owner.moving)[index]
        if volume is None: return
        data=volume.data.ravel()[::max(1,volume.data.size//500000)]
        data=data[np.isfinite(data)&(data>0)]
        limits=np.percentile(data,(1,99.8)).tolist() if len(data) else [0,1]
        if limits[1]-limits[0]<1e-6: limits=[0,1]
        self.slice_display.records[index]['contrast']=limits
        self.slice_display.sync(); self.owner.request_render()

    def update_report(self):
        p=self.owner.project
        if p is None: return
        key=(p.coarse.tobytes(),p.residual.tobytes(),repr(p.pairs),repr(p.fit_result),
             p.moving.spacing_um_zyx,p.reference.spacing_um_zyx)
        if key==self.report_key: return
        self.report_key=key
        report=registration_report(p)
        self.report=report
        self.summary.setText(f"{(report['model'] or 'Registration').title()} · {report['pairs']} pairs")
        for (stage,key),widget in self.metrics.items():
            widget.setText(f"{report[stage][key]:.1f}" if report[stage] else '—')
        loo=report['leave_one_out_rms']
        self.validation.setText('Leave-one-out RMS · '+(f'{loo:.1f} µm' if loo is not None else 'unavailable'))
        voxels=report['rms_atlas_voxels']; reduction=report['rms_reduction_percent']
        self.scale_context.setText('RMS · '+(f'{voxels:.1f} atlas voxels' if voxels is not None else '—')+
            '\nRMS reduction · '+(f'{reduction:.1f}%' if reduction is not None else '—'))
        self.scale_context.setToolTip('Voxel RMS uses 3D distances on the native atlas voxel grid.\nReduction compares Before (unscaled placement) with After.')
        self.largest.setText(f"Largest residual · pair {report.get('largest_pair','—')}")
        self.more_button.setEnabled(bool(report['pair_errors']))
        stretches=' / '.join(f'{s:.3g}' for s in report['principal_stretches'])
        self.details.setText(f"Principal stretches · {stretches}\nVolume ratio · {report['volume_ratio']:.3g}×")
        self.details.setToolTip('Principal stretches and volume ratio describe the full Placement + Registration transform.\nRulers and probe geometry use the original scan physical frame.')
        xyz=lambda values:' / '.join(f'{v:g}' for v in values)
        dimensions=' / '.join(f'{value:.2f}' for value in report['atlas_size_mm_xyz'])
        self.calibration.setText(f"Atlas image size · XYZ\n{dimensions} mm\n\n"
            f"Voxel XYZ · µm\nAtlas · {xyz(report['atlas_voxel_um_xyz'])}\nBrain · {xyz(report['source_voxel_um_xyz'])}")
        self.calibration.setToolTip('Atlas dimensions describe its complete image field, including any padding.')

    def attach_volume_controls(self,enabled):
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
        if enabled:
            view.set_placement_mode(False,compact=True)
            self.refresh_rendering(); self.rendering.sync()

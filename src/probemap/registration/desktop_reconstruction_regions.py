"""Display-only region controls for the final 2D and 3D views."""
import copy

from qtpy.QtCore import Qt,Signal
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (QAbstractItemView,QCheckBox,QColorDialog,QFormLayout,QHBoxLayout,
    QLineEdit,QSlider,QTableWidget,QTableWidgetItem,QVBoxLayout,QWidget)

from .desktop_regions import region_side
from .desktop_tables import CheckHeader,ColorDotDelegate,TableHeightGrip
from .desktop_widgets import PopupComboBox


class ReconstructionRegions(QWidget):
    changed=Signal()

    def __init__(self):
        super().__init__(); self.regions=[]; self._all_regions=[]; self.styles={}; self._updating=False
        layout=QVBoxLayout(self); layout.setContentsMargins(6,10,6,6)
        row=QHBoxLayout(); self.search=QLineEdit(); self.search.setPlaceholderText('Find region…')
        self.search.textChanged.connect(self._filter); row.addWidget(self.search,1)
        self.side=PopupComboBox()
        for name,key in (('All sides','all'),('Left','left'),('Right','right'),('Unspecified','unspecified')): self.side.addItem(name,key)
        self.side.currentIndexChanged.connect(self._filter); row.addWidget(self.side); layout.addLayout(row)
        self.table=QTableWidget(0,3); table=self.table
        table.setHorizontalHeaderLabels(['','Region','Color']); table.verticalHeader().hide()
        table.verticalHeader().setDefaultSectionSize(29); table.setShowGrid(False)
        table.setStyleSheet('QTableWidget{background:transparent;border:0;}')
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        header=CheckHeader(table,{0}); header.setStretchLastSection(False); table.setHorizontalHeader(header)
        header.toggled.connect(self._all); table.itemChanged.connect(self._cell_changed)
        table.cellClicked.connect(self._color); table.setItemDelegateForColumn(2,ColorDotDelegate(table))
        table.setColumnWidth(0,30); table.setColumnWidth(1,190); table.setColumnWidth(2,48)
        table.setFixedHeight(300); layout.addWidget(table); layout.addWidget(TableHeightGrip(table))
        form=QFormLayout(); self.opacity=QSlider(Qt.Orientation.Horizontal); self.opacity.setRange(0,100); self.opacity.setValue(40)
        self.opacity.valueChanged.connect(self._changed); form.addRow('Opacity',self.opacity)
        self.names=QCheckBox('Names'); self.names.toggled.connect(self._changed); form.addRow(self.names)
        layout.addLayout(form); layout.addStretch()

    def initialize(self,regions,base_styles,saved):
        self._updating=True; self.regions=list(regions); self._all_regions=list(regions)
        # Region visibility/color is shared with Tracks, Probes and Review.
        # Import older Reconstruction-only preferences once, without replacing
        # a more recent setting made on another page.
        self.styles=base_styles
        for key,record in saved.get('styles',{}).items():
            style=self.styles.setdefault(key,{})
            for name,value in record.items(): style.setdefault(name,copy.deepcopy(value))
        self.opacity.setValue(saved.get('opacity',40)); self.names.setChecked(saved.get('names',False))
        self.search.setText(saved.get('search','')); self.side.setCurrentIndex(max(0,self.side.findData(saved.get('side','all'))))
        self._populate(); self._updating=False; self._filter()

    def _populate(self):
        selected={cell.data(Qt.ItemDataRole.UserRole) for cell in self.table.selectedItems() if cell.column()==0}
        self.table.blockSignals(True); self.table.setRowCount(len(self.regions))
        self.table.clearSelection()
        for row,region in enumerate(self.regions):
            style=self.styles.get(str(region['id']),{})
            check=QTableWidgetItem(); check.setData(Qt.ItemDataRole.UserRole,region['id'])
            check.setFlags(Qt.ItemFlag.ItemIsEnabled|Qt.ItemFlag.ItemIsSelectable|Qt.ItemFlag.ItemIsUserCheckable)
            check.setCheckState(Qt.CheckState.Checked if style.get('visible',True) else Qt.CheckState.Unchecked)
            self.table.setItem(row,0,check)
            name=QTableWidgetItem(style.get('name',region['name'])); name.setToolTip(f"{region['name']} · {region['id']}"); self.table.setItem(row,1,name)
            color=QTableWidgetItem(); color.setData(Qt.ItemDataRole.UserRole,style.get('color_rgb',region['color_rgb'])); self.table.setItem(row,2,color)
            if region['id'] in selected: self.table.selectRow(row)
        self.table.blockSignals(False)

    def prioritize(self,identifiers):
        """Keep regions on the sampled plane first, including unchecked ones."""
        present=set(map(int,identifiers))
        ordered=sorted(self._all_regions,key=lambda r:r['id'] not in present)
        if [r['id'] for r in ordered]==[r['id'] for r in self.regions]: return
        self.regions=ordered; self._populate(); self._filter()

    def _filter(self,*args):
        if self._updating: return
        text=self.search.text().casefold(); side=self.side.currentData()
        for row,region in enumerate(self.regions):
            name=self.table.item(row,1).text().casefold()
            self.table.setRowHidden(row,(text not in name and text not in region['name'].casefold()) or (side!='all' and region_side(region)!=side))
        self.table.horizontalHeader().viewport().update()

    def _cell_changed(self,cell):
        if self._updating or cell.column()!=0: return
        key=str(cell.data(Qt.ItemDataRole.UserRole))
        self.styles.setdefault(key,{})['visible']=cell.checkState()==Qt.CheckState.Checked
        self.table.horizontalHeader().viewport().update(); self._changed()

    def _all(self,column,visible):
        self.table.blockSignals(True)
        for row in range(self.table.rowCount()):
            if self.table.isRowHidden(row): continue
            cell=self.table.item(row,0); cell.setCheckState(Qt.CheckState.Checked if visible else Qt.CheckState.Unchecked)
            self.styles.setdefault(str(cell.data(Qt.ItemDataRole.UserRole)),{})['visible']=visible
        self.table.blockSignals(False); self.table.horizontalHeader().viewport().update(); self._changed()

    def _color(self,row,column):
        if column!=2: return
        region=self.regions[row]; style=self.styles.setdefault(str(region['id']),{})
        color=QColorDialog.getColor(QColor(*style.get('color_rgb',region['color_rgb'])),self,'Region color')
        if color.isValid():
            style['color_rgb']=[color.red(),color.green(),color.blue()]
            self.table.item(row,2).setData(Qt.ItemDataRole.UserRole,style['color_rgb']); self._changed()

    def _changed(self,*args):
        if not self._updating: self.changed.emit()

    def state_record(self):
        return dict(opacity=self.opacity.value(),names=self.names.isChecked(),
            search=self.search.text(),side=self.side.currentData())

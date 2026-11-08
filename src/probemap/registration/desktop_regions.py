"""Sortable atlas regions, with bulk checkboxes acting on the filtered rows."""
import re
from pathlib import Path

from qtpy.QtCore import Qt
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (QColorDialog, QDoubleSpinBox, QFormLayout, QHBoxLayout,
    QHeaderView, QLineEdit, QMenu, QSlider, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from .desktop_tables import CheckHeader, ColorDotDelegate, TableHeightGrip
from .desktop_widgets import PopupComboBox


def region_side(region):
    value=str(region.get('hemisphere',region.get('side',''))).lower()
    if value in ('l','left'): return 'left'
    if value in ('r','right'): return 'right'
    words=re.split(r'[^a-z0-9]+',Path(region.get('mask_path','')).stem.lower())
    for side,aliases in (('left',('l','left')),('right',('r','right'))):
        if any(alias in words for alias in aliases): return side
    words=re.split(r'[^a-z0-9]+',region.get('name','').lower())
    if 'left' in words: return 'left'
    if 'right' in words: return 'right'
    return 'unspecified'


class RegionControlsMixin:
    def _region_controls(self,include_display=True):
        panel=QWidget(); layout=QVBoxLayout(panel); layout.setContentsMargins(0,0,0,0)
        row=QHBoxLayout()
        self.region_search=QLineEdit(); self.region_search.setPlaceholderText('Find region…')
        self.region_search.textChanged.connect(self._filter_regions)
        row.addWidget(self.region_search,1)
        self.region_side_filter=PopupComboBox()
        for label,value in (('All sides','all'),('Left','left'),('Right','right'),('Unspecified','unspecified')):
            self.region_side_filter.addItem(label,value)
        self.region_side_filter.currentIndexChanged.connect(self._filter_regions)
        self.region_side_filter.setToolTip('Filter by side recorded in the atlas labels')
        row.addWidget(self.region_side_filter)
        layout.addLayout(row)
        self.region_table=QTableWidget(0,4)
        table=self.region_table
        table.setHorizontalHeaderLabels(['','Label','Region','Color'])
        table.verticalHeader().hide(); table.verticalHeader().setDefaultSectionSize(27)
        table.setShowGrid(False); table.setMinimumHeight(80); table.setFixedHeight(250)
        header=CheckHeader(table,(0,1)); table.setHorizontalHeader(header)
        header.toggled.connect(self._region_header_toggled)
        for column,width in enumerate((28,65,110,38)): table.setColumnWidth(column,width)
        table.setItemDelegateForColumn(3,ColorDotDelegate(table))
        table.itemChanged.connect(self._region_changed)
        table.cellClicked.connect(self._region_color)
        table.setSortingEnabled(True); table.sortItems(2,Qt.SortOrder.AscendingOrder)
        header.sortIndicatorChanged.connect(self._filter_regions)
        table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        table.customContextMenuRequested.connect(self._region_menu)
        layout.addWidget(table); layout.addWidget(TableHeightGrip(table))
        form=QFormLayout()
        self.region_opacity=QSlider(Qt.Orientation.Horizontal)
        self.region_opacity.setRange(0,100); self.region_opacity.setValue(45)
        self.region_opacity.valueChanged.connect(self.draw_cached)
        form.addRow('Opacity',self.region_opacity)
        self.region_opacity_label=form.labelForField(self.region_opacity)
        self.atlas_gamma=QDoubleSpinBox(); self.atlas_gamma.setRange(.1,5)
        self.atlas_gamma.setValue(1); self.atlas_gamma.setSingleStep(.1)
        self.atlas_gamma.valueChanged.connect(self.draw_cached)
        if include_display: form.addRow('Atlas gamma',self.atlas_gamma)
        else: self.atlas_gamma.hide()
        layout.addLayout(form)
        return panel

    def _region_for_row(self,row):
        if self.labels is None or row<0: return None
        cell=self.region_table.item(row,2)
        identifier=cell.data(Qt.ItemDataRole.UserRole) if cell else None
        return next((r for r in self.labels.regions if r['id']==identifier),None)

    def _populate_regions(self):
        table=self.region_table
        table.blockSignals(True); table.setSortingEnabled(False)
        table.setRowCount(len(self.labels.regions))
        for row,region in enumerate(self.labels.regions):
            style=self.session['regions'].get(str(region['id']),{})
            for column,key,default in ((0,'visible',True),(1,'label',False)):
                cell=QTableWidgetItem()
                cell.setFlags(Qt.ItemFlag.ItemIsEnabled|Qt.ItemFlag.ItemIsSelectable|Qt.ItemFlag.ItemIsUserCheckable)
                cell.setCheckState(Qt.CheckState.Checked if style.get(key,default) else Qt.CheckState.Unchecked)
                table.setItem(row,column,cell)
            name=QTableWidgetItem(style.get('name',region['name']))
            name.setData(Qt.ItemDataRole.UserRole,region['id'])
            name.setToolTip(f"{region['name']} · ID {region['id']} · {region_side(region).title()}")
            table.setItem(row,2,name)
            color=QTableWidgetItem()
            color.setFlags(Qt.ItemFlag.ItemIsEnabled|Qt.ItemFlag.ItemIsSelectable)
            color.setData(Qt.ItemDataRole.UserRole,style.get('color_rgb',region['color_rgb']))
            table.setItem(row,3,color)
        table.setSortingEnabled(True); table.blockSignals(False)
        self._filter_regions()

    def _filter_regions(self,*args):
        text=self.region_search.text().casefold()
        side=self.region_side_filter.currentData()
        for row in range(self.region_table.rowCount()):
            region=self._region_for_row(row)
            if region is None: continue
            name=self.region_table.item(row,2).text().casefold()
            hidden=(text not in name and text not in region['name'].casefold()) or (side!='all' and region_side(region)!=side)
            self.region_table.setRowHidden(row,hidden)
        self.region_table.horizontalHeader().viewport().update()

    def _region_changed(self,cell):
        if self.labels is None or self.session is None: return
        region=self._region_for_row(cell.row())
        if region is None: return
        style=self.session['regions'].setdefault(str(region['id']),{})
        if cell.column() in (0,1):
            style['visible' if cell.column()==0 else 'label']=cell.checkState()==Qt.CheckState.Checked
        elif cell.column()==2:
            style['name']=cell.text().strip() or region['name']; self.report_timer.start(150)
        self.region_table.horizontalHeader().viewport().update()
        self.draw_cached()

    def _region_color(self,row,column):
        if column!=3: return
        region=self._region_for_row(row)
        if region is None: return
        style=self.session['regions'].setdefault(str(region['id']),{})
        color=QColorDialog.getColor(QColor(*style.get('color_rgb',region['color_rgb'])),self,'Region color')
        if color.isValid():
            self._remember(); style['color_rgb']=[color.red(),color.green(),color.blue()]
            self._populate_regions(); self.draw_cached(); self.owner.schedule_save()

    def _region_header_toggled(self,column,checked):
        if self.labels is None or self.session is None: return
        self._remember()
        for row in range(self.region_table.rowCount()):
            if self.region_table.isRowHidden(row): continue
            region=self._region_for_row(row)
            self.session['regions'].setdefault(str(region['id']),{})['visible' if column==0 else 'label']=checked
        self._populate_regions(); self.draw_cached(); self.owner.schedule_save()

    def _set_all_regions(self,visible):
        if self.labels is None or self.session is None: return
        for region in self.labels.regions: self.session['regions'].setdefault(str(region['id']),{})['visible']=visible
        self._populate_regions(); self.draw_cached()

    def _isolate_region_side(self,side):
        self._remember()
        for region in self.labels.regions:
            self.session['regions'].setdefault(str(region['id']),{})['visible']=region_side(region)==side
        self._populate_regions(); self.draw_cached(); self.owner.schedule_save()

    def _region_menu(self,position):
        if self.labels is None: return
        menu=QMenu(self)
        menu.addAction('Show only left regions',lambda:self._isolate_region_side('left'))
        menu.addAction('Show only right regions',lambda:self._isolate_region_side('right'))
        menu.exec(self.region_table.viewport().mapToGlobal(position))

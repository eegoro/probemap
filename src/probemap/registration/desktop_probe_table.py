"""Editable insertion table for native probe placement."""
import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (QAbstractItemView,QHeaderView,QLabel,QMenu,
    QTableWidget,QTableWidgetItem,QStyledItemDelegate)

from .desktop_placement import EditedSpin
from .desktop_tables import ColorDotDelegate,TableHeightGrip
from .probe_placement import contact_side,source_plane
from .neuropixels_model import probe_settings
from .track_model import plane_geometry


class ProbeAngleDelegate(QStyledItemDelegate):
    def createEditor(self,parent,option,index):
        editor=EditedSpin(parent); editor.setRange(-180,180); editor.setDecimals(1)
        editor.setSuffix('°'); editor.setKeyboardTracking(False)
        return editor

    def setEditorData(self,editor,index):
        editor.setValue(float(str(index.data() or '0').replace('°','').replace('—','0')))

    def setModelData(self,editor,model,index):
        model.setData(index,f'{editor.value():.1f}°')


class ProbePlacementTableMixin:
    def _build_placement_table(self,layout):
        self._placement_row_active=True
        layout.addWidget(QLabel('Insertions'))
        table=QTableWidget(0,4); self.placement_table=table
        table.setHorizontalHeaderLabels(['Insertion','Color','Rotation','Contacts'])
        table.verticalHeader().hide(); table.verticalHeader().setDefaultSectionSize(32)
        table.setShowGrid(False)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        table.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked|QAbstractItemView.EditTrigger.EditKeyPressed)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        table.horizontalHeader().setStretchLastSection(True)
        for col,width in enumerate((100,50,65,65)): table.setColumnWidth(col,width)
        table.setFixedHeight(155)
        table.setItemDelegateForColumn(1,ColorDotDelegate(table))
        table.setItemDelegateForColumn(2,ProbeAngleDelegate(table))
        table.itemSelectionChanged.connect(self._placement_row_selected)
        table.itemChanged.connect(self._placement_cell_edited)
        table.cellClicked.connect(self._placement_cell_clicked)
        table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        table.customContextMenuRequested.connect(self._placement_table_menu)
        layout.addWidget(table); layout.addWidget(TableHeightGrip(table))

    def _refresh_placement_table(self):
        if self.session is None: return
        table=self.placement_table
        table.blockSignals(True); table.setRowCount(len(self.session['insertions']))
        table.clearSelection(); table.setCurrentItem(None)
        for row,item in enumerate(self.session['insertions']):
            s=probe_settings(item); p=s.get('pose')
            angle='—'; face='Front'
            if p:
                native=source_plane(item,self.session['plane'],self.project.total); g=plane_geometry(native)
                angle=f"{np.rad2deg(np.arctan2(-np.dot(p['down_source_zyx'],g['u_zyx']),-np.dot(p['down_source_zyx'],g['v_zyx']))):.1f}°"
                face='Front' if contact_side(item,native)>0 else 'Back'
            elif s.get('placement_contact_side',1)<0: face='Back'
            physical=s['geometry'].get('geometry_version',1)>=2
            if physical: face='3D'
            for col,text in enumerate((item['name'],'',angle,face)):
                cell=QTableWidgetItem(text); cell.setData(Qt.ItemDataRole.UserRole,item['color'] if col==1 else item['id'])
                if col==0:
                    cell.setFlags(cell.flags()|Qt.ItemFlag.ItemIsUserCheckable)
                    cell.setCheckState(Qt.CheckState.Checked if item.get('probe_visible',True) else Qt.CheckState.Unchecked)
                    cell.setToolTip('Double-click to rename · Check to show probe')
                elif col==1:
                    cell.setFlags(Qt.ItemFlag.ItemIsEnabled|Qt.ItemFlag.ItemIsSelectable); cell.setToolTip('Change insertion color')
                elif col==2:
                    cell.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    cell.setToolTip('Rotate in the scan plane · You can also drag the circular handle on the probe')
                    if p is None: cell.setFlags(cell.flags()&~Qt.ItemFlag.ItemIsEditable)
                else:
                    cell.setFlags(Qt.ItemFlag.ItemIsEnabled|Qt.ItemFlag.ItemIsSelectable)
                    cell.setToolTip(('Rotate 180° about the element axis · ' if physical else 'Flip recording face and shank order · ')+s['geometry']['model'])
                    cell.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                table.setItem(row,col,cell)
            if item['id']==self.insertion_id and self._placement_row_active: table.selectRow(row)
        table.blockSignals(False)

    def _placement_select_row(self,identifier):
        self._placement_row_active=True
        for row in range(self.placement_table.rowCount()):
            if self.placement_table.item(row,0).data(Qt.ItemDataRole.UserRole)==identifier:
                self.placement_table.blockSignals(True); self.placement_table.selectRow(row); self.placement_table.blockSignals(False)
                return

    def _placement_row_selected(self):
        if self._updating or self._mode!=1 or self.session is None: return
        rows=self.placement_table.selectionModel().selectedRows()
        if not rows:
            self._placement_row_active=False
            self.track_slice.placement_selected=False
            self._placement_tool('select')
            return
        self._placement_row_active=True
        identifier=self.placement_table.item(rows[0].row(),0).data(Qt.ItemDataRole.UserRole)
        self.placement_insertion.setCurrentIndex(self.placement_insertion.findData(identifier))
        if probe_settings(self.current_insertion()).get('pose'): self._placement_select(True)
        else:
            self.track_slice.placement_selected=False
            self.track_slice.update()

    def _placement_cell_clicked(self,row,col):
        if self._mode!=1: return
        if col==1:
            self.color_insertion(); self._placement_refresh_controls()
        elif col==3: self._placement_flip_face()

    def _placement_cell_edited(self,cell):
        if self._updating or self.session is None or self._mode!=1: return
        identifier=self.placement_table.item(cell.row(),0).data(Qt.ItemDataRole.UserRole)
        item=next(i for i in self.session['insertions'] if i['id']==identifier)
        if cell.column()==0:
            name=cell.text().strip(); visible=cell.checkState()==Qt.CheckState.Checked
            if not self._valid_insertion_name(name,identifier): self._refresh_placement_table(); return
            if name==item['name'] and visible==item.get('probe_visible',True): return
            self._remember(); item['name']=name; item['probe_visible']=visible
            self._refresh_insertions(); self._placement_refresh_controls(); self.draw_cached()
        elif cell.column()==2:
            try: angle=float(cell.text().replace('°',''))
            except ValueError: self._refresh_placement_table(); return
            if not np.isfinite(angle): self._refresh_placement_table(); return
            self.placement_insertion.setCurrentIndex(self.placement_insertion.findData(identifier))
            self.placement_probe_fields['angle'].setValue(angle)
            self._refresh_placement_table()
        self.owner.schedule_save()

    def _placement_table_menu(self,position):
        row=self.placement_table.rowAt(position.y())
        if row<0: return
        self.placement_table.selectRow(row)
        menu=QMenu(self)
        menu.addAction('Rename',lambda:self.placement_table.editItem(self.placement_table.item(row,0)))
        menu.addAction('Color…',self.color_insertion)
        models=menu.addMenu('Probe model')
        for model in [self.placement_model.itemText(i) for i in range(self.placement_model.count())]:
            action=models.addAction(model,lambda checked=False,m=model:self.placement_model.setCurrentText(m))
            action.setCheckable(True); action.setChecked(self.placement_model.currentText()==model)
        menu.addAction('Rotate 180° about axis' if self.current_insertion().get('probe',{}).get('geometry',{}).get('geometry_version',1)>=2 else 'Flip recording face',self._placement_flip_face)
        menu.addAction('Reverse tip direction',self._placement_flip)
        menu.addSeparator(); menu.addAction('Remove probe',self._placement_delete)
        menu.exec(self.placement_table.viewport().mapToGlobal(position))
        self._placement_refresh_controls()

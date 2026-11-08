"""Compact recording library and native-contact overlays for probe placement."""
from pathlib import Path

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (QAbstractItemView,QColorDialog,QFileDialog,QHBoxLayout,
    QLabel,QMenu,QStyledItemDelegate,QTableWidget,QTableWidgetItem,QVBoxLayout,QWidget)

from .desktop_placement import EditedSpin,SelectedCombo
from .desktop_tables import CheckHeader,ColorDotDelegate,TableHeightGrip
from .desktop_widgets import view_button
from .landmarks import transform_points
from .neuropixels_model import contact_mesh,recording_sites
from .recordings import assign_recording,import_recordings,recording_entries,remove_recordings


def recording_rgba(color,opacity=1.):
    """One contact appearance for Probes and all reconstruction views."""
    alpha=float(np.clip(opacity,0,1)); brightness=.35+.65*alpha
    return (*(brightness*c for c in QColor(color).getRgbF()[:3]),alpha)


class RecordingInsertionDelegate(QStyledItemDelegate):
    def __init__(self,workspace,table):
        super().__init__(table); self.workspace=workspace

    def createEditor(self,parent,option,index):
        editor=SelectedCombo(parent); editor.addItem('—',None)
        for item in self.workspace.session['insertions']: editor.addItem(item['name'],item['id'])
        editor.activated.connect(lambda:self._commit(editor))
        return editor

    def _commit(self,editor):
        self.commitData.emit(editor); self.closeEditor.emit(editor)

    def setEditorData(self,editor,index):
        editor.setCurrentIndex(max(0,editor.findData(index.data(Qt.ItemDataRole.UserRole))))

    def setModelData(self,editor,model,index):
        # One edit signal, with both the stable ID and displayed name in place.
        model.setItemData(index,{Qt.ItemDataRole.UserRole:editor.currentData(),
            Qt.ItemDataRole.DisplayRole:editor.currentText()})


class RecordingOpacityDelegate(QStyledItemDelegate):
    def createEditor(self,parent,option,index):
        editor=EditedSpin(parent); editor.setRange(0,100); editor.setDecimals(0)
        editor.setSuffix('%'); editor.setKeyboardTracking(False)
        return editor

    def setEditorData(self,editor,index): editor.setValue(float(str(index.data()).rstrip('%')))

    def setModelData(self,editor,model,index): model.setData(index,f'{editor.value():.0f}%')


class RecordingTableMixin:
    def _build_recording_library(self,layout):
        panel=QWidget(); content=QVBoxLayout(panel); content.setContentsMargins(0,0,0,0)
        self.placement_recordings_panel=panel
        header=QHBoxLayout(); header.addWidget(QLabel('Recordings'),1)
        self.recordings_add_button=view_button('add','Add recording files or folder',self._recording_import_menu)
        header.addWidget(self.recordings_add_button); content.addLayout(header)
        table=QTableWidget(0,5); self.placement_recordings_table=table
        table.setHorizontalHeaderLabels(['Name','Insertion','File','Color','Opacity'])
        table.verticalHeader().hide(); table.verticalHeader().setDefaultSectionSize(30)
        table.setShowGrid(False); table.setWordWrap(False); table.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        table.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked|QAbstractItemView.EditTrigger.EditKeyPressed)
        header=CheckHeader(table,{0}); header.toggled.connect(self._recording_visibility_all)
        table.setHorizontalHeader(header); header.setStretchLastSection(True)
        for col,width in enumerate((64,64,74,42,58)): table.setColumnWidth(col,width)
        table.setItemDelegateForColumn(1,RecordingInsertionDelegate(self,table))
        table.setItemDelegateForColumn(3,ColorDotDelegate(table))
        table.setItemDelegateForColumn(4,RecordingOpacityDelegate(table))
        table.setFixedHeight(124)
        table.itemChanged.connect(self._recording_cell_edited)
        table.cellClicked.connect(self._recording_cell_clicked)
        table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        table.customContextMenuRequested.connect(self._recording_table_menu)
        content.addWidget(table); content.addWidget(TableHeightGrip(table)); layout.addWidget(panel)
        self._placement_recording_visuals={}

    def _recording_import_menu(self):
        menu=QMenu(self)
        menu.addAction('Files…',self.import_recording_files)
        menu.addAction('Folder…',self.import_recording_folder)
        menu.exec(self.recordings_add_button.mapToGlobal(self.recordings_add_button.rect().bottomLeft()))

    def import_recording_folder(self):
        if self.project is None: return
        path=QFileDialog.getExistingDirectory(self,'Add recordings folder',str(self.project.output_dir))
        if path: self.import_recording_files(paths=[path])

    def import_recording_files(self,checked=False,paths=None):
        if self.session is None: return
        if paths is None:
            paths,_=QFileDialog.getOpenFileNames(self,'Add recordings',str(self.project.output_dir),
                'SpikeGLX channel maps (*.meta *.imro *.txt);;All files (*)')
        if not paths: return
        self._remember()
        added,errors=import_recordings(self.session,paths)
        self._recording_library_changed()
        self.owner.statusBar().showMessage(f'Added {len(added)} recordings'+(f' · {len(errors)} files could not be loaded' if errors else ''))
        if errors: self.owner._error('\n'.join(errors))

    def _refresh_recording_library(self):
        if self.session is None: return
        table=self.placement_recordings_table; table.blockSignals(True)
        selected=table.currentItem()
        identifier=table.item(selected.row(),0).data(Qt.ItemDataRole.UserRole) if selected is not None else None
        entries=recording_entries(self.session); table.setRowCount(len(entries)); table.clearSelection()
        for row,(owner,rec) in enumerate(entries):
            color=rec.get('color') or (owner['color'] if owner else '#9aafc1')
            values=(rec['name'],owner['name'] if owner else '—',Path(rec['path']).name,'',f"{rec.get('opacity',1)*100:.0f}%")
            for col,text in enumerate(values):
                cell=QTableWidgetItem(text)
                value=color if col==3 else (owner['id'] if owner else None) if col==1 else rec['id']
                cell.setData(Qt.ItemDataRole.UserRole,value)
                if col==0:
                    cell.setFlags(cell.flags()|Qt.ItemFlag.ItemIsUserCheckable)
                    cell.setCheckState(Qt.CheckState.Checked if rec.get('visible',True) else Qt.CheckState.Unchecked)
                if col in (2,3): cell.setFlags(Qt.ItemFlag.ItemIsEnabled|Qt.ItemFlag.ItemIsSelectable)
                if col==1: cell.setToolTip('Choose an insertion · Unassigned recordings are not displayed')
                elif col==2: cell.setToolTip(rec['path'])
                elif col==3: cell.setToolTip('Change color · Default: insertion color')
                elif col==4: cell.setToolTip('Recording opacity · 0–100%')
                table.setItem(row,col,cell)
            if rec['id']==identifier: table.selectRow(row)
        table.blockSignals(False); table.horizontalHeader().viewport().update()

    def _recording_at_row(self,row):
        cell=self.placement_recordings_table.item(row,0)
        if cell is None: return None,None
        return next(((owner,r) for owner,r in recording_entries(self.session) if r['id']==cell.data(Qt.ItemDataRole.UserRole)),(None,None))

    def _recording_cell_edited(self,cell):
        if self._updating or self.session is None: return
        owner,rec=self._recording_at_row(cell.row())
        if rec is None: return
        column=cell.column()
        if column==0:
            name=cell.text().strip(); visible=cell.checkState()==Qt.CheckState.Checked
            if not name: self._refresh_recording_library(); return
            if name==rec['name'] and visible==rec.get('visible',True): return
            self._remember(); rec.update(name=name,visible=visible)
        elif column==1:
            target=cell.data(Qt.ItemDataRole.UserRole)
            if target==(owner['id'] if owner else None): return
            try:
                self._remember(); assign_recording(self.session,rec['id'],target)
            except ValueError as error:
                self.owner._error(str(error)); self._refresh_recording_library(); return
        elif column==4:
            try: value=float(cell.text().rstrip('%'))/100
            except ValueError: self._refresh_recording_library(); return
            if not np.isfinite(value) or not 0<=value<=1: self._refresh_recording_library(); return
            if value==rec.get('opacity',1): return
            self._remember(); rec['opacity']=value
        else: return
        self._recording_library_changed()

    def _recording_cell_clicked(self,row,column):
        if column==1: self.placement_recordings_table.editItem(self.placement_recordings_table.item(row,column))
        elif column==3:
            owner,rec=self._recording_at_row(row)
            if rec is None: return
            color=QColorDialog.getColor(QColor(rec.get('color') or (owner['color'] if owner else '#9aafc1')),self,'Recording color')
            if color.isValid(): self._remember(); rec['color']=color.name(); self._recording_library_changed()

    def _recording_visibility_all(self,column,visible):
        if self.session is None: return
        self._remember()
        for _,rec in recording_entries(self.session): rec['visible']=visible
        self._recording_library_changed()

    def _recording_table_menu(self,position):
        table=self.placement_recordings_table; row=table.rowAt(position.y())
        owner,rec=self._recording_at_row(row)
        if rec is None: return
        table.selectRow(row); menu=QMenu(self)
        menu.addAction('Rename',lambda:table.editItem(table.item(row,0)))
        menu.addAction('Use insertion color',lambda:self._recording_inherit_color(rec))
        menu.addSeparator(); menu.addAction('Remove',lambda:self._remove_library_recording(rec['id']))
        menu.exec(table.viewport().mapToGlobal(position))

    def _recording_inherit_color(self,rec):
        self._remember(); rec['color']=None; self._recording_library_changed()

    def _remove_library_recording(self,identifier):
        self._remember(); remove_recordings(self.session,[identifier]); self._recording_library_changed()

    def _recording_library_changed(self):
        self._refresh_recording_library()
        if self._mode==1: self._draw_placement_annotations()
        if self._mode==4: self._refresh_probe_controls(); self.request_render()
        self.owner.schedule_save()

    def _hide_placement_recordings(self,*,dispose=False):
        for mesh in getattr(self,'_placement_recording_visuals',{}).values():
            mesh.visible=False
            if dispose: mesh.parent=None
        if dispose: self._placement_recording_visuals={}

    def _draw_placement_recordings(self,item,display,*,draw_3d=True):
        from vispy import scene
        for rec in item.get('recordings',[]):
            if not rec.get('visible',True) or rec.get('opacity',1)<=0: continue
            channels,heights,sources=recording_sites(item,rec)
            if not channels: continue
            rgba=recording_rgba(rec.get('color') or item['color'],rec.get('opacity',1))
            if item['id']==self.insertion_id:
                self.track_slice.probe_sites.extend(dict(atlas_zyx_um=point,rgba=rgba,
                    label=f"{rec['name']} · ch {channel['channel']}",depth=None,recording_id=rec['id'])
                    for point,channel in zip(transform_points(sources,display),channels))
            if not draw_3d: continue
            mesh=self._placement_recording_visuals.get(rec['id'])
            if mesh is None:
                mesh=scene.visuals.Mesh(parent=self.volume_view.view.scene,shading=None)
                # Contact annotations must remain legible inside the brain
                # and on their coplanar translucent shank faces. Their world
                # coordinates and physical 12 um geometry remain exact.
                mesh.set_gl_state('translucent',depth_test=False,cull_face=False); mesh.order=27
                self._placement_recording_visuals[rec['id']]=mesh
            vertices,faces=contact_mesh(item,sources,self.project.total)
            mesh.set_data(vertices=vertices,faces=faces,color=rgba); mesh.visible=True

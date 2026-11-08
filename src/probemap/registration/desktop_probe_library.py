"""Native probe library and dimensioned, orthographic geometry editor."""
from __future__ import annotations
import copy
import json
import math
from pathlib import Path

import numpy as np
from qtpy.QtCore import QPointF,QRectF,Qt,Signal
from qtpy.QtGui import QColor,QPainter,QPen,QPolygonF,QKeySequence,QShortcut
from qtpy.QtWidgets import (QAbstractItemView,QCheckBox,QComboBox,QDialog,QDialogButtonBox,
    QDoubleSpinBox,QFileDialog,QFormLayout,QGroupBox,QHBoxLayout,QHeaderView,QLabel,QLineEdit,
    QListWidget,QListWidgetItem,QMessageBox,QPushButton,QScrollArea,QSpinBox,QSplitter,
    QTabWidget,QTableWidget,QTableWidgetItem,QVBoxLayout,QWidget)

from ..probe_library import (ProbeLibrary,contacts_for,grid_geometry,validate_geometry,
                             contacts_csv,parse_contacts_csv,MAX_CONTACTS)


from .desktop_probe_canvas_base import note,button


class ProbesLibraryDialog(QDialog):
    libraryChanged=Signal()
    def __init__(self,parent=None,library=None):
        super().__init__(parent); self.library=library or ProbeLibrary(); self.setWindowTitle('Probes Library')
        self.resize(1120,800); self.setMinimumSize(920,620)
        root=QVBoxLayout(self); header=QHBoxLayout()
        title=QLabel('Probes Library'); title.setObjectName('heading'); header.addWidget(title,1)
        header.addWidget(button('Import…',self.import_probe)); header.addWidget(button('Create probe…',self.create_probe,True)); root.addLayout(header)
        root.addWidget(note('Reusable probe templates. Projects keep their own copy of the geometry.'))
        split=QSplitter(Qt.Orientation.Horizontal); root.addWidget(split,1)
        left=QWidget(); ll=QVBoxLayout(left); ll.setContentsMargins(0,0,8,0)
        self.search=QLineEdit(); self.search.setPlaceholderText('Search name or manufacturer…'); ll.addWidget(self.search)
        self.filter=QComboBox(); self.filter.addItems(['All probes','Built-in','My probes']); ll.addWidget(self.filter)
        self.list=QListWidget(); self.list.setSpacing(3); ll.addWidget(self.list,1); split.addWidget(left)
        right=QWidget(); rl=QVBoxLayout(right); rl.setContentsMargins(8,0,0,0)
        self.name=QLabel(); self.name.setStyleSheet('font-size:18px;font-weight:600;'); self.name.setWordWrap(True); rl.addWidget(self.name)
        self.summary=note(); rl.addWidget(self.summary)
        self.preview=ProbePreview(); rl.addWidget(self.preview,1)
        rl.addWidget(note('Scroll to zoom · Drag to pan · Fit contacts to inspect the recording array'))
        self.details=QLabel(); self.details.setWordWrap(True); self.details.setOpenExternalLinks(True); rl.addWidget(self.details)
        self.notes=note(); rl.addWidget(self.notes)
        row=QHBoxLayout(); self.edit_button=button('Edit…',self.edit_probe); self.duplicate_button=button('Duplicate…',self.duplicate_probe)
        self.export_button=button('Export…',self.export_probe); self.delete_button=button('Delete',self.delete_probe)
        for b in (self.edit_button,self.duplicate_button,self.export_button,self.delete_button): row.addWidget(b)
        row.addStretch(); rl.addLayout(row); split.addWidget(right); split.setSizes([280,800])
        self.message=note(); root.addWidget(self.message)
        bottom=QHBoxLayout(); self.location=note(str(self.library.path)); self.location.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        bottom.addWidget(self.location,1); bottom.addWidget(button('Close',self.accept)); root.addLayout(bottom)
        self.search.textChanged.connect(self.refresh); self.filter.currentIndexChanged.connect(self.refresh)
        self.list.currentRowChanged.connect(self.selection_changed); self.list.itemDoubleClicked.connect(lambda *_:self.duplicate_probe() if self.current().get('builtin') else self.edit_probe())
        self.refresh()

    def current(self):
        item=self.list.currentItem(); return item.data(Qt.ItemDataRole.UserRole) if item else None

    def refresh(self,*args,select_id=None):
        previous=self.current(); identifier=select_id or (previous or {}).get('id')
        self.list.blockSignals(True); self.list.clear(); query=self.search.text().casefold(); mode=self.filter.currentIndex()
        for entry in self.library.entries():
            if query not in (entry['name']+' '+entry.get('manufacturer','')).casefold(): continue
            if mode==1 and not entry.get('builtin') or mode==2 and entry.get('builtin'): continue
            item=QListWidgetItem(entry['name']+'\n'+entry.get('manufacturer','Custom')+(' · Built-in' if entry.get('builtin') else ' · My probe'))
            item.setData(Qt.ItemDataRole.UserRole,entry); self.list.addItem(item)
            if entry['id']==identifier: self.list.setCurrentItem(item)
        if self.list.currentRow()<0 and self.list.count(): self.list.setCurrentRow(0)
        self.list.blockSignals(False); self.selection_changed()
        self.message.setText('\n'.join(self.library.errors) if self.library.errors else ('No matching probes.' if not self.list.count() else ''))

    def selection_changed(self,*args):
        entry=self.current()
        for b in (self.edit_button,self.duplicate_button,self.export_button,self.delete_button): b.setEnabled(entry is not None)
        self.edit_button.setEnabled(bool(entry and not entry.get('builtin'))); self.delete_button.setEnabled(bool(entry and not entry.get('builtin')))
        if entry is None:
            self.name.setText('No probe selected'); self.summary.clear(); self.details.clear(); self.notes.clear(); self.preview.canvas.geometry=None; self.preview.canvas.update(); return
        g=entry['geometry']; contacts=contacts_for(g)
        self.name.setText(entry['name']); self.summary.setText(f'{entry.get("manufacturer","Custom")} · {g["shanks"]} shank'+('s' if g['shanks']!=1 else '')+f' · {len(contacts):,} physical sites')
        self.preview.canvas.set_geometry(g,reset=True)
        self.details.setText(f'Length {g["shank_length_um"]:g} µm   ·   Width {g["shank_width_um"]:g} µm   ·   Thickness {g["shank_thickness_um"]:g} µm\n'
            f'Shank pitch {g["shank_pitch_um"]:g} µm   ·   Tip {g["taper_length_um"]:g} µm   ·   Contacts {g["site_size_um"]:g} × {g.get("site_height_um",g["site_size_um"]):g} µm\n'
            f'First centre {min(c["y_um"] for c in contacts):g} µm from tip   ·   Nominal row pitch {g["site_pitch_um"]:g} µm   ·   Column pitch {g["column_pitch_um"]:g} µm')
        if g.get('geometry_version',1)>=2:
            self.details.setText(('Body unspecified · only contact coordinates are known.' if g.get('body_shape')=='none' else f'Body: {g.get("body_shape","tapered")} · Length {g["shank_length_um"]:g} µm · Width {g["shank_width_um"]:g} µm · Thickness {g["shank_thickness_um"]:g} µm')+'\nIndividual recording surfaces keep their own shape, dimensions and orientation.')
        import html
        sources=' · '.join(f'<a style="color:#7ac7e5" href="{html.escape(url,quote=True)}">Source {i+1}</a>' for i,url in enumerate(entry.get('sources',[])) if url.startswith('https://'))
        self.notes.setText(html.escape(entry.get('notes',''))+('<br>'+sources if sources else '')); self.notes.setOpenExternalLinks(True)

    def _editor(self,template=None):
        editor=ProbeEditor(self,template)
        while editor.exec()==QDialog.DialogCode.Accepted:
            try:
                saved=self.library.save(editor.result_probe)
                self.search.clear(); self.filter.setCurrentIndex(0)
                self.refresh(select_id=saved['id']); self.libraryChanged.emit(); return
            except (OSError,ValueError) as error:
                QMessageBox.warning(self,'Could not save probe',str(error))

    def create_probe(self): self._editor()
    def edit_probe(self):
        if self.current() and not self.current().get('builtin'): self._editor(self.current())
    def duplicate_probe(self):
        if not self.current(): return
        entry=copy.deepcopy(self.current()); entry.pop('id',None); entry['builtin']=False
        entry['geometry']['model']+=' · copy'; self._editor(entry)

    def import_probe(self):
        path,_=QFileDialog.getOpenFileName(self,'Import probe template','','ProbeMap probe (*.json)')
        if not path: return
        try:
            entry=self.library.read(path)
            existing=next((e for e in self.library.entries() if e['id']==entry['id']),None)
            if existing:
                entry.pop('id'); entry['geometry']['model']+=' · imported'
            saved=self.library.save(entry); self.search.clear(); self.filter.setCurrentIndex(0); self.refresh(select_id=saved['id']); self.libraryChanged.emit()
        except (OSError,ValueError,KeyError,TypeError) as error: QMessageBox.warning(self,'Could not import probe',str(error))

    def export_probe(self):
        entry=self.current()
        if not entry: return
        path,_=QFileDialog.getSaveFileName(self,'Export probe template','probe.json','ProbeMap probe (*.json)')
        if not path: return
        try:
            exported=copy.deepcopy(entry); exported.pop('id',None); exported['builtin']=False
            from uuid import uuid4
            exported['id']='custom-'+uuid4().hex
            self.library.export(exported,path)
        except (OSError,ValueError) as error: QMessageBox.warning(self,'Could not export probe',str(error))

    def delete_probe(self):
        entry=self.current()
        if not entry or entry.get('builtin'): return
        answer=QMessageBox.question(self,'Delete probe?',f'Remove “{entry["name"]}” from your library? Existing projects keep their saved geometry.',QMessageBox.StandardButton.Yes|QMessageBox.StandardButton.Cancel,QMessageBox.StandardButton.Cancel)
        if answer!=QMessageBox.StandardButton.Yes: return
        try: self.library.delete(entry); self.refresh(); self.libraryChanged.emit()
        except OSError as error: QMessageBox.warning(self,'Could not delete probe',str(error))


# Import after the reusable canvas helpers: the designer shares their conventions.
from .desktop_probe_designer import ProbeDesigner as ProbeEditor, DesignerPreview as ProbePreview

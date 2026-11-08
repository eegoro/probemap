"""Editable ruler presentation; physical lengths are always stored in micrometres."""
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (QAbstractItemView, QColorDialog, QHBoxLayout, QHeaderView,
    QLabel, QMenu, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from .desktop_tables import CheckHeader, ColorDotDelegate
from .desktop_widgets import PopupComboBox, view_button

DISTANCE_UNITS={'µm':1.,'mm':1000.,'nm':.001,'cm':10000.}


def distance_text(length_um,unit='µm'):
    divisor=DISTANCE_UNITS.get(unit,1.)
    value=length_um/divisor
    return f'{value:.3f} {unit}' if unit in ('mm','cm') else f'{value:.1f} {unit}'


class MeasurementControlsMixin:
    def _measurement_controls(self):
        panel=QWidget(); layout=QVBoxLayout(panel); layout.setContentsMargins(0,0,0,0)
        row=QHBoxLayout(); row.addWidget(QLabel('Measurements'),1)
        self.measurement_unit=PopupComboBox()
        self.measurement_unit.addItems(DISTANCE_UNITS)
        self.measurement_unit.setToolTip('Measurement display units')
        self.measurement_unit.setFixedWidth(66)
        self.measurement_unit.currentTextChanged.connect(self._measurement_units_changed)
        row.addWidget(self.measurement_unit)
        self.clear_measurements_button=view_button('clear','Delete all measurements',self.clear_measurements)
        row.addWidget(self.clear_measurements_button); layout.addLayout(row)
        self.measurement_table=QTableWidget(0,4)
        table=self.measurement_table
        table.setHorizontalHeaderLabels(['Name','Length','Label','Color'])
        table.verticalHeader().hide(); table.verticalHeader().setDefaultSectionSize(28)
        table.setShowGrid(False); table.setMinimumHeight(85)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        table.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked|QAbstractItemView.EditTrigger.EditKeyPressed)
        table.setItemDelegateForColumn(3,ColorDotDelegate(table))
        header=CheckHeader(table,(0,2)); table.setHorizontalHeader(header)
        header.toggled.connect(self._measurement_header_toggled)
        for i,width in enumerate((80,75,65,38)): table.setColumnWidth(i,width)
        table.itemChanged.connect(self._measurement_edited)
        table.cellClicked.connect(self._measurement_clicked)
        table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        table.customContextMenuRequested.connect(self._measurement_menu)
        layout.addWidget(table,1)
        return panel

    def _refresh_measurements(self):
        if self.session is None: return
        table=self.measurement_table
        table.blockSignals(True)
        records=self.session.get('measurements',[])
        table.setRowCount(len(records))
        table.horizontalHeaderItem(1).setText('Length')
        for row,record in enumerate(records):
            name=QTableWidgetItem(record.setdefault('name',f'M{row+1}'))
            name.setData(Qt.ItemDataRole.UserRole,record['id'])
            name.setFlags(name.flags()|Qt.ItemFlag.ItemIsUserCheckable)
            name.setCheckState(Qt.CheckState.Checked if record.get('visible',True) else Qt.CheckState.Unchecked)
            table.setItem(row,0,name)
            value=distance_text(record['length_um'],self.measurement_unit.currentText()).rsplit(' ',1)[0]
            length=QTableWidgetItem(value)
            length.setFlags(Qt.ItemFlag.ItemIsEnabled|Qt.ItemFlag.ItemIsSelectable)
            table.setItem(row,1,length)
            label=QTableWidgetItem()
            label.setFlags(Qt.ItemFlag.ItemIsEnabled|Qt.ItemFlag.ItemIsUserCheckable|Qt.ItemFlag.ItemIsSelectable)
            label.setCheckState(Qt.CheckState.Checked if record.get('show_label',True) else Qt.CheckState.Unchecked)
            table.setItem(row,2,label)
            color=QTableWidgetItem()
            color.setFlags(Qt.ItemFlag.ItemIsEnabled|Qt.ItemFlag.ItemIsSelectable)
            color.setData(Qt.ItemDataRole.UserRole,record.get('color','#79e7ff'))
            table.setItem(row,3,color)
        table.blockSignals(False)
        table.horizontalHeader().viewport().update()

    def _measurement_for_row(self,row):
        if row<0 or self.session is None: return None
        cell=self.measurement_table.item(row,0)
        return next((m for m in self.session.get('measurements',[]) if cell and m['id']==cell.data(Qt.ItemDataRole.UserRole)),None)

    def _measurement_edited(self,cell):
        if self._updating: return
        record=self._measurement_for_row(cell.row())
        if record is None: return
        self._remember()
        if cell.column()==0:
            record['name']=cell.text().strip() or f'M{cell.row()+1}'
            record['visible']=cell.checkState()==Qt.CheckState.Checked
        elif cell.column()==2: record['show_label']=cell.checkState()==Qt.CheckState.Checked
        self.measurement_table.horizontalHeader().viewport().update()
        self.draw_cached(annotations_only=True)

    def _measurement_clicked(self,row,column):
        record=self._measurement_for_row(row)
        if column!=3 or record is None: return
        color=QColorDialog.getColor(QColor(record.get('color','#79e7ff')),self,'Measurement color')
        if color.isValid():
            self._remember(); record['color']=color.name()
            self._refresh_measurements(); self.draw_cached(annotations_only=True)

    def _measurement_header_toggled(self,column,visible):
        if self.session is None: return
        self._remember()
        key='visible' if column==0 else 'show_label'
        for record in self.session.get('measurements',[]): record[key]=visible
        self._refresh_measurements(); self.draw_cached(annotations_only=True)

    def _measurement_units_changed(self,*args):
        if self.session is None or self._updating: return
        self._refresh_measurements(); self.draw_cached(annotations_only=True)
        self.owner.schedule_save()

    def measurement_context_menu(self,identifier):
        menu=QMenu(self)
        menu.addAction('Delete measurement',lambda:self.delete_measurement_id(identifier))
        return menu

    def _measurement_menu(self,position):
        record=self._measurement_for_row(self.measurement_table.rowAt(position.y()))
        if record is not None:
            self.measurement_context_menu(record['id']).exec(self.measurement_table.viewport().mapToGlobal(position))

    def delete_measurement_id(self,identifier):
        if self.session is None: return
        self._remember()
        self.session['measurements']=[m for m in self.session.get('measurements',[]) if m['id']!=identifier]
        self._refresh_measurements(); self.draw_cached(annotations_only=True)

    def clear_measurements(self):
        if self.session is None or not self.session.get('measurements'): return
        self._remember(); self.session['measurements']=[]
        self._refresh_measurements(); self.draw_cached(annotations_only=True)

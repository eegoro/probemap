"""Compact insertion editing and the unscaled scan frame used for probe tracing."""
import numpy as np

from qtpy.QtCore import Qt
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (QAbstractButton, QAbstractItemView, QAbstractSlider,
    QAbstractSpinBox, QComboBox, QFormLayout, QHBoxLayout, QHeaderView,
    QLabel, QMenu, QSlider, QStackedWidget, QTabBar,
    QSplitter, QTableWidget, QTableWidgetItem, QToolButton, QVBoxLayout, QWidget)

from .desktop_slice_tools import tool_icon
from .desktop_tables import ColorDotDelegate
from .desktop_widgets import PopupComboBox, view_button
from .landmarks import transform_points
from .review import placement_without_scale


class WorkflowControls(QStackedWidget):
    """An unrelated later step must not force the current sidebar to scroll."""
    def sizeHint(self):
        return self.currentWidget().sizeHint() if self.currentWidget() is not None else super().sizeHint()

    def minimumSizeHint(self):
        return self.currentWidget().minimumSizeHint() if self.currentWidget() is not None else super().minimumSizeHint()


class ProbeTracingMixin:
    def _tracing_controls(self,layout):
        self.annotation_splitter=QSplitter(Qt.Orientation.Vertical)
        self.annotation_splitter.setChildrenCollapsible(False)
        panel=QWidget(); top=QVBoxLayout(panel); top.setContentsMargins(0,0,0,0)
        layout.addWidget(self.annotation_splitter,1)
        self.annotation_splitter.addWidget(panel)
        header=QHBoxLayout()
        header.addWidget(QLabel('Insertions'),1)
        self.add_insertion_button=view_button('add','New insertion',self.create_insertion)
        header.addWidget(self.add_insertion_button)
        self.clear_points_button=view_button('clear','Delete all insertion points',self.clear_points)
        header.addWidget(self.clear_points_button)
        top.addLayout(header)
        self.insertion_table=QTableWidget(0,3)
        table=self.insertion_table
        table.setHorizontalHeaderLabels(['Insertion','Color','Points'])
        table.verticalHeader().hide()
        table.verticalHeader().setDefaultSectionSize(32)
        table.setShowGrid(False)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        table.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked|QAbstractItemView.EditTrigger.EditKeyPressed)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        table.horizontalHeader().setStretchLastSection(True)
        for col,width in enumerate((145,60,55)): table.setColumnWidth(col,width)
        table.setMinimumHeight(70)
        table.setItemDelegateForColumn(1,ColorDotDelegate(table))
        table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        table.customContextMenuRequested.connect(self._insertion_menu)
        table.itemChanged.connect(self._insertion_edited)
        table.itemSelectionChanged.connect(self._insertion_row_selected)
        table.cellClicked.connect(self._insertion_cell_clicked)
        top.addWidget(table,1)
        form=QFormLayout()
        row=QHBoxLayout()
        self.point_size=QSlider(Qt.Orientation.Horizontal)
        self.point_size.setRange(4,24); self.point_size.setValue(8)
        self.point_size.setToolTip('Point diameter in screen pixels')
        self.point_size.valueChanged.connect(self._point_size_changed)
        self.point_size_value=QLabel('8')
        self.point_size_value.setFixedWidth(25)
        row.addWidget(self.point_size,1); row.addWidget(self.point_size_value)
        form.addRow('Point size',row)
        top.insertLayout(1,form)
        self.move_target=PopupComboBox()
        self.move_target.setParent(panel); self.move_target.hide()
        self.measurement_panel=self._measurement_controls()
        self.annotation_splitter.addWidget(self.measurement_panel)
        self.annotation_splitter.setStretchFactor(0,0)
        self.annotation_splitter.setStretchFactor(1,1)
        self.annotation_splitter.setSizes([260,180])

    def _build_tracing_toolbar(self):
        self.tracing_toolbar=QWidget()
        row=QHBoxLayout(self.tracing_toolbar)
        row.setContentsMargins(0,0,0,0); row.setSpacing(4)
        self.tracing_tools={}
        for name,title in (('add','Add points'),('select','Select points · Drag 3D background: rotate · Ctrl/⌘+drag: select a rectangle'),('measure','Measure distance')):
            button=QToolButton()
            button.setIcon(tool_icon(name)); button.setCheckable(True)
            button.setFixedSize(32,32)
            button.setToolTip(title); button.setAccessibleName(title)
            button.clicked.connect(lambda checked=False,name=name:self._tracing_tool_selected(name,checked))
            row.addWidget(button); self.tracing_tools[name]=button
        self.tracing_toolbar.setStyleSheet('''QToolButton {background:transparent;border:0;border-radius:5px;}
            QToolButton:hover {background:rgba(122,199,229,28);}
            QToolButton:checked {background:rgba(51,141,177,75);border:1px solid #7ac7e5;}''')

    def _tracing_tool_selected(self,name,checked=True):
        selected=name if checked else 'pan'
        if selected=='add' and self.current_insertion() is None:
            if not self.session['insertions']: self.create_insertion()
            else:
                self.insertion_id=self.session['insertions'][0]['id']
                self._refresh_insertions()
        self._canvas_tools=[selected]*3
        self._canvas_tool_modes['0']=list(self._canvas_tools)
        self._tool_modes['0']={'add':0,'select':1,'pan':2,'measure':4}[selected]
        self.point_tool.blockSignals(True)
        self.point_tool.setCurrentIndex(self._tool_modes['0'])
        self.point_tool.blockSignals(False)
        self._apply_slice_tools()
        self.owner.schedule_save()

    def _refresh_insertion_table(self):
        if self.session is None: return
        table=self.insertion_table
        widths=[table.columnWidth(i) for i in range(3)]
        table.blockSignals(True)
        table.setRowCount(len(self.session['insertions']))
        for row,item in enumerate(self.session['insertions']):
            name=QTableWidgetItem(item['name'])
            name.setData(Qt.ItemDataRole.UserRole,item['id'])
            name.setFlags(name.flags()|Qt.ItemFlag.ItemIsUserCheckable)
            name.setCheckState(Qt.CheckState.Checked if item.get('points_visible',item.get('visible',True)) else Qt.CheckState.Unchecked)
            name.setToolTip('Double-click to rename · Check to show points')
            table.setItem(row,0,name)
            color=QTableWidgetItem()
            color.setData(Qt.ItemDataRole.UserRole,item['color'])
            color.setForeground(QColor(item['color']))
            color.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            color.setFlags(Qt.ItemFlag.ItemIsEnabled|Qt.ItemFlag.ItemIsSelectable)
            color.setToolTip('Change insertion color')
            table.setItem(row,1,color)
            count=QTableWidgetItem(str(len(item['points'])))
            count.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            count.setFlags(Qt.ItemFlag.ItemIsEnabled|Qt.ItemFlag.ItemIsSelectable)
            table.setItem(row,2,count)
            if item['id']==self.insertion_id: table.selectRow(row)
        for i,width in enumerate(widths): table.setColumnWidth(i,width)
        table.blockSignals(False)

    def _insertion_row_selected(self):
        if self._updating or self.session is None: return
        rows=self.insertion_table.selectionModel().selectedRows()
        if rows:
            identifier=self.insertion_table.item(rows[0].row(),0).data(Qt.ItemDataRole.UserRole)
            self.insertion_combo.setCurrentIndex(self.insertion_combo.findData(identifier))
            if self._mode==0: self._tracing_tool_selected('add')
        elif self._mode==0:
            self.insertion_id=None
            self.insertion_combo.blockSignals(True); self.insertion_combo.setCurrentIndex(-1); self.insertion_combo.blockSignals(False)
            self._tracing_tool_selected('select')

    def _insertion_cell_clicked(self,row,column):
        if column==1: self.color_insertion()

    def _insertion_edited(self,cell):
        if self._updating or cell.column()!=0 or self.session is None: return
        identifier=cell.data(Qt.ItemDataRole.UserRole)
        item=next(item for item in self.session['insertions'] if item['id']==identifier)
        name=cell.text().strip()
        visible=cell.checkState()==Qt.CheckState.Checked
        if name!=item['name'] and not self._valid_insertion_name(name,identifier):
            self._refresh_insertion_table(); return
        if name==item['name'] and visible==item.get('points_visible',item.get('visible',True)): return
        self._remember()
        item['name']=name
        item.setdefault('tracks_visible',item.get('visible',True))
        item['points_visible']=visible
        item['visible']=visible or item['tracks_visible']
        from .insertion_model import sync_tracks
        sync_tracks(self.session,self.project)
        self._refresh_insertions()
        self._refresh_tracks()

    def _insertion_menu(self,position):
        row=self.insertion_table.rowAt(position.y())
        if row<0: return
        self.insertion_table.selectRow(row)
        menu=QMenu(self)
        menu.addAction('Rename',lambda:self.insertion_table.editItem(self.insertion_table.item(row,0)))
        menu.addAction('Color…',self.color_insertion)
        if self.selected_points: menu.addAction('Move selected points here',self._move_points_to_active)
        menu.addSeparator()
        menu.addAction('Delete insertion',self.delete_tracing_insertion)
        menu.exec(self.insertion_table.viewport().mapToGlobal(position))

    def _move_points_to_active(self):
        self.move_target.setCurrentIndex(self.move_target.findData(self.insertion_id))
        self.move_selected()

    def delete_tracing_insertion(self):
        item=self.current_insertion()
        if item is None: return
        self._remember()
        self.selected_points.difference_update(p['id'] for p in item['points'])
        self.session.setdefault('unassigned_recordings',[]).extend(item.get('recordings',[]))
        self.session['insertions'].remove(item)
        self.insertion_id=self.session['insertions'][0]['id'] if self.session['insertions'] else None
        self.selected_id=None
        self._refresh_insertions(); self._refresh_tracks()

    def _point_size_changed(self,value):
        self.point_size_value.setText(str(value))
        self.draw_cached(annotations_only=True)
        if not self._updating: self.owner.schedule_save()

    def keep_insertion_for_click(self,widget):
        if self._mode!=0 or self.project is None: return False
        if self._canvas_tools[0]!='add': return False
        # Browsing slices, orbiting 3D and editing display controls do not end
        # point placement. A click on empty sidebar/title space still deselects.
        for surface in (self.view_container,self.tracing_toolbar):
            if surface is widget or surface.isAncestorOf(widget): return True
        target=widget
        while target is not None and target is not self:
            if isinstance(target,(QAbstractButton,QAbstractSlider,QAbstractSpinBox,QComboBox,QTabBar)):
                return True
            target=target.parentWidget()
        return False

    def clear_points(self):
        if self.session is None or not any(i['points'] for i in self.session['insertions']): return
        self._remember()
        for item in self.session['insertions']: item['points']=[]
        self.selected_points.clear(); self._refresh_insertions()

    def point_context_menu(self,identifier):
        if identifier not in self.selected_points: self.select_points([identifier])
        menu=QMenu(self)
        menu.addAction('Center slices',self.go_to_point)
        targets=menu.addMenu('Move to insertion')
        for item in self.session['insertions']:
            targets.addAction(item['name'],lambda checked=False,target=item['id']:self._move_points_to(target))
        menu.addAction('New insertion from selection',self.group_selected)
        menu.addSeparator(); menu.addAction('Delete points',self.delete_points)
        return menu

    def _move_points_to(self,target):
        self.move_target.setCurrentIndex(self.move_target.findData(target)); self.move_selected()

    def _annotation_context(self,kind,identifier,position):
        if self.session is None: return
        menu=self.point_context_menu(identifier) if kind=='point' else self.measurement_context_menu(identifier)
        menu.exec(position)

    def _tracing_matrix(self):
        return placement_without_scale(self.project)

    def _tracing_to_atlas(self):
        return self.project.total@np.linalg.inv(self._tracing_matrix())

    def _to_atlas(self,points):
        return transform_points(np.asarray(points,float).reshape(-1,3),self._tracing_to_atlas()) if self._mode==0 else np.asarray(points,float)

    def _slice_positions(self):
        return self._tracing_positions if self._mode==0 else self._ortho_positions

    def apply_tracing(self):
        if self.session is None: return
        self.compute_planes()
        self.capture_state()
        self.owner.schedule_save()


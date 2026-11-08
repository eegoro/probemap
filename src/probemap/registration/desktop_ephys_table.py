"""Inline editing for the independent electrophysiology recording library."""
from datetime import datetime
from pathlib import Path

from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import QMenu, QStyledItemDelegate

from .desktop_widgets import PopupComboBox


class EphysInsertionDelegate(QStyledItemDelegate):
    def __init__(self, workspace, table):
        super().__init__(table); self.workspace = workspace

    def createEditor(self, parent, option, index):
        editor = PopupComboBox(parent); editor.addItem('—', None)
        for item in self.workspace._items(): editor.addItem(item['name'], item['id'])
        editor.activated.connect(lambda: self._commit(editor))
        return editor

    def _commit(self, editor):
        self.commitData.emit(editor); self.closeEditor.emit(editor)

    def setEditorData(self, editor, index):
        editor.setCurrentIndex(max(0, editor.findData(index.data(Qt.ItemDataRole.UserRole))))

    def setModelData(self, editor, model, index):
        model.setItemData(index, {Qt.ItemDataRole.UserRole: editor.currentData(),
                                  Qt.ItemDataRole.DisplayRole: editor.currentText()})


class SortingCombo(PopupComboBox):
    def __init__(self, workspace, record, parent=None):
        super().__init__(parent); self.workspace = workspace; self.record_id = record['id']
        self.addItem('—', None)
        for source in record['sortings']:
            name = source['name']
            if sum(s['name'] == name for s in record['sortings']) > 1:
                name = f"{Path(source['path']).parent.name}/{name}"
            self.addItem(name, source['id'])
            self.setItemData(self.count()-1, source['path'], Qt.ItemDataRole.ToolTipRole)
        self.addItem('Choose folder…', '__add__')
        self.setCurrentIndex(max(0, self.findData(record.get('selected_sorting'))))
        self.setToolTip(next((s['path'] for s in record['sortings'] if s['id'] == record.get('selected_sorting')), 'Choose or connect Kilosort results'))
        self.setStyleSheet('QComboBox{background:transparent;border:0;border-radius:0;padding:2px 4px;}')
        self.activated.connect(self._activate)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(lambda pos: self._menu(self.currentIndex(), self.mapToGlobal(pos)))
        self.view().setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.view().customContextMenuRequested.connect(self._popup_menu)

    def _activate(self, index):
        if self.workspace._busy: return
        value = self.itemData(index)
        if value == '__add__':
            record = self.workspace.record_by_id(self.record_id)
            self.blockSignals(True)
            self.setCurrentIndex(max(0, self.findData(record.get('selected_sorting'))))
            self.blockSignals(False)
            self.workspace.attach_sorting(record_id=self.record_id)
        else: self.workspace.switch_sorting(self.record_id, value)

    def _popup_menu(self, position):
        index = self.view().indexAt(position)
        if index.isValid(): self._menu(index.row(), self.view().viewport().mapToGlobal(position))

    def _menu(self, index, position):
        value = self.itemData(index)
        if value in (None, '__add__') or self.workspace._busy: return
        menu = QMenu(self)
        workspace, record_id = self.workspace, self.record_id
        def remove():
            self.hidePopup()
            QTimer.singleShot(0, lambda: workspace.remove_sorting(record_id, value))
        menu.addAction('Remove path', remove)
        menu.exec(position)

    def _remove(self, value):
        self.hidePopup()
        self.workspace.remove_sorting(self.record_id, value)


def recording_time(record):
    try:
        return datetime.fromisoformat(record.get('recorded_at', '').replace('Z', '+00:00')).strftime('%Y-%m-%d %H:%M:%S')
    except ValueError: return '—'


class TimestampDelegate(QStyledItemDelegate):
    """Keep full timestamps legible in a narrow table without changing edits."""
    def paint(self, painter, option, index):
        value = str(index.data(Qt.ItemDataRole.DisplayRole) or '')
        try: stamp = datetime.fromisoformat(value.replace('Z', '+00:00'))
        except ValueError:
            super().paint(painter, option, index); return
        from qtpy.QtGui import QColor
        from qtpy.QtWidgets import QStyle, QStyleOptionViewItem
        background = QStyleOptionViewItem(option); self.initStyleOption(background, index); background.text = ''
        from qtpy.QtWidgets import QApplication
        style = option.widget.style() if option.widget is not None else QApplication.style()
        style.drawControl(QStyle.ControlElement.CE_ItemViewItem, background, painter, option.widget)
        painter.save()
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        painter.setPen(QColor('#e5eaf2'))
        rect = option.rect.adjusted(7, 3, -4, -3)
        painter.drawText(rect.adjusted(0, 0, 0, -rect.height()//2), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, stamp.strftime('%H:%M:%S'))
        font = painter.font(); font.setPointSizeF(max(8, font.pointSizeF()-1)); painter.setFont(font)
        painter.setPen(QColor('#c6d9e7' if selected else '#9aafc1'))
        painter.drawText(rect.adjusted(0, rect.height()//2, 0, 0), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, stamp.strftime('%Y-%m-%d'))
        painter.restore()

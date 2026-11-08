"""Quiet, reusable table interactions for the desktop workspaces."""
from qtpy.QtCore import QEvent, QObject, QRect, QRectF, Qt, Signal
from qtpy.QtGui import QColor, QPainter
from qtpy.QtWidgets import (QApplication, QHeaderView, QMenu, QStyle, QStyledItemDelegate,
    QStyleOptionButton, QStyleOptionViewItem, QTableWidget, QWidget)


class ColorDotDelegate(QStyledItemDelegate):
    def paint(self,painter,option,index):
        background=QStyleOptionViewItem(option)
        self.initStyleOption(background,index)
        background.text=''
        super().paint(painter,background,index)
        value=index.data(Qt.ItemDataRole.UserRole)
        if not value: return
        painter.save(); painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(*value) if isinstance(value,(list,tuple)) else QColor(value))
        center=option.rect.center()
        painter.drawEllipse(QRectF(center.x()-5,center.y()-5,10,10))
        painter.restore()


class CheckHeader(QHeaderView):
    toggled=Signal(int,bool)

    def __init__(self,table,columns):
        super().__init__(Qt.Orientation.Horizontal,table)
        self.table=table
        self.columns=set(columns)
        self.setSectionsClickable(True)
        self.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)

    def state(self,column):
        states=[self.table.item(row,column).checkState() for row in range(self.table.rowCount())
                if not self.table.isRowHidden(row) and self.table.item(row,column) is not None]
        if states and all(state==Qt.CheckState.Checked for state in states): return Qt.CheckState.Checked
        if any(state!=Qt.CheckState.Unchecked for state in states): return Qt.CheckState.PartiallyChecked
        return Qt.CheckState.Unchecked

    def _check_rect(self,section):
        return QRect(self.sectionViewportPosition(section)+7,(self.height()-14)//2,14,14)

    def paintSection(self,painter,rect,section):
        if section not in self.columns:
            super().paintSection(painter,rect,section); return
        painter.save()
        painter.fillRect(rect,QColor('#242b36'))
        box=QStyleOptionButton()
        box.rect=self._check_rect(section)
        state=self.state(section)
        box.state=QStyle.StateFlag.State_Enabled|(
            QStyle.StateFlag.State_On if state==Qt.CheckState.Checked else
            QStyle.StateFlag.State_NoChange if state==Qt.CheckState.PartiallyChecked else QStyle.StateFlag.State_Off)
        self.style().drawControl(QStyle.ControlElement.CE_CheckBox,box,painter,self)
        painter.setPen(QColor('#9aabc0'))
        label=self.model().headerData(section,self.orientation(),Qt.ItemDataRole.DisplayRole) or ''
        painter.drawText(rect.adjusted(27,0,-3,0),Qt.AlignmentFlag.AlignVCenter,str(label))
        painter.restore()

    def mouseReleaseEvent(self,event):
        section=self.logicalIndexAt(event.position().toPoint())
        if section in self.columns and self._check_rect(section).adjusted(-3,-3,3,3).contains(event.position().toPoint()):
            self.toggled.emit(section,self.state(section)!=Qt.CheckState.Checked)
            self.viewport().update(); event.accept(); return
        super().mouseReleaseEvent(event)


class TableHeightGrip(QWidget):
    def __init__(self,table):
        super().__init__()
        self.table=table; self._start=None
        self.setFixedHeight(9); self.setCursor(Qt.CursorShape.SizeVerCursor)
        self.setToolTip('Drag to resize table')

    def paintEvent(self,event):
        painter=QPainter(self); painter.setPen(QColor('#526173'))
        painter.drawLine(self.width()//2-14,4,self.width()//2+14,4)

    def mousePressEvent(self,event):
        self._start=(event.globalPosition().y(),self.table.height())

    def mouseMoveEvent(self,event):
        if self._start is not None:
            self.table.setFixedHeight(max(75,int(self._start[1]+event.globalPosition().y()-self._start[0])))

    def mouseReleaseEvent(self,event): self._start=None


class TableSelectionGuard(QObject):
    """Clicking another surface clears table selection without swallowing its click."""
    def __init__(self,window):
        super().__init__(window)
        self.window=window
        QApplication.instance().installEventFilter(self)

    def eventFilter(self,watched,event):
        if event.type()!=QEvent.Type.MouseButtonPress or not isinstance(watched,QWidget): return False
        if event.button()!=Qt.MouseButton.LeftButton: return False
        if watched.window() is not self.window and not self.window.isAncestorOf(watched): return False
        if isinstance(QApplication.activePopupWidget(),QMenu): return False
        target=watched
        table=None
        while target is not None:
            if isinstance(target,QTableWidget): table=target; break
            target=target.parentWidget()
        workspace=getattr(self.window,'track_workspace',None)
        for candidate in watched.window().findChildren(QTableWidget):
            if candidate is table or not candidate.isVisible() or candidate.property("persistent_selection"): continue
            # Navigation and controls keep the active drawing destination.
            if workspace is not None and candidate is getattr(workspace,'insertion_table',None):
                if table is None and workspace.keep_insertion_for_click(watched): continue
            # Probe manipulation on the image keeps its table row; clicking
            # empty image space clears it through the canvas selection signal.
            if workspace is not None and candidate is getattr(workspace,'placement_table',None):
                canvas=getattr(workspace,'track_slice',None)
                if workspace._mode==1 and canvas is not None and (watched is canvas or canvas.isAncestorOf(watched)): continue
                toolbar=getattr(workspace,'placement_toolbar',None)
                if workspace._mode==1 and toolbar is not None and toolbar.isAncestorOf(watched): continue
            if candidate.selectionModel().hasSelection() or candidate.currentItem() is not None:
                candidate.clearSelection()
                candidate.setCurrentItem(None)
                candidate.selectionModel().clearCurrentIndex()
        return False

"""Probe placement gestures layered over the shared slice navigation."""
import numpy as np
from qtpy.QtCore import QPoint,QPointF,QRectF,Qt,Signal
from qtpy.QtGui import QPainter, QColor, QPen

from .desktop_insertion_views import InsertionSliceCanvas


class ProbePlacementSlice(InsertionSliceCanvas):
    probePreview=Signal(object,object)
    probePlaced=Signal(object,object)
    probeMoved=Signal(object,bool)
    probeRotated=Signal(object,object,bool)
    probeSelected=Signal(bool)
    probeDeleted=Signal()
    probePreviewLeft=Signal()
    probeCancelled=Signal()

    def __init__(self):
        super().__init__()
        self.placement_enabled=False
        self.placement_read_only=False
        self.placement_tool='select'
        self.placement_selected=False
        self._probe_anchor=None
        self._probe_moving=False
        self._probe_rotating=False
        self.placement_pivot=None
        self.placement_base=None
        self.placement_contacts=[]
        self.placement_contact_front=True

    def rotation_handle(self):
        if not self.placement_selected or self.image is None or self.placement_pivot is None: return None
        pivot=self._pixel(self.placement_pivot)
        base=self._pixel(self.placement_base)-pivot
        length=max(1.,np.hypot(base.x(),base.y()))
        direction=QPointF(base.x()/length,base.y()/length)
        return pivot+direction*65+QPointF(-direction.y(),direction.x())*45

    def _hit_probe(self,position):
        if self.image is None: return False
        for site in self.placement_contacts:
            if (self._pixel(site)-position).manhattanLength()<10: return True
        for poly,_ in self.probe_polygons:
            from qtpy.QtGui import QPolygonF
            if QPolygonF([self._pixel(p) for p in poly]).containsPoint(position,Qt.FillRule.OddEvenFill): return True
            # Narrow shanks must remain selectable when zoomed out.
            a,b=self._pixel(poly[0]),self._pixel((np.asarray(poly[2])+poly[3])/2)
            a,b=np.array([a.x(),a.y()]),np.array([b.x(),b.y()])
            pos=np.array([position.x(),position.y()]); d=b-a
            t=np.clip(np.dot(pos-a,d)/max(np.dot(d,d),1e-12),0,1)
            if np.linalg.norm(pos-a-t*d)<9: return True
        return False

    def mousePressEvent(self,event):
        super().mousePressEvent(event)
        if not self.placement_enabled or self.placement_read_only: return
        if self.placement_tool=='measure':
            if event.button()==Qt.MouseButton.LeftButton: self.probeSelected.emit(False)
            return
        if self._panning or self._zooming: return
        if event.button()!=Qt.MouseButton.LeftButton or not self.image_rect().contains(event.position()): return
        self._probe_anchor=self._world(event.position())
        if self.placement_tool=='draw': self.probePreview.emit(self._probe_anchor,None)
        else:
            handle=self.rotation_handle()
            if handle is not None and (event.position()-handle).manhattanLength()<18:
                self._probe_rotating=True
                self.probeSelected.emit(True)
                return
            self._probe_moving=self._hit_probe(event.position())
            self.probeSelected.emit(self._probe_moving)
            if not self._probe_moving:
                self._probe_anchor=None
                self._panning=True

    def mouseMoveEvent(self,event):
        if self.placement_enabled and not self.placement_read_only and not self._panning and not self._zooming and self.image is not None:
            world=self._world(event.position())
            if self.placement_tool=='draw' and self.image_rect().contains(event.position()):
                self.probePreview.emit(self._probe_anchor if self._probe_anchor is not None else world,
                                       world if self._probe_anchor is not None else None)
            elif self._probe_rotating and self._probe_anchor is not None:
                self.probeRotated.emit(self._probe_anchor,world,False)
            elif self._probe_moving and self._probe_anchor is not None:
                self.probeMoved.emit(world-self._probe_anchor,False)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self,event):
        if self.placement_enabled and not self.placement_read_only and self._probe_anchor is not None and event.button()==Qt.MouseButton.LeftButton:
            world=self._world(event.position())
            if self.placement_tool=='draw':
                direction=world if (event.position()-self._start_pixel).manhattanLength()>=5 else None
                self.probePlaced.emit(self._probe_anchor,direction)
            elif self._probe_rotating: self.probeRotated.emit(self._probe_anchor,world,True)
            elif self._probe_moving: self.probeMoved.emit(world-self._probe_anchor,True)
            self._probe_anchor=None; self._probe_moving=self._probe_rotating=False
        super().mouseReleaseEvent(event)

    def leaveEvent(self,event):
        if self.placement_enabled and not self.placement_read_only and self._probe_anchor is None: self.probePreviewLeft.emit()
        super().leaveEvent(event)

    def keyPressEvent(self,event):
        if self.placement_enabled and not self.placement_read_only and event.key()==Qt.Key.Key_Escape:
            self._probe_anchor=None; self._probe_moving=self._probe_rotating=False
            self.probeCancelled.emit()
            event.accept(); return
        if self.placement_enabled and not self.placement_read_only and self.placement_selected and event.key() in (Qt.Key.Key_Delete,Qt.Key.Key_Backspace):
            self.probeDeleted.emit(); event.accept(); return
        super().keyPressEvent(event)

    def paintEvent(self,event):
        super().paintEvent(event)
        if not self.placement_enabled or self.image is None: return
        painter=QPainter(self); painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setClipRect(self.rect())
        if self.probe_polygons:
            painter.setPen(QPen(QColor('#d5f5fc'),1))
            painter.setBrush(QColor('#d5f5fc') if self.placement_contact_front else Qt.BrushStyle.NoBrush)
            for site in self.placement_contacts:
                painter.drawEllipse(self._pixel(site),1.5,1.5)
            text='Contacts · front' if self.placement_contact_front else 'Contacts · back'
            bounds=painter.fontMetrics().boundingRect(text).adjusted(-6,-3,6,3)
            bounds.moveTopLeft(self.rect().topLeft()+QPoint(10,10))
            painter.fillRect(bounds,QColor(0,0,0,140)); painter.drawText(bounds,Qt.AlignmentFlag.AlignCenter,text)
        if not self.placement_selected: return
        painter.setPen(QPen(QColor('#e8f8ff'),1.5)); painter.setBrush(Qt.BrushStyle.NoBrush)
        for shank,(poly,_) in enumerate(self.probe_polygons):
            painter.drawEllipse(self._pixel(poly[0]),4,4)
            painter.drawText(self._pixel(poly[0])+QPointF(7,-7),str(shank))
        handle=self.rotation_handle()
        if handle is not None:
            painter.setPen(QPen(QColor('#9ac6d8'),1,Qt.PenStyle.DashLine))
            painter.drawLine(self._pixel(self.placement_pivot),handle)
            painter.setPen(QPen(QColor('#7ac7e5'),1.5)); painter.setBrush(QColor('#1d3a48'))
            painter.drawEllipse(handle,12,12)
            from .desktop_slice_tools import tool_icon
            tool_icon('rotate').paint(painter,QRectF(handle.x()-8,handle.y()-8,16,16).toRect())

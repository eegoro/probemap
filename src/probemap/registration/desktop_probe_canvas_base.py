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


def note(text=''):
    label=QLabel(text); label.setWordWrap(True); label.setObjectName('muted'); return label


def button(text,callback,primary=False):
    b=QPushButton(text); b.clicked.connect(callback)
    if primary: b.setObjectName('primary')
    return b


def contour(g,shank=0):
    x=shank*g['shank_pitch_um']; w=g['shank_width_um']/2; length=g['shank_length_um']
    if g.get('body_shape') in ('rectangle','wire'):
        return [(x-w,0),(x-w,length),(x+w,length),(x+w,0)]
    return [(x,0),(x-w,g['taper_length_um']),(x-w,length),(x+w,length),(x+w,g['taper_length_um'])]


class ProbeDrawing(QWidget):
    """Metric projections; selection and drag are always in physical units."""
    selected=Signal(int)
    addContact=Signal(object)
    moveContact=Signal(int,object,bool)
    deleteRequested=Signal()
    def __init__(self,parent=None):
        super().__init__(parent)
        self.setMinimumSize(360,360); self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.geometry=None; self.projection='Front · X/Y'; self.tool='inspect'
        self.selected_index=-1; self.show_labels=False; self.show_dimensions=True
        self.snap=5.; self.active_shank=0; self.depth=0.; self.zoom=1.; self.pan=QPointF()
        self._drag=None; self._drag_transform=None; self._metric_cache=None; self._positions=np.empty((0,3)); self._panning=None; self._hover=None; self._focus_contacts=False
        self.setMouseTracking(True)

    def set_geometry(self,g,reset=False):
        self.geometry=copy.deepcopy(g)
        self.geometry['contacts']=copy.deepcopy(contacts_for(g))
        self._positions=np.array([[c['x_um']+c['shank']*g['shank_pitch_um'],c['y_um'],c['z_um']] for c in self.geometry['contacts']],float).reshape(-1,3)
        self._metric_cache=None
        if self.selected_index>=len(contacts_for(g)): self.selected_index=-1
        if reset: self.fit()
        self.update()

    def fit(self,contacts=False):
        self.zoom=1.; self.pan=QPointF(); self._focus_contacts=contacts; self.update()

    def axes(self): return (0,1) if self.projection.startswith('Front') else (2,1) if self.projection.startswith('Side') else (0,2)

    def positions(self):
        return self._positions

    def transform(self):
        if self._drag_transform is not None: return self._drag_transform
        key=(id(self.geometry),self.width(),self.height(),self.projection,self._focus_contacts,self.zoom,self.pan.x(),self.pan.y())
        if self._metric_cache and self._metric_cache[0]==key: return self._metric_cache[1]
        g=self.geometry; a,b=self.axes(); xyz=self.positions()
        if g is None: return 1.,QPointF(self.width()/2,self.height()/2)
        if self._focus_contacts and len(xyz):
            lo=xyz[:,[a,b]].min(0); hi=xyz[:,[a,b]].max(0)
            pad=max(g['site_size_um']*3,20); lo-=pad; hi+=pad
        else:
            lo=np.array([-g['shank_width_um']/2,0.,-g['shank_thickness_um']/2])
            hi=np.array([(g['shanks']-1)*g['shank_pitch_um']+g['shank_width_um']/2,g['shank_length_um'],g['shank_thickness_um']/2])
            if len(xyz): lo=np.minimum(lo,xyz.min(0)); hi=np.maximum(hi,xyz.max(0))
            lo,hi=lo[[a,b]],hi[[a,b]]
        extent=np.maximum(hi-lo,20); scale=min(max(1,self.width()-150)/extent[0],max(1,self.height()-130)/extent[1])*self.zoom
        center=(lo+hi)/2
        origin=QPointF(self.width()/2-center[0]*scale,self.height()/2+center[1]*scale)+self.pan
        self._metric_cache=(key,(scale,origin))
        return scale,origin

    def pixel(self,xy):
        scale,origin=self.transform(); return origin+QPointF(float(xy[0])*scale,-float(xy[1])*scale)

    def world(self,pos):
        scale,origin=self.transform(); return np.array([(pos.x()-origin.x())/scale,-(pos.y()-origin.y())/scale])

    def nearest(self,pos):
        positions=self.positions(); a,b=self.axes()
        if not len(positions): return -1
        scale,origin=self.transform()
        pts=np.column_stack((origin.x()+positions[:,a]*scale,origin.y()-positions[:,b]*scale))
        distances=np.linalg.norm(pts-[pos.x(),pos.y()],axis=1)
        index=int(np.argmin(distances))
        return index if distances[index]<=max(9,self.geometry['site_size_um']*scale/2+3) else -1

    def paintEvent(self,event):
        p=QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(),QColor('#18202b'))
        if not self.geometry: return
        g=self.geometry; a,b=self.axes(); scale,_=self.transform()
        # Adaptive metric grid, bounded independently of zoom and probe length.
        step=10**math.floor(math.log10(max(1e-9,70/scale)))
        while step*scale<35: step*=2
        lower=self.world(QPointF(0,self.height())); upper=self.world(QPointF(self.width(),0))
        p.setPen(QPen(QColor('#222f3d'),1))
        for axis in (0,1):
            start=math.ceil(lower[axis]/step)*step
            for value in np.arange(start,upper[axis]+step,step)[:200]:
                if axis==0: p.drawLine(self.pixel((value,lower[1])),self.pixel((value,upper[1])))
                else: p.drawLine(self.pixel((lower[0],value)),self.pixel((upper[0],value)))
        p.setPen(QPen(QColor('#627a8e'),1)); p.setBrush(QColor('#2b3d4c'))
        for s in range(g['shanks']):
            if a==0 and b==1: points=contour(g,s)
            elif a==2 and b==1:
                t=g['shank_thickness_um']/2; points=[(-t,0),(-t,g['shank_length_um']),(t,g['shank_length_um']),(t,0)]
                if s: continue
            else:
                x=s*g['shank_pitch_um']; w=g['shank_width_um']/2; t=g['shank_thickness_um']/2
                points=[(x-w,-t),(x-w,t),(x+w,t),(x+w,-t)]
            p.drawPolygon(QPolygonF([self.pixel(pt) for pt in points]))
        xyz=self.positions()
        for i,pt in enumerate(xyz):
            pos=self.pixel(pt[[a,b]])
            if not self.rect().adjusted(-20,-20,20,20).contains(pos.toPoint()): continue
            selected=i==self.selected_index
            p.setPen(QPen(QColor('#ffffff' if selected else '#6dc9db'),2 if selected else 1))
            p.setBrush(QColor('#f6ce7c' if selected else '#75c9da'))
            w=max(3,g['site_size_um']*scale); h=max(3,g.get('site_height_um',g['site_size_um'])*scale)
            if b==2: h=max(3,g['site_size_um']*scale)
            rect=QRectF(pos.x()-w/2,pos.y()-h/2,w,h)
            if g.get('contact_shape')=='circle': p.drawEllipse(rect)
            else: p.drawRect(rect)
            if self.show_labels and len(xyz)<600 or selected:
                p.setPen(QColor('#e5eaf2')); p.drawText(pos+QPointF(w/2+5,-4),str(contacts_for(g)[i]['id']))
        if self.show_dimensions:
            p.setPen(QPen(QColor('#9aabc0'),1))
            if b==1:
                x=lower[0]+25/scale
                start,end=self.pixel((x,0)),self.pixel((x,g['shank_length_um']))
                p.drawLine(start,end)
                for pt in (start,end): p.drawLine(pt+QPointF(-4,0),pt+QPointF(4,0))
                p.drawText(start+QPointF(7,-8),f'{g["shank_length_um"]:g} µm')
            if a==0 and b==1:
                y=upper[1]-25/scale
                start,end=self.pixel((-g['shank_width_um']/2,y)),self.pixel((g['shank_width_um']/2,y))
                p.drawLine(start,end); p.drawText(end+QPointF(7,4),f'{g["shank_width_um"]:g} µm width')
                if g['shanks']>1:
                    start,end=self.pixel((0,y-30/scale)),self.pixel((g['shank_pitch_um'],y-30/scale))
                    p.drawLine(start,end); p.drawText(end+QPointF(7,4),f'{g["shank_pitch_um"]:g} µm pitch')
        # Physical scale bar always uses the exact same scale as the geometry.
        bar=10**math.floor(math.log10(max(1e-9,100/scale)))
        while bar*scale<55: bar*=2
        p.setPen(QPen(QColor('#b5c9d8'),2)); start=QPointF(24,self.height()-30)
        p.drawLine(start,start+QPointF(bar*scale,0)); p.drawText(start+QPointF(0,-8),f'{bar:g} µm')
        p.setPen(QColor('#9aabc0')); p.drawText(QPointF(24,25),'X →   Y ↑' if b==1 and a==0 else 'Z →   Y ↑' if b==1 else 'X →   Z ↑')
        if self._hover is not None:
            xy=self.world(self._hover)
            p.drawText(QPointF(24,self.height()-9),f'{"XYZ"[a]} {xy[0]:.1f} · {"XYZ"[b]} {xy[1]:.1f} µm')
        p.end()

    def mousePressEvent(self,event):
        self.setFocus(); pos=event.position()
        if self.tool=='inspect':
            self._panning=(pos,self.pan); return
        if event.button()==Qt.MouseButton.MiddleButton or self.tool=='pan': self._panning=(pos,self.pan); return
        if event.button()!=Qt.MouseButton.LeftButton or not self.geometry: return
        index=self.nearest(pos)
        if index>=0:
            self.selected_index=index; self.selected.emit(index)
            self._drag_transform=self.transform(); self._drag=(index,self.positions()[index].copy(),self.world(pos)); self.update(); return
        if self.tool=='add':
            xyz=np.array([self.active_shank*self.geometry['shank_pitch_um'],0.,self.geometry['shank_thickness_um']/2])
            hidden=({0,1,2}-set(self.axes())).pop(); xyz[hidden]=self.depth
            a,b=self.axes(); xyz[[a,b]]=self.world(pos)
            xyz[0]-=self.active_shank*self.geometry['shank_pitch_um']
            if self.snap: xyz[[a,b]]=np.round(xyz[[a,b]]/self.snap)*self.snap
            self.addContact.emit(dict(shank=self.active_shank,x_um=float(xyz[0]),y_um=float(xyz[1]),z_um=float(xyz[2])))
        else: self.selected_index=-1; self.selected.emit(-1); self.update()

    def mouseMoveEvent(self,event):
        pos=event.position(); self._hover=pos
        if self._panning:
            start,pan=self._panning; self.pan=pan+pos-start
        elif self._drag:
            index,xyz,start=self._drag; updated=xyz.copy(); a,b=self.axes(); updated[[a,b]]+=self.world(pos)-start
            updated[0]-=contacts_for(self.geometry)[index]['shank']*self.geometry['shank_pitch_um']
            if self.snap: updated[[a,b]]=np.round(updated[[a,b]]/self.snap)*self.snap
            self.moveContact.emit(index,updated.tolist(),False)
        self.update()

    def mouseReleaseEvent(self,event):
        if self._drag:
            index,_,_=self._drag
            c=contacts_for(self.geometry)[index]
            self.moveContact.emit(index,[c['x_um'],c['y_um'],c['z_um']],True)
        self._drag=None; self._drag_transform=None; self._panning=None

    def wheelEvent(self,event):
        before=self.world(event.position()); delta=event.angleDelta().y() or event.pixelDelta().y()
        self.zoom=float(np.clip(self.zoom*math.exp(delta/600),.1,100))
        after=self.pixel(before); self.pan+=event.position()-after; self.update(); event.accept()

    def keyPressEvent(self,event):
        if event.key() in (Qt.Key.Key_Delete,Qt.Key.Key_Backspace): self.deleteRequested.emit()
        elif event.key()==Qt.Key.Key_F: self.fit()
        else: super().keyPressEvent(event)



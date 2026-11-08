"""Oblique slice drawing and a transform-only 3D electrode-channel view."""
from __future__ import annotations

import numpy as np
from qtpy.QtCore import QPointF, Qt, Signal
from qtpy.QtGui import QColor, QPainter, QPainterPath, QPen

from .desktop_views import SliceCanvas, VolumeView
from .track_model import plane_corners, plane_uv, plane_world


class TrackSliceCanvas(SliceCanvas):
    vertexAdded=Signal(object)
    finished=Signal()
    hovered=Signal(object)

    def __init__(self,drawable=False):
        super().__init__(0)
        self.geometry_record=None
        self.drawable=drawable
        self.drawing=False
        self.annotation_mode=False
        self.track_lines=[]
        self.region_labels=[]
        self.selected_track=None
        self.setMinimumSize(180,180)
        self.setToolTip("Left-click: add a track vertex · Double-click: finish\nWheel: zoom · Middle-drag: pan")

    def set_plane_image(self,pixels,geometry):
        self.geometry_record=geometry
        size=geometry["pixel_size_um"]
        self.set_image(pixels,(size,size,size),0)

    def _world(self,position):
        rect=self.image_rect()
        if self.image is None or rect.isEmpty() or self.geometry_record is None:
            return None
        uv=[(position.x()-rect.left())/rect.width()*self.image.width()-.5,
            (position.y()-rect.top())/rect.height()*self.image.height()-.5]
        u,v=(uv[0]+.5)/self.image.width(),(uv[1]+.5)/self.image.height()
        for _ in range(self.view_rotation//90): u,v=v,1-u
        uv=[u*self.image.width()-.5,v*self.image.height()-.5]
        return plane_world(self.geometry_record,[uv])[0]

    def _pixel(self,world):
        uv,_=plane_uv(self.geometry_record,[world])
        rect=self.image_rect()
        u,v=(uv[0,0]+.5)/self.image.width(),(uv[0,1]+.5)/self.image.height()
        for _ in range(self.view_rotation//90): u,v=1-v,u
        return QPointF(rect.left()+u*rect.width(),rect.top()+v*rect.height())

    def paintEvent(self,event):
        painter=QPainter(self)
        painter.fillRect(self.rect(),QColor(self.background_color))
        if self.image is None:
            painter.setPen(QColor("#91a5bb"))
            painter.drawText(self.rect(),Qt.AlignmentFlag.AlignCenter,"Choose a track scan to begin")
            return
        self.paint_image(painter,getattr(self,'smooth_interpolation',True))
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QColor("#91a5bb"))
        if self.show_axes: painter.drawText(10,18,getattr(self,"axis_label","Plane U →   V ↓"))
        for track in self.track_lines:
            points=np.asarray(track["points_atlas_zyx_um"],dtype=float).reshape(-1,3)
            if not len(points): continue
            _,distance=plane_uv(self.geometry_record,points)
            if np.max(np.abs(distance))>self.geometry_record["pixel_size_um"]*.75:
                continue
            color=QColor(track.get("color","#ffc966"))
            selected=track["id"]==self.selected_track
            painter.setPen(QPen(color,2.5 if selected else 1.8))
            path=QPainterPath(self._pixel(points[0]))
            for point in points[1:]: path.lineTo(self._pixel(point))
            painter.drawPath(path)
            if selected:
                for point in points: painter.drawEllipse(self._pixel(point),3,3)
            painter.drawText(self._pixel(points[-1])+QPointF(6,-6),track["name"])
        for text,uv,color in self.region_labels:
            world=plane_world(self.geometry_record,[uv])[0]
            position=self._pixel(world)
            painter.setPen(QPen(QColor("#0b1018"),3))
            # A dark backing keeps labels legible on both bright and dark atlas tissue.
            bounds=painter.fontMetrics().boundingRect(text)
            painter.fillRect(int(position.x()-2),int(position.y()-bounds.height()),bounds.width()+4,bounds.height()+3,QColor(8,12,18,180))
            painter.setPen(QColor(*color))
            painter.drawText(position,text)

    def mouseMoveEvent(self,event):
        world=self._world(event.position())
        if world is not None: self.hovered.emit(world)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self,event):
        if (self.drawable and self.drawing and self._button==Qt.MouseButton.LeftButton
                and self._start_pixel is not None and (event.position()-self._start_pixel).manhattanLength()<5
                and self.image_rect().contains(event.position())):
            world=self._world(event.position())
            if world is not None: self.vertexAdded.emit(world)
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self,event):
        if self.drawable and self.drawing:
            self.finished.emit()
        else:
            super().mouseDoubleClickEvent(event)

    def wheelEvent(self,event):
        if self.annotation_mode:
            self.zoom=float(np.clip(self.zoom*(1.15 if event.angleDelta().y()>0 else 1/1.15),.25,10))
            self.update()
            self.viewChanged.emit()
            event.accept()
        else:
            super().wheelEvent(event)


class TrackVolumeView(VolumeView):
    def __init__(self):
        super().__init__(("Atlas context","Track scan"))
        self.heading.setText("Track scan in atlas space · drag to orbit, wheel to zoom")
        self.show_slice_planes.setChecked(False)
        self.show_slice_planes.hide()
        self.fill_slice_planes.hide()
        self.plane_mesh=self.plane_outline=None
        self.track_visuals={}
        self._plane_geometry=None

    def load(self,reference,moving):
        super().load(reference,moving)
        if self.plane_mesh is None:
            from vispy import scene
            vertices=np.zeros((4,3),dtype=np.float32)
            self.plane_mesh=scene.visuals.Mesh(vertices=vertices,faces=np.array([[0,1,2],[0,2,3]],np.uint32),
                                              color=(.25,.8,1,.065),parent=self.view.scene)
            self.plane_outline=scene.visuals.Line(vertices,color=(.35,.85,1,1),width=2,
                                                 connect="strip",parent=self.view.scene)
            for index,visual in enumerate((self.plane_mesh,self.plane_outline),start=10):
                visual.order=index
                visual.set_gl_state("translucent",depth_test=False)
        self._plane_geometry=None
        self.plane_mesh.visible=self.plane_outline.visible=False

    def set_plane(self,geometry):
        if self.plane_mesh is None or geometry==self._plane_geometry: return
        self._plane_geometry=geometry
        vertices=np.ascontiguousarray(plane_corners(geometry)[:,::-1],dtype=np.float32)
        self.plane_mesh.set_data(vertices=vertices,faces=np.array([[0,1,2],[0,2,3]],np.uint32))
        self.plane_outline.set_data(pos=vertices[[0,1,2,3,0]])
        self.plane_mesh.visible=self.plane_outline.visible=True
        self.canvas.update()

    def set_tracks(self,tracks):
        if self.canvas is None: return
        from vispy import scene
        from vispy.color import Color
        active=set()
        for track in tracks:
            identifier=track["id"]
            points=np.asarray(track["points_atlas_zyx_um"],dtype=np.float32).reshape(-1,3)
            if len(points)<2: continue
            active.add(identifier)
            if identifier not in self.track_visuals:
                visual=scene.visuals.Line(points[:,::-1],connect="strip",width=2,parent=self.view.scene)
                visual.order=12
                visual.set_gl_state("translucent",depth_test=False)
                self.track_visuals[identifier]=visual
            visual=self.track_visuals[identifier]
            visual.set_data(pos=np.ascontiguousarray(points[:,::-1]),color=Color(track.get("color","#ffc966")).rgba)
            visual.visible=True
        for identifier,visual in self.track_visuals.items():
            visual.visible=identifier in active
        self.canvas.update()

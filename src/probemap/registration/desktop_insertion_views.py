"""Linked insertion annotations, slab-aware track projections and 3D picking."""
from __future__ import annotations

import numpy as np
from qtpy.QtCore import QPointF, Qt, Signal
from qtpy.QtGui import QColor, QPainter, QPen, QPolygonF

from .desktop_track_views import TrackSliceCanvas, TrackVolumeView
from .insertion_model import clip_segment_to_slab,track_slice_points
from .track_model import plane_uv


class InsertionSliceCanvas(TrackSliceCanvas):
    pointAdded=Signal(object)
    pointsSelected=Signal(object,bool)
    pointsDeleted=Signal()
    trackPreview=Signal(object,object)
    trackDragged=Signal(object,object)
    drawingCancelled=Signal()
    trackPicked=Signal(str)
    trackShifted=Signal(str,object,bool)
    measurementAdded=Signal(object,object)
    annotationContext=Signal(str,str,object)

    def __init__(self):
        super().__init__(drawable=True)
        self.insertion_points=[]
        self.insertion_tracks=[]
        self.selected_points=set()
        self.point_mode="view"
        self.slab_um=50.
        self.project_tracks=True
        self.show_points=True
        self.point_size=8
        self.slice_thickness_um=None
        self.click_navigation=False
        self.show_tracks=True
        self.track_hits=False
        self.mask_pixels=None
        self.interaction_tool='annotate'
        self.measurements=[]
        self.measurement_unit='µm'
        self.measurement_inverse=np.eye(3)
        self.probe_sites=[]
        self.probe_polygons=[]
        self._measure_start=self._measure_end=None
        self._track_drag=None
        self._zooming=False
        from .desktop_slice_tools import SliceTools
        self.tools=SliceTools(self)
        self.setToolTip("Click: add/select a point or define a track · Ctrl-click: extend selection\n"
                        "Pinch or wheel: zoom · Shift+drag: pan · Right-drag: zoom · Slider: slice")

    def set_plane_image(self,pixels,geometry):
        if geometry!=self.geometry_record:
            self._measure_start=self._measure_end=self._track_drag=None
        super().set_plane_image(pixels,geometry)

    def set_interaction_tool(self,tool):
        if tool!=self.interaction_tool:
            self._measure_start=self._measure_end=self._track_drag=None
        self.interaction_tool=tool
        self.pan_mode=tool=='pan'
        self.setCursor(Qt.CursorShape.OpenHandCursor if self.pan_mode else Qt.CursorShape.CrossCursor)
        self.update()

    def _nearest_track(self,position):
        candidates=[]
        pos=np.array([position.x(),position.y()])
        for track in self.insertion_tracks:
            if self.track_hits:
                segments=[(hit['atlas_zyx_um'],hit['atlas_zyx_um']) for hit in track_slice_points(track,self.geometry_record)]
            else:
                segment=clip_segment_to_slab(*np.asarray(track['points_atlas_zyx_um'])[[0,-1]],self.geometry_record,self.slab_um)
                segments=[] if segment is None else [segment]
            for a,b in segments:
                a,b=self._pixel(a),self._pixel(b)
                a,b=np.array([a.x(),a.y()]),np.array([b.x(),b.y()])
                vector=b-a
                t=np.clip(np.dot(pos-a,vector)/max(np.dot(vector,vector),1e-12),0,1)
                candidates.append((np.linalg.norm(pos-a-t*vector),track['id']))
        distance,identifier=min(candidates,default=(np.inf,None))
        return identifier if distance<12 else None

    def visible_points(self):
        if self.geometry_record is None or not self.show_points: return []
        result=[]
        for point in self.insertion_points:
            _,distance=plane_uv(self.geometry_record,[point["atlas_zyx_um"]])
            thickness=self.slice_thickness_um or self.geometry_record['pixel_size_um']
            if int(np.floor(distance[0]/thickness+.5+1e-10))==0:
                result.append((point,float(distance[0])))
        return result

    def paintEvent(self,event):
        super().paintEvent(event)
        if self.image is None or self.geometry_record is None: return
        painter=QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setClipRect(self.rect() if getattr(self,'placement_enabled',False) else self.image_rect())
        for positions,rgba in self.probe_polygons:
            color=QColor.fromRgbF(*rgba)
            painter.setPen(QPen(color,1)); painter.setBrush(color)
            painter.drawPolygon(QPolygonF([self._pixel(p) for p in positions]))
        if self.probe_sites:
            _,distances=plane_uv(self.geometry_record,[site['atlas_zyx_um'] for site in self.probe_sites])
            for site,distance in zip(self.probe_sites,distances):
                if abs(distance)>max(50,self.geometry_record['pixel_size_um']): continue
                color=QColor.fromRgbF(*site['rgba'])
                painter.setPen(QPen(color,1)); painter.setBrush(color)
                painter.drawEllipse(self._pixel(site['atlas_zyx_um']),2.5,2.5)
        if self.track_hits:
            for track in self.insertion_tracks:
                painter.setPen(QPen(QColor(track['color']),1.5))
                painter.setBrush(QColor(track['color']))
                for hit in track_slice_points(track,self.geometry_record):
                    painter.drawEllipse(self._pixel(hit['atlas_zyx_um']),4,4)
        if self.show_tracks and not self.track_hits:
            for track in self.insertion_tracks:
                points=np.asarray(track["points_atlas_zyx_um"],float).reshape(-1,3)
                if len(points)<2: continue
                color=QColor(track["color"])
                for a,b in zip(points[:-1],points[1:]):
                    if self.project_tracks:
                        painter.setPen(QPen(QColor(color.red(),color.green(),color.blue(),145),1.4,Qt.PenStyle.DashLine))
                        painter.drawLine(self._pixel(a),self._pixel(b))
                    clipped=clip_segment_to_slab(a,b,self.geometry_record,self.slab_um)
                    if clipped is not None:
                        if track['id']==self.selected_track:
                            painter.setPen(QPen(QColor('white'),6))
                            painter.drawLine(self._pixel(clipped[0]),self._pixel(clipped[1]))
                        painter.setPen(QPen(color,3))
                        painter.drawLine(self._pixel(clipped[0]),self._pixel(clipped[1]))
                if self.project_tracks or abs(plane_uv(self.geometry_record,[points[0]])[1][0])<=self.slab_um/2:
                    painter.setPen(QPen(color,2))
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    painter.drawEllipse(self._pixel(points[0]),5,5)
                    painter.drawText(self._pixel(points[0])+QPointF(7,-7),track["name"])
        for point,distance in self.visible_points():
            position=self._pixel(point["atlas_zyx_um"])
            selected=point["id"] in self.selected_points
            color=QColor(point["color"])
            pen=QPen(QColor("white") if selected else color,2)
            painter.setPen(pen)
            painter.setBrush(color)
            radius=self.point_size/2+(1 if selected else 0)
            painter.drawEllipse(position,radius,radius)
            if selected:
                painter.drawText(position+QPointF(8,-6),point.get("label","")+f" ({distance:+.0f} µm)")
        rulers=list(self.measurements)
        if self._measure_start is not None and self._measure_end is not None:
            rulers.append({'points_atlas_zyx_um':[self._measure_start,self._measure_end]})
        for measurement in rulers:
            if not measurement.get('visible',True): continue
            points=np.asarray(measurement['points_atlas_zyx_um'],float)
            if np.max(np.abs(plane_uv(self.geometry_record,points)[1]))>self.geometry_record['pixel_size_um']/2: continue
            a,b=(self._pixel(point) for point in points)
            color=QColor(measurement.get('color','#79e7ff'))
            painter.setPen(QPen(color,2))
            painter.setBrush(color)
            painter.drawLine(a,b)
            for point in (a,b): painter.drawEllipse(point,3,3)
            length=float(np.linalg.norm(self.measurement_inverse@(points[1]-points[0])))
            if not measurement.get('show_label',True): continue
            from .desktop_measurements import distance_text
            text=distance_text(length,self.measurement_unit)
            center=(a+b)/2+QPointF(5,-8)
            bounds=painter.fontMetrics().boundingRect(text)
            painter.fillRect(int(center.x()-3),int(center.y()-bounds.height()),bounds.width()+6,bounds.height()+4,QColor('#11161e'))
            painter.drawText(center,text)

    def mousePressEvent(self,event):
        super().mousePressEvent(event)
        self._zooming=event.button()==Qt.MouseButton.RightButton and not (event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        if self._zooming:
            self._panning=False
            return
        if self._panning or self.drawing or not self.image_rect().contains(event.position()): return
        if self.interaction_tool=='track' and event.button()==Qt.MouseButton.LeftButton:
            identifier=self._nearest_track(event.position())
            if identifier:
                self._track_drag=(identifier,self._world(event.position()))
                self.trackPicked.emit(identifier)

    def mouseMoveEvent(self,event):
        if self._zooming and self._last_pixel is not None:
            dy=event.position().y()-self._last_pixel.y()
            self.zoom_at(self._start_pixel,float(np.exp(-dy*.01)))
            self._last_pixel=event.position()
            return
        if (self.click_navigation and not self._panning and self._start_pixel is not None
                and event.buttons() & Qt.MouseButton.LeftButton and self.interaction_tool=='annotate'
                and (event.position()-self._start_pixel).manhattanLength()>=5):
            self._panning=True
        if not self._panning and not self.drawing and self.image is not None:
            world=self._world(event.position())
            if self.interaction_tool=='measure' and self.image_rect().contains(event.position()):
                if self._measure_start is None and event.buttons() & Qt.MouseButton.LeftButton and self._start_pixel is not None:
                    self._measure_start=self._world(self._start_pixel)
                self._measure_end=world
                self.update()
            elif self._track_drag is not None and event.buttons() & Qt.MouseButton.LeftButton:
                identifier,start=self._track_drag
                self.trackShifted.emit(identifier,(world-start).tolist(),False)
        if self.drawing and not self._panning and self.image_rect().contains(event.position()):
            world=self._world(event.position())
            start=(self._world(self._start_pixel) if self._button==Qt.MouseButton.LeftButton
                   and self._start_pixel is not None else None)
            self.trackPreview.emit(start,world)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self,event):
        if self._zooming:
            self._zooming=False
            if self._start_pixel is not None and (event.position()-self._start_pixel).manhattanLength()<5:
                hit=self._context_hit(event.position())
                if hit is not None:
                    self.annotationContext.emit(*hit,event.globalPosition().toPoint())
            super(TrackSliceCanvas,self).mouseReleaseEvent(event)
            return
        if self._panning:
            super(TrackSliceCanvas,self).mouseReleaseEvent(event)
            return
        if not self.drawing and event.button()==Qt.MouseButton.LeftButton:
            if self.interaction_tool=='measure' and self.image_rect().contains(event.position()):
                world=self._world(event.position())
                if self._measure_start is None: self._measure_start=world
                elif np.linalg.norm(world-self._measure_start)>1e-6:
                    self.measurementAdded.emit(self._measure_start,world)
                    self._measure_start=self._measure_end=None
                super(TrackSliceCanvas,self).mouseReleaseEvent(event)
                self.update()
                return
            if self._track_drag is not None:
                identifier,start=self._track_drag
                self._track_drag=None
                self.trackShifted.emit(identifier,(self._world(event.position())-start).tolist(),True)
                super(TrackSliceCanvas,self).mouseReleaseEvent(event)
                return
        if (self.drawing and event.button()==Qt.MouseButton.LeftButton and self._start_pixel is not None
              and (event.position()-self._start_pixel).manhattanLength()>=5
              and self.image_rect().contains(self._start_pixel) and self.image_rect().contains(event.position())):
            self.trackDragged.emit(self._world(self._start_pixel),self._world(event.position()))
        click=(event.button()==Qt.MouseButton.LeftButton and self._start_pixel is not None
               and (event.position()-self._start_pixel).manhattanLength()<5
               and self.image_rect().contains(event.position()))
        if click and self.interaction_tool=='annotate' and self.point_mode=="add":
            world=self._world(event.position())
            if world is not None: self.pointAdded.emit(world)
        elif click and self.interaction_tool=='annotate' and self.point_mode=="select":
            candidates=[(np.linalg.norm(np.array([self._pixel(p["atlas_zyx_um"]).x(),self._pixel(p["atlas_zyx_um"]).y()])-
                                        [event.position().x(),event.position().y()]),p["id"]) for p,_ in self.visible_points()]
            nearest=min(candidates,default=(np.inf,None))
            self.pointsSelected.emit([nearest[1]] if nearest[0]<max(10,self.point_size/2+4) else [],
                                     bool(event.modifiers() & (Qt.KeyboardModifier.ControlModifier|Qt.KeyboardModifier.MetaModifier)))
        super().mouseReleaseEvent(event)

    def _context_hit(self,position):
        if self.geometry_record is None or self.image is None: return None
        candidates=[((self._pixel(p['atlas_zyx_um'])-position).manhattanLength(),('point',p['id']))
                    for p,_ in self.visible_points()] if self.point_mode in ('add','select') else []
        distance,hit=min(candidates,default=(float('inf'),None))
        if distance<max(12,self.point_size/2+4): return hit
        pos=np.array([position.x(),position.y()])
        for measurement in self.measurements:
            if not measurement.get('visible',True): continue
            points=np.asarray(measurement['points_atlas_zyx_um'])
            if np.max(np.abs(plane_uv(self.geometry_record,points)[1]))>self.geometry_record['pixel_size_um']/2: continue
            a,b=(self._pixel(point) for point in points)
            a,b=np.array([a.x(),a.y()]),np.array([b.x(),b.y()])
            vector=b-a
            t=np.clip(np.dot(pos-a,vector)/max(np.dot(vector,vector),1e-12),0,1)
            if np.linalg.norm(pos-a-t*vector)<9: return ('measurement',measurement['id'])
        return None

    def keyPressEvent(self,event):
        if event.key() in (Qt.Key.Key_Delete,Qt.Key.Key_Backspace) and self.point_mode in ('add','select'):
            self.pointsDeleted.emit()
            event.accept()
        elif event.key()==Qt.Key.Key_Escape:
            if self.drawing: self.drawingCancelled.emit()
            self._measure_start=self._measure_end=None
            if self._track_drag is not None:
                self.trackShifted.emit(self._track_drag[0],[0,0,0],True)
                self._track_drag=None
            self.update()
            event.accept()
        else: super().keyPressEvent(event)

    def wheelEvent(self,event):
        delta=event.angleDelta().y()/120 if event.angleDelta().y() else event.pixelDelta().y()/40
        if delta: self.zoom_at(event.position(),1.1**delta)
        event.accept()

    def resizeEvent(self,event):
        super().resizeEvent(event)
        if hasattr(self,'tools'):
            self.tools.place()

    def focusOutEvent(self,event):
        self._zooming=False
        super().focusOutEvent(event)


class InsertionVolumeView(TrackVolumeView):
    pointsSelected=Signal(object,bool)
    trackSelected=Signal(str)
    annotationContext=Signal(str,str,object)

    def __init__(self):
        self.surface_entries_active=False
        self.surface_entry_bounds=None
        super().__init__()
        from .desktop_rendering import default_render_settings
        self.set_layer_titles(("Atlas","Track scan","Brain scan"))
        self.render_settings=default_render_settings(3)
        self.brain=None
        self.point_markers=None
        self.point_records=[]
        self.point_size=8
        self.selection_enabled=False
        self.selection_mode='points'
        self._selection_origin=None
        self._background_selection_origin=self._context_origin=None
        self._pick_connected=False
        self.probe_face=None
        self.heading.setText("3D · drag to orbit · select points with the selection tool")
        self.reset_button.setText("Fit scans and tracks")
        self.inline_settings_handler=None

    def _apply_layer_visibility(self):
        super()._apply_layer_visibility()
        if self.surface_entries_active:
            for visual in [*self.visuals,*self.label_visuals.values()]: visual.visible=False

    def show_probe_face(self,insertion,matrix):
        if self.canvas is None: return
        if self.probe_face is None:
            from .desktop_probe_face import ProbeFaceIndicator
            self.probe_face=ProbeFaceIndicator(self)
        self.probe_face.set_probe(insertion,matrix)

    def hide_probe_face(self):
        if self.probe_face is not None: self.probe_face.hide()

    def open_settings(self):
        if self.inline_settings_handler is not None: self.inline_settings_handler()
        else: super().open_settings()

    def _slice_visibility_changed(self):
        super()._slice_visibility_changed()
        if not getattr(self,'orthogonal_planes_enabled',False):
            for mesh,line,text in self.slice_planes: mesh.visible=line.visible=text.visible=False

    def load_channels(self,atlas,track,brain):
        from vispy import scene
        from vispy.visuals.transforms import MatrixTransform
        from .desktop_rendering import apply_render_settings
        from .volumes import Volume
        factors=np.maximum(1,np.ceil(np.asarray(brain.data.shape)/192).astype(int))
        brain=Volume(np.ascontiguousarray(brain.data[tuple(slice(None,None,int(n)) for n in factors)],dtype=np.float32),
                     tuple(np.asarray(brain.spacing_um_zyx)*factors),brain.path,brain.intensity_limits)
        self.brain=brain
        super().load(atlas,track)
        from .desktop_rendering import volume_visual
        visual=volume_visual(brain,self.view.scene,self.render_settings[2])
        apply_render_settings(visual,self.render_settings[2])
        visual.transform=MatrixTransform()
        self.visuals.append(visual)
        self.render_volumes=(self.reference,self.moving,brain)
        if self.point_markers is None:
            self.point_markers=scene.visuals.Markers(parent=self.view.scene)
            self.point_markers.order=20
            self.point_markers.set_gl_state("translucent",depth_test=False)
            self.selection_outline=scene.visuals.Line(np.zeros((5,2),np.float32),color="white",parent=self.canvas.scene)
            self.selection_outline.order=100
            self.selection_outline.visible=False
        if not self._pick_connected:
            self.canvas.events.mouse_press.connect(self._pick_press)
            self.canvas.events.mouse_move.connect(self._pick_move)
            self.canvas.events.mouse_release.connect(self._pick_release)
            self._pick_connected=True
        self.update_alignment(np.eye(4),0,1,.25)

    def update_alignment(self,matrix,reference_opacity,moving_opacity,brain_opacity=None):
        super().update_alignment(matrix,reference_opacity,moving_opacity)
        if len(self.visuals)>2:
            from .desktop_views import visual_matrix
            visual=self.visuals[2]
            visual.transform.matrix=visual_matrix(self.brain.spacing_um_zyx,matrix)
            if brain_opacity is not None:
                self.layers.sliders[2].blockSignals(True)
                self.layers.sliders[2].setValue(round(brain_opacity*100))
                self.layers.sliders[2].blockSignals(False)
                self.layers.amounts[2].setText(f'{round(brain_opacity*100)}%')
                visual.opacity=float(brain_opacity)
                visual.visible=brain_opacity>0 and self.layers.checks[2].isChecked() and not self.surface_entries_active
            if self.settings_dialog is not None: self.settings_dialog.sync()

    def set_points(self,points,selected=(),visible=True):
        self.selected_point_ids=set(selected)
        self.point_records=points if visible else []
        if self.point_markers is None: return
        if not self.point_records:
            self.point_markers.visible=False
            self.canvas.update()
            return
        from vispy.color import Color
        positions=np.array([p["atlas_zyx_um"][::-1] for p in points],np.float32)
        colors=np.array([Color(p["color"]).rgba for p in points],np.float32)
        edges=np.array([Color("white" if p["id"] in selected else p["color"]).rgba for p in points],np.float32)
        sizes=np.array([self.point_size+4 if p["id"] in selected else self.point_size for p in points],np.float32)
        self.point_markers.set_data(pos=positions,face_color=colors,edge_color=edges,size=sizes,edge_width=2)
        self.point_markers.visible=True
        self.canvas.update()

    def set_selection_enabled(self,enabled):
        self.selection_enabled=enabled
        if not enabled:
            self._selection_origin=None
            if self.point_markers is not None: self.selection_outline.visible=False
        if self.canvas is not None: self.view.camera.interactive=self._selection_origin is None

    def projected_points(self):
        if not self.point_records or self.point_markers is None: return np.empty((0,2))
        positions=np.array([p["atlas_zyx_um"][::-1] for p in self.point_records],float)
        projected=self.point_markers.get_transform(map_from="visual",map_to="canvas").map(positions)
        return projected[:,:2]/projected[:,3,None]

    def _nearest_track_screen(self,position):
        candidates=[]
        for track in getattr(self,'_display_tracks',[]):
            visual=self.track_visuals.get(track['id'])
            if visual is None or not visual.visible: continue
            points=np.asarray(track['points_atlas_zyx_um'],float)[:,::-1]
            mapped=visual.get_transform(map_from='visual',map_to='canvas').map(points)
            pixels=mapped[:,:2]/mapped[:,3,None]
            for a,b in zip(pixels[:-1],pixels[1:]):
                vector=b-a
                t=np.clip(np.dot(position-a,vector)/max(np.dot(vector,vector),1e-12),0,1)
                candidates.append((np.linalg.norm(position-a-t*vector),track['id']))
        distance,identifier=min(candidates,default=(np.inf,None))
        return identifier if distance<12 else None

    def _pick_press(self,event):
        if event.button==2:
            self._context_origin=np.asarray(event.pos,float)
            return
        if self.selection_enabled and event.button==1:
            modifiers={getattr(key,'name',str(key)).lower() for key in (event.modifiers or ())}
            if 'shift' in modifiers: return
            pixels=self.projected_points()
            near=bool(len(pixels) and np.min(np.linalg.norm(pixels-np.asarray(event.pos),axis=1))<max(12,self.point_size/2+4))
            if self.selection_mode=='tracks': near=self._nearest_track_screen(np.asarray(event.pos,float)) is not None
            if not near and not modifiers.intersection({'control','meta'}):
                self._background_selection_origin=np.asarray(event.pos,float)
                return
            self._selection_origin=np.asarray(event.pos,float)
            self.view.camera.interactive=False
            event.handled=True

    def _pick_move(self,event):
        if self._selection_origin is None: return
        start,end=self._selection_origin,np.asarray(event.pos,float)
        self.selection_outline.set_data(pos=np.array([start,[end[0],start[1]],end,[start[0],end[1]],start],np.float32))
        self.selection_outline.visible=True
        self.canvas.update()
        event.handled=True

    def _pick_release(self,event):
        if event.button==2 and self._context_origin is not None:
            origin=self._context_origin; self._context_origin=None
            pixels=self.projected_points()
            if self.selection_mode=='points' and len(pixels) and np.linalg.norm(np.asarray(event.pos)-origin)<5:
                distances=np.linalg.norm(pixels-np.asarray(event.pos),axis=1)
                nearest=int(np.argmin(distances))
                if distances[nearest]<max(12,self.point_size/2+4):
                    from qtpy.QtCore import QPoint
                    position=self.canvas.native.mapToGlobal(QPoint(*map(int,event.pos)))
                    self.annotationContext.emit('point',self.point_records[nearest]['id'],position)
            return
        if self._background_selection_origin is not None:
            start=self._background_selection_origin; self._background_selection_origin=None
            if np.linalg.norm(np.asarray(event.pos)-start)<5: self.pointsSelected.emit([],False)
            return
        if self._selection_origin is None: return
        start,end=self._selection_origin,np.asarray(event.pos,float)
        self._selection_origin=None
        self.view.camera.interactive=True
        self.selection_outline.visible=False
        if self.selection_mode=='tracks':
            identifier=self._nearest_track_screen(end)
            if identifier is not None: self.trackSelected.emit(identifier)
            self.canvas.update()
            event.handled=True
            return
        pixels=self.projected_points()
        selected=[]
        if len(pixels):
            if np.linalg.norm(end-start)<5:
                distances=np.linalg.norm(pixels-end,axis=1)
                nearest=int(np.argmin(distances))
                if distances[nearest]<max(12,self.point_size/2+4): selected=[self.point_records[nearest]["id"]]
            else:
                lo,hi=np.minimum(start,end),np.maximum(start,end)
                selected=[p["id"] for p,xy in zip(self.point_records,pixels) if np.all((xy>=lo)&(xy<=hi))]
        additive=any(getattr(key,'name',str(key)).lower() in ('control','meta') for key in (event.modifiers or ()))
        if additive and np.linalg.norm(end-start)>=5:
            selected=list(set(selected)|self.selected_point_ids)
            additive=False
        self.pointsSelected.emit(selected,additive)
        self.canvas.update()
        event.handled=True

    def reset_view(self):
        if self.canvas is None or not self.visuals: return
        if self.surface_entries_active and self.surface_entry_bounds is not None:
            xyz=np.asarray(self.surface_entry_bounds)[:,::-1]
            self.view.camera.set_range(x=tuple(xyz[:,0]),y=tuple(xyz[:,1]),z=tuple(xyz[:,2]),margin=.12)
            self.canvas.update(); return
        from .coarse import transformed_corners
        volumes=getattr(self,"render_volumes",(self.reference,self.moving))
        corners=[transformed_corners(volume.data.shape,volume.spacing_um_zyx,np.eye(4) if index==0 else self.matrix)
                 for index,(volume,visual) in enumerate(zip(volumes,self.visuals)) if visual.visible and visual.opacity>0]
        if self.layers.labels_check.isChecked() and self.layers.labels_check.isEnabled():
            corners.append(transformed_corners(self.reference.data.shape,self.reference.spacing_um_zyx,np.eye(4)))
        points=[point for track in getattr(self,"_display_tracks",[]) for point in track["points_atlas_zyx_um"]]
        points.extend(p["atlas_zyx_um"] for p in self.point_records)
        points.extend(getattr(self,'annotation_bounds_zyx',[]))
        if points: corners.append(np.asarray(points,float))
        if not corners: corners=[transformed_corners(self.moving.data.shape,self.moving.spacing_um_zyx,self.matrix)]
        xyz=np.vstack(corners)[:,::-1]
        self.view.camera.set_range(x=(xyz[:,0].min(),xyz[:,0].max()),y=(xyz[:,1].min(),xyz[:,1].max()),
                                   z=(xyz[:,2].min(),xyz[:,2].max()),margin=.12)
        self.canvas.update()

    def set_tracks(self,tracks):
        self._display_tracks=tracks
        super().set_tracks(tracks)
        for identifier,visual in self.track_visuals.items():
            visual.set_data(width=4 if identifier==getattr(self,'selected_track_id',None) else 2)

"""Body-first probe designer with a shared metric canvas and surface-aware tools."""
from __future__ import annotations
import copy
import math
from pathlib import Path
import numpy as np
from qtpy.QtCore import QEvent,QPointF,QRectF,Qt,Signal
from qtpy.QtGui import QColor,QPainter,QPen,QPolygonF,QKeySequence,QShortcut
from qtpy.QtWidgets import (QAbstractItemView,QCheckBox,QComboBox,QDialog,QDoubleSpinBox,QFileDialog,
    QFormLayout,QHBoxLayout,QHeaderView,QLabel,QLineEdit,QMessageBox,QPushButton,QScrollArea,
    QSpinBox,QSplitter,QTabWidget,QTableWidget,QTableWidgetItem,QVBoxLayout,QWidget)
from ..probe_library import contacts_for,grid_geometry,validate_geometry,contacts_csv,parse_contacts_csv,MAX_CONTACTS
from ..probe_geometry import body_mesh,body_polygon,contact_center,contact_mesh_local,new_contact,perimeter_contacts,polygon_valid
from .desktop_probe_canvas_base import ProbeDrawing,note,button


class QuietSpin(QDoubleSpinBox):
    def wheelEvent(self,event):event.ignore()


class QuietCount(QSpinBox):
    def wheelEvent(self,event):event.ignore()


class QuietCombo(QComboBox):
    def wheelEvent(self,event):event.ignore()


class PhysicalCanvas(ProbeDrawing):
    outlineFinished=Signal(object)
    nudgeRequested=Signal(object)
    def __init__(self,parent=None):
        super().__init__(parent);self.rotation=np.array([25.,-30.]);self._orbit=None;self.outline=[]
        self.bodies=(np.empty((0,3)),np.empty((0,3),int));self.contact_surfaces=[]
        self._space=False;self._all_points=np.empty((0,3))

    def set_geometry(self,g,reset=False):
        super().set_geometry(g,False)
        try:self.bodies=body_mesh(g)
        except (ValueError,KeyError,IndexError):self.bodies=(np.empty((0,3)),np.empty((0,3),int))
        self.contact_surfaces=[contact_mesh_local(c,g) for c in self.geometry['contacts']]
        self._contact_edges=[]
        for vertices,faces in self.contact_surfaces:
            counts={}
            for face in faces:
                for a,b in zip(face,np.roll(face,-1)):
                    edge=tuple(sorted((int(a),int(b))));counts[edge]=counts.get(edge,0)+1
            self._contact_edges.append([edge for edge,count in counts.items() if count==1])
        self._all_points=np.vstack([self.bodies[0],self._positions,*[v for v,f in self.contact_surfaces]])
        if reset:self.fit()
        self.update()

    def basis(self):
        if self.projection.startswith('3D'):
            yaw,pitch=np.deg2rad(self.rotation)
            return np.array([[np.cos(yaw),np.sin(yaw)*np.sin(pitch)],[0,np.cos(pitch)],[np.sin(yaw),-np.cos(yaw)*np.sin(pitch)]])
        a,b=self.axes();return np.eye(3)[:,[a,b]]

    def project(self,points):return np.asarray(points)@self.basis()

    def positions(self):
        if self.projection.startswith('3D'):
            p=self.project(self._positions);return np.column_stack([p,np.zeros(len(p))])
        return self._positions

    def axes(self):
        if self.projection.startswith('3D'):return 0,1
        return super().axes()

    def transform(self):
        if self._drag_transform is not None:return self._drag_transform
        if self.geometry is None:return 1.,QPointF(self.width()/2,self.height()/2)
        key=(id(self.geometry),self.width(),self.height(),self.projection,tuple(self.rotation),self._focus_contacts,self.zoom,self.pan.x(),self.pan.y())
        if self._metric_cache and self._metric_cache[0]==key:return self._metric_cache[1]
        points=np.vstack([v for v,f in self.contact_surfaces]) if self._focus_contacts and self.contact_surfaces else self._all_points
        if not len(points):points=np.array([[-50,0,0],[50,100,0]])
        p=self.project(points);lo=p.min(0);hi=p.max(0);pad=max(self.geometry['site_size_um']*2,5)
        extent=np.maximum(hi-lo+2*pad,20);scale=min(max(1,self.width()-100)/extent[0],max(1,self.height()-100)/extent[1])*self.zoom
        center=(lo+hi)/2;origin=QPointF(self.width()/2-center[0]*scale,self.height()/2+center[1]*scale)+self.pan
        self._metric_cache=key,(scale,origin);return scale,origin

    def screen(self,points):return [self.pixel(p) for p in self.project(points)]

    def nearest(self,pos):
        index=super().nearest(pos)
        if index>=0:return index
        for i,(verts,faces) in reversed(list(enumerate(self.contact_surfaces))):
            points=self.screen(verts)
            for face in faces:
                if QPolygonF([points[j] for j in face]).containsPoint(pos,Qt.FillRule.OddEvenFill):return i
            if any((pt-pos).manhattanLength()<8 for pt in points):return i
        return -1

    def paintEvent(self,event):
        p=QPainter(self);p.setRenderHint(QPainter.RenderHint.Antialiasing);p.fillRect(self.rect(),QColor('#18202b'))
        if self.geometry is None:return
        scale,_=self.transform();step=10**math.floor(math.log10(max(1e-9,65/scale)))
        while step*scale<35:step*=2
        lo=self.world(QPointF(0,self.height()));hi=self.world(QPointF(self.width(),0))
        p.setPen(QPen(QColor('#24313f'),1))
        for axis in (0,1):
            for value in np.arange(math.ceil(lo[axis]/step)*step,hi[axis]+step,step)[:150]:
                p.drawLine(self.pixel((value,lo[1])),self.pixel((value,hi[1]))) if axis==0 else p.drawLine(self.pixel((lo[0],value)),self.pixel((hi[0],value)))
        vertices,faces=self.bodies
        p.setBrush(QColor(63,91,110,100));p.setPen(Qt.PenStyle.NoPen)
        normal=np.cross(*self.basis().T)
        for face in sorted(faces,key=lambda f:float(vertices[f].mean(0)@normal)):
            p.drawPolygon(QPolygonF(self.screen(vertices[face])))
        if len(vertices):
            outline=None;kind=self.geometry.get('body_shape','tapered')
            if not self.projection.startswith('3D'):
                if self.axes()==(0,1) and kind not in ('wire','cylinder','prism','none'):
                    contour=body_polygon(self.geometry)
                    for element in range(self.geometry['shanks']):
                        pts=np.array([[x+element*self.geometry['shank_pitch_um'],y,0] for x,y in contour]);p.setPen(QPen(QColor('#698399'),1));p.setBrush(Qt.BrushStyle.NoBrush);p.drawPolygon(QPolygonF(self.screen(pts)))
                else:
                    from scipy.spatial import ConvexHull,QhullError
                    projected=self.project(vertices)
                    try:
                        hull=ConvexHull(projected);p.setPen(QPen(QColor('#698399'),1));p.setBrush(Qt.BrushStyle.NoBrush);p.drawPolygon(QPolygonF([self.pixel(v) for v in projected[hull.vertices]]))
                    except QhullError:pass
        for i,((verts,triangles),contact) in enumerate(zip(self.contact_surfaces,self.geometry['contacts'])):
            selected=i==self.selected_index
            p.setBrush(QColor('#f6ce7c' if selected else '#7ad4df'));p.setPen(Qt.PenStyle.NoPen)
            screen=self.screen(verts)
            bounds=QPolygonF(screen).boundingRect()
            if not self.rect().adjusted(-20,-20,20,20).intersects(bounds.toAlignedRect()):continue
            if max(bounds.width(),bounds.height())>=3:
                for face in triangles:p.drawPolygon(QPolygonF([screen[j] for j in face]))
                p.setPen(QPen(QColor('#fff4d4' if selected else '#75c9da'),1.5 if selected else .6))
                for a,b in self._contact_edges[i]:p.drawLine(screen[a],screen[b])
            center=self.screen([contact_center(contact,self.geometry)])[0]
            # A screen-size marker is just a selection aid; the surface keeps metric size.
            bounds=QPolygonF(screen).boundingRect()
            if max(bounds.width(),bounds.height())<3:p.drawEllipse(center,1.5,1.5)
            if self.show_labels and len(self.contact_surfaces)<300 or selected:
                p.setPen(QColor('#edf3fa'));p.drawText(center+QPointF(7,-7),contact['id'])
        if self.outline:
            p.setPen(QPen(QColor('#f6ce7c'),2));p.setBrush(Qt.BrushStyle.NoBrush)
            pixels=[self.pixel(v) for v in self.outline]
            p.drawPolyline(QPolygonF(pixels))
            for pt in pixels:p.drawEllipse(pt,4,4)
        p.setPen(QColor('#adbfce'))
        caption='3D · drag to orbit' if self.projection.startswith('3D') else 'X →   Y ↑' if self.axes()==(0,1) else 'Z →   Y ↑' if self.axes()==(2,1) else 'X →   Z ↑'
        p.drawText(QPointF(18,25),caption)
        if self.show_dimensions:
            g=self.geometry;p.drawText(QPointF(18,45),'Body unspecified · contact coordinates only' if g.get('body_shape')=='none' else f'{g["shank_length_um"]:g} µm length · {g["shank_width_um"]:g} µm width · {g["shank_thickness_um"]:g} µm thickness')
        bar=10**math.floor(math.log10(max(1e-9,90/scale)))
        while bar*scale<45:bar*=2
        start=QPointF(20,self.height()-28);p.setPen(QPen(QColor('#adbfce'),2));p.drawLine(start,start+QPointF(bar*scale,0));p.drawText(start+QPointF(0,-8),f'{bar:g} µm')
        p.end()

    def zoom_at(self,pos,factor):
        before=self.world(pos);self.zoom=float(np.clip(self.zoom*factor,.02,2000));self.pan+=pos-self.pixel(before);self.update()

    def event(self,event):
        if event.type()==QEvent.Type.NativeGesture:
            if event.gestureType()==Qt.NativeGestureType.ZoomNativeGesture:
                ratio=1-float(event.value())
                if np.isfinite(ratio) and ratio>0:self.zoom_at(QPointF(self.mapFromGlobal(event.globalPosition())),1/ratio)
                event.accept();return True
            if event.gestureType() in (Qt.NativeGestureType.BeginNativeGesture,Qt.NativeGestureType.EndNativeGesture):event.accept();return True
        return super().event(event)

    def wheelEvent(self,event):
        if not event.pixelDelta().isNull() and not event.modifiers()&Qt.KeyboardModifier.ControlModifier:
            self.pan+=QPointF(event.pixelDelta());self.update()
        else:
            delta=event.angleDelta().y() or event.pixelDelta().y();self.zoom_at(event.position(),math.exp(delta/600))
        event.accept()

    def mousePressEvent(self,event):
        self.setFocus();pos=event.position()
        if event.button()==Qt.MouseButton.MiddleButton or self._space or self.tool=='pan':self._panning=(pos,self.pan);return
        if self.tool=='outline' and event.button()==Qt.MouseButton.LeftButton:
            if len(self.outline)>=3 and (pos-self.pixel(self.outline[0])).manhattanLength()<12:self.finish_outline();return
            xy=self.world(pos)
            if self.snap:xy=np.round(xy/self.snap)*self.snap
            self.outline.append(xy.tolist());self.update();return
        if self.projection.startswith('3D'):
            if event.button()==Qt.MouseButton.LeftButton:self.selected_index=self.nearest(pos);self.selected.emit(self.selected_index)
            self._orbit=(pos,self.rotation.copy());self.update();return
        super().mousePressEvent(event)

    def mouseMoveEvent(self,event):
        if self._orbit:
            start,angles=self._orbit;delta=event.position()-start;self.rotation=angles+np.array([delta.x(),delta.y()])*.4;self.update();return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self,event):self._orbit=None;super().mouseReleaseEvent(event)
    def mouseDoubleClickEvent(self,event):
        if self.tool=='outline':self.finish_outline()

    def finish_outline(self):
        poly=self.outline.copy()
        if len(poly)>1 and np.allclose(poly[-1],poly[-2]):poly.pop()
        if len(poly)>=3:self.outlineFinished.emit(poly)

    def keyPressEvent(self,event):
        if event.key()==Qt.Key.Key_Space:self._space=True;event.accept();return
        if self.tool=='outline':
            if event.key() in (Qt.Key.Key_Return,Qt.Key.Key_Enter):self.finish_outline();return
            if event.key()==Qt.Key.Key_Escape:self.outline=[];self.update();return
            if event.key() in (Qt.Key.Key_Delete,Qt.Key.Key_Backspace):
                if self.outline:self.outline.pop();self.update()
                return
        arrows={Qt.Key.Key_Left:(-1,0),Qt.Key.Key_Right:(1,0),Qt.Key.Key_Up:(0,1),Qt.Key.Key_Down:(0,-1)}
        if event.key() in arrows and not self.projection.startswith('3D'):
            delta=np.zeros(3);a,b=self.axes();delta[[a,b]]=np.array(arrows[event.key()])*(self.snap or 1)*(10 if event.modifiers()&Qt.KeyboardModifier.ShiftModifier else 1);self.nudgeRequested.emit(delta.tolist());event.accept();return
        super().keyPressEvent(event)
    def focusOutEvent(self,event):self._space=False;super().focusOutEvent(event)
    def keyReleaseEvent(self,event):
        if event.key()==Qt.Key.Key_Space:self._space=False
        super().keyReleaseEvent(event)


class DesignerPreview(QWidget):
    def __init__(self,parent=None):
        super().__init__(parent);layout=QVBoxLayout(self);layout.setContentsMargins(0,0,0,0)
        tools=QHBoxLayout();self.view=QuietCombo();self.view.addItems(['Front · X/Y','Side · Z/Y','Top · X/Z','3D'])
        tools.addWidget(self.view);tools.addStretch();self.canvas=PhysicalCanvas()
        tools.addWidget(button('−',lambda:self.zoom(.8)));tools.addWidget(button('+',lambda:self.zoom(1.25)))
        tools.addWidget(button('Fit body',lambda:self.canvas.fit(False)));tools.addWidget(button('Fit contacts',lambda:self.canvas.fit(True)))
        tools.addWidget(button('Selected',self.fit_selected));layout.addLayout(tools)
        layout.addWidget(self.canvas,1)
        foot=QHBoxLayout();self.ids=QCheckBox('IDs');self.dims=QCheckBox('Dimensions');self.dims.setChecked(True)
        foot.addWidget(self.ids);foot.addWidget(self.dims);foot.addStretch();foot.addWidget(note('Pinch / wheel: zoom · Two fingers / Space + drag: pan'))
        layout.addLayout(foot)
        self.ids.toggled.connect(lambda v:self.display('show_labels',v));self.dims.toggled.connect(lambda v:self.display('show_dimensions',v))
        self.view.currentTextChanged.connect(self.projection)
    def projection(self,text):self.canvas.projection=text;self.canvas.fit()
    def display(self,key,v):setattr(self.canvas,key,v);self.canvas.update()
    def zoom(self,factor):self.canvas.zoom_at(QPointF(self.canvas.width()/2,self.canvas.height()/2),factor)
    def fit_selected(self):
        c=self.canvas;i=c.selected_index
        if i<0:return
        c.fit(True);pos=c.screen([contact_center(c.geometry['contacts'][i],c.geometry)])[0]
        c.zoom_at(pos,4);pos=c.screen([contact_center(c.geometry['contacts'][i],c.geometry)])[0]
        c.pan+=QPointF(c.width()/2,c.height()/2)-pos;c.update()


class ProbeDesigner(QDialog):
    def __init__(self,parent=None,template=None):
        super().__init__(parent);self.template=copy.deepcopy(template or {});self.result_probe=None
        self.g=copy.deepcopy(template['geometry']) if template else grid_geometry('My probe',rows=1,first_site=0,width=25,thickness=25,taper=0,site_width=25,body='wire')
        self.g['contacts']=copy.deepcopy(contacts_for(self.g))
        if self.g.get('body_shape')=='rectangle':
            poly=body_polygon(self.g).tolist();self.g.update(body_shape='polygon',body_polygon_um=poly,geometry_version=2,contour_preset='Rectangle')
        if template is None:
            self.g.update(geometry_version=2)
            self.g['contacts'][0].update(x_um=0,z_um=0,surface='terminal',shape='circle',u_xyz=[1,0,0],v_xyz=[0,0,1],width_um=25,height_um=25)
        self._undo=[];self._redo=[];self._loading=True;self._drag_saved=False
        self._initial=copy.deepcopy(self.g);self._saved_maker=self.template.get('manufacturer','Custom')
        self.setWindowTitle('Probe designer');self.resize(1230,850);self.setMinimumSize(1000,680)
        root=QVBoxLayout(self);head=QHBoxLayout();self.name=QLineEdit(self.g['model']);self.name.setPlaceholderText('Probe name')
        head.addWidget(QLabel('Name'));head.addWidget(self.name,1);root.addLayout(head)
        split=QSplitter(Qt.Orientation.Horizontal);root.addWidget(split,1);self.tabs=QTabWidget();split.addWidget(self.tabs)
        self.fields={};self.body_form=self.page('Body');self.body=QuietCombo()
        for title,key in [('Wire / needle','wire'),('Flat shape','polygon'),('Cylinder','cylinder'),('Shank / grid','tapered'),('Custom cross-section','prism'),('Contacts only','none')]:self.body.addItem(title,key)
        shape=self.g.get('body_shape','tapered');index=self.body.findData('polygon' if shape=='rectangle' else shape);self.body.setCurrentIndex(max(0,index));self.body_form.addRow('Physical shape',self.body)
        self.body_hint=note();self.body_form.addRow(self.body_hint)
        for key,title,gkey,lo,hi in [('length','Length','shank_length_um',1,100000),('width','Width / diameter','shank_width_um',.1,100000),('thickness','Thickness','shank_thickness_um',.1,10000),('taper','Sharpened length','taper_length_um',0,10000)]:self.field(self.body_form,key,title,lo,hi,self.g[gkey])
        self.preset=QuietCombo();self.preset.addItems(['Triangle','Rectangle','Custom contour']);self.body_form.addRow('Contour',self.preset)
        self.outline_tools=QWidget();row=QHBoxLayout(self.outline_tools);row.setContentsMargins(0,0,0,0)
        row.addWidget(button('Draw contour',self.draw_outline));row.addWidget(button('Close contour',lambda:self.preview.canvas.finish_outline()))
        self.body_form.addRow(self.outline_tools)
        self.vertices=QTableWidget(0,2);self.vertices.setHorizontalHeaderLabels(['X µm','Y µm']);self.vertices.setMaximumHeight(170)
        self.vertices.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch);self.body_form.addRow(self.vertices)
        self.draw_hint=note('Drawing: click vertices; click the first point or press Enter to close. Esc cancels. Body changes preserve contacts.');self.body_form.addRow(self.draw_hint)
        self.details=QLineEdit(self._saved_maker);self.maker=self.details;self.body_form.addRow('Manufacturer · optional',self.details)
        self.layout_form=self.page('Add contacts')
        self.pattern=QuietCombo()
        for title,key in [('Single contact','single'),('Row / grid','grid'),('Along contour / edges','perimeter'),('Terminal contact','terminal'),('Full rings','ring'),('Directional segments','sector'),('Exposed wire end','exposed_tip')]:self.pattern.addItem(title,key)
        self.layout_form.addRow('Arrangement',self.pattern)
        self.surface=QuietCombo();self.layout_form.addRow('Surface',self.surface)
        specs=[('count','Number',1,20000,8,True),('rows','Sites per column',1,5000,16,True),('columns','Columns',1,64,1,True),
            ('row_pitch','Row spacing',.1,100000,50,False),('column_pitch','Column spacing',.1,100000,20,False),
            ('first_site','Distance along body',0,100000,self.g['first_site_center_from_tip_um'],False),
            ('angle','Start angle',-360,360,0,False),('span','Angular width',.1,360,90,False),('offset','Inset from edge',0,10000,10,False),
            ('site_width','Contact width / diameter',.01,10000,self.g['site_size_um'],False),('site_height','Contact height / exposed length',.01,10000,self.g.get('site_height_um',self.g['site_size_um']),False)]
        for key,title,lo,hi,val,integer in specs:self.field(self.layout_form,key,title,lo,hi,val,integer)
        self.shank=QuietCount();self.shank.setRange(0,self.g['shanks']-1);self.layout_form.addRow('Element',self.shank)
        self.edge=QuietCombo();self.layout_form.addRow('Edges',self.edge)
        self.shape=QuietCombo();self.shape.addItems(['Circle','Rectangle','Square','Ellipse']);self.layout_form.addRow('Contact shape',self.shape)
        self.generate=button('Add contacts',self.generate_layout,True);self.layout_form.addRow(self.generate)
        self.layout_form.addRow(note('New contacts are added to the existing geometry. Delete a selected group to replace it.'))
        self.assembly_form=self.page('Array')
        self.field(self.assembly_form,'shanks','Repeated elements',1,64,self.g['shanks'],True)
        self.field(self.assembly_form,'shank_pitch','Centre spacing · X',.1,100000,self.g['shank_pitch_um'])
        self.assembly_form.addRow(button('Apply array',self.apply_array))
        self.assembly_form.addRow(note('Copies of element 0 replace existing copies; Undo restores them. The array shares one rigid placement. Independently moving electrodes belong to separate insertions.'))
        self.contact_page=QWidget();cl=QVBoxLayout(self.contact_page);tools=QHBoxLayout()
        self.select_button=button('Select / move',lambda:self.set_tool('select'));self.add_button=button('Add',lambda:self.set_tool('add'));self.pan_button=button('Pan',lambda:self.set_tool('pan'))
        for b in (self.select_button,self.add_button,self.pan_button):b.setCheckable(True);tools.addWidget(b)
        cl.addLayout(tools)
        self.selection_info=note('Select a contact on the canvas or in the table.');cl.addWidget(self.selection_info)
        self.table=QTableWidget(0,6);self.table.setHorizontalHeaderLabels(['ID','Element','Site','X µm','Y µm','Z µm'])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows);self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents);self.table.verticalHeader().hide();self.table.setShowGrid(False);cl.addWidget(self.table,1)
        self.contact_form=QFormLayout();self.contact_width=QuietSpin();self.contact_height=QuietSpin()
        for box in (self.contact_width,self.contact_height):box.setRange(.01,10000);box.setSuffix(' µm');box.setKeyboardTracking(False)
        self.contact_form.addRow('Selected width',self.contact_width);self.contact_form.addRow('Selected height',self.contact_height);cl.addLayout(self.contact_form)
        self.selected_shape=QuietCombo();self.selected_shape.addItems(['circle','rectangle','square','ellipse'])
        self.contact_form.addRow('Selected shape',self.selected_shape)
        self.selected_angle=QuietSpin();self.selected_angle.setRange(-360,360);self.selected_angle.setSuffix('°');self.selected_angle.setKeyboardTracking(False)
        self.selected_span=QuietSpin();self.selected_span.setRange(.1,360);self.selected_span.setSuffix('°');self.selected_span.setKeyboardTracking(False)
        self.contact_form.addRow('Start angle',self.selected_angle);self.contact_form.addRow('Angular width',self.selected_span)
        ops=QHBoxLayout();ops.addWidget(button('Delete',self.delete_selected));ops.addWidget(button('Duplicate',self.duplicate_selected));cl.addLayout(ops)
        io=QHBoxLayout();io.addWidget(button('Import CSV…',self.import_csv));io.addWidget(button('Export CSV…',self.export_csv));cl.addLayout(io)
        self.tabs.addTab(self.contact_page,'Contacts')
        self.preview=DesignerPreview();split.addWidget(self.preview);split.setSizes([360,850]);self.tabs.setMinimumWidth(310)
        row=QHBoxLayout();self.snap=QuietSpin();self.snap.setRange(0,1000);self.snap.setValue(5);self.snap.setSuffix(' µm');self.snap.setSpecialValueText('Off')
        row.addWidget(QLabel('Snap'));row.addWidget(self.snap);row.addWidget(note('All coordinates are in µm. Origin = chosen distal reference.'),1)
        root.addLayout(row)
        bottom=QHBoxLayout();self.undo_button=button('Undo',self.undo);self.redo_button=button('Redo',self.redo)
        bottom.addWidget(self.undo_button);bottom.addWidget(self.redo_button);self.status=note();bottom.addWidget(self.status,1)
        bottom.addWidget(button('Cancel',self.reject));self.save_button=button('Save probe',self.finish,True);bottom.addWidget(self.save_button);root.addLayout(bottom)
        self.body.currentIndexChanged.connect(self.body_changed);self.preset.currentIndexChanged.connect(self.preset_changed)
        for k in ('length','width','thickness','taper'):self.fields[k].valueChanged.connect(self.body_dimensions)
        self.vertices.itemChanged.connect(self.vertex_edited);self.pattern.currentIndexChanged.connect(self.pattern_changed);self.surface.currentIndexChanged.connect(self.surface_changed)
        self.name.textChanged.connect(self.name_changed);self.table.itemChanged.connect(self.table_edited);self.table.itemSelectionChanged.connect(self.table_selected)
        self.contact_width.valueChanged.connect(self.contact_dimensions);self.contact_height.valueChanged.connect(self.contact_dimensions)
        self.selected_shape.currentTextChanged.connect(self.contact_dimensions);self.selected_angle.valueChanged.connect(self.contact_dimensions);self.selected_span.valueChanged.connect(self.contact_dimensions)
        canvas=self.preview.canvas;canvas.selected.connect(self.select_contact);canvas.addContact.connect(self.add_contact)
        canvas.moveContact.connect(self.move_contact);canvas.deleteRequested.connect(self.delete_selected);canvas.outlineFinished.connect(self.accept_outline);canvas.nudgeRequested.connect(self.nudge_selected)
        self.snap.valueChanged.connect(lambda v:setattr(canvas,'snap',v))
        self.undo_shortcut=QShortcut(QKeySequence.StandardKey.Undo,self);self.undo_shortcut.activated.connect(self.undo)
        self.redo_shortcut=QShortcut(QKeySequence.StandardKey.Redo,self);self.redo_shortcut.activated.connect(self.redo)
        self._loading=False;self.update_context();self.refresh(reset=True);self.set_tool('select')
        if self.g.get('body_shape') in ('wire','cylinder'):
            self.preview.view.setCurrentText('3D')
        if self.g['contacts']:self.preview.canvas.fit(True)

    def page(self,title):
        scroll=QScrollArea();scroll.setWidgetResizable(True);scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        widget=QWidget();form=QFormLayout(widget);form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows);scroll.setWidget(widget);self.tabs.addTab(scroll,title);return form

    def field(self,form,key,title,lo,hi,value,integer=False):
        box=QuietCount() if integer else QuietSpin();box.setRange(lo,hi);box.setValue(value);box.setKeyboardTracking(False)
        if not integer:box.setDecimals(2);box.setSuffix('°' if key in ('angle','span') else ' µm')
        self.fields[key]=box;form.addRow(title,box)

    def show_field(self,form,key,visible):
        box=self.fields[key];box.setVisible(visible);label=form.labelForField(box)
        if label:label.setVisible(visible)
    def show_row(self,form,widget,visible):
        widget.setVisible(visible);label=form.labelForField(widget)
        if label:label.setVisible(visible)

    def remember(self):self._undo.append(copy.deepcopy(self.g));self._undo=self._undo[-50:];self._redo=[]
    def undo(self):
        if self._undo:self._redo.append(copy.deepcopy(self.g));self.g=self._undo.pop();self.refresh(sync=True)
    def redo(self):
        if self._redo:self._undo.append(copy.deepcopy(self.g));self.g=self._redo.pop();self.refresh(sync=True)
    def name_changed(self,text):
        if not self._loading:self.g['model']=text.strip();self.refresh(table=False)

    def update_context(self):
        kind=self.g.get('body_shape','tapered');flat=kind in ('polygon','rectangle');poly=kind in ('polygon','prism')
        self.draw_hint.setVisible(poly)
        for w in (self.preset,self.vertices,self.outline_tools):self.show_row(self.body_form,w,poly)
        self.vertices.setHorizontalHeaderLabels(['X µm','Z µm' if kind=='prism' else 'Y µm'])
        self.show_field(self.body_form,'thickness',kind not in ('wire','cylinder','none'));self.show_field(self.body_form,'taper',kind in ('wire','tapered'))
        self.show_field(self.body_form,'width',kind!='none');self.show_field(self.body_form,'length',kind!='none')
        self.body_hint.setText({'wire':'One insulated wire with a recording surface at its end. Add rows only if the physical wire has side contacts.',
            'polygon':'Draw any simple flat outline. Contacts can sit on either face or on the thin perimeter edge.',
            'prism':'Draw the X/Z cross-section. It is extruded along Y; contacts may sit on any side face.',
            'cylinder':'Place rings, directional sectors, tip contacts or individual side contacts around the body.',
            'none':'Only known recording surfaces. The body is unknown; contact coordinates remain fully editable.'}.get(kind,'Regular shank geometry. Rows and grids are optional contact-placement tools.'))
        current=self.surface.currentData();self.surface.blockSignals(True);self.surface.clear()
        choices=[('Front face','front'),('Back face','back'),('Perimeter edge','edge')] if flat else [('Side face','edge')] if kind=='prism' else [('Cylindrical surface','cylinder'),('Distal end','terminal')] if kind in ('wire','cylinder') else [('Front face','front'),('Back face','back')]
        if kind=='none':choices=[('X/Y plane · Z = 0','front')]
        for title,key in choices:self.surface.addItem(title,key)
        if self.surface.findData(current)>=0:self.surface.setCurrentIndex(self.surface.findData(current))
        elif kind=='wire':self.surface.setCurrentIndex(self.surface.findData('terminal'))
        self.surface.blockSignals(False)
        self.edge.clear();self.edge.addItem('Whole perimeter',None)
        if poly:
            for i in range(len(self.g.get('body_polygon_um',[]))):self.edge.addItem(f'Edge {i+1}',i)
        for i in range(self.pattern.count()):
            key=self.pattern.itemData(i);enabled=key!='perimeter' or poly
            if key in ('ring','sector','exposed_tip'):enabled=kind in ('wire','cylinder','none')
            self.pattern.model().item(i).setEnabled(enabled)
        self.pattern_changed()

    def pattern_changed(self,*args):
        mode=self.pattern.currentData()
        visible={'count':mode in ('perimeter','ring','sector'),'rows':mode=='grid','columns':mode=='grid',
            'row_pitch':mode in ('grid','ring','sector'),'column_pitch':mode=='grid','first_site':mode not in ('single',),
            'angle':mode in ('sector','single','grid') and self.g.get('body_shape') in ('cylinder','wire'),
            'span':mode=='sector','offset':mode=='perimeter' and self.g.get('body_shape')=='polygon',
            'site_width':mode not in ('ring','sector','exposed_tip'),'site_height':True}
        for key,value in visible.items():self.show_field(self.layout_form,key,value)
        self.show_row(self.layout_form,self.edge,mode=='perimeter');self.show_row(self.layout_form,self.surface,mode in ('perimeter','grid','single'))
        self.show_row(self.layout_form,self.shape,mode not in ('ring','sector','exposed_tip'))
        self.generate.setText('Place by clicking' if mode=='single' else 'Add contacts')

    def surface_changed(self,*args):
        if self._loading:return
        if self.g.get('body_shape')=='prism':self.preview.view.setCurrentText('Top · X/Z')
        elif self.surface.currentData()=='terminal':self.preview.view.setCurrentText('Top · X/Z')
        else:self.preview.view.setCurrentText('Front · X/Y')

    def body_changed(self,*args):
        if self._loading:return
        self.remember();kind=self.body.currentData();self.g.update(body_shape=kind,geometry_version=2)
        if kind in ('polygon','prism'):
            self._set_preset('Triangle')
        if kind in ('wire','cylinder'):self.g['shank_thickness_um']=self.g['shank_width_um']
        self.update_context();self.refresh(reset=True,sync=True)

    def _set_preset(self,name):
        w=self.g['shank_width_um']/2;h=self.g['shank_length_um'] if self.g['body_shape']=='polygon' else self.g['shank_thickness_um']
        self.g['contour_preset']=name
        self.g['body_polygon_um']=[[0,0],[-w,h],[w,h]] if name=='Triangle' else [[-w,0],[-w,h],[w,h],[w,0]]
        if self.g['body_shape']=='prism':self.g['body_polygon_um']=[[x,y-h/2] for x,y in self.g['body_polygon_um']]
    def preset_changed(self,*args):
        if self._loading or self.preset.currentText()=='Custom contour':return
        self.remember();self._set_preset(self.preset.currentText());self.refresh(reset=True)
    def body_dimensions(self,*args):
        if self._loading:return
        self.remember();old=[self.g['shank_width_um'],self.g['shank_length_um'],self.g['shank_thickness_um']]
        for key,gkey in [('width','shank_width_um'),('length','shank_length_um'),('thickness','shank_thickness_um'),('taper','taper_length_um')]:self.g[gkey]=self.fields[key].value()
        if self.g.get('body_shape') in ('wire','cylinder'):self.g['shank_thickness_um']=self.g['shank_width_um']
        if 'body_polygon_um' in self.g:
            a,b=(self.g['shank_width_um']/old[0],self.g['shank_thickness_um']/old[2] if self.g.get('body_shape')=='prism' else self.g['shank_length_um']/old[1])
            self.g['body_polygon_um']=(np.asarray(self.g['body_polygon_um'])*[a,b]).tolist()
        self.refresh(table=False)

    def draw_outline(self):
        self.preview.view.setCurrentText('Top · X/Z' if self.g.get('body_shape')=='prism' else 'Front · X/Y')
        self.preview.canvas.outline=[];self.set_tool('outline');self.status.setText('Click vertices · Enter closes · Esc cancels')
    def accept_outline(self,poly):
        if not polygon_valid(poly):self.status.setText('Contour crosses itself or contains duplicate points.');return
        if self.g.get('body_shape')!='prism' and min(p[1] for p in poly)<0:self.status.setText('Keep the flat body above the distal reference Y = 0.');return
        self.remember();self.g.update(body_polygon_um=poly,geometry_version=2,contour_preset='Custom contour')
        arr=np.asarray(poly);self.g['shank_width_um']=max(1.,2*max(abs(arr[:,0])))
        if self.g.get('body_shape')=='prism':self.g['shank_thickness_um']=max(1.,2*max(abs(arr[:,1])))
        else:self.g['shank_length_um']=max(1.,max(arr[:,1]))
        self.preview.canvas.outline=[];self.set_tool('select');self.refresh(sync=True)
    def vertex_edited(self,item):
        if self._loading:return
        try:
            poly=[[float(self.vertices.item(r,c).text()) for c in range(2)] for r in range(self.vertices.rowCount())]
            if not polygon_valid(poly):raise ValueError()
            self.accept_outline(poly)
        except ValueError:self.status.setText('Enter a simple closed contour with finite coordinates.');self.refresh()

    def add_specs(self,specs):
        if len(specs)+len(self.g['contacts'])>MAX_CONTACTS:raise ValueError(f'Use at most {MAX_CONTACTS:,} contacts.')
        self.remember()
        for spec in specs:
            xyz=spec.pop('xyz');c=new_contact(self.g,xyz,shank=self.shank.value(),width_um=self.fields['site_width'].value(),height_um=self.fields['site_height'].value(),shape=self.shape.currentText().lower(),**spec)
            self.g['contacts'].append(c)
        self.g['geometry_version']=2;self.g.pop('generator',None)
        self.preview.canvas.selected_index=len(self.g['contacts'])-1;self.refresh()

    def generate_layout(self):
        mode=self.pattern.currentData();f=lambda key:self.fields[key].value();g=self.g;specs=[];surface=self.surface.currentData()
        try:
            if mode=='single':self.set_tool('add');return
            if mode=='perimeter':
                gcopy=copy.deepcopy(g);gcopy['contact_level_um']=f('first_site')
                specs=perimeter_contacts(gcopy,int(f('count')),face=surface,edge=self.edge.currentData(),offset=f('offset'))
            elif mode in ('ring','sector','exposed_tip'):
                count=int(f('count')) if mode!='exposed_tip' else 1
                specs=[dict(xyz=[0,f('first_site')+i*f('row_pitch'),0],surface=mode,radius_um=g['shank_width_um']/2,angle_deg=f('angle'),span_deg=f('span')) for i in range(count)]
            elif mode=='terminal':specs=[dict(xyz=[0,f('first_site'),0],surface='terminal',u_xyz=[1,0,0],v_xyz=[0,0,1])]
            elif mode=='grid':
                rows,cols=int(f('rows')),int(f('columns'))
                if rows*cols>MAX_CONTACTS:raise ValueError('Too many contacts.')
                for r in range(rows):
                    for col in range(cols):
                        x=(col-(cols-1)/2)*f('column_pitch');y=f('first_site')+r*f('row_pitch')
                        if surface=='cylinder':
                            angle=np.deg2rad(f('angle'))+col*f('column_pitch')/(g['shank_width_um']/2);radius=g['shank_width_um']/2
                            specs.append(dict(xyz=[radius*np.cos(angle),y,radius*np.sin(angle)],surface='cylindrical_patch',u_xyz=[-np.sin(angle),0,np.cos(angle)],v_xyz=[0,1,0]))
                        else:specs.append(dict(xyz=[x,y,0 if g.get('body_shape')=='none' else g['shank_thickness_um']/2*(-1 if surface=='back' else 1)],surface='patch'))
            self.add_specs(specs)
        except (ValueError,IndexError) as error:self.status.setText(str(error))

    def set_tool(self,tool):
        self.preview.canvas.tool=tool
        for b,key in [(self.select_button,'select'),(self.add_button,'add'),(self.pan_button,'pan')]:b.setChecked(tool==key)
        self.preview.canvas.setCursor(Qt.CursorShape.CrossCursor if tool in ('add','outline') else Qt.CursorShape.OpenHandCursor if tool=='pan' else Qt.CursorShape.ArrowCursor)
        if tool=='add':
            self.surface_changed()
            if self.preview.view.currentText()=='3D':self.preview.view.setCurrentText('Front · X/Y')
            self.status.setText('Click a position. Surface is chosen in Add contacts.')

    def add_contact(self,values):
        xyz=np.array([values['x_um'],values['y_um'],values['z_um']]);surface=self.surface.currentData();g=self.g
        props=dict(surface='patch')
        if g.get('body_shape') in ('wire','cylinder'):
            a=np.deg2rad(self.fields['angle'].value());r=g['shank_width_um']/2
            if surface=='terminal':xyz=[xyz[0],0,xyz[2]];props.update(surface='terminal',u_xyz=[1,0,0],v_xyz=[0,0,1])
            else:xyz=[r*np.cos(a),xyz[1],r*np.sin(a)];props.update(surface='cylindrical_patch',u_xyz=[-np.sin(a),0,np.cos(a)],v_xyz=[0,1,0])
        elif surface=='edge' and g.get('body_shape') in ('polygon','prism'):
            poly=body_polygon(g);probe=xyz[[0,2]] if g['body_shape']=='prism' else xyz[:2]
            options=[]
            for a,b in zip(poly,np.roll(poly,-1,axis=0)):
                d=b-a;t=np.clip(np.dot(probe-a,d)/max(np.dot(d,d),1e-20),0,1);point=a+t*d
                options.append((np.linalg.norm(probe-point),point,d/np.linalg.norm(d)))
            _,point,d=min(options,key=lambda value:value[0])
            if g['body_shape']=='prism':xyz=[point[0],self.fields['first_site'].value(),point[1]];props.update(u_xyz=[d[0],0,d[1]],v_xyz=[0,1,0])
            else:xyz=[point[0],point[1],0];props.update(u_xyz=[d[0],d[1],0],v_xyz=[0,0,1])
        else:xyz[2]=0 if g.get('body_shape')=='none' else g['shank_thickness_um']/2*(-1 if surface=='back' else 1)
        self.add_specs([dict(xyz=xyz,**props)])

    def apply_array(self):
        count=int(self.fields['shanks'].value());pitch=self.fields['shank_pitch'].value()
        if count*self.g['shank_width_um']>0 and pitch<self.g['shank_width_um'] and count>1:self.status.setText('Element spacing is smaller than the body width.');return
        original=[copy.deepcopy(c) for c in self.g['contacts'] if c['shank']==0]
        if len(original)*count>MAX_CONTACTS:self.status.setText('Too many contacts.');return
        self.remember();self.g.update(shanks=count,shank_pitch_um=pitch)
        self.g['contacts']=[]
        for s in range(count):
            for c in original:
                c=copy.deepcopy(c);c.update(shank=s,id=f's{s}e{c["electrode"]}');self.g['contacts'].append(c)
        self.refresh(reset=True)

    def refresh(self,reset=False,table=True,sync=False):
        self._loading=True
        if sync:
            self.preset.setCurrentText(self.g.get('contour_preset','Custom contour'))
            self.name.setText(self.g['model']);kind=self.g.get('body_shape','tapered');index=self.body.findData(kind if kind!='rectangle' else 'polygon');self.body.setCurrentIndex(max(0,index))
            for key,gkey in [('length','shank_length_um'),('width','shank_width_um'),('thickness','shank_thickness_um'),('taper','taper_length_um'),('shanks','shanks'),('shank_pitch','shank_pitch_um')]:self.fields[key].setValue(self.g[gkey])
            self.update_context()
        self.shank.setMaximum(self.g['shanks']-1);self.preview.canvas.active_shank=self.shank.value()
        self.preview.canvas.set_geometry(self.g,reset)
        poly=self.g.get('body_polygon_um',[]);self.vertices.setRowCount(len(poly))
        for r,point in enumerate(poly):
            for c,value in enumerate(point):self.vertices.setItem(r,c,QTableWidgetItem(f'{value:g}'))
        if table:
            self.table.blockSignals(True);self.table.setRowCount(len(self.g['contacts']))
            for r,contact in enumerate(self.g['contacts']):
                for c,key in enumerate(('id','shank','electrode','x_um','y_um','z_um')):
                    value=contact[key];self.table.setItem(r,c,QTableWidgetItem(f'{value:g}' if isinstance(value,float) else str(value)))
            self.table.blockSignals(False)
        try:
            validate_geometry(self.g);self.save_button.setEnabled(True);self.status.setStyleSheet('');self.status.setText(f'{len(self.g["contacts"]):,} recording surfaces · {self.g["shanks"]} element'+('s' if self.g['shanks']!=1 else ''))
        except (ValueError,TypeError,KeyError) as error:self.save_button.setEnabled(False);self.status.setStyleSheet('color:#f2ba83;');self.status.setText(str(error))
        self.undo_button.setEnabled(bool(self._undo));self.redo_button.setEnabled(bool(self._redo));self._loading=False

    def select_contact(self,index):
        self.table.blockSignals(True);self.table.clearSelection()
        if index>=0:self.table.selectRow(index);self.table.scrollToItem(self.table.item(index,0))
        self.table.blockSignals(False);self.update_selection(index)
    def table_selected(self):
        if self._loading:return
        rows=self.table.selectionModel().selectedRows();i=rows[0].row() if rows else -1;self.preview.canvas.selected_index=i;self.preview.canvas.update();self.update_selection(i)
    def update_selection(self,index):
        self._loading=True
        if index>=0:
            c=self.g['contacts'][index];self.selection_info.setText(c['id']+' · '+c.get('surface','patch')+' · '+c.get('shape',self.g.get('contact_shape','square')))
            self.contact_width.setValue(c.get('width_um',self.g['site_size_um']));self.contact_height.setValue(c.get('height_um',self.g.get('site_height_um',self.g['site_size_um'])))
            self.selected_shape.setCurrentText(c.get('shape',self.g.get('contact_shape','square')))
            self.selected_angle.setValue(c.get('angle_deg',0));self.selected_span.setValue(c.get('span_deg',360))
            curved=c.get('surface')=='sector'
            self.show_row(self.contact_form,self.selected_angle,curved);self.show_row(self.contact_form,self.selected_span,curved)
            self.show_row(self.contact_form,self.selected_shape,c.get('surface','patch') not in ('ring','sector','exposed_tip'))
        else:self.selection_info.setText('Select a contact on the canvas or in the table.')
        self.contact_width.setEnabled(index>=0);self.contact_height.setEnabled(index>=0);self._loading=False
    def selected_rows(self):
        rows={i.row() for i in self.table.selectionModel().selectedRows()}
        if not rows and self.preview.canvas.selected_index>=0:rows={self.preview.canvas.selected_index}
        return rows
    def contact_dimensions(self,*args):
        if self._loading:return
        rows=self.selected_rows()
        if not rows:return
        self.remember()
        for i in rows:
            c=self.g['contacts'][i];c.update(width_um=self.contact_width.value(),height_um=self.contact_height.value(),shape=self.selected_shape.currentText())
            if c.get('surface')=='sector':c.update(angle_deg=self.selected_angle.value(),span_deg=self.selected_span.value())
        self.refresh(table=False)
    def table_edited(self,item):
        if self._loading:return
        r,col=item.row(),item.column()
        try:
            value=item.text().strip() if col==0 else int(item.text()) if col<3 else float(item.text())
            if col>=3 and not math.isfinite(value):raise ValueError()
        except ValueError:self.refresh();return
        self.remember();self.g['contacts'][r][('id','shank','electrode','x_um','y_um','z_um')[col]]=value;self.refresh(table=False)
    def move_contact(self,index,xyz,finished):
        if not self._drag_saved:self.remember();self._drag_saved=True
        contact=self.g['contacts'][index]
        if contact.get('surface')=='terminal':xyz[1]=contact['y_um']
        elif contact.get('surface')=='cylindrical_patch':
            angle=np.arctan2(xyz[2],xyz[0]);radius=self.g['shank_width_um']/2;xyz[0]=radius*np.cos(angle);xyz[2]=radius*np.sin(angle)
        contact.update(zip(('x_um','y_um','z_um'),map(float,xyz)));self.refresh(table=finished)
        if finished:self._drag_saved=False;self.select_contact(index)
    def nudge_selected(self,delta):
        rows=self.selected_rows()
        if not rows:return
        self.remember()
        for i in rows:
            c=self.g['contacts'][i]
            for key,step in zip(('x_um','y_um','z_um'),delta):c[key]+=step
        self.refresh()
        if len(rows)==1:self.select_contact(next(iter(rows)))

    def delete_selected(self):
        rows=self.selected_rows()
        if rows:self.remember();self.g['contacts']=[c for i,c in enumerate(self.g['contacts']) if i not in rows];self.preview.canvas.selected_index=-1;self.refresh()
    def duplicate_selected(self):
        rows=self.selected_rows()
        if not rows:return
        if len(rows)+len(self.g['contacts'])>MAX_CONTACTS:return
        self.remember()
        for i in sorted(rows):
            c=copy.deepcopy(self.g['contacts'][i]);c['y_um']+=self.snap.value() or 10
            extra={k:v for k,v in c.items() if k not in ('id','electrode','shank','x_um','y_um','z_um')}
            self.g['contacts'].append(new_contact(self.g,[c['x_um'],c['y_um'],c['z_um']],shank=c['shank'],**extra))
        self.refresh()
    def import_csv(self):
        path,_=QFileDialog.getOpenFileName(self,'Import contact centres','','CSV (*.csv *.tsv)')
        if not path:return
        try:
            contacts=parse_contacts_csv(Path(path).read_text(encoding='utf-8-sig'))
            if not contacts or any(c['shank']<0 or c['shank']>=64 for c in contacts):raise ValueError('Check contact and element IDs.')
            self.remember();self.g['contacts']=contacts;self.g['shanks']=max(c['shank'] for c in contacts)+1;self.refresh(reset=True,sync=True)
        except (OSError,ValueError) as e:QMessageBox.warning(self,'Could not import',str(e))
    def export_csv(self):
        path,_=QFileDialog.getSaveFileName(self,'Export contact centres','contacts.csv','CSV (*.csv)')
        if path:
            try:Path(path).write_text(contacts_csv(self.g),encoding='utf-8')
            except OSError as e:QMessageBox.warning(self,'Could not export',str(e))
    def finish(self):
        try:validate_geometry(self.g)
        except (ValueError,KeyError,TypeError) as e:self.status.setText(str(e));return
        self.result_probe=dict(schema_version=1,name=self.g['model'],manufacturer=self.maker.text().strip(),geometry=copy.deepcopy(self.g),
            notes=self.template.get('notes',''),sources=self.template.get('sources',[]),builtin=False)
        if self.template.get('id') and not self.template.get('builtin'):self.result_probe['id']=self.template['id']
        self.accept()
    def reject(self):
        if self.g!=self._initial or self.maker.text()!=self._saved_maker:
            result=QMessageBox.question(self,'Discard changes?','Close without saving this probe?',QMessageBox.StandardButton.Discard|QMessageBox.StandardButton.Cancel,QMessageBox.StandardButton.Cancel)
            if result!=QMessageBox.StandardButton.Discard:return
        super().reject()

"""A quiet atlas-surface view for matching insertions to photographs."""
import numpy as np

from qtpy.QtCore import QThreadPool,QTimer
from qtpy.QtWidgets import QFormLayout,QSizePolicy,QWidget

from .desktop_widgets import PopupComboBox
from .surface_entries import build_atlas_surface,probe_surface_entries
from .landmarks import transform_points


def surface_label_positions(points,texts):
    """Keep compact numbers close to their dots without overlapping each other."""
    positions=np.empty((len(points),2)); boxes=[]
    offsets=[(x,y) for x in range(-30,37,6) for y in range(-30,31,6)]
    for i in sorted(range(len(points)),key=lambda i:(points[i,1],points[i,0])):
        width=max(7,len(texts[i])*5.5); height=10
        candidates=[]
        for x,y in offsets:
            position=points[i]+[x,y]; box=(position[0]-1,position[1]-height/2-1,position[0]+width+1,position[1]+height/2+1)
            overlap=sum(max(0,min(box[2],b[2])-max(box[0],b[0]))*max(0,min(box[3],b[3])-max(box[1],b[1])) for b in boxes)
            covered=np.count_nonzero((points[:,0]>box[0]-3)&(points[:,0]<box[2]+3)&(points[:,1]>box[1]-3)&(points[:,1]<box[3]+3))
            candidates.append((overlap*10000+covered*1000+(x-7)**2+(y+6)**2,position,box))
        _,positions[i],box=min(candidates,key=lambda c:c[0]); boxes.append(box)
    return positions


class SurfaceEntryVisuals:
    def __init__(self,view,surface,*,show_surface=True,matrix=None):
        from vispy import scene
        self.view,self.surface=view,surface
        self.show_surface=show_surface; self.matrix=np.eye(4) if matrix is None else np.asarray(matrix)
        self.records=[]; self.active=False
        self._extent=float(np.linalg.norm(np.ptp(surface.vertices_zyx_um,axis=0))*2)
        self._update_timer=QTimer(view); self._update_timer.setSingleShot(True)
        self._update_timer.timeout.connect(self.update_labels)
        self.mesh=scene.visuals.Mesh(vertices=transform_points(surface.vertices_zyx_um,self.matrix)[:,::-1],faces=surface.faces,
            color=(.70,.74,.78,1.),shading='smooth',parent=view.view.scene)
        self.mesh.set_gl_state('opaque',depth_test=True,cull_face=False)
        self.mesh.shading_filter.ambient_light=(1,1,1,.55)
        self.mesh.shading_filter.diffuse_light=(1,1,1,.45)
        self.mesh.shading_filter.specular_light=(1,1,1,.1)
        self.mesh.order=1
        self.markers=scene.visuals.Markers(parent=view.view.scene)
        self.markers.set_gl_state('translucent',depth_test=False); self.markers.order=35
        self.labels=scene.visuals.Text('',parent=view.canvas.scene,font_size=7,
            anchor_x='left',anchor_y='center',color='white')
        self.labels.set_gl_state('translucent',depth_test=False); self.labels.order=42
        self.label_shadow=scene.visuals.Text('',parent=view.canvas.scene,font_size=7,
            anchor_x='left',anchor_y='center',color='#080c12')
        self.label_shadow.set_gl_state('translucent',depth_test=False); self.label_shadow.order=41
        self.leaders=scene.visuals.Line(parent=view.canvas.scene,connect='segments',width=1)
        self.leaders.set_gl_state('translucent',depth_test=False); self.leaders.order=40
        view.view.camera.transform.changed.connect(self.schedule_update)
        view.canvas.events.resize.connect(self.schedule_update)
        self.set_visible(False)

    def dispose(self):
        self.view.view.camera.transform.changed.disconnect(self.schedule_update)
        self.view.canvas.events.resize.disconnect(self.schedule_update)
        self._update_timer.stop(); self._update_timer.deleteLater()
        for visual in (self.mesh,self.markers,self.labels,self.label_shadow,self.leaders): visual.parent=None

    def schedule_update(self,event=None):
        if self.active: self._update_timer.start(0)

    def set_visible(self,visible):
        self.active=visible; self.mesh.visible=visible and self.show_surface
        if visible: self.update_labels()
        else:
            self._update_timer.stop()
            self.markers.visible=self.labels.visible=self.label_shadow.visible=self.leaders.visible=False

    def set_entries(self,records):
        self.records=records
        self.set_visible(True)

    def update_labels(self,event=None):
        if not self.active: return
        from vispy.color import Color
        points=np.array([r['atlas_zyx_um'] for r in self.records],float).reshape(-1,3)
        toward=np.asarray(self.view.view.camera.transform.map([0,0,1,0]))[:3][::-1]
        toward=np.linalg.solve(self.matrix[:3,:3],toward)
        toward/=max(np.linalg.norm(toward),1e-12)
        distance=self._extent
        epsilon=min(self.surface.spacing_um_zyx)*.02
        # Hide entries behind the brain, including after a display-only flip.
        visible=[i for i,p in enumerate(points)
                 if self.surface.entry(p+toward*epsilon,p+toward*distance) is None]
        self.visible_records=[self.records[i] for i in visible]
        if not visible:
            self.markers.visible=self.labels.visible=self.label_shadow.visible=self.leaders.visible=False; return
        xyz=transform_points(points[visible],self.matrix)[:,::-1]
        colors=np.array([Color(self.records[i]['color']).rgba for i in visible])
        self.markers.set_data(pos=xyz.astype(np.float32),face_color=colors,edge_color='#f2f7fa',
                              size=7,edge_width=.8)
        self.markers.visible=True
        transform=self.markers.get_transform(map_from='visual',map_to='canvas')
        projected=transform.map(xyz); projected=projected[:,:2]/projected[:,3,None]
        texts=[f"{self.records[i]['name']}{self.records[i]['shank']}" for i in visible]
        positions=surface_label_positions(projected,texts)
        self.labels.text=texts; self.labels.pos=np.asarray(positions); self.labels.color=colors
        self.labels.visible=True
        self.label_shadow.text=texts; self.label_shadow.pos=np.asarray(positions)+(1,1)
        self.label_shadow.visible=True
        self.leaders.visible=False


class SurfaceEntriesMixin:
    def _build_surface_entries_controls(self,layout):
        self._entry_surface=None; self._entry_surface_loading=None
        self._entry_visuals=None; self._entry_cache={}
        self.surface_entries_controls=QWidget()
        self.surface_entries_controls.setSizePolicy(QSizePolicy.Policy.Preferred,QSizePolicy.Policy.Maximum)
        form=QFormLayout(self.surface_entries_controls); form.setContentsMargins(0,0,0,0)
        self.placement_view=PopupComboBox()
        self.placement_view.addItem('Probes','probes')
        self.placement_view.addItem('Surface entries','entries')
        self.placement_view.setToolTip('Intersections of registered shanks with the estimated outer atlas surface · Shanks 0–3')
        self.placement_view.currentIndexChanged.connect(self._surface_entries_changed)
        form.addRow('View',self.placement_view)
        layout.insertWidget(0,self.surface_entries_controls)

    def _surface_entries_active(self):
        return self._mode==1 and self.placement_view.currentData()=='entries'

    def _surface_entries_changed(self,*args):
        if self._updating: return
        self._sync_surface_entries_mode()
        if self.project is not None and self.session is not None:
            self.draw_cached(annotations_only=True)
            self._placement_show_plane()
            self.owner.schedule_save()

    def _sync_surface_entries_mode(self):
        active=self._surface_entries_active()
        self.surface_entries_controls.setVisible(self._mode==1)
        self.volume_view.surface_entries_active=active
        self.volume_view.layers.setVisible(not active)
        self.volume_box.setTitle('3D view' if active else '3D layers / opacity')
        self.rendering_box.setVisible(not active)
        self.region_group.setVisible(self._mode in (0,1) and not active)
        self.visibility_table.setColumnHidden(1,active)
        self.volume_view._apply_layer_visibility()
        if not active:
            self.volume_view.surface_entry_bounds=None
            if self._entry_visuals is not None: self._entry_visuals.set_visible(False)
        elif self.project is not None and self.atlas_native is not None:
            self._draw_surface_entries()

    def _ensure_entry_surface(self):
        if self._entry_surface is not None: return True
        project=self.project
        if self._entry_surface_loading is project: return False
        from .desktop_app import Job
        volume=self.atlas_native
        self._entry_surface_loading=project
        job=Job(lambda progress:build_atlas_surface(volume)); self._jobs.add(job)
        self.owner.statusBar().showMessage('Preparing atlas surface…')
        def finish(surface=None,error=None):
            self._jobs.discard(job)
            if self._entry_surface_loading is project: self._entry_surface_loading=None
            if self._closing or self.project is not project: return
            if error:
                self.placement_view.setCurrentIndex(0)
                self.owner.statusBar().showMessage(error); return
            self._entry_surface=surface; self._entry_cache={}
            self._draw_surface_entries()
        job.signals.done.connect(lambda surface:finish(surface=surface))
        job.signals.failed.connect(lambda error:finish(error=error))
        QThreadPool.globalInstance().start(job)
        return False

    def _draw_surface_entries(self):
        if not self._surface_entries_active() or self.volume_view.view is None: return
        if not self._ensure_entry_surface(): return
        if self._entry_visuals is None:
            self._entry_visuals=SurfaceEntryVisuals(self.volume_view,self._entry_surface)
        records=[]; matrix=self.project.total
        for item in self.session['insertions']:
            pose=(item.get('probe') or {}).get('pose')
            if pose is None or not item.get('probe_visible',True): continue
            geometry=item['probe']['geometry']
            key=(tuple(pose['tip0_source_zyx_um']),tuple(pose['down_source_zyx']),tuple(pose['lateral_source_zyx']),
                 geometry['shank_pitch_um'],geometry['shank_length_um'],matrix.tobytes())
            cached=self._entry_cache.get(item['id'])
            if cached is None or cached[0]!=key:
                cached=(key,probe_surface_entries(item,matrix,self._entry_surface))
                self._entry_cache[item['id']]=cached
            records.extend(dict(r,name=item['name'],color=item['color']) for r in cached[1])
        self.volume_view.surface_entry_bounds=np.array([self._entry_surface.vertices_zyx_um.min(0),
                                                      self._entry_surface.vertices_zyx_um.max(0)])
        self._entry_visuals.set_entries(records)

    def _reset_surface_entries(self):
        if self._entry_visuals is not None: self._entry_visuals.dispose()
        self._entry_visuals=None; self._entry_surface=None
        self._entry_surface_loading=None; self._entry_cache={}
        self.volume_view.surface_entry_bounds=None

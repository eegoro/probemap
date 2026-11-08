"""Contact-side cue for the selected probe, following registration and camera."""
import numpy as np

from .landmarks import transform_points
from .neuropixels_model import probe_coordinates


class ProbeFaceIndicator:
    def __init__(self,volume_view):
        from vispy import scene
        self.view=volume_view
        self.normal_xyz=None
        self.side=None
        self.arrow=scene.visuals.Arrow(parent=volume_view.view.scene,width=2,
            arrow_size=8,arrow_type='triangle_60',color='#7adbe5',arrow_color='#7adbe5')
        # Screen-space text stays readable when the scene is mirrored.
        self.label=scene.visuals.Text('',parent=volume_view.canvas.scene,pos=(14,18),
            anchor_x='left',anchor_y='top',font_size=10,color='#7adbe5')
        for visual in (self.arrow,self.label):
            visual.order=40
            visual.set_gl_state('translucent',depth_test=False)
        # VisPy reuses the camera's MatrixTransform while orbiting; node
        # transform_change only fires when the object itself is replaced.
        volume_view.view.camera.transform.changed.connect(self.update_side)
        self.hide()

    def set_probe(self,insertion,matrix):
        settings=insertion['probe']; pose=settings['pose']
        # A single front/back arrow does not describe terminal, curved or multi-face contacts.
        if settings['geometry'].get('geometry_version',1)>=2:
            self.hide(); return
        self.name=insertion['name']
        normal=np.asarray(pose['normal_source_zyx'],float)
        # The leader points out of the actual recording face. It is an
        # annotation, not an additional component of the physical probe.
        height=min(1000.,settings['geometry']['shank_length_um']/2)
        anchor=probe_coordinates(settings,settings['geometry']['shanks']-1,height,face=True)
        self.endpoints_xyz=transform_points([anchor,anchor+normal*300],matrix)[:,::-1]
        # Under scale/shear a face normal is a covector, not A @ normal.
        self.normal_xyz=np.linalg.solve(np.asarray(matrix)[:3,:3].T,normal)[::-1]
        self.normal_xyz/=np.linalg.norm(self.normal_xyz)
        self.arrow.set_data(pos=self.endpoints_xyz.astype(np.float32),
            arrows=self.endpoints_xyz.reshape(1,6).astype(np.float32))
        self.label.visible=True
        self.update_side()

    def update_side(self,event=None):
        if self.normal_xyz is None or not self.label.visible: return
        camera=self.view.view.camera
        # Camera +Z points towards the viewer, including camera reflections.
        toward_viewer=np.asarray(camera.transform.map([0,0,1,0]))[:3]
        cosine=float(np.dot(self.normal_xyz,toward_viewer)/max(np.linalg.norm(toward_viewer),1e-12))
        side='front' if cosine>.12 else 'back' if cosine<-.12 else 'edge'
        self.facing_cosine=cosine
        # Suppress a degenerate arrow when looking directly along its shaft.
        self.arrow.visible=abs(cosine)<.97
        text=f'{self.name} · Contacts · {side}'
        if text!=self.label.text:
            self.label.text=text
        if side!=self.side:
            self.side=side
            color='#7adbe5' if side=='front' else '#9aafc1'
            self.label.color=color
            self.arrow.set_data(color=color)
            self.arrow.arrow_color=color

    def hide(self):
        self.arrow.visible=self.label.visible=False
        self.normal_xyz=None

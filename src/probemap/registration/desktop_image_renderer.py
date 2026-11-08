"""VisPy image kernels shared with napari, for the filtered 2D viewport."""
import numpy as np

_renderer=None


class ImageRenderer:
    def __init__(self):
        from vispy import scene
        self.canvas=scene.SceneCanvas(show=False,bgcolor='black',size=(64,64))
        self.view=self.canvas.central_widget.add_view()
        self.view.camera=scene.cameras.PanZoomCamera(aspect=None,flip=(False,True,False))
        self.image=scene.visuals.Image(np.zeros((2,2,3),np.float32),parent=self.view.scene,method='subdivide')
        self.image.set_gl_state('opaque',depth_test=False)
        from qtpy.QtWidgets import QApplication
        QApplication.instance().aboutToQuit.connect(self.canvas.close)

    def render(self,data,interpolation,size,rect,rotation):
        from vispy.visuals.transforms import MatrixTransform
        self.image.set_data(data); self.image.interpolation=interpolation
        width,height=size; self.view.camera.rect=(0,0,width,height)
        rw,rh=(rect.height(),rect.width()) if rotation%180 else (rect.width(),rect.height())
        angle=np.deg2rad(rotation); rotate=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        matrix=np.eye(4); matrix[:2,:2]=rotate@np.diag([rw/data.shape[1],rh/data.shape[0]])
        matrix[:2,3]=np.array([rect.center().x(),rect.center().y()])-matrix[:2,:2]@np.array([data.shape[1]/2,data.shape[0]/2])
        self.image.transform=MatrixTransform(matrix.T)
        from .desktop_gl import render_frame
        return render_frame(self.canvas,(width,height))


def render_image(data,interpolation,size,rect,rotation):
    global _renderer
    if _renderer is None: _renderer=ImageRenderer()
    return _renderer.render(data,interpolation,size,rect,rotation)

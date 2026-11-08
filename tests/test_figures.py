import json
import copy
import numpy as np
import pytest
from numpy.testing import assert_allclose


def test_region_palette_is_atlas_stable_and_separates_neighbors():
    from probemap.registration.region_colors import region_palette
    from skimage.color import rgb2lab
    data=np.zeros((24,24,24),np.uint16); data[:12,:,:12]=1; data[12:,:,:12]=2; data[:,:,12:]=3
    regions=[dict(id=i,name=str(i)) for i in (1,2,3)]
    palette=region_palette(data,regions)
    assert palette==region_palette(data.copy(),list(reversed(regions)))
    lab=rgb2lab(np.array([palette[str(i)] for i in (1,2,3)])[None]/255)[0]
    assert np.min(np.linalg.norm(lab[:,None]-lab[None,:],axis=2)+np.eye(3)*1000)>40


def test_reconstruction_tilt_is_measured_after_affine_in_atlas_axes():
    from test_reconstruction import sample_probe
    from probemap.registration.reconstruction import reconstruction_report
    session,item,matrix=sample_probe()
    direction=matrix[:3,:3]@np.array(item['probe']['pose']['down_source_zyx'])
    expected=np.degrees(np.arccos(abs(direction[1])/np.linalg.norm(direction)))
    assert_allclose(reconstruction_report(session,matrix)['probes'][0]['tilt_atlas_y_deg'],expected)
    item['probe']['pose']['down_source_zyx']=[0,-1,0]
    assert_allclose(reconstruction_report(session,np.eye(4))['probes'][0]['tilt_atlas_y_deg'],0)
    item['probe']['pose']['down_source_zyx']=[-1,0,0]
    assert_allclose(reconstruction_report(session,np.eye(4))['probes'][0]['tilt_atlas_y_deg'],90)


@pytest.mark.gui
def test_native_png_figures_restore_view_and_use_sequential_paths(tmp_path):
    from probemap.registration.macos_qt import initialize_qt
    app=initialize_qt()
    from probemap.registration.desktop_app import ProbeMapWindow
    from probemap.registration.desktop_model import AlignmentProject,ImageSpec
    from probemap.registration.volumes import write_volume
    from probemap.registration.desktop_figures import next_figure_path,export_slice,export_volume
    from qtpy.QtGui import QImage
    from test_desktop_tracks import wait_until
    z,y,x=np.indices((95,100,105)); data=(50000*np.exp(-((z-47)**2+(y-50)**2+(x-52)**2)/500)+x*60).astype(np.uint16)
    path=write_volume(tmp_path/'scan.tif',data,(20,25,30))
    p=AlignmentProject(ImageSpec(path,data.shape,(20,25,30)),ImageSpec(path,data.shape,(20,25,30)),tmp_path/'project',preview_dimension=32)
    w=ProbeMapWindow();w.resize(1400,950);w.show();errors=[];w._error=errors.append
    def ready(): return not w.busy and not w._render_running and not w.render_timer.isActive()
    try:
        w._load_project_volumes(p,1,project_path=tmp_path/'project/project.json')
        wait_until(app,lambda:ready() and w.overlay_panes[0].canvas.image is not None)
        w.overlay_panes[0].mode_selector.setCurrentIndex(w.overlay_panes[0].mode_selector.findData('side_by_side'))
        wait_until(app,lambda:ready() and w.overlay_panes[0].reference_canvas.image is not None)
        pane=w.overlay_panes[0];canvas=pane.reference_canvas
        original=copy.deepcopy(p.as_record());state=canvas.view_record(); image=canvas.image.copy()
        first=next_figure_path(p.output_dir); export_slice(w,canvas,first,long_edge=640)
        wait_until(app,ready)
        png=QImage(str(first));metadata=json.loads(png.text('Description'))
        assert first.name=='1.png' and first.stat().st_size>1000
        assert metadata['source']=='original TIFF samples' and png.width()>=2*canvas.width()
        assert canvas.view_record()==state and canvas.image==image
        svg=next_figure_path(p.output_dir,'.svg'); export_slice(w,canvas,svg,long_edge=640)
        wait_until(app,ready)
        import xml.etree.ElementTree as ET
        assert ET.parse(svg).getroot().tag.endswith('svg')
        assert svg.stat().st_size>1000
        view=w.volume_view; camera=view.camera_record(); shapes=[v._vol_shape for v in view.visuals]
        settings=view.render_settings_record(); transforms=[v.transform for v in view.visuals]
        # Camera framing must survive a different framebuffer size (including
        # Retina scaling); checking camera state alone misses a cropped PNG.
        from probemap.registration.desktop_gl import render_frame
        from PIL import Image
        view.layers.checks[1].setChecked(False)
        app.processEvents()
        camera=view.camera_record();settings=view.render_settings_record()
        preview=render_frame(view.canvas,view.canvas.physical_size)
        def bounds(pixels):
            yy,xx=np.where(pixels[:,:,1]>100)
            return np.array([xx.min()/pixels.shape[1],yy.min()/pixels.shape[0],
                             xx.max()/pixels.shape[1],yy.max()/pixels.shape[0]])
        second=next_figure_path(p.output_dir); export_volume(w,view,second,long_edge=640)
        wait_until(app,ready)
        assert second.name=='3.png' and second.stat().st_size>1000
        png=QImage(str(second));metadata=json.loads(png.text('Description'))
        assert all(stride==[1,1,1] for stride in metadata['native_strides'].values())
        assert_allclose(bounds(np.array(Image.open(second))),bounds(preview),atol=.07)
        assert view.camera_record()==camera and view.render_settings_record()==settings
        assert [v._vol_shape for v in view.visuals]==shapes and [v.transform for v in view.visuals]==transforms
        from vispy import scene
        triangle=scene.visuals.Mesh(vertices=np.array([[0,0,0],[200,0,0],[0,200,0]],np.float32),
            faces=np.array([[0,1,2]],np.uint32),color=(1,0,0,.5),parent=view.view.scene)
        vector_path=next_figure_path(p.output_dir,'.svg'); export_volume(w,view,vector_path)
        wait_until(app,ready)
        root=ET.parse(vector_path).getroot(); ns={'s':'http://www.w3.org/2000/svg'}
        assert root.findall('.//s:path',ns) and root.findall('.//s:image',ns)
        assert triangle.visible
        triangle.parent=None
        assert p.as_record()['coarse']==original['coarse'] and p.pairs==original['pairs'] and not errors,errors
        # Filtered 2D kernels use the same framebuffer path. A centered image
        # must stay centered instead of being clipped into a corner.
        from probemap.registration.desktop_image_renderer import render_image
        from qtpy.QtCore import QRectF
        pixels=np.zeros((50,50,3),np.float32);pixels[10:40,10:40]=1
        rendered=render_image(pixels,'cubic',(400,300),QRectF(100,50,200,200),0)
        yy,xx=np.where(rendered[:,:,:3].max(2)>80)
        assert_allclose([xx.min(),yy.min(),xx.max(),yy.max()],[139,89,260,210],atol=2)
    finally:
        wait_until(app,lambda:not w.busy); w.close();app.processEvents()


import pytest

pytestmark = pytest.mark.gui
import sys
import time

import numpy as np
from numpy.testing import assert_allclose

from probemap.registration.coarse import pose_matrix
from probemap.registration.desktop_model import AlignmentProject, ImageSpec
from probemap.registration.landmarks import transform_points
from probemap.registration.volumes import write_volume


def wait_until(app, predicate, timeout=15):
    until = time.monotonic()+timeout
    while time.monotonic() < until:
        app.processEvents()
        if predicate():
            return
        time.sleep(.01)
    raise AssertionError("Background task did not finish.")


def test_standalone_steps_points_export_and_resume(tmp_path):
    from probemap.registration.macos_qt import initialize_qt
    app = initialize_qt()
    before = {name for name in sys.modules if name.startswith("napari")}
    from probemap.registration.desktop_app import AlignWindow

    z,y,x = np.indices((24,28,32))
    data = (50000*np.exp(-((z-12)**2+(y-14)**2+(x-16)**2)/45)).astype(np.uint16)
    raw = write_volume(tmp_path/"raw.tif", data, (3,4,5))
    atlas = write_volume(tmp_path/"atlas.tif", data, (2,3,4))
    window = AlignWindow()
    errors = []
    window._error = errors.append
    window.atlas_input.set_spec(ImageSpec(atlas,data.shape,(2,3,4)))
    window.scan_input.set_spec(ImageSpec(raw,data.shape,(3,4,5)))
    window.output_path.setText(str(tmp_path/"results"))
    window.show()
    try:
        window.load_images()
        wait_until(app,lambda: not window.busy and window.reference is not None)
        assert not errors,errors
        assert window.step == 1
        window.pose_controls["rotation",2][1].setValue(90)
        window.pose_controls["scale",0][1].setValue(.8)
        assert_allclose(window.project.coarse[:3,:3], [[1,0,0],[0,0,1],[0,-.8,0]], atol=1e-10)
        window.go_step(2)
        wait_until(app,lambda: not window.busy and window.placed is not None)
        assert window.step == 2
        assert all(p.canvas.image.isGrayscale() for panes in window.landmark_panes.values() for p in panes)
        placed = np.array([[10,20,30],[25,20,30],[10,40,30],[10,20,60],[20,35,50.]])
        fine = pose_matrix(data.shape,(2,3,4),(1,2,3),(3,-2,4),(1.05,.95,1.1))
        ref = transform_points(placed,fine)
        for r,m in zip(ref,placed):
            window.add_point("reference",r)
            window.add_point("moving",m)
        coarse = window.project.coarse.copy()
        window.fit_points()
        assert window.step == 3
        assert window.project.fit_is_current
        assert_allclose(window.project.total,fine@coarse,atol=1e-10)
        window.review_opacity[0].setValue(32)
        window.review_opacity[1].setValue(71)
        wait_until(app,lambda:not window._render_running and not window.render_timer.isActive())
        # The overlay and 3D layer opacities are independently adjustable.
        window.volume_view.layers.sliders[0].setValue(32)
        window.volume_view.layers.sliders[1].setValue(71)
        assert window.volume_view.visuals[0].opacity == .32
        assert window.volume_view.visuals[1].opacity == .71
        voxel_zyx = np.array([3,4,5.])
        actual = window.volume_view.visuals[1].transform.map(np.r_[voxel_zyx[::-1],1])
        expected = window.project.total@np.r_[voxel_zyx*[3,4,5],1]
        assert_allclose(actual[:3],expected[:3][::-1])
        camera = window.volume_view.camera_record()
        window.save_project()
        project_file = tmp_path/"results/project.json"
        loaded = AlignmentProject.load(project_file)
        assert loaded.fit_is_current
        assert loaded.display["step"] == 3
        assert loaded.display["camera_3d"] == camera
        resumed = AlignWindow()
        resumed._error = errors.append
        resumed.show()
        try:
            resumed.load_project_path(project_file)
            wait_until(app,lambda:not resumed.busy and resumed.reference is not None)
            app.processEvents()
            assert resumed.step == 3
            assert resumed.project.fit_is_current
            assert resumed.pose_controls["rotation",2][1].value() == 90
            assert resumed.pose_controls["scale",0][1].value() == .8
            assert_allclose(resumed.volume_view.camera_record()["quaternion_wxyz"],camera["quaternion_wxyz"])
            resumed._pose_changed()
            assert_allclose(resumed.project.coarse,coarse)
        finally:
            resumed.close()
            wait_until(app,lambda:not resumed._render_running)
        # A failed project save must stop export instead of exporting an older
        # project snapshot that happens to exist on disk.
        output = window.project.output_dir
        invalid_output = tmp_path/"output_is_a_file"
        invalid_output.write_text("not a directory")
        window.project.output_dir = invalid_output
        saved_path=window.project_path
        window.project_path=invalid_output/'project.json'
        window.export(dict(path=tmp_path/'chosen-transform.npz',kind='transforms',stage='registration',format='npz'))
        assert errors
        assert not window.busy and window.last_export is None
        errors.clear()
        window.project.output_dir = output
        window.project_path=saved_path
        window.export(dict(path=tmp_path/'chosen-transform.npz',kind='transforms',stage='registration',format='npz'))
        wait_until(app,lambda:not window.busy)
        assert not errors,errors
        assert window.last_export.is_file()
        assert_allclose(np.load(window.last_export)["total_zyx_um"],fine@coarse)
        window.go_step(1)
        window.pose_controls["translation",0][1].setValue(12)
        assert not window.project.fit_is_current
        window.go_step(3)
        assert window.step == 1
        assert_allclose(window.reference.data.max(),1)
        # No napari import is needed to build, render, annotate, fit or export.
        assert not ({name for name in sys.modules if name.startswith("napari")}-before)
    finally:
        wait_until(app,lambda:not window.busy)
        window.close()
        wait_until(app,lambda:not window._render_running)


def test_skip_placement_keeps_preplaced_sidecar_origin(tmp_path):
    from probemap.registration.macos_qt import initialize_qt
    app=initialize_qt()
    from probemap.registration.desktop_app import AlignWindow
    data=np.random.default_rng(4).random((10,12,14),dtype=np.float32)
    path=write_volume(tmp_path/"placed.tif",data,(2,3,4))
    import json
    path.with_suffix(".geometry.json").write_text(json.dumps({"spacing_um_zyx":[2,3,4],"origin_um_zyx":[5,6,7]}))
    atlas=write_volume(tmp_path/"atlas.tif",data,(2,3,4))
    w=AlignWindow()
    errors=[]
    w._error=errors.append
    w.atlas_input.set_spec(ImageSpec(atlas,data.shape,(2,3,4)))
    w.scan_input.set_spec(ImageSpec(path,data.shape,(2,3,4),(5,6,7)))
    w.output_path.setText(str(tmp_path/"out"))
    w.show()
    try:
        w.load_images(skip_placement=True)
        wait_until(app,lambda:not w.busy and w.placed is not None)
        assert not errors,errors
        assert w.step==2
        assert_allclose(w.project.coarse[:3,3],[5,6,7])
    finally:
        w.close()


def test_full_bounds_landmarks_keep_legacy_coordinates_and_negative_positions(tmp_path):
    from probemap.registration.macos_qt import initialize_qt
    app=initialize_qt()
    from probemap.registration.desktop_app import AlignWindow
    data=np.random.default_rng(12).random((12,14,16),dtype=np.float32)
    path=write_volume(tmp_path/'raw.tif',data,(2,3,4))
    p=AlignmentProject(ImageSpec(path,data.shape,(2,3,4)),ImageSpec(path,data.shape,(2,3,4)),tmp_path/'out')
    p.coarse[:3,3]=[-30,-20,-10]
    point=[-18,-11,-2]
    p.pairs=[{'reference':[4,9,16],'moving':point.copy()}]
    # These are index-based fields written by the original cropped-view app.
    p.display={'step':2,'landmark_slices':{'reference':[2,3,4],'moving':[3,4,5]}}
    w=AlignWindow()
    errors=[]
    w._error=errors.append
    w.show()
    try:
        w._load_project_volumes(p,2)
        wait_until(app,lambda:not w.busy and w.placed is not None)
        assert not errors,errors
        assert w.project.pairs[0]['moving']==point
        assert_allclose([pane.world_depth for pane in w.landmark_panes['reference']],[4,9,16])
        assert_allclose([pane.world_depth for pane in w.landmark_panes['moving']],[6,12,20])
        assert np.asarray(w.landmark_grid['origin_um_zyx']).min()<0
        pane=w.landmark_panes['moving'][0]
        pane.set_world_depth(point[0])
        w.draw_landmarks()
        pixel=pane.canvas._pixel(point)
        assert pane.canvas.image_rect().contains(pixel)
        assert_allclose(pane.canvas._world(pixel),point,atol=1e-10)
        w.link_slices.setChecked(True)
        ref=w.landmark_panes['reference'][0]
        previous=ref.world_depth
        pane.set_world_depth(-20)
        # An individual adjustment preserves the other image's depth. The
        # shared control subsequently moves both with their offset intact.
        assert ref.world_depth==previous
        shared=w.landmark_views.shared_sliders[0]
        shared.setValue(shared.value()+1)
        assert pane.world_depth==-20+pane.spacing[0]
        assert ref.world_depth==previous+ref.spacing[0]
        file=w.save_project()
        restored=AlignmentProject.load(file)
        assert restored.display['landmark_positions_um_zyx']['moving'][0]==pane.world_depth
        again=AlignWindow()
        again._error=errors.append
        again.show()
        try:
            again.load_project_path(file)
            wait_until(app,lambda:not again.busy and again.placed is not None)
            assert again.landmark_panes['moving'][0].world_depth==pane.world_depth
            assert again.project.pairs==p.pairs
        finally:
            again.close()
    finally:
        w.close()


def test_independent_3d_render_controls_and_real_shader_modes(tmp_path):
    from probemap.registration.macos_qt import initialize_qt
    app=initialize_qt()
    from probemap.registration.desktop_app import AlignWindow
    from probemap.registration.desktop_rendering import METHODS
    z,y,x=np.indices((16,18,20))
    data=np.exp(-((z-8)**2+(y-9)**2+(x-10)**2)/20).astype(np.float32)
    path=write_volume(tmp_path/'raw.tif',data,(2,3,4))
    p=AlignmentProject(ImageSpec(path,data.shape,(2,3,4)),ImageSpec(path,data.shape,(2,3,4)),tmp_path/'out')
    w=AlignWindow()
    errors=[]
    w._error=errors.append
    w.show()
    try:
        w._load_project_volumes(p,1)
        wait_until(app,lambda:not w.busy and w.reference is not None)
        view=w.volume_view
        matrix=w.project.coarse.copy()
        gpu_transform=view.visuals[1].transform
        for offset in range(20):
            moved=matrix.copy()
            moved[2,3]+=offset
            view.update_alignment(moved,.65,.9)
            assert view.visuals[1].transform is gpu_transform
        view.update_alignment(matrix,.65,.9)
        view.open_settings()
        controls=view.settings_dialog.controls[1]
        controls['gamma'].setValue(.65)
        controls['colormap'].setCurrentIndex(controls['colormap'].findData('gray'))
        controls['blending'].setCurrentIndex(controls['blending'].findData('translucent_no_depth'))
        controls['low'].setValue(.1)
        controls['high'].setValue(.8)
        controls['opacity'].setValue(44)
        assert view.render_settings[0]['gamma']==1
        assert view.render_settings[1]['gamma']==.65
        assert view.visuals[1].opacity==.44
        assert view.layers.sliders[1].value()==44
        assert w.placement_opacity[1].value()==90
        assert_allclose(view.visuals[1].clim,[.1,.8])
        for method in METHODS.values():
            settings=view.render_settings_record()[1]
            settings['rendering']=method
            view.set_render_settings(1,settings)
            image=view.canvas.render(size=(160,130))
            assert image.shape==(130,160,4)
            assert view.visuals[1].method==method
        assert_allclose(w.project.coarse,matrix)
        w.save_project()
        restored=AlignmentProject.load(tmp_path/'out/project.json')
        assert restored.display['rendering_3d']==view.render_settings_record()
        assert_allclose(w.moving.data,w.reference.data)
        assert not errors,errors
    finally:
        wait_until(app,lambda:not w._render_running and not w.render_timer.isActive())
        w.close()


def test_review_comparison_shared_physical_slices_stages_and_resume(tmp_path,monkeypatch):
    from probemap.registration.macos_qt import initialize_qt
    app=initialize_qt()
    from probemap.registration.desktop_app import AlignWindow
    from probemap.registration.desktop_model import resample_preview_grid
    from probemap.registration.review import placement_without_scale
    z,y,x=np.indices((10,12,14))
    data=(100+400*z+30*y+x).astype(np.uint16)
    path=write_volume(tmp_path/'scan.tif',data,(2,3,4))
    atlas=write_volume(tmp_path/'atlas.tif',np.flip(data,axis=1).copy(),(2,3,4))
    p=AlignmentProject(ImageSpec(atlas,data.shape,(2,3,4),(10,0,-4)),
                       ImageSpec(path,data.shape,(2,3,4),(-8,6,4)),tmp_path/'out')
    p.coarse=pose_matrix(data.shape,(2,3,4),(-12,6,-8),(0,0,90),(1,.8,1.2))
    placed=transform_points(np.array([[2,3,4],[12,3,4],[2,24,4],[2,3,40],[12,24,40]]),p.coarse)
    p.pairs=[{'reference':(point+[4,-6,8]).tolist(),'moving':point.tolist()} for point in placed]
    p.fit()
    original_record=p.as_record()
    errors=[]
    w=AlignWindow()
    w._error=errors.append
    w.show()
    try:
        w._load_project_volumes(p,3)
        wait_until(app,lambda:not w.busy and w.reference is not None)
        w.preview_tabs.setCurrentIndex(1)
        view=w.slice_comparison
        def rendered():
            return (not w._render_running and not w.render_timer.isActive()
                    and all(canvas.image is not None for canvas in view.canvases))
        wait_until(app,rendered)
        assert w.preview_tabs.tabBar().isVisible()
        assert view.grid['origin_um_zyx'][0]<0
        from qtpy.QtCore import QPoint, QPointF, Qt
        from qtpy.QtGui import QWheelEvent
        canvas=view.canvases[0]
        canvas.wheelEvent(QWheelEvent(QPointF(50,50),QPointF(50,50),QPoint(),QPoint(0,120),
                                     Qt.MouseButton.NoButton,Qt.KeyboardModifier.ControlModifier,
                                     Qt.ScrollPhase.NoScrollPhase,False))
        assert all(c.zoom>1 for c in view.canvases)
        assert_allclose([c.zoom for c in view.canvases],[canvas.zoom]*3)
        canvas.mouseDoubleClickEvent(None)
        assert_allclose([c.zoom for c in view.canvases],[1]*3)
        for axis in range(3):
            view.plane.setCurrentIndex(axis)
            view.slider.setValue(view.slider.maximum()//2)
            wait_until(app,rendered)
            expected=[resample_preview_grid(volume,matrix,view.grid)
                      for volume,matrix in ((w.reference,np.eye(4)),(w.moving,placement_without_scale(p)),(w.moving,p.total))]
            for canvas,volume in zip(view.canvases,expected):
                plane=np.take(volume,view.slider.value(),axis=axis)
                assert canvas.axis==axis and canvas.image.isGrayscale()
                assert canvas.image.width()==plane.shape[1] and canvas.image.height()==plane.shape[0]
                pixels=np.array([[canvas.image.pixelColor(col,row).red() for col in range(plane.shape[1])]
                                 for row in range(plane.shape[0])])
                assert_allclose(pixels,plane*255,atol=1.01)
                assert_allclose(canvas.origin,view.grid['origin_um_zyx'])
                assert_allclose(canvas.spacing,view.grid['spacing_um_zyx'])
                assert_allclose(canvas.origin[axis]+canvas.index*canvas.spacing[axis],view.positions[axis],atol=.005)
        # Appearance is controlled on the left and shared by Before and After.
        w.review_tools.setCurrentIndex(2)
        view.plane.setCurrentIndex(0)
        view.slider.setValue(1)
        w.review_tools.slice_display.channel.setCurrentIndex(1)
        w.review_tools.slice_display.gamma.setValue(.7)
        wait_until(app,rendered)
        plane=np.take(resample_preview_grid(w.moving,placement_without_scale(p),view.grid),view.slider.value(),axis=0)
        canvas=view.canvases[1]
        assert_allclose([[canvas.image.pixelColor(col,row).red() for col in range(plane.shape[1])]
                         for row in range(plane.shape[0])],np.power(plane,.7)*255,atol=1.01)
        assert w.workspace_title.text()==w.nav_buttons[3].text()=='Review'
        from qtpy.QtWidgets import QPushButton,QFileDialog
        assert [b.text() for b in w.footer.findChildren(QPushButton) if b.isVisible()]==['Back','Save','Next']
        assert not w.source_summary.isVisible() and not w.save_button.isVisible()
        assert not w.volume_view.zoom_controls.isVisible() and not w.volume_view.settings_button.isVisible()
        assert w.volume_view.layers.parentWidget() is w.review_tools.volume_layout.parentWidget()
        w.resize(1080,720); app.processEvents()
        for tab in range(3):
            w.review_tools.setCurrentIndex(tab); app.processEvents()
            assert w.review_tools.widget(tab).horizontalScrollBar().maximum()==0
        w.resize(1440,940); w.review_tools.setCurrentIndex(0); app.processEvents()
        w.grab().save(str(tmp_path/'review-comparison.png'))
        w.preview_tabs.setCurrentIndex(0); wait_until(app,lambda:not w._render_running and not w.render_timer.isActive())
        w.grab().save(str(tmp_path/'review-overlay.png'))
        w.preview_tabs.setCurrentIndex(1); wait_until(app,rendered)
        # Native save dialog starts in the project folder; chosen names are used
        # as-is and exporting elsewhere never relocates the open project.
        def destination(parent,title,default,filters):
            assert str(p.output_dir) in default and 'brain_registration' in default
            return str(tmp_path/'selected.tif'),filters
        monkeypatch.setattr(QFileDialog,'getSaveFileName',destination)
        w.open_export(); app.processEvents()
        dialog=w.export_dialog
        dialog.grab().save(str(tmp_path/'export-dialog.png'))
        dialog.choose_destination()
        wait_until(app,lambda:not w.busy)
        assert w.last_export==tmp_path/'selected.tif'
        assert p.output_dir==tmp_path/'out'
        state=view.state_record()
        assert state['position_um_zyx'][0]<0
        file=w.save_project()
        saved=AlignmentProject.load(file)
        assert saved.display['slice_comparison']==state
        assert saved.display['review_tab']==1
        for key in ('pairs','coarse','residual','fit_result'):
            assert saved.as_record()[key]==original_record[key]
        # Visiting Placement retains the separate Review tab choice.
        w.go_step(1)
        assert not w.preview_tabs.tabBar().isVisible()
        assert w.preview_tabs.currentIndex()==0
        w.go_step(3)
        wait_until(app,rendered)
        assert w.preview_tabs.currentIndex()==1
        resumed=AlignWindow()
        resumed._error=errors.append
        resumed.show()
        try:
            resumed.load_project_path(file)
            wait_until(app,lambda:not resumed.busy and resumed.slice_comparison.grid is not None
                       and not resumed._render_running and not resumed.render_timer.isActive())
            assert resumed.preview_tabs.currentIndex()==1
            assert resumed.slice_comparison.state_record()==state
            assert resumed.project.fit_is_current
            assert all(c.image.isGrayscale() for c in resumed.slice_comparison.canvases)
        finally:
            resumed.close()
        assert not errors,errors
    finally:
        wait_until(app,lambda:not w.busy)
        w.close()


def test_placement_split_views_scan_drag_and_physical_3d_planes(tmp_path):
    from probemap.registration.macos_qt import initialize_qt
    app=initialize_qt()
    from qtpy.QtCore import QEvent, QPoint, QPointF, Qt
    from qtpy.QtGui import QMouseEvent, QWheelEvent
    from probemap.registration.desktop_app import AlignWindow
    from probemap.registration.desktop_model import resample_preview_grid
    z,y,x=np.indices((12,14,16))
    data=(100+400*z+30*y+x).astype(np.uint16)
    scan=write_volume(tmp_path/'scan.tif',data,(2,3,4))
    atlas=write_volume(tmp_path/'atlas.tif',np.flip(data,axis=1).copy(),(2,3,4))
    p=AlignmentProject(ImageSpec(atlas,data.shape,(2,3,4)),ImageSpec(scan,data.shape,(2,3,4)),tmp_path/'out')
    p.coarse[:3,3]=[-18,9,-12]
    p.display={'overlay_position_um_zyx':[-4,24,16]}
    w=AlignWindow()
    errors=[]
    w._error=errors.append
    w.show()
    def ready():
        return not w.busy and not w._render_running and not w.render_timer.isActive() and w.overlay_grid is not None
    def drag(canvas,start,end,mouse_button,modifiers=Qt.KeyboardModifier.NoModifier):
        for kind,pos,button,buttons in ((QEvent.Type.MouseButtonPress,start,mouse_button,mouse_button),
                                       (QEvent.Type.MouseMove,end,Qt.MouseButton.NoButton,mouse_button),
                                       (QEvent.Type.MouseButtonRelease,end,mouse_button,Qt.MouseButton.NoButton)):
            app.sendEvent(canvas,QMouseEvent(kind,pos,canvas.mapToGlobal(pos),button,buttons,modifiers))
    try:
        w._load_project_volumes(p,1)
        wait_until(app,ready)
        baseline=p.coarse.copy()
        assert [pane.views.currentIndex() for pane in w.overlay_panes]==[0,0,0]
        w.overlay_panes[0].mode_selector.setCurrentIndex(1)
        assert [pane.views.currentIndex() for pane in w.overlay_panes]==[1,0,0]
        for pane in w.overlay_panes:
            pane.mode_selector.setCurrentIndex(1)
        wait_until(app,ready)
        assert_allclose(p.coarse,baseline)
        expected=[resample_preview_grid(volume,matrix,w.overlay_grid)
                  for volume,matrix in ((w.reference,np.eye(4)),(w.moving,p.coarse))]
        for axis,pane in enumerate(w.overlay_panes):
            assert pane.reference_canvas.mode=='view' and pane.scan_canvas.mode=='placement'
            for canvas,volume in zip((pane.reference_canvas,pane.scan_canvas),expected):
                plane=np.take(volume,pane.slider.value(),axis=axis)
                assert canvas.image.isGrayscale()
                pixels=[[canvas.image.pixelColor(col,row).red() for col in range(plane.shape[1])]
                        for row in range(plane.shape[0])]
                assert_allclose(pixels,plane*255,atol=1.01)
                point=np.asarray(w.overlay_grid['origin_um_zyx'])+np.asarray(w.overlay_grid['spacing_um_zyx'])*2
                point[axis]=pane.world_depth
                assert_allclose(canvas._world(canvas._pixel(point)),point,atol=1e-9)
        # Plane vertices are XYZ on the GPU, while slice positions and grids
        # are physical ZYX. Check negatives, anisotropy and all three normals.
        view=w.volume_view
        assert view.slice_positions[0]<0
        spacing=np.asarray(w.overlay_grid['spacing_um_zyx'])
        lower=np.asarray(w.overlay_grid['origin_um_zyx'])-spacing/2
        upper=np.asarray(w.overlay_grid['origin_um_zyx'])+(np.asarray(w.overlay_grid['shape_zyx'])-.5)*spacing
        for axis,(mesh,line,text) in enumerate(view.slice_planes):
            vertices=mesh.mesh_data.get_vertices()[:,::-1]
            assert_allclose(vertices[:,axis],w.overlay_panes[axis].world_depth)
            for other in range(3):
                if other!=axis:
                    assert_allclose([vertices[:,other].min(),vertices[:,other].max()],[lower[other],upper[other]])
            assert line.visible and text.visible and not mesh.visible
        pane=w.overlay_panes[0]
        pane.slider.setValue(pane.slider.value()+1)
        assert_allclose(view.slice_positions,[item.world_depth for item in w.overlay_panes])
        wait_until(app,ready)
        ref,mov=pane.reference_canvas,pane.scan_canvas
        ref.wheelEvent(QWheelEvent(QPointF(40,40),QPointF(40,40),QPoint(),QPoint(0,120),
                                  Qt.MouseButton.NoButton,Qt.KeyboardModifier.ControlModifier,
                                  Qt.ScrollPhase.NoScrollPhase,False))
        assert ref.zoom>1
        assert_allclose(ref.zoom,mov.zoom)
        start=ref.image_rect().center()
        drag(ref,start,start+QPointF(5,7),Qt.MouseButton.MiddleButton)
        assert ref.pan==mov.pan and ref.pan!=QPointF()
        ref.mouseDoubleClickEvent(None)
        start=ref.image_rect().center()
        drag(ref,start,start+QPointF(8,5),Qt.MouseButton.LeftButton)
        assert_allclose(p.coarse,baseline)
        start=mov.image_rect().center()
        end=start+QPointF(8,5)
        delta=mov._world(end)-mov._world(start)
        drag(mov,start,end,Qt.MouseButton.MiddleButton)
        wait_until(app,ready)
        assert_allclose(p.coarse[:3,3],baseline[:3,3]+delta,atol=1e-5)
        linear=p.coarse[:3,:3].copy()
        start=mov.image_rect().center()+QPointF(15,0)
        drag(mov,start,start+QPointF(0,10),Qt.MouseButton.RightButton,Qt.KeyboardModifier.AltModifier)
        wait_until(app,ready)
        assert not np.allclose(p.coarse[:3,:3],linear)
        # Exercise real GPU plane filling and visibility, not just Qt controls.
        view.fill_slice_planes.setChecked(True)
        assert all(mesh.visible for mesh,_,_ in view.slice_planes)
        shown=view.canvas.render(size=(240,180))
        view.show_slice_planes.setChecked(False)
        hidden=view.canvas.render(size=(240,180))
        assert np.any(shown!=hidden)
        assert all(not visual.visible for plane in view.slice_planes for visual in plane)
        saved_matrix=p.coarse.copy()
        file=w.save_project()
        saved=AlignmentProject.load(file)
        assert saved.display['placement_slice_modes']==['side_by_side']*3
        assert saved.display['slice_planes_3d']=={'visible':False,'fill':True}
        resumed=AlignWindow()
        resumed._error=errors.append
        resumed.show()
        try:
            resumed.load_project_path(file)
            wait_until(app,lambda:not resumed.busy and resumed.overlay_grid is not None
                       and not resumed._render_running and not resumed.render_timer.isActive())
            assert [pane.views.currentIndex() for pane in resumed.overlay_panes]==[1,1,1]
            assert resumed.volume_view.slice_settings_record()=={'visible':False,'fill':True}
            assert_allclose(resumed.project.coarse,saved_matrix)
        finally:
            resumed.close()
        assert not errors,errors
    finally:
        wait_until(app,lambda:not w.busy)
        w.close()

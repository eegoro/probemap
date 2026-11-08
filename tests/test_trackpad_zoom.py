"""macOS native pinch events, independently of wheel and mouse drag events."""
import copy

import numpy as np
import pytest
from numpy.testing import assert_allclose

from test_desktop_tracks import wait_until
from test_track_model import make_track_project

pytestmark=pytest.mark.gui


def pinch(app,widget,position,values,*,wrong_local=False,check_accept=True):
    from qtpy.QtCore import QPointF,Qt
    from qtpy.QtGui import QInputDevice,QNativeGestureEvent,QPointingDevice
    device=QPointingDevice('Test trackpad',42,QInputDevice.DeviceType.TouchPad,
        QPointingDevice.PointerType.Finger,QInputDevice.Capability.Position,2,0)
    global_position=widget.mapToGlobal(position)
    scene_position=widget.window().mapFromGlobal(global_position)
    local=QPointF(-20,-30) if wrong_local else position
    sequence=[(Qt.NativeGestureType.BeginNativeGesture,0)]
    sequence.extend((Qt.NativeGestureType.ZoomNativeGesture,value) for value in values)
    sequence.append((Qt.NativeGestureType.EndNativeGesture,0))
    for kind,value in sequence:
        event=QNativeGestureEvent(kind,device,2,local,scene_position,global_position,value,QPointF())
        app.sendEvent(widget,event)
        if check_accept: assert event.isAccepted()


def test_native_pinch_matches_3d_across_overlay_split_landmark_and_review_slices(tmp_path):
    from probemap.registration.macos_qt import initialize_qt
    from probemap.registration.desktop_app import ProbeMapWindow
    from qtpy.QtCore import QPointF
    app=initialize_qt()
    project=make_track_project(tmp_path)
    window=ProbeMapWindow(); window.show()
    errors=[]; window._error=errors.append
    values=[.04,.08,-.03,.06]
    expected=float(np.prod([1/(1-value) for value in values]))
    def ready(): return not window.busy and not window._render_running and not window.render_timer.isActive()
    def zoom_slice(canvas):
        point=QPointF(canvas.width()*.61,canvas.height()*.47)
        before=canvas._world(point); zoom=canvas.zoom
        pinch(app,canvas,point,values,wrong_local=True)
        assert_allclose(canvas.zoom/zoom,expected)
        assert_allclose(canvas._world(point),before,atol=1e-9)
    try:
        window._load_project_volumes(project,1); wait_until(app,ready)
        matrix=project.coarse.copy(); pairs=copy.deepcopy(project.pairs)
        slices=[pane.slider.value() for pane in window.overlay_panes]
        camera=window.volume_view.view.camera
        before=camera.scale_factor
        native=window.volume_view.canvas.native
        pinch(app,native,QPointF(native.width()/2,native.height()/2),values,check_accept=False)
        assert_allclose(before/camera.scale_factor,expected)
        for pane in window.overlay_panes:
            zoom_slice(pane.canvas)
            pane.mode_selector.setCurrentIndex(1)
            wait_until(app,ready)
            for canvas in (pane.reference_canvas,pane.scan_canvas):
                zoom_slice(canvas)
                assert pane.reference_canvas.view_record()==pane.scan_canvas.view_record()
        assert [pane.slider.value() for pane in window.overlay_panes]==slices
        assert_allclose(project.coarse,matrix)
        assert project.pairs==pairs
        window.go_step(2)
        wait_until(app,lambda:ready() and window.placed is not None)
        for panes in window.landmark_panes.values():
            for pane in panes: zoom_slice(pane.canvas)
        assert project.pairs==pairs
        window.go_step(3); wait_until(app,ready)
        for pane in window.overlay_panes: zoom_slice(pane.canvas)
        assert_allclose(project.coarse,matrix)
        assert not errors,errors
    finally:
        wait_until(app,lambda:not window.busy)
        window.close()


def test_oblique_insertion_pinch_keeps_plane_annotations_and_slice_pixels():
    from probemap.registration.macos_qt import initialize_qt
    from probemap.registration.desktop_insertion_views import InsertionSliceCanvas
    from probemap.registration.track_model import plane_geometry
    from qtpy.QtCore import QPointF
    app=initialize_qt()
    canvas=InsertionSliceCanvas(); canvas.resize(600,450); canvas.show()
    geometry=plane_geometry(dict(center_um_zyx=[500,700,900],rotation_xyz_deg=[20,30,40],
        width_um=1800,height_um=1200,max_dimension=128))
    pixels=np.linspace(0,1,np.prod(geometry['shape_hw'])).reshape(geometry['shape_hw'])
    canvas.set_plane_image(pixels,geometry)
    canvas.set_interaction_tool('add')
    signals=[]
    for signal in (canvas.pointAdded,canvas.vertexAdded,canvas.trackDragged,canvas.measurementAdded):
        signal.connect(lambda *args:signals.append(args))
    image=canvas.image.copy(); saved=copy.deepcopy(canvas.geometry_record)
    point=QPointF(340,200); anchor=canvas._world(point)
    try:
        pinch(app,canvas,point,[.1,.2],wrong_local=True)
        assert canvas.zoom>1
        assert_allclose(canvas._world(point),anchor,atol=1e-9)
        previous=canvas.zoom
        pinch(app,canvas,point,[-.1,-.2])
        assert canvas.zoom<previous
        assert_allclose(canvas._world(point),anchor,atol=1e-9)
        assert canvas.geometry_record==saved and canvas.image==image and not signals
    finally:
        canvas.close()

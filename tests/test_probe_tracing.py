"""Native physical annotation with a separately transformed atlas preview."""
import copy

import numpy as np
import pytest
from numpy.testing import assert_allclose

from probemap.registration.coarse import pose_matrix, transformed_corners
from probemap.registration.desktop_model import AlignmentProject, ImageSpec
from probemap.registration.landmarks import transform_points
from probemap.registration.review import placement_without_scale
from probemap.registration.volumes import write_volume
from test_desktop_tracks import wait_until


@pytest.mark.gui
def test_probe_tracing_native_points_table_navigation_and_resume(tmp_path):
    from probemap.registration.macos_qt import initialize_qt
    app=initialize_qt()
    from qtpy.QtCore import QEvent, QPoint, QPointF, Qt
    from qtpy.QtGui import QMouseEvent
    from qtpy.QtTest import QTest
    from qtpy.QtWidgets import QApplication
    from probemap.registration.desktop_app import ProbeMapWindow
    shape=(25,30,35); spacing=(160,200,240)
    z,y,x=np.indices(shape)
    data=(50000*np.exp(-((z-12)**2+(y-15)**2+(x-17)**2)/50)).astype(np.uint16)
    brain=write_volume(tmp_path/'brain.tif',data,spacing)
    electrodes=write_volume(tmp_path/'electrodes.tif',data//2,spacing)
    spec=ImageSpec(brain,shape,spacing)
    p=AlignmentProject(spec,spec,tmp_path/'output',track=ImageSpec(electrodes,shape,spacing))
    p.coarse=pose_matrix(shape,spacing,(-500,800,400),(20,-30,40),(.6,1.4,.9))
    raw=np.array([[400,600,800],[3000,600,800],[400,5000,800],[400,600,6000],[2800,4000,5500.]])
    placed=transform_points(raw,p.coarse)
    residual=np.array([[.8,.1,0,300],[0,1.2,.15,-100],[.05,0,1.1,50],[0,0,0,1]])
    atlas=transform_points(placed,residual)
    p.pairs=[dict(reference=a.tolist(),moving=b.tolist()) for a,b in zip(atlas,placed)]
    p.fit(); p.display={'step':4}
    path=p.save(tmp_path/'output/project.json')
    errors=[]
    w=ProbeMapWindow(); w._error=errors.append
    w.resize(1500,1040); w.show()
    t=w.track_workspace
    def ready(): return not w.busy and t._cache_current() and not t._running and not t.render_timer.isActive()
    try:
        w.load_project_path(path); wait_until(app,ready); app.processEvents()
        assert t.title.text()==w.nav_buttons[4].text()=='Tracks'
        assert [t.sidebar_tabs.tabText(i) for i in range(t.sidebar_tabs.count())]==['Points','3D','Slices']
        assert t.current_insertion()['name']=='A'
        assert all(not c.tools.isVisible() for c in t.slices)
        assert not w.save_button.isVisible()
        assert [b.text() for b in (t.back_button,t.apply_button,t.save_button,t.next_button)]==['Back','Apply','Save','Next']
        expected=placement_without_scale(w.project)
        assert_allclose(expected[:3,:3].T@expected[:3,:3],np.eye(3),atol=1e-12)
        corners=transformed_corners(shape,spacing,expected)
        assert_allclose(t.lower,corners.min(0)); assert_allclose(t.upper,corners.max(0))
        c=t.slices[0]
        center=QPointF(c.rect().center())
        displayed=c._world(center)
        QTest.mouseClick(c,Qt.MouseButton.LeftButton,pos=center.toPoint()); app.processEvents()
        point=t.current_insertion()['points'][0]
        native=transform_points([displayed],np.linalg.inv(expected))[0]
        assert_allclose(point['source_track_zyx_voxel'],native/spacing,atol=1e-9)
        assert_allclose(point['atlas_zyx_um'],transform_points([native],w.project.total)[0],atol=1e-9)
        assert_allclose(t.volume_view.point_records[0]['atlas_zyx_um'],point['atlas_zyx_um'])
        assert_allclose(c.insertion_points[0]['atlas_zyx_um'],displayed)
        assert t.insertion_table.item(0,2).text()=='1'
        t.measure_distance(displayed,displayed+[0,300,400])
        assert_allclose(t.session['measurements'][-1]['length_um'],500,atol=1e-9)
        assert abs(t.session['measurements'][-1]['atlas_length_um']-500)>10
        t.undo()
        wait_until(app,ready)
        t.select_points([point['id']]); t.go_to_point(); wait_until(app,ready)
        assert all(any(p['id']==point['id'] for p,_ in canvas.visible_points()) for canvas in t.slices)
        t.point_size.setValue(18)
        assert all(c.point_size==18 for c in t.slices) and t.volume_view.point_size==18
        assert_allclose(t.volume_view.point_markers._data['a_size'],[22])
        t.insertion_table.item(0,0).setText('Left dorsal')
        assert t.current_insertion()['name']=='Left dorsal'
        t.add_insertion_button.click(); assert t.current_insertion()['name']=='B'
        t.add_insertion_button.click(); assert t.current_insertion()['name']=='C'
        t.insertion_table.selectRow(0)
        assert t.current_insertion()['name']=='Left dorsal'
        t.insertion_table.item(0,0).setCheckState(Qt.CheckState.Unchecked)
        assert not t.volume_view.point_records
        t.insertion_table.item(0,0).setCheckState(Qt.CheckState.Checked)
        assert len(t.volume_view.point_records)==1
        # All slice tools act together; navigation never changes depth.
        t.tracing_tools['select'].click()
        assert all(c.point_mode=='select' for c in t.slices) and t.volume_view.selection_enabled
        t.tracing_tools['add'].click()
        positions=t._tracing_positions.copy()
        c.zoom_at(center,2)
        count=len(t.current_insertion()['points'])
        QTest.mousePress(c,Qt.MouseButton.LeftButton,pos=center.toPoint())
        QTest.mouseMove(c,(center+QPointF(30,20)).toPoint(),delay=10)
        QTest.mouseRelease(c,Qt.MouseButton.LeftButton,pos=(center+QPointF(30,20)).toPoint())
        assert len(t.current_insertion()['points'])==count
        assert c.pan!=QPointF()
        assert_allclose(t._tracing_positions,positions)
        from test_trackpad_zoom import pinch
        for canvas in t.slices:
            position=QPointF(canvas.rect().center())
            before=canvas._world(position); zoom=canvas.zoom
            pinch(app,canvas,position,[.1],wrong_local=True)
            assert_allclose(canvas.zoom,zoom/.9)
            assert_allclose(canvas._world(position),before,atol=1e-9)
        assert_allclose(t._tracing_positions,positions)
        for angle in (90,180,270,0):
            c.rotate_view()
            assert c.view_rotation==angle
            assert_allclose(c._world(c._pixel(displayed)),displayed,atol=1e-9)
        c.fit_image()
        # Real navigation clicks used to clear the insertion table and turn
        # Draw into Select. Keep B as the destination through sliders, 3D
        # orbit and display controls, then place another source-anchored point.
        t.insertion_table.selectRow(1)
        destination=t.current_insertion()['id']
        def assert_drawing():
            assert t.insertion_id==destination
            assert t.insertion_table.selectionModel().hasSelection()
            assert t.tracing_tools['add'].isChecked()
            assert all(canvas.point_mode=='add' for canvas in t.slices)
        for axis,slider in enumerate(t.slice_sliders):
            previous=slider.value()
            QTest.mouseClick(slider,Qt.MouseButton.LeftButton,pos=QPoint(slider.width()-5,slider.height()//2))
            wait_until(app,ready)
            assert slider.value()!=previous
            assert_drawing()
        native=t.volume_view.canvas.native
        camera=t.volume_view.view.camera
        before=tuple(camera.get_state()['_quaternion'].get_axis_angle())
        start=QPointF(25,25); end=start+QPointF(50,30)
        QTest.mousePress(native,Qt.MouseButton.LeftButton,pos=start.toPoint())
        event=QMouseEvent(QEvent.Type.MouseMove,end,QPointF(native.mapToGlobal(end.toPoint())),
            Qt.MouseButton.NoButton,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier)
        QApplication.sendEvent(native,event)
        QTest.mouseRelease(native,Qt.MouseButton.LeftButton,pos=end.toPoint()); app.processEvents()
        assert tuple(camera.get_state()['_quaternion'].get_axis_angle())!=before
        assert_drawing()
        tabs=t.sidebar_tabs.tabBar()
        QTest.mouseClick(tabs,Qt.MouseButton.LeftButton,pos=tabs.tabRect(1).center())
        app.processEvents()
        axes=t.volume_view.show_axes; previous=axes.isChecked()
        t.sidebar_tabs.widget(1).ensureWidgetVisible(axes)
        QTest.mouseClick(axes,Qt.MouseButton.LeftButton,pos=QPoint(8,axes.height()//2))
        assert axes.isChecked()!=previous
        assert_drawing()
        QTest.mouseClick(tabs,Qt.MouseButton.LeftButton,pos=tabs.tabRect(0).center())
        count=sum(len(i['points']) for i in t.session['insertions'])
        position=QPointF(c.rect().center())
        displayed=c._world(position)
        QTest.mouseClick(c,Qt.MouseButton.LeftButton,pos=position.toPoint()); app.processEvents()
        assert sum(len(i['points']) for i in t.session['insertions'])==count+1
        added=t.current_insertion()['points'][-1]
        assert_allclose(added['atlas_zyx_um'],transform_points([displayed],t._tracing_to_atlas())[0])
        assert_drawing()
        t.undo(); wait_until(app,ready)
        # Selection transfer, deletion and undo preserve point anchors.
        t.select_points([point['id']]); t.move_target.setCurrentIndex(1); t.move_selected()
        assert not t.session['insertions'][0]['points'] and t.session['insertions'][1]['points'][0]['id']==point['id']
        t.insertion_table.selectRow(1); t.delete_tracing_insertion()
        assert len(t.session['insertions'])==2
        t.undo(); assert len(t.session['insertions'])==3
        t.select_points([point['id']]); c.setFocus()
        QTest.keyClick(c,Qt.Key.Key_Backspace); app.processEvents()
        assert not any(i['points'] for i in t.session['insertions'])
        w.undo_current_step(); assert sum(len(i['points']) for i in t.session['insertions'])==1
        t.sidebar_tabs.setCurrentIndex(2); t.slice_source.setCurrentIndex(1); wait_until(app,ready)
        t.slice_display.channel.setCurrentIndex(1); t.slice_display.gamma.setValue(1.4)
        t.insertion_table.setColumnWidth(0,135)
        w.save_project()
        saved=AlignmentProject.load(path)
        view=saved.track_analysis['sessions'][0]['insertion_view']
        assert view['point_size_px']==18 and view['slice_source']==1 and view['slice_display'][1]['gamma']==1.4
        assert view['insertion_columns'][0]==135
        w.go_step(5); wait_until(app,ready)
        w.go_step(4); wait_until(app,ready)
        assert t.sidebar_tabs.count()==3 and t.point_size.value()==18
        panes=[t.volume_view,*t.slice_panels]
        assert max(p.width() for p in panes)-min(p.width() for p in panes)<=1
        assert max(p.height() for p in panes)-min(p.height() for p in panes)<=1
        t.grab().save(str(tmp_path/'probe-tracing.png'))
        assert not errors,errors
    finally:
        wait_until(app,lambda:not w.busy)
        w.close()

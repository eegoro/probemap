"""Native plane/probe placement, affine display and downstream reconstruction."""
import copy

import numpy as np
import pytest
from numpy.testing import assert_allclose

from probemap.registration.landmarks import transform_points
from probemap.registration.insertion_model import add_insertion_point,fit_insertion_plane,insertion_session,new_insertion
from probemap.registration.neuropixels_model import probe_settings,probe_coordinates,pose_current
from probemap.registration.probe_placement import pose_on_plane,place_probe,source_plane
from probemap.registration.track_model import plane_geometry,plane_corners,transform_plane_geometry
from test_track_model import make_track_project


def test_source_plane_probe_pitch_and_transformed_grid(tmp_path):
    p=make_track_project(tmp_path)
    session=insertion_session(p); item=new_insertion(session)
    matrix=np.array([[.6,.2,0,500],[0,1.4,.1,-200],[.1,0,.9,100],[0,0,0,1.]])
    source=np.array([[500,1200,700],[4000,1200,700],[500,1200,1450],[4000,1200,1450.]])
    for point in transform_points(source,matrix): add_insertion_point(p,item,point)
    fit_insertion_plane(item,session['plane'],matrix)
    g=plane_geometry(item['source_plane'])
    assert_allclose((source-g['center_um_zyx'])@g['normal_zyx'],0,atol=1e-9)
    # The GL field is the mapped original grid, including shear, not a newly
    # fitted atlas rectangle with an unrelated field of view.
    transformed=transform_plane_geometry(g,matrix)
    assert_allclose(plane_corners(transformed),transform_points(plane_corners(g),matrix))
    pose=pose_on_plane(item,item['source_plane'],source[0],70)
    place_probe(item,pose,p,session)
    s=probe_settings(item)
    tips=probe_coordinates(s,np.arange(4),np.zeros(4))
    assert_allclose(np.linalg.norm(np.diff(tips,axis=0),axis=1),250,atol=1e-10)
    assert len(item['tracks'])==4 and pose_current(item,p.total)
    assert_allclose(transform_points([t['control_points_atlas_zyx_um'][0] for t in item['tracks']],np.linalg.inv(p.total)),tips)
    ids=[t['id'] for t in item['tracks']]
    moved=copy.deepcopy(pose); moved['tip0_source_zyx_um']=(tips[0]+[10,20,30]).tolist()
    place_probe(item,moved,p,session)
    assert [t['id'] for t in item['tracks']]==ids
    assert_allclose(probe_coordinates(s,0,0),tips[0]+[10,20,30])


@pytest.mark.gui
def test_native_probe_placement_two_views_gestures_frames_save_and_undo(tmp_path):
    from probemap.registration.macos_qt import initialize_qt
    app=initialize_qt()
    from qtpy.QtCore import QEvent,QPointF,Qt
    from qtpy.QtGui import QMouseEvent
    from qtpy.QtWidgets import QApplication
    from qtpy.QtTest import QTest
    from probemap.registration.desktop_app import ProbeMapWindow
    from probemap.registration.desktop_model import AlignmentProject,ImageSpec
    from probemap.registration.coarse import pose_matrix
    from probemap.registration.volumes import write_volume
    from test_desktop_tracks import wait_until
    shape=(35,35,40); spacing=(150,150,150)
    z,y,x=np.indices(shape); data=(50000*np.exp(-((z-17)**2+(y-17)**2+(x-20)**2)/100)).astype(np.uint16)
    brain=write_volume(tmp_path/'brain.tif',data,spacing)
    electrodes=write_volume(tmp_path/'electrodes.tif',data//2,spacing)
    p=AlignmentProject(ImageSpec(brain,shape,spacing),ImageSpec(brain,shape,spacing),tmp_path/'out',track=ImageSpec(electrodes,shape,spacing))
    p.coarse=pose_matrix(shape,spacing,(100,-300,200),(15,-20,30),(.6,1.2,.8))
    raw=np.array([[400,500,600],[4000,500,600],[400,4000,600],[400,500,5000],[3500,4000,5000.]])
    placed=transform_points(raw,p.coarse)
    residual=np.array([[1,.2,0,100],[0,.9,.1,-150],[.1,0,1.1,200],[0,0,0,1.]])
    p.pairs=[dict(reference=a.tolist(),moving=b.tolist()) for a,b in zip(transform_points(placed,residual),placed)]
    p.fit(); session=insertion_session(p); item=new_insertion(session)
    raw_points=np.array([[1000,2400,1800],[4000,2400,1800],[1000,2400,2550],[4000,2400,2550.]])
    for point in transform_points(raw_points,p.total): add_insertion_point(p,item,point)
    second=new_insertion(session)
    for point in transform_points(raw_points+[0,600,0],p.total): add_insertion_point(p,second,point)
    session['insertion_view']={'placement_scene':'source','ui_version':4}
    p.display={'step':4}; path=p.save(tmp_path/'out/project.json')
    w=ProbeMapWindow(); errors=[]; w._error=errors.append; w.resize(1500,1000); w.show()
    t=w.track_workspace
    def ready(): return not w.busy and t._cache_current() and not t._running and not t.render_timer.isActive()
    try:
        w.load_project_path(path); wait_until(app,ready)
        t.apply_button.click(); assert t.session['insertions'][0].get('source_plane')
        w.go_step(5); wait_until(app,ready)
        assert t.title.text()==w.nav_buttons[5].text()=='Probes'
        assert [t.sidebar_tabs.tabText(i) for i in range(t.sidebar_tabs.count())]==['Placement','3D','Slices']
        assert t.slice_panels[0].isVisible() and all(not panel.isVisible() for panel in t.slice_panels[1:])
        assert all(not c.tools.isVisible() for c in t.slices)
        assert not t.info.isVisible() and not t.hover_info.isVisible() and not t.export_button.isVisible()
        assert t.controls.currentWidget() is t.placement_panel
        # Selecting a different empty insertion must survive a toolbar click.
        row_rect=t.placement_table.visualItemRect(t.placement_table.item(1,0))
        QTest.mouseClick(t.placement_table.viewport(),Qt.MouseButton.LeftButton,pos=row_rect.center())
        wait_until(app,ready)
        assert t.current_insertion()['name']=='B' and t.placement_table.selectionModel().hasSelection()
        QTest.mouseClick(t.placement_tools['draw'],Qt.MouseButton.LeftButton)
        assert t.current_insertion()['name']=='B'
        row_rect=t.placement_table.visualItemRect(t.placement_table.item(0,0))
        QTest.mouseClick(t.placement_table.viewport(),Qt.MouseButton.LeftButton,pos=row_rect.center())
        wait_until(app,ready)
        assert t.current_insertion()['name']=='A'
        item=t.current_insertion(); source=copy.deepcopy(item['source_plane'])
        original_slice=t._cache['images'][0][0].copy()
        canvas=t.track_slice; center=QPointF(canvas.rect().center())
        assert t.placement_table.isVisible() and not t.placement_insertion.isVisible()
        QTest.mouseClick(t.placement_tools['draw'],Qt.MouseButton.LeftButton)
        assert t.placement_tools['draw'].isChecked() and not t.placement_tools['select'].isChecked()
        assert 'QToolButton:checked' in t.placement_toolbar.styleSheet()
        # Cocoa can coalesce pointer-only QTest moves after a toolbar click;
        # deliver the corresponding native Qt hover event deterministically.
        hover=QMouseEvent(QEvent.Type.MouseMove,center,QPointF(canvas.mapToGlobal(center.toPoint())),Qt.MouseButton.NoButton,Qt.MouseButton.NoButton,Qt.KeyboardModifier.NoModifier)
        QApplication.sendEvent(canvas,hover); app.processEvents()
        assert len(canvas.probe_polygons)==4 and not item['probe'].get('pose')
        assert_allclose(np.linalg.norm(np.diff([poly[0][0] for poly in canvas.probe_polygons],axis=0),axis=1),250,atol=1e-8)
        QTest.mousePress(canvas,Qt.MouseButton.LeftButton,pos=center.toPoint())
        end=center+QPointF(0,-80)
        QTest.mouseMove(canvas,end.toPoint(),delay=20)
        QTest.mouseRelease(canvas,Qt.MouseButton.LeftButton,pos=end.toPoint()); app.processEvents()
        assert item['probe'].get('pose') and len(item['tracks'])==4
        assert canvas.placement_tool=='select' and canvas.placement_selected
        pose=copy.deepcopy(item['probe']['pose'])
        # A 500 um source-space ruler stays 500 despite nonuniform scale/shear.
        g=canvas.geometry_record; a=np.array(g['center_um_zyx']); b=a+500*np.array(g['u_zyx'])
        t.measure_distance(a,b)
        assert_allclose(t.session['measurements'][-1]['length_um'],500,atol=1e-8)
        # Move via the actual select gesture at one shank's deep tip.
        displayed=transform_points([probe_coordinates(item['probe'],1,500)],t._tracing_matrix())[0]
        pixel=canvas._pixel(displayed).toPoint(); delta=QPointF(30,10)
        before=np.array(item['probe']['pose']['tip0_source_zyx_um'])
        expected=np.linalg.inv(t._tracing_matrix())[:3,:3]@(canvas._world(QPointF(pixel)+delta)-canvas._world(QPointF(pixel)))
        QTest.mousePress(canvas,Qt.MouseButton.LeftButton,pos=pixel)
        QTest.mouseMove(canvas,(QPointF(pixel)+delta).toPoint(),delay=20)
        QTest.mouseRelease(canvas,Qt.MouseButton.LeftButton,pos=(QPointF(pixel)+delta).toPoint()); app.processEvents()
        assert_allclose(item['probe']['pose']['tip0_source_zyx_um'],before+expected,atol=1e-7)
        moved=copy.deepcopy(item['probe']['pose'])
        t.undo(); wait_until(app,ready); item=t.current_insertion()
        assert_allclose(item['probe']['pose']['tip0_source_zyx_um'],before)
        # Rotate with the visible handle, about the centre of the four tips.
        t._placement_select(True)
        assert canvas.rotation_handle() is not None
        pivot=canvas._pixel(canvas.placement_pivot)
        handle=canvas.rotation_handle(); vector=handle-pivot
        end=pivot+QPointF(-vector.y(),vector.x())
        pivot_before=probe_coordinates(item['probe'],1.5,0).copy()
        down_before=np.array(item['probe']['pose']['down_source_zyx'])
        QTest.mousePress(canvas,Qt.MouseButton.LeftButton,pos=handle.toPoint())
        QTest.mouseMove(canvas,end.toPoint(),delay=20)
        QTest.mouseRelease(canvas,Qt.MouseButton.LeftButton,pos=end.toPoint()); app.processEvents()
        assert abs(np.dot(down_before,item['probe']['pose']['down_source_zyx']))<.05
        assert_allclose(probe_coordinates(item['probe'],1.5,0),pivot_before,atol=1e-8)
        t.undo(); wait_until(app,ready); item=t.current_insertion()
        # Flip the physical recording face, then rotate again: it stays flipped.
        tips_before=probe_coordinates(item['probe'],np.arange(4),np.zeros(4))
        face=t.volume_view.probe_face
        normal_before=face.normal_xyz.copy(); facing_before=face.facing_cosine
        t.placement_face_button.click(); app.processEvents()
        assert_allclose(probe_coordinates(item['probe'],np.arange(4),np.zeros(4)),tips_before[::-1],atol=1e-8)
        assert_allclose(face.normal_xyz,-normal_before,atol=1e-8)
        assert_allclose(face.facing_cosine,-facing_before,atol=1e-8)
        assert t.placement_table.item(0,3).text()=='Back' and not canvas.placement_contact_front
        t.placement_probe_fields['angle'].setValue(t.placement_probe_fields['angle'].value()+10)
        assert not canvas.placement_contact_front
        before=np.array(item['probe']['pose']['tip0_source_zyx_um'])
        t.placement_table.item(0,0).setText('Dorsal')
        assert item['name']=='Dorsal'
        point_ids={p['id'] for p in t.volume_view.point_records}
        t.placement_table.item(0,0).setCheckState(Qt.CheckState.Unchecked)
        assert not canvas.probe_polygons
        assert {p['id'] for p in t.volume_view.point_records}==point_ids
        assert not t._placement_visuals[item['id']].visible
        t.placement_table.item(0,0).setCheckState(Qt.CheckState.Checked)
        assert len(canvas.probe_polygons)==4
        # Points and probes have independent per-insertion controls in 3D.
        t.sidebar_tabs.setCurrentIndex(1); app.processEvents()
        table=t.visibility_table
        assert table.isVisible() and table.horizontalHeaderItem(2).text()=='Probe'
        own_ids={p['id'] for p in item['points']}
        table.item(0,1).setCheckState(Qt.CheckState.Unchecked)
        assert {p['id'] for p in t.volume_view.point_records}==point_ids-own_ids
        assert len(canvas.probe_polygons)==4 and t._placement_visuals[item['id']].visible
        table.item(0,2).setCheckState(Qt.CheckState.Unchecked)
        assert not canvas.probe_polygons and not t._placement_visuals[item['id']].visible
        assert t.placement_table.item(0,0).checkState()==Qt.CheckState.Unchecked
        table.item(0,1).setCheckState(Qt.CheckState.Checked)
        assert {p['id'] for p in t.volume_view.point_records}==point_ids
        assert not canvas.probe_polygons and not t._placement_visuals[item['id']].visible
        table.item(0,2).setCheckState(Qt.CheckState.Checked)
        assert len(canvas.probe_polygons)==4 and t._placement_visuals[item['id']].visible
        assert t.placement_table.item(0,0).checkState()==Qt.CheckState.Checked
        t.sidebar_tabs.setCurrentIndex(0)
        # The left 3D scene ALWAYS uses the full registration, including when
        # Atlas is hidden. Native planes/probes and the right slice stay intact.
        from probemap.registration.neuropixels_model import shank_mesh
        native_corners=plane_corners(t._placement_geometry())
        expected_plane=transform_points(native_corners,w.project.total)
        assert t._placement_scene=='atlas'
        assert_allclose(t.volume_view.matrix,w.project.total)
        assert_allclose(plane_corners(t.volume_view._plane_geometry),expected_plane)
        vertices,_=shank_mesh(item,w.project.total)
        assert_allclose(t._placement_visuals[item['id']].mesh_data.get_vertices(),vertices)
        assert not np.allclose(expected_plane,transform_points(native_corners,t._tracing_matrix()))
        t.sidebar_tabs.setCurrentIndex(1); app.processEvents()
        atlas_check=t.volume_view.layers.checks[0]
        for visible in (True,False,True):
            if atlas_check.isChecked()!=visible:
                QTest.mouseClick(atlas_check,Qt.MouseButton.LeftButton); app.processEvents()
            assert atlas_check.isChecked()==visible
            assert_allclose(plane_corners(t.volume_view._plane_geometry),expected_plane)
            assert_allclose(t.volume_view.matrix,w.project.total)
            assert_allclose(t._placement_visuals[item['id']].mesh_data.get_vertices(),vertices)
            assert_allclose(t._cache['images'][0][0],original_slice)
            assert_allclose(item['probe']['pose']['tip0_source_zyx_um'],before)
        assert t.opacity[0].value()>0 and t.volume_view.visuals[0].visible
        # Display reflection happens after registration, through the shared
        # camera: every actor and point-picking follows it; the native slice,
        # calibration, ruler and pose are untouched. Fit retains reflection.
        view=t.volume_view; camera=view.view.camera
        from vispy.util.quaternion import Quaternion
        original_camera=view.camera_record()
        sides=set()
        for angles in ((0,0,0),(180,0,0),(90,0,0),(-90,0,0),(0,0,90),(0,0,-90)):
            camera.set_state(_quaternion=Quaternion.create_from_euler_angles(*angles,degrees=True))
            app.processEvents(); sides.add(face.side)
        assert {'front','back'}<=sides
        view.restore_camera(original_camera)
        registration=w.project.total.copy(); native_pose=copy.deepcopy(item['probe']['pose'])
        native_plane=copy.deepcopy(item['source_plane'])
        baseline_points=view.projected_points().copy()
        points_xyz=np.array([p['atlas_zyx_um'][::-1] for p in view.point_records])
        center=np.array(camera.center)
        for axis,check in enumerate(view.flip_checks):
            reflected=points_xyz.copy(); reflected[:,axis]=2*center[axis]-reflected[:,axis]
            transform=view.point_markers.get_transform(map_from='visual',map_to='canvas')
            expected=transform.map(reflected); expected=expected[:,:2]/expected[:,3,None]
            QTest.mouseClick(check,Qt.MouseButton.LeftButton); app.processEvents()
            assert camera.flip==tuple(i==axis for i in range(3))
            assert_allclose(view.projected_points(),expected,atol=1e-5)
            assert_allclose(view.matrix,registration)
            assert_allclose(t._placement_visuals[item['id']].mesh_data.get_vertices(),vertices)
            assert_allclose(t._cache['images'][0][0],original_slice)
            assert item['probe']['pose']==native_pose and item['source_plane']==native_plane
            assert_allclose(t.session['measurements'][-1]['length_um'],500,atol=1e-8)
            QTest.mouseClick(check,Qt.MouseButton.LeftButton); app.processEvents()
            assert_allclose(view.projected_points(),baseline_points,atol=1e-5)
        view.flip_checks[0].setChecked(True); view.reset_view()
        assert camera.flip==(True,False,False)
        t.grab().save(str(tmp_path/'mirrored-3d.png'))
        t.sidebar_tabs.setCurrentIndex(0)
        assert t.placement_table.horizontalScrollBar().maximum()==0
        # Plane movement updates the source slice and carries the probe rigidly.
        center_before=np.array(item['source_plane']['center_um_zyx'])
        plane_before=copy.deepcopy(item['source_plane'])
        pose_before=copy.deepcopy(item['probe']['pose'])
        other_before=copy.deepcopy(t.session['insertions'][1])
        t.placement_target.setCurrentIndex(0); t.placement_normal.setValue(150); wait_until(app,ready)
        assert_allclose(np.linalg.norm(np.subtract(item['source_plane']['center_um_zyx'],center_before)),150)
        assert not np.array_equal(t._cache['images'][0][0],original_slice)
        assert_allclose(np.linalg.norm(np.diff(probe_coordinates(item['probe'],np.arange(4),np.zeros(4)),axis=0),axis=1),250)
        rotation=t.placement_plane_fields['rotation',0]
        rotation.setValue(rotation.value()+13); wait_until(app,ready)
        adjusted_plane=copy.deepcopy(item['source_plane'])
        adjusted_pose=copy.deepcopy(item['probe']['pose'])
        w.save_project(); w.load_project_path(path); wait_until(app,ready)
        item=t.current_insertion()
        assert t.placement_plane_reset.isVisible() and t.placement_plane_reset.text()=='Reset plane'
        QTest.mouseClick(t.placement_plane_reset,Qt.MouseButton.LeftButton); wait_until(app,ready)
        restored=plane_geometry(item['source_plane']); initial=plane_geometry(plane_before)
        for key in ('center_um_zyx','u_zyx','v_zyx','normal_zyx'):
            assert_allclose(restored[key],initial[key],atol=1e-8)
        for key in ('tip0_source_zyx_um','down_source_zyx','lateral_source_zyx','normal_source_zyx'):
            assert_allclose(item['probe']['pose'][key],pose_before[key],atol=1e-8)
        assert_allclose(t._cache['images'][0][0],original_slice)
        assert_allclose(plane_corners(t.volume_view._plane_geometry),expected_plane)
        assert not item['plane_fit']['manual'] and t.placement_normal.value()==0
        assert t.session['insertions'][1]==other_before
        t.undo(); wait_until(app,ready); item=t.current_insertion()
        assert item['source_plane']==adjusted_plane and item['probe']['pose']==adjusted_pose
        t.placement_plane_reset.click(); wait_until(app,ready)
        t.apply_button.click(); w.save_project()
        saved=copy.deepcopy(item)
        t.grab().save(str(tmp_path/'probe-placement.png'))
        for index in range(3):
            t.sidebar_tabs.setCurrentIndex(index); app.processEvents()
            assert t.sidebar_tabs.widget(index).horizontalScrollBar().maximum()==0
        w.go_step(6)
        r=w.reconstruction
        wait_until(app,lambda:not w.busy and r.report is not None and not r._running)
        assert r.items[0]['probe']['pose']==saved['probe']['pose']
        assert r.volume_view.point_records==[]
        assert len(r.report['probes'])==1
        w.save_project(); w.load_project_path(path)
        wait_until(app,lambda:not w.busy and r.report is not None and not r._running)
        assert w.step==6 and r.items[0]['source_plane']==saved['source_plane']
        assert r.items[0]['probe']['pose']==saved['probe']['pose']
        # Save/reopen retains different visibility states without touching pose.
        w.go_step(5); wait_until(app,ready)
        t.visibility_table.item(0,1).setCheckState(Qt.CheckState.Unchecked)
        w.save_project(); w.load_project_path(path); wait_until(app,ready)
        item=t.current_insertion()
        assert not item['points_visible'] and item['probe_visible']
        assert t._placement_visuals[item['id']].visible and len(t.track_slice.probe_polygons)==4
        assert not ({p['id'] for p in item['points']}&{p['id'] for p in t.volume_view.point_records})
        assert item['probe']['pose']==saved['probe']['pose']
        assert not errors,errors
    finally:
        wait_until(app,lambda:not w.busy)
        w.close()


def test_native_plane_and_rigid_probe_rebuild_after_voxel_recalibration(tmp_path):
    from test_calibration import annotated_project
    from probemap.registration.calibration import recalibrate_project
    p,session,item,points=annotated_project(tmp_path)
    fit_insertion_plane(item,session['plane'],p.total)
    native=source_plane(item,session['plane'],p.total)
    pose=pose_on_plane(item,native,np.mean(points,axis=0),30)
    place_probe(item,pose,p,session)
    original=copy.deepcopy(p.as_record())
    updated=recalibrate_project(p,(3,4,5),(4,3,6),track_spacing=(5,6,7))
    assert p.as_record()==original
    result=insertion_session(updated)['insertions'][0]
    geometry=plane_geometry(result['source_plane'])
    source=transform_points([q['atlas_zyx_um'] for q in result['points']],np.linalg.inv(updated.total))
    assert_allclose((source-geometry['center_um_zyx'])@geometry['normal_zyx'],0,atol=1e-9)
    tips=probe_coordinates(result['probe'],np.arange(4),np.zeros(4))
    assert_allclose(np.linalg.norm(np.diff(tips,axis=0),axis=1),250,atol=1e-9)
    assert pose_current(result,updated.total)


def test_flip_recording_face_preserves_shanks_and_survives_yaw(tmp_path):
    from probemap.registration.probe_placement import flip_contact_face,contact_side,contact_preview
    p=make_track_project(tmp_path); session=insertion_session(p); item=new_insertion(session)
    plane=dict(center_um_zyx=[500,600,700],rotation_xyz_deg=[20,30,40],width_um=3000,height_um=5000,max_dimension=384)
    item['source_plane']=plane
    pose=pose_on_plane(item,plane,[500,600,700],35)
    place_probe(item,pose,p,session); original=copy.deepcopy(item['probe']['pose'])
    tips=probe_coordinates(item['probe'],np.arange(4),np.zeros(4))
    place_probe(item,flip_contact_face(item,plane),p,session)
    assert contact_side(item,plane)==-1
    assert_allclose(probe_coordinates(item['probe'],np.arange(4),np.zeros(4)),tips[::-1],atol=1e-9)
    assert_allclose(item['probe']['pose']['normal_source_zyx'],-np.array(original['normal_source_zyx']))
    center=probe_coordinates(item['probe'],1.5,0)
    rotated=pose_on_plane(item,plane,center,70)
    place_probe(item,rotated,p,session)
    assert contact_side(item,plane)==-1 and pose_current(item,p.total)
    assert_allclose(np.linalg.norm(np.diff(probe_coordinates(item['probe'],np.arange(4),np.zeros(4)),axis=0),axis=1),250)
    matrix=np.array([[.8,.2,0,50],[0,1.1,.1,-20],[.2,0,.7,15],[0,0,0,1.]])
    assert_allclose(contact_preview(item,matrix),transform_points(contact_preview(item,np.eye(4)),matrix))
    place_probe(item,flip_contact_face(item,plane),p,session)
    assert contact_side(item,plane)==1

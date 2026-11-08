from legacy_workspace import open_legacy
"""Native Qt regression for old-project attachment and the insertion workflow."""

import pytest

pytestmark = pytest.mark.gui
import copy
import json
import time

import numpy as np
import tifffile
from numpy.testing import assert_allclose

from probemap.registration.coarse import pose_matrix
from probemap.registration.desktop_model import AlignmentProject, ImageSpec
from probemap.registration.landmarks import transform_points
from probemap.registration.insertion_model import plane_is_stale
from probemap.registration.track_model import plane_geometry, plane_uv
from probemap.registration.volumes import write_volume


def wait_until(app,predicate,timeout=30):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        app.processEvents()
        if predicate(): return
        time.sleep(.01)
    raise AssertionError('Insertion workflow did not finish its background work.')


def test_existing_registration_attach_insertion_points_planes_tracks_views_export_resume(tmp_path,monkeypatch):
    from probemap.registration.macos_qt import initialize_qt
    app=initialize_qt()
    from qtpy.QtCore import QPointF, Qt
    from qtpy.QtGui import QColor
    from qtpy.QtTest import QTest
    from qtpy.QtWidgets import QFileDialog
    from probemap.registration.desktop_app import AlignWindow
    from probemap.registration.desktop_rendering import METHODS
    shape=(25,30,35)
    spacing=(200,200,200)
    z,y,x=np.indices(shape)
    data=(50000*np.exp(-((z-12)**2+(y-15)**2+(x-17)**2)/50)).astype(np.uint16)
    brain=write_volume(tmp_path/'brain.tif',data,spacing)
    track=write_volume(tmp_path/'electrodes.tif',(data//2+x*50).astype(np.uint16),spacing)
    atlas=write_volume(tmp_path/'atlas.tif',data,spacing)
    p=AlignmentProject(ImageSpec(atlas,shape,spacing),ImageSpec(brain,shape,spacing),tmp_path/'results')
    p.coarse=pose_matrix(shape,spacing,(-100,200,300),(0,0,15),(1.1,.8,1.2))
    points=transform_points(np.array([[400,600,800],[4000,600,800],[400,5000,800],
                                    [400,600,6000],[4000,5000,6000.]]),p.coarse)
    p.pairs=[{'reference':(point+[100,-200,300]).tolist(),'moving':point.tolist()} for point in points]
    p.fit()
    p.display={'step':3}
    original=p.as_record()
    project_file=p.save(tmp_path/'results/project.json')
    masks=tmp_path/'labels'
    masks.mkdir()
    write_volume(masks/'1_left_mask.tif',(x<17).astype(np.uint8),spacing)
    write_volume(masks/'2_right_mask.tif',(x>=17).astype(np.uint8),spacing)
    w=AlignWindow()
    errors=[]
    w._error=errors.append
    w.show()
    try:
        w.load_project_path(project_file)
        wait_until(app,lambda:not w.busy and w.reference is not None)
        assert w.step==3
        # Real checkable-button click used to leave two nav buttons checked and
        # only display a tiny status-bar instruction.
        w.nav_buttons[4].click()
        assert w.step==0 and w.project_page.mode=='summary'
        assert 'electrodes scan' in w.project_page.progress_summary.text()
        assert [b.isChecked() for b in w.nav_buttons]==[True]+[False]*6
        monkeypatch.setattr(QFileDialog,'getExistingDirectory',lambda *args,**kwargs:str(masks))
        w.choose_labels()
        monkeypatch.setattr(QFileDialog,'getOpenFileName',lambda *args,**kwargs:(str(track),''))
        w.change_electrodes()
        t=w.track_workspace
        def ready():
            return not w.busy and not t._running and not t.render_timer.isActive() and t._cache_current()
        wait_until(app,ready)
        assert w.step==4
        for key in ('pairs','coarse','residual','fit_result'):
            assert w.project.as_record()[key]==original[key]
        matrix=w.project.total.copy()
        item=t.current_insertion()
        world=transform_points(np.array([[2400,2600,2800],[2400,3400,4000],
                                        [2400,2200,4000],[2400,3200,3000.]]),matrix)
        tracing=transform_points(world,np.linalg.inv(t._tracing_to_atlas()))
        for point in tracing: t.add_point(point)
        assert len(item['points'])==4 and item['plane'] is None
        assert len(t.volume_view.point_records)==4
        # Selection travels from the rendered 3D scene back to the point table.
        t.volume_view.canvas.render(size=(400,300))
        pixels=t.volume_view.projected_points()
        assert np.isfinite(pixels).all() and len(pixels)==4
        t.tracing_tools['select'].click()
        QTest.mouseClick(t.volume_view.canvas.native,Qt.MouseButton.LeftButton,
                        pos=QPointF(*pixels[0]).toPoint())
        app.processEvents()
        assert item['points'][0]['id'] in t.selected_points
        low=pixels.min(0)-15
        high=pixels.max(0)+15
        QTest.mousePress(t.volume_view.canvas.native,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.ControlModifier,pos=QPointF(*low).toPoint())
        QTest.mouseMove(t.volume_view.canvas.native,QPointF(*high).toPoint())
        QTest.mouseRelease(t.volume_view.canvas.native,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.ControlModifier,pos=QPointF(*high).toPoint())
        app.processEvents()
        assert len(t.selected_points)==4
        t.tracing_tools['add'].click()
        # Group and move selected points without altering their raw coordinates.
        ids=[p['id'] for p in item['points']]
        t.select_points(ids[-1:])
        t.group_selected()
        other=t.current_insertion()
        assert other['color']!=item['color'] and len(other['points'])==1
        t.move_target.setCurrentIndex(t.move_target.findData(item['id']))
        t.move_selected()
        assert len(item['points'])==4 and not other['points']
        t.insertion_combo.setCurrentIndex(t.insertion_combo.findData(item['id']))
        w.go_step(5)
        wait_until(app,ready)
        t._ensure_labels()
        wait_until(app,lambda:ready() and t.labels is not None)
        assert not plane_is_stale(item)
        cached_fit=copy.deepcopy(item['plane_fit'])
        _,distance=plane_uv(plane_geometry(item['plane']),world)
        assert_allclose(distance,0,atol=1e-8)
        # Reopening does not recompute an unchanged fit.
        w.go_step(4)
        wait_until(app,ready)
        w.go_step(5)
        wait_until(app,ready)
        assert item['plane_fit']==cached_fit
        assert t.track_slice.image.isGrayscale() and t.brain_slice.image.isGrayscale()
        assert len(t.slices)==3 and len(t.volume_view.visuals)==3
        t.volume_view.open_settings()
        assert [t.volume_view.settings_dialog.tabs.tabText(i) for i in range(3)]==['Atlas','Track scan','Brain scan']
        for index in (1,2):
            for method in METHODS.values():
                settings=t.volume_view.render_settings_record()[index]
                settings['rendering']=method
                t.volume_view.set_render_settings(index,settings)
                assert t.volume_view.canvas.render(size=(150,120)).shape==(120,150,4)
        t.volume_view.settings_dialog.controls[2]['opacity'].setValue(41)
        assert t.opacity[2].value()==41 and t.volume_view.visuals[2].opacity==.41
        # Edit and persist the original scan plane; registration stays fixed.
        center=np.array(t.current_insertion()['source_plane']['center_um_zyx'])
        t.placement_normal.setValue(25)
        wait_until(app,ready)
        adjusted=copy.deepcopy(item['source_plane'])
        assert not np.allclose(adjusted['center_um_zyx'],center)
        open_legacy(w,6); wait_until(app,ready)
        w.go_step(5); wait_until(app,ready)
        assert item['source_plane']==adjusted
        t.undo(); wait_until(app,ready)
        item=t.current_insertion()
        start=transform_points([world.mean(0)],np.linalg.inv(t._tracing_to_atlas()))[0]
        direction=start+np.array(t.track_slice.geometry_record['u_zyx'])*800
        t._placement_commit(start,direction)
        t.apply_placement()
        assert len(item['tracks'])==4
        trace=item['tracks'][0]
        assert trace['start_validation']['inside']
        assert_allclose(np.linalg.norm(np.diff(trace['points_source_scan_zyx_um'],axis=0)),4000)
        assert trace['color']==item['color']
        monkeypatch.setattr('probemap.registration.desktop_insertions.QColorDialog.getColor',lambda *args,**kwargs:QColor('#55aaff'))
        t.color_insertion()
        assert all(track['color']=='#55aaff' for track in item['tracks'])
        # A second insertion owns a separate plane and differently colored
        # tracks; selecting it never reassigns the first insertion's geometry.
        w.go_step(4)
        wait_until(app,ready)
        t.insertion_combo.setCurrentIndex(t.insertion_combo.findData(other['id']))
        for point in transform_points(world+np.array([200,0,0]),np.linalg.inv(t._tracing_to_atlas())): t.add_point(point)
        w.go_step(5)
        wait_until(app,ready)
        other=t.current_insertion()
        start=np.array(t.track_slice.geometry_record['center_um_zyx'])
        direction=start+np.array(t.track_slice.geometry_record['u_zyx'])*800
        t._placement_commit(start,direction); t.apply_placement()
        assert len(other['tracks'])==4 and len(item['tracks'])==4
        assert other['tracks'][0]['color']!=item['tracks'][0]['color']
        assert_allclose(np.linalg.norm(np.diff(other['tracks'][0]['points_source_scan_zyx_um'],axis=0)),4000)
        # Atlas region visibility/styles remain available on the matched plane.
        t.region_table.item(0,0).setCheckState(Qt.CheckState.Unchecked)
        t.region_table.item(0,2).setText('Left test region')
        t._update_report()
        assert t.report['segments']
        t.rename_insertion('Posterior insertion')
        open_legacy(w,6)
        wait_until(app,ready)
        assert all(not panel.isVisible() for panel in t.slice_panels)
        assert t.volume_view.track_visuals[trace['id']].visible
        t.volume_view.layers.checks[1].setChecked(False)
        t.volume_view.layers.checks[2].setChecked(True)
        assert not t.volume_view.visuals[1].visible and t.volume_view.visuals[2].visible
        open_legacy(w,7)
        wait_until(app,ready)
        assert all(panel.isVisible() for panel in t.slice_panels)
        assert all(c.track_hits and not c.show_tracks and not c.show_points for c in t.slices)
        t.xyz_source.setCurrentIndex(1)
        wait_until(app,ready)
        assert all('Brain' in label.text() for label in t.slice_titles)
        assert all({line['insertion_id'] for line in c.insertion_tracks}=={item['id'],other['id']} for c in t.slices)
        t.export_analysis()
        wait_until(app,lambda:not w.busy and t.last_export is not None)
        manifest=json.loads(t.last_export.read_text())
        assert manifest['status']=='complete' and manifest['track_length_um']==4000
        assert tifffile.imread(t.last_export.parent/f"insertion_{item['id']}_brain_plane.tif").max()>1
        saved=AlignmentProject.load(project_file)
        assert saved.display['step']==5 and saved.track is not None
        assert_allclose(saved.total,matrix)
        saved_session=copy.deepcopy(t.session)
        again=AlignWindow()
        again._error=errors.append
        again.show()
        try:
            again.load_project_path(project_file)
            open_legacy(again,7)
            a=again.track_workspace
            wait_until(app,lambda:not again.busy and a._cache_current() and not a._running and not a.render_timer.isActive())
            assert again.step==5
            assert a.session['insertions']==saved_session['insertions']
            assert a.session['regions']==saved_session['regions']
            assert a.xyz_source.currentIndex()==1
            assert_allclose(again.project.total,matrix)
            # Export after resuming XYZ still includes the configured native
            # atlas region report, even before visiting step 6 this session.
            assert a.labels is None
            a.export_analysis()
            wait_until(app,lambda:not again.busy and a.last_export is not None)
            assert 'report' in json.loads(a.last_export.read_text())
        finally:
            again.close()
        assert not errors,errors
    finally:
        wait_until(app,lambda:not w.busy)
        w.close()


def test_points_only_in_current_slice_with_nonzero_origin_and_oblique_plane():
    from probemap.registration.macos_qt import initialize_qt
    app=initialize_qt()
    from probemap.registration.desktop_views import SliceCanvas
    from probemap.registration.desktop_insertion_views import InsertionSliceCanvas
    from probemap.registration.track_model import plane_world
    # Registration step 3: neighbouring depth bins and selected out-of-slice
    # points must not leak into the current slice, including a shifted origin.
    canvas=SliceCanvas(0)
    canvas.set_image(np.zeros((16,18)),(20,30,40),3,(-200,50,80))
    canvas.points=[(1,np.array([-140,100,120]),False),
                   (2,np.array([-120,100,120]),False),
                   (3,np.array([-160,100,120]),True)]
    assert [number for number,_,_ in canvas.visible_points()]==[1]
    plane={'center_um_zyx':[210,-120,340],'rotation_xyz_deg':[23,-37,56],
           'width_um':630,'height_um':630,'max_dimension':64}
    geometry=plane_geometry(plane)
    origin=plane_world(geometry,[[30,30]])[0]
    normal=np.array(geometry['normal_zyx'])
    oblique=InsertionSliceCanvas()
    oblique.set_plane_image(np.zeros((64,64)),geometry)
    oblique.slab_um=2000  # Track slab must not broaden point visibility.
    oblique.insertion_points=[{'id':str(i),'atlas_zyx_um':(origin+d*normal).tolist(),'color':'#ffaa44'}
                              for i,d in enumerate([0,9.9,-9.9,30,-30])]
    assert [p['id'] for p,_ in oblique.visible_points()]==['0']
    oblique.selected_points={'3','4'}
    assert [p['id'] for p,_ in oblique.visible_points()]==['0']
    canvas.close()
    oblique.close()
    app.processEvents()


def test_probe_preview_plane_adjustment_insertion_switch_delete_and_resume(tmp_path):
    from probemap.registration.macos_qt import initialize_qt
    app=initialize_qt()
    from qtpy.QtCore import QEvent, QPointF, Qt
    from qtpy.QtGui import QMouseEvent
    from qtpy.QtWidgets import QApplication
    from qtpy.QtTest import QTest
    from probemap.registration.desktop_app import AlignWindow
    from probemap.registration.insertion_model import (add_insertion_point, insertion_session, new_insertion,
                                                       save_manual_plane)
    shape=(25,30,35)
    z,y,x=np.indices(shape)
    spacing=(200,200,200)
    data=(50000*np.exp(-((z-12)**2+(y-15)**2+(x-17)**2)/50)).astype(np.uint16)
    path=write_volume(tmp_path/'brain.tif',data,spacing)
    track=write_volume(tmp_path/'electrode.tif',data,spacing)
    spec=ImageSpec(path,shape,spacing)
    p=AlignmentProject(spec,spec,tmp_path/'result',track=ImageSpec(track,shape,spacing))
    anchors=[[400,600,800],[4000,600,800],[400,5000,800],[400,600,6000]]
    p.pairs=[{'reference':point,'moving':point} for point in anchors]
    p.fit()
    session=insertion_session(p)
    a=new_insertion(session,'Insertion A')
    b=new_insertion(session,'Insertion B')
    plane={'center_um_zyx':[2400,3000,3400],'rotation_xyz_deg':[0,0,0],
           'width_um':6000,'height_um':5000,'max_dimension':384}
    save_manual_plane(a,plane)
    save_manual_plane(b,plane)
    for item in (a,b):
        for point in ([2400,3000,3400],[2600,3000,3400],[2200,3000,3400]):
            add_insertion_point(p,item,point)
    p.display={'step':5}
    project_file=p.save(tmp_path/'result/project.json')
    errors=[]
    w=AlignWindow()
    w._error=errors.append
    w.show()
    try:
        w.load_project_path(project_file)
        t=w.track_workspace
        def ready():
            return not w.busy and t._cache_current() and not t._running and not t.render_timer.isActive()
        wait_until(app,ready)
        a,b=t.session['insertions']
        assert t.title.text()=='Probes'
        assert not t.track_slice.visible_points()  # This page does not add points.
        t.placement_normal.setValue(25); wait_until(app,ready)
        assert_allclose(a['source_plane']['center_um_zyx'][0],2425)
        c=t.track_slice
        first=c._pixel([2425,3000,3400]).toPoint()
        last=c._pixel([2425,3000,4200]).toPoint()
        t.placement_tools['draw'].click()
        QTest.mousePress(c,Qt.MouseButton.LeftButton,pos=first)
        QTest.mouseMove(c,last); app.processEvents()
        assert len(c.probe_polygons)==4 and not a['tracks']
        QTest.mouseRelease(c,Qt.MouseButton.LeftButton,pos=last)
        assert len(a['tracks'])==4 and a['probe'].get('pose')
        original=copy.deepcopy(a['probe']['pose'])
        t.placement_tools['draw'].click()
        QTest.mouseMove(c,first); app.processEvents()
        QTest.keyClick(c,Qt.Key.Key_Escape)
        assert a['probe']['pose']==original
        t.placement_insertion.setCurrentIndex(1); wait_until(app,ready)
        assert t.current_insertion() is b and not c.probe_polygons
        t.placement_insertion.setCurrentIndex(0); wait_until(app,ready)
        assert len(c.probe_polygons)==4
        t._placement_select(True)
        QTest.keyClick(c,Qt.Key.Key_Delete)
        assert not a['tracks'] and not a['probe'].get('pose')
        t.undo(); wait_until(app,ready); a=t.current_insertion()
        assert a['probe']['pose']==original and len(a['tracks'])==4
        w.save_project()
        restored=AlignmentProject.load(project_file).track_analysis['sessions'][0]['insertions']
        assert restored[0]['probe']['pose']==original
        assert len(restored[0]['tracks'])==4 and not restored[1]['tracks']
        assert not errors,errors
    finally:
        wait_until(app,lambda:not w.busy)
        w.close()

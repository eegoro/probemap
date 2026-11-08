from legacy_workspace import open_legacy
"""Physical ruler, named insertions and mouse tools on native Qt canvases."""

import pytest

pytestmark = pytest.mark.gui
import copy

import numpy as np
from numpy.testing import assert_allclose

from probemap.registration.desktop_model import AlignmentProject, ImageSpec
from probemap.registration.insertion_model import (add_insertion_point, add_measurement,
    insertion_session, move_tracks, new_insertion, save_manual_plane, sync_tracks)
from probemap.registration.volumes import write_volume
from test_desktop_tracks import wait_until


def test_transfer_preserves_geometry_and_measurement_uses_3d_units(tmp_path):
    from test_track_model import make_track_project
    p=make_track_project(tmp_path)
    session=insertion_session(p)
    a,b=new_insertion(session,'Left dorsal'),new_insertion(session,'Right ventral')
    a['tracks']=[dict(id='t',name='Shank',control_points_atlas_zyx_um=[[4,9,16],[9,12,25]],plane=copy.deepcopy(session['plane']))]
    sync_tracks(session,p)
    before=copy.deepcopy(a['tracks'][0])
    assert move_tracks(session,{'t'},b['id'])==1
    sync_tracks(session,p)
    moved=b['tracks'][0]
    assert not a['tracks'] and moved['insertion_name']=='Right ventral'
    for key in ('id','plane','control_points_atlas_zyx_um','points_atlas_zyx_um','points_source_track_zyx_voxel'):
        assert moved[key]==before[key]
    assert moved['color']==b['color']
    ruler=add_measurement(session,[100,200,300],[400,600,1500])
    assert_allclose(ruler['length_um'],1300)


def test_sidebar_pan_ruler_legacy_track_transfer_and_roundtrip(tmp_path,monkeypatch):
    from probemap.registration.macos_qt import initialize_qt
    app=initialize_qt()
    from qtpy.QtCore import QPointF,Qt,QEvent
    from qtpy.QtGui import QMouseEvent
    from qtpy.QtTest import QTest
    from qtpy.QtWidgets import QApplication,QInputDialog
    from probemap.registration.desktop_app import AlignWindow
    shape=(25,30,35)
    spacing=(200,200,200)
    z,y,x=np.indices(shape)
    data=(50000*np.exp(-((z-12)**2+(y-15)**2+(x-17)**2)/50)).astype(np.uint16)
    brain=write_volume(tmp_path/'brain.tif',data,spacing)
    track=write_volume(tmp_path/'track.tif',data//2,spacing)
    spec=ImageSpec(brain,shape,spacing)
    p=AlignmentProject(spec,spec,tmp_path/'out',track=ImageSpec(track,shape,spacing))
    anchors=[[400,600,800],[4000,600,800],[400,5000,800],[400,600,6000]]
    p.pairs=[dict(reference=point,moving=point) for point in anchors]
    p.fit()
    session=insertion_session(p)
    a=new_insertion(session,'Dorsal A')
    for point in ([2400,3000,3400],[2400,3400,4000],[2400,2200,4000]): add_insertion_point(p,a,point)
    save_manual_plane(a,dict(center_um_zyx=[2400,3000,3400],rotation_xyz_deg=[0,0,0],width_um=6000,height_um=5000,max_dimension=384))
    a['tracks']=[dict(id='t',name='Shank',control_points_atlas_zyx_um=[[2400,3000,3400],[2400,3000,4200]],plane=copy.deepcopy(a['plane']))]
    sync_tracks(session,p)
    p.display={'step':4}
    path=p.save(tmp_path/'out/project.json')
    errors=[]
    w=AlignWindow()
    w._error=errors.append
    w.show()
    t=w.track_workspace
    def ready(): return not w.busy and t._cache_current() and not t._running and not t.render_timer.isActive()
    def drag(canvas,start,end,button=Qt.MouseButton.LeftButton):
        start,end=QPointF(start),QPointF(end)
        QTest.mousePress(canvas,button,pos=start.toPoint())
        event=QMouseEvent(QEvent.Type.MouseMove,end,QPointF(canvas.mapToGlobal(end.toPoint())),
                          Qt.MouseButton.NoButton,button,Qt.KeyboardModifier.NoModifier)
        QApplication.sendEvent(canvas,event)
        QTest.mouseRelease(canvas,button,pos=end.toPoint())
        app.processEvents()
    try:
        w.load_project_path(path)
        wait_until(app,ready)
        # No invisible automatic name field: New and Rename accept arbitrary labels.
        monkeypatch.setattr(QInputDialog,'getText',lambda *a,**kw:('Anterior left',True))
        t.rename_insertion_button.click()
        assert t.current_insertion()['name']=='Anterior left'
        t.create_insertion('Posterior right')
        b=t.current_insertion()
        t.insertion_combo.setCurrentIndex(0)
        assert t.current_insertion()['name']=='Anterior left'
        assert t.session['insertions'][1]['name']=='Posterior right'
        c=t.track_slice
        c.zoom_at(QPointF(c.width()/2,c.height()/2),3)
        points_before=copy.deepcopy(t.current_insertion()['points'])
        point_owner=t.current_insertion()
        t.point_tool.setCurrentIndex(2)
        before=QPointF(c.pan)
        drag(c,QPointF(c.width()/2,c.height()/2),QPointF(c.width()/2+35,c.height()/2+22))
        assert_allclose([c.pan.x()-before.x(),c.pan.y()-before.y()],[35,22],atol=1)
        assert point_owner['points']==points_before
        # Right-drag zooms, matching the 3D camera, without adding a point.
        t.point_tool.setCurrentIndex(0)
        before=QPointF(c.pan)
        old_zoom=c.zoom
        drag(c,QPointF(c.width()/2,c.height()/2),QPointF(c.width()/2-20,c.height()/2+12),Qt.MouseButton.RightButton)
        assert c.zoom<old_zoom
        assert point_owner['points']==points_before
        t.point_tool.setCurrentIndex(4)
        center=QPointF(c.width()/2,c.height()/2)
        end=center+QPointF(45,10)
        expected=np.linalg.norm(c._world(end)-c._world(center))
        drag(c,center,end)
        assert len(t.session['measurements'])==1
        assert_allclose(t.session['measurements'][0]['length_um'],expected,rtol=.03)
        assert point_owner['points']==points_before
        w.go_step(5)
        wait_until(app,ready)
        # Rendering and Draw are sidebar descendants; image panels contain no settings.
        assert t.control_content.isAncestorOf(t.volume_view.layers)
        assert t.control_content.isAncestorOf(t.rendering_panel)
        assert t.control_content.isAncestorOf(t.track_draw_button)
        t.volume_view.open_settings()
        assert t.sidebar_tabs.currentIndex()==1 and t.rendering_box.isChecked()
        app.processEvents()
        assert t.sidebar_tabs.widget(1).horizontalScrollBar().maximum()==0
        old_sizes=t.splitter.sizes()
        t.splitter.setSizes([330,sum(old_sizes)-330])
        app.processEvents()
        assert t.sidebar_tabs.widget(1).horizontalScrollBar().maximum()==0
        t.splitter.setSizes(old_sizes)
        t.rendering_panel.controls[1]['gamma'].setValue(1.4)
        assert t.volume_view.render_settings[1]['gamma']==1.4
        t.sidebar_tabs.setCurrentIndex(0)
        # Existing manually traced tracks are retained while inspecting the
        # new probe placement page; transfer remains available in 3D tracks.
        a=t.current_insertion()
        moved=copy.deepcopy(a['tracks'][0]['points_atlas_zyx_um'])
        open_legacy(w,6); wait_until(app,ready)
        t.select_track_by_id('t')
        assert a['tracks'][0]['points_atlas_zyx_um']==moved
        t.track_move_target.setCurrentIndex(t.track_move_target.findData(b['id']))
        t.move_selected_track()
        assert not a['tracks'] and b['tracks'][0]['points_atlas_zyx_um']==moved
        assert b['tracks'][0]['color']==b['color']
        open_legacy(w,7)
        wait_until(app,ready)
        assert t.current_insertion()['name']=='Posterior right'
        assert t.control_content.isAncestorOf(t.track_editor) and t.track_editor.isVisible()
        # Selection in the native 3D canvas chooses the same physical track.
        open_legacy(w,6)
        wait_until(app,ready)
        v=t.volume_view
        v.reset_view()
        v.canvas.render(size=(400,300))
        visual=v.track_visuals['t']
        position=np.mean(b['tracks'][0]['points_atlas_zyx_um'],axis=0)[::-1]
        pixel=visual.get_transform(map_from='visual',map_to='canvas').map(position)
        t.select_3d.setChecked(True)
        QTest.mouseClick(v.canvas.native,Qt.MouseButton.LeftButton,pos=QPointF(*(pixel[:2]/pixel[3])).toPoint())
        app.processEvents()
        assert t.selected_id=='t' and t.current_insertion()['id']==b['id']
        w.save_project()
        restored=AlignmentProject.load(path).track_analysis['sessions'][0]
        assert restored['measurements']==t.session['measurements']
        assert restored['insertions']==t.session['insertions']
        assert not errors,errors
    finally:
        wait_until(app,lambda:not w.busy)
        w.close()

from legacy_workspace import open_legacy
"""Stage-specific overlay tools, navigation, all-plane hits and atlas-region views."""

import pytest

pytestmark = pytest.mark.gui
import copy
import numpy as np
from numpy.testing import assert_allclose
from probemap.registration.desktop_model import AlignmentProject,ImageSpec
from probemap.registration.insertion_model import (insertion_session,new_insertion,add_insertion_point,save_manual_plane,sync_tracks,track_slice_points)
from probemap.registration.volumes import write_volume
from test_desktop_tracks import wait_until


def test_slice_contrast_gamma_brightness_and_inversion_leave_source_unchanged():
    from probemap.registration.desktop_slice_display import slice_grayscale
    data=np.array([[0.,10.,20.,30.]])
    before=data.copy()
    settings=dict(contrast=[10,30],gamma=2,brightness=.1,invert=False)
    assert_allclose(slice_grayscale(data,settings),[[.1,.1,.35,1]])
    settings['invert']=True
    assert_allclose(slice_grayscale(data,settings),[[.9,.9,.65,0]])
    assert_allclose(data,before)


def test_equal_panes_icon_tools_navigation_undo_labels_and_xyz(tmp_path):
    from probemap.registration.macos_qt import initialize_qt
    app=initialize_qt()
    from qtpy.QtCore import Qt,QPoint,QPointF,QEvent
    from qtpy.QtGui import QWheelEvent,QMouseEvent
    from qtpy.QtWidgets import QApplication
    from qtpy.QtTest import QTest
    from probemap.registration.desktop_app import AlignWindow
    shape=(25,30,35); spacing=(200,200,200)
    z,y,x=np.indices(shape)
    data=(50000*np.exp(-((z-12)**2+(y-15)**2+(x-17)**2)/50)).astype(np.uint16)
    brain=write_volume(tmp_path/'brain.tif',data,spacing)
    track=write_volume(tmp_path/'track.tif',data//2,spacing)
    spec=ImageSpec(brain,shape,spacing)
    p=AlignmentProject(spec,spec,tmp_path/'result',track=ImageSpec(track,shape,spacing))
    anchors=[[400,600,800],[4000,600,800],[400,5000,800],[400,600,6000]]
    p.pairs=[dict(reference=a,moving=a) for a in anchors]; p.fit()
    masks=tmp_path/'labels'; masks.mkdir()
    write_volume(masks/'1_left.tif',(x<17).astype(np.uint8),spacing)
    write_volume(masks/'2_right.tif',(x>=17).astype(np.uint8),spacing)
    p.labels_dir=masks
    session=insertion_session(p); item=new_insertion(session,'Dorsal')
    for a in ([2400,3000,3400],[2400,3400,4000],[2400,2200,4000]): add_insertion_point(p,item,a)
    save_manual_plane(item,dict(center_um_zyx=[2400,3000,3400],rotation_xyz_deg=[0,0,0],width_um=6000,height_um=5000,max_dimension=384))
    item['tracks']=[dict(id='t',name='Track',control_points_atlas_zyx_um=[[2400,2800,3200],[2800,3200,3600]],plane=copy.deepcopy(item['plane']))]
    sync_tracks(session,p); p.display={'step':4}
    path=p.save(tmp_path/'result/project.json')
    errors=[]; w=AlignWindow(); w._error=errors.append; w.resize(1500,1040); w.show()
    t=w.track_workspace
    def ready(): return not w.busy and t._cache_current() and not t._running and not t.render_timer.isActive()
    def wheel(c):
        pos=QPointF(c.width()*.6,c.height()*.5)
        e=QWheelEvent(pos,QPointF(c.mapToGlobal(pos.toPoint())),QPoint(),QPoint(0,120),Qt.MouseButton.NoButton,Qt.KeyboardModifier.NoModifier,Qt.ScrollPhase.NoScrollPhase,False)
        QApplication.sendEvent(c,e)
    def equal_panes():
        panes=[t.volume_view,*t.slice_panels]
        assert max(p.width() for p in panes)-min(p.width() for p in panes)<=1
        assert max(p.height() for p in panes)-min(p.height() for p in panes)<=1
    try:
        w.load_project_path(path); wait_until(app,ready); app.processEvents()
        equal_panes()
        c=t.track_slice
        assert t.tracing_tools['add'].isVisible() and not c.tools.isVisible()
        assert all(not button.text() and not button.icon().isNull() for button in c.tools.buttons.values())
        slices=[s.value() for s in t.slice_sliders]; position=t._ortho_positions.copy()
        old_zoom=c.zoom; wheel(c)
        assert_allclose(c.zoom,old_zoom*1.1)
        assert [s.value() for s in t.slice_sliders]==slices
        assert_allclose(t._ortho_positions,position)
        count=len(t.current_insertion()['points'])
        t.add_point(t._tracing_positions)
        assert len(t.current_insertion()['points'])==count+1
        w.activateWindow(); c.setFocus(); app.processEvents()
        QTest.keyClick(c,Qt.Key.Key_Z,Qt.KeyboardModifier.ControlModifier)
        app.processEvents()
        assert len(t.current_insertion()['points'])==count
        width=c.width(); t.sidebar_toggle.click(); app.processEvents()
        assert not t.control_content.isVisible() and c.width()>width
        equal_panes()
        t.sidebar_toggle.click(); app.processEvents()
        assert t.control_content.isVisible()
        w.go_step(5); wait_until(app,ready); app.processEvents()
        assert abs(t.volume_view.width()-t.slice_panels[0].width())<=1
        assert all(not panel.isVisible() for panel in t.slice_panels[1:])
        assert all(not c.tools.buttons['add'].isVisible() for c in t.slices)
        assert t.placement_tools['draw'].isVisible() and not t.track_slice.tools.isVisible()
        assert all(not c.tools.buttons['draw'].isVisible() for c in t.slices[1:])
        plane=copy.deepcopy(t.session['plane']); token=t._token
        for c in t.slices:
            wheel(c)
            pos=QPointF(c.width()*.6,c.height()*.5)
            e=QMouseEvent(QEvent.Type.MouseMove,pos,QPointF(c.mapToGlobal(pos.toPoint())),Qt.MouseButton.NoButton,Qt.MouseButton.NoButton,Qt.KeyboardModifier.NoModifier)
            QApplication.sendEvent(c,e)
        app.processEvents()
        assert t.session['plane']==plane and t._token==token
        spin=t.placement_plane_fields['center',0]
        spin.setValue(spin.value()+100)
        wait_until(app,ready)
        assert t.session['plane']!=plane
        t.track_slice.setFocus(); w.undo_current_step(); wait_until(app,ready)
        assert t.session['plane']==plane
        open_legacy(w,7); wait_until(app,ready); app.processEvents(); equal_panes()
        assert all(not c.tools.buttons['add'].isVisible() and not c.tools.buttons['draw'].isVisible() for c in t.slices)
        for c in t.slices:
            hits=track_slice_points(t.session['tracks'][0],c.geometry_record)
            assert hits and all(c.rect().contains(c._pixel(hit['atlas_zyx_um']).toPoint()) for hit in hits)
        assert all(line.visible for mesh,line,text in t.volume_view.slice_planes)
        old=t.volume_view.slice_positions.copy()
        t.slice_sliders[0].setValue(t.slice_sliders[0].value()+1)
        wait_until(app,ready)
        assert t.volume_view.slice_positions[0]!=old[0]
        assert_allclose(t.volume_view.slice_positions,t._ortho_positions)
        t.xyz_source.setCurrentIndex(2); wait_until(app,ready)
        assert all(image[1] is None for image in t._cache['images'])
        t.xyz_source.setCurrentIndex(3); wait_until(app,lambda:ready() and t.labels is not None)
        assert all(image[1] is not None for image in t._cache['images'])
        t.sidebar_tabs.setCurrentIndex(3); app.processEvents()
        assert t.region_table.isVisible() and t.region_table.rowCount()==2
        t.region_table.item(0,0).setCheckState(Qt.CheckState.Unchecked)
        assert t.session['regions']['1']['visible'] is False
        c=t.slices[0]; pixels,ids,geometry,_=t._cache['images'][0]
        row,col=np.argwhere(ids==1)[len(np.argwhere(ids==1))//2]
        color=c.image.pixelColor(int(col),int(row))
        assert color.red()==color.green()==color.blue()
        t.volume_view.layers.labels_check.setChecked(True)
        wait_until(app,lambda:ready() and len(t.volume_view.label_visuals)==2)
        assert not t.volume_view.label_visuals[1].visible and t.volume_view.label_visuals[2].visible
        t.region_table.item(0,0).setCheckState(Qt.CheckState.Checked)
        assert t.volume_view.label_visuals[1].visible
        t.slice_display.channel.setCurrentIndex(2)
        t.slice_display.gamma.setValue(1.8); t.slice_display.invert.setChecked(True)
        assert t.slice_display.records[2]['gamma']==1.8
        matrix=w.project.total.copy()
        w.save_project(); saved=AlignmentProject.load(path)
        view=saved.track_analysis['sessions'][0]['insertion_view']
        assert view['slice_display'][2]['invert'] and view['xyz_source']==3
        assert_allclose(saved.total,matrix)
        assert not errors,errors
    finally:
        wait_until(app,lambda:not w.busy)
        w.close()

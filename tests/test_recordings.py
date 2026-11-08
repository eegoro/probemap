"""Recording libraries, real SpikeGLX metadata and complete affine geometry."""
import copy
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from probemap.registration.insertion_model import (add_insertion_point,fit_insertion_plane,
    insertion_session,new_insertion)
from probemap.registration.landmarks import transform_points
from probemap.registration.neuropixels_model import (channel_rows,contact_mesh,parse_channelmap,
    probe_coordinates,recording_sites)
from probemap.registration.probe_placement import place_probe,pose_on_plane
from probemap.registration.recordings import assign_recording,import_recordings,recording_entries,remove_recordings

META=Path(__file__).parent/'fixtures/np2014_example.meta'


def test_metadata_blank_standby_real_four_shank_geommap_and_saved_subset():
    text=META.read_text(); parsed=parse_channelmap(text)
    assert parsed['probe_type']==2013 and parsed['probe_model']=='NP2014'
    assert len(parsed['channels'])==384 and all(c['recorded'] for c in parsed['channels'])
    assert [sum(c['shank']==i for c in parsed['channels']) for i in range(4)]==[96]*4
    changed=text.replace('imStdby=\n','imStdby=0:2,383\n').replace('snsSaveChanSubset=0:384','snsSaveChanSubset=0:10,383:384')
    assert [c['channel'] for c in parse_channelmap(changed)['channels'] if c['recorded']]==list(range(3,11))
    assert parse_channelmap(text.replace('\n','\r\n'))['channels']==parsed['channels']


def test_library_bulk_import_assignment_unassignment_and_embedded_resume(tmp_path):
    from test_track_model import make_track_project
    from probemap.registration.desktop_model import AlignmentProject
    p=make_track_project(tmp_path); session=insertion_session(p)
    a=new_insertion(session); b=new_insertion(session)
    folder=tmp_path/'maps'; folder.mkdir(); (folder/'nested').mkdir()
    (folder/'one.meta').write_text(META.read_text()); (folder/'nested/two.meta').write_text(META.read_text())
    (folder/'notes.txt').write_text('Unrelated notes'); (folder/'large.bin').write_bytes(b'')
    added,errors=import_recordings(session,[folder]); assert not errors
    assert [r['name'] for r in added]==['a','b']
    assert added[0]['opacity']>added[1]['opacity'] and added[0]['color'] is None
    assert all(owner is None for owner,_ in recording_entries(session))
    assert not channel_rows(a,p.total) and not channel_rows(b,p.total)
    assert import_recordings(session,[folder])[0]==[]
    first,second=added; assign_recording(session,first['id'],a['id'])
    assert a['recordings']==[first] and session['unassigned_recordings']==[second]
    assign_recording(session,first['id'],b['id'])
    assert not a['recordings'] and b['recordings']==[first]
    assert [r['id'] for _,r in recording_entries(session)]==[first['id'],second['id']]
    path=p.save(tmp_path/'saved/project.json')
    restored=insertion_session(AlignmentProject.load(path))
    assert [r['source_text'] for _,r in recording_entries(restored)]==[META.read_text()]*2
    assign_recording(restored,first['id'],None)
    assert all(owner is None for owner,_ in recording_entries(restored))
    remove_recordings(restored,[first['id']]); assert len(recording_entries(restored))==1


def test_contact_corners_and_channel_centres_use_entire_source_to_atlas_affine(tmp_path):
    from test_track_model import make_track_project
    p=make_track_project(tmp_path); session=insertion_session(p); item=new_insertion(session)
    plane=dict(center_um_zyx=[2500,2500,2500],rotation_xyz_deg=[25,15,40],width_um=6000,height_um=6000,max_dimension=256)
    item['source_plane']=plane
    place_probe(item,pose_on_plane(item,plane,[2500,2500,2500],35),p,session)
    rec=import_recordings(session,[META])[0][0]; assign_recording(session,rec['id'],item['id'])
    affine=np.array([[.6,.2,.1,300],[.1,1.3,.2,-200],[0,.15,.85,400],[0,0,0,1.]])
    _,_,native=recording_sites(item,rec)
    base,faces=contact_mesh(item,native,np.eye(4)); vertices,faces2=contact_mesh(item,native,affine)
    assert_allclose(vertices,transform_points(base[:,::-1],affine)[:,::-1],atol=.002)
    assert np.array_equal(faces,faces2) and len(vertices)==384*4
    rows=channel_rows(item,affine)
    atlas=np.array([[row[f'atlas_{axis}_um'] for axis in 'zyx'] for row in rows])
    assert_allclose(atlas,transform_points(native,affine),atol=1e-9)
    # Affine distortion changes spacing, but straight lines and planes remain
    # straight and planar. Native hardware pitch remains exactly 250 um.
    native_tips=probe_coordinates(item['probe'],np.arange(4),np.zeros(4))
    assert_allclose(np.linalg.norm(np.diff(native_tips,axis=0),axis=1),250)
    assert not np.isclose(np.linalg.norm(np.diff(transform_points(native_tips,affine),axis=0)[0]),250)


@pytest.mark.gui
def test_recording_table_assignment_rendering_visibility_color_undo_and_resume(tmp_path,monkeypatch):
    from probemap.registration.macos_qt import initialize_qt
    app=initialize_qt()
    from qtpy.QtCore import Qt
    from qtpy.QtGui import QColor
    from qtpy.QtWidgets import QColorDialog
    from qtpy.QtTest import QTest
    from probemap.registration.desktop_app import ProbeMapWindow
    from probemap.registration.desktop_model import AlignmentProject,ImageSpec
    from probemap.registration.desktop_placement import SelectedCombo
    from probemap.registration.track_model import plane_corners
    from probemap.registration.volumes import write_volume
    from test_desktop_tracks import wait_until
    shape=(50,50,55); spacing=(100,100,100)
    z,y,x=np.indices(shape); data=(50000*np.exp(-((z-25)**2+(y-25)**2+(x-27)**2)/180)).astype(np.uint16)
    brain=write_volume(tmp_path/'brain.tif',data,spacing); electrodes=write_volume(tmp_path/'electrodes.tif',data//2,spacing)
    p=AlignmentProject(ImageSpec(brain,shape,spacing),ImageSpec(brain,shape,spacing),tmp_path/'out',track=ImageSpec(electrodes,shape,spacing))
    masks=tmp_path/'labels'; masks.mkdir()
    write_volume(masks/'1_left.tif',((data>5000)&(x<27)).astype(np.uint8),spacing)
    write_volume(masks/'2_right.tif',((data>5000)&(x>=27)).astype(np.uint8),spacing)
    p.labels_dir=masks
    p.coarse=np.array([[.65,.2,0,500],[0,1.2,.15,-100],[.05,0,.85,200],[0,0,0,1.]])
    anchors=transform_points([[500,500,500],[4000,500,500],[500,4000,500],[500,500,4500]],p.coarse)
    p.pairs=[dict(reference=a.tolist(),moving=a.tolist()) for a in anchors]; p.fit()
    session=insertion_session(p); a=new_insertion(session); b=new_insertion(session)
    points=np.array([[1000,2500,2000],[4000,2500,2000],[1000,2500,2750],[4000,2500,2750.]])
    for item,shift in ((a,0),(b,500)):
        for point in transform_points(points+[0,0,shift],p.total): add_insertion_point(p,item,point)
        fit_insertion_plane(item,session['plane'],p.total)
        plane=item['source_plane']; place_probe(item,pose_on_plane(item,plane,plane['center_um_zyx'],30),p,session)
    p.display={'step':5}; path=p.save(tmp_path/'out/project.json')
    maps=tmp_path/'maps'; maps.mkdir()
    for name in ('first.meta','second.meta','third.meta'): (maps/name).write_text(META.read_text())
    w=ProbeMapWindow(); w.resize(1520,1020); w.show(); errors=[]; w._error=errors.append; t=w.track_workspace
    def ready(): return not w.busy and t._cache_current() and not t._running and not t.render_timer.isActive()
    def assign(row,identifier):
        cell=table.item(row,1); QTest.mouseClick(table.viewport(),Qt.MouseButton.LeftButton,pos=table.visualItemRect(cell).center())
        app.processEvents(); editor=next((e for e in table.findChildren(SelectedCombo) if e.isVisible()),None)
        assert editor is not None
        editor.setCurrentIndex(editor.findData(identifier)); editor.activated.emit(editor.currentIndex()); app.processEvents()
    try:
        w.load_project_path(path); wait_until(app,ready)
        table=t.placement_recordings_table
        t.import_recording_files(paths=[maps]); app.processEvents()
        assert table.rowCount()==3 and not t._placement_recording_visuals and not t.track_slice.probe_sites
        assert [table.item(i,0).text() for i in range(3)]==['a','b','c']
        assert table.item(0,2).text()=='first.meta'
        slice_before=t._cache['images'][0][0].copy(); pose_before=copy.deepcopy(t.current_insertion()['probe']['pose'])
        current=t.current_insertion(); assign(0,current['id']); assign(1,current['id'])
        assert len(t.track_slice.probe_sites)==768
        records=current['recordings']; assert records[0]['opacity']>records[1]['opacity']
        for rec in records:
            _,_,native=recording_sites(current,rec); expected,_=contact_mesh(current,native,w.project.total)
            assert_allclose(t._placement_recording_visuals[rec['id']].mesh_data.get_vertices(),expected)
        # Resolve real 12 um squares at a useful zoom and face-on angle.
        from vispy.util.quaternion import Quaternion
        camera=t.volume_view.view.camera; original_camera=t.volume_view.camera_record()
        _,_,centres=recording_sites(current,records[0])
        camera.center=transform_points(centres,w.project.total).mean(0)[::-1]
        camera.scale_factor=1800
        best=None
        for angles in ((0,0,0),(180,0,0),(90,0,0),(-90,0,0),(0,0,90),(0,0,-90)):
            q=Quaternion.create_from_euler_angles(*angles,degrees=True); camera.set_state(_quaternion=q)
            score=t.volume_view.probe_face.facing_cosine
            if best is None or score>best[0]: best=(score,q)
        camera.set_state(_quaternion=best[1]); app.processEvents()
        rendered=t.volume_view.canvas.render().copy()
        table.horizontalHeader().toggled.emit(0,False); app.processEvents()
        hidden=t.volume_view.canvas.render()
        assert not any(mesh.visible for mesh in t._placement_recording_visuals.values())
        assert np.count_nonzero(np.any(rendered!=hidden,axis=2))>20
        table.horizontalHeader().toggled.emit(0,True); app.processEvents()
        t.volume_view.restore_camera(original_camera)
        expected_plane=transform_points(plane_corners(t._placement_geometry()),w.project.total)
        assert_allclose(plane_corners(t.volume_view._plane_geometry),expected_plane)
        table.item(0,0).setCheckState(Qt.CheckState.Unchecked); app.processEvents()
        assert not t._placement_recording_visuals[records[0]['id']].visible and len(t.track_slice.probe_sites)==384
        table.item(0,0).setCheckState(Qt.CheckState.Checked)
        monkeypatch.setattr(QColorDialog,'getColor',lambda *args,**kwargs:QColor('#38d3a2'))
        QTest.mouseClick(table.viewport(),Qt.MouseButton.LeftButton,pos=table.visualItemRect(table.item(0,3)).center())
        assert records[0]['color']=='#38d3a2'
        table.item(0,4).setText('40%'); app.processEvents()
        mesh=t._placement_recording_visuals[records[0]['id']]
        assert_allclose(mesh.color.rgba[3],.4)
        assert_allclose(mesh.color.rgba[:3],(.35+.65*.4)*np.array(QColor('#38d3a2').getRgbF()[:3]))
        # Contact overlays follow the same pixel transform as the probe when
        # the user rotates a slice. Compare actual painted pixels at contacts.
        canvas=t.track_slice
        for angle in (0,90,180,270):
            canvas.view_rotation=angle; sites=canvas.probe_sites
            painted=canvas.grab().toImage()
            canvas.probe_sites=[]; background=canvas.grab().toImage(); canvas.probe_sites=sites
            changed=0
            for site in sites[:150]:
                pixel=canvas._pixel(site['atlas_zyx_um']).toPoint()
                if canvas.rect().contains(pixel):
                    x=round(pixel.x()*painted.devicePixelRatio()); y=round(pixel.y()*painted.devicePixelRatio())
                    if painted.pixelColor(x,y)!=background.pixelColor(x,y): changed+=1
            assert changed>10
        canvas.view_rotation=0; canvas.update()
        assign(1,None); assert len(t.track_slice.probe_sites)==384
        assert t._placement_recording_visuals[records[0]['id']].visible
        t.undo(); wait_until(app,ready)
        assert len(t.current_insertion()['recordings'])==2 and len(t.track_slice.probe_sites)==768
        current=t.current_insertion(); assign(1,b['id'])
        b_current=next(i for i in t.session['insertions'] if i['id']==b['id'])
        assert len(b_current['recordings'])==1 and t._placement_recording_visuals[b_current['recordings'][0]['id']].visible
        assert len(t.track_slice.probe_sites)==384
        assert_allclose(t._cache['images'][0][0],slice_before)
        assert t.current_insertion()['probe']['pose']==pose_before
        table.setColumnWidth(2,105); table.setFixedHeight(150)
        w.save_project(); w.load_project_path(path); wait_until(app,ready)
        assert table.rowCount()==3 and table.item(0,4).text()=='40%'
        assert table.columnWidth(2)==105 and table.height()==150
        assert table.item(2,1).text()=='—'
        assert len([m for m in t._placement_recording_visuals.values() if m.visible])==2
        assert t.session['unassigned_recordings'][0]['source_text']==META.read_text()
        t.grab().save(str(tmp_path/'recordings.png'))
        QTest.mouseClick(t.title,Qt.MouseButton.LeftButton); assert not table.selectionModel().hasSelection()
        # Region changes must redraw 3D immediately in Probes, and remain
        # consistent when moving between workspaces.
        t.volume_view.layers.labels_check.setChecked(True)
        wait_until(app,lambda:ready() and len(t.volume_view.label_visuals)==2)
        region_row=next(i for i in range(t.region_table.rowCount()) if t._region_for_row(i)['id']==1)
        t.region_table.item(region_row,0).setCheckState(Qt.CheckState.Unchecked)
        assert not t.volume_view.label_visuals[1].visible
        t.placement_insertion.setCurrentIndex(t.placement_insertion.findData(a['id'])); wait_until(app,ready)
        table.item(0,4).setText('20%'); table.item(0,0).setCheckState(Qt.CheckState.Checked)
        current=t.current_insertion(); rec=current['recordings'][0]; identifier=rec['id']
        source_sites=copy.deepcopy(t.track_slice.probe_sites)
        source_rgba=t._placement_recording_visuals[identifier].color.rgba.copy()
        # An old Reconstruction-only filter must not overrule Probes settings.
        t.session.setdefault('reconstruction_view',{})['hidden_recordings']=[identifier]
        t.session['reconstruction_view']['show_recordings']=False
        w.go_step(6); r=w.reconstruction
        def final_ready(): return not w.busy and not r._running and not r.timer.isActive() and r.report is not None
        wait_until(app,final_ready)
        r.view_choice.setCurrentIndex(3); r.plane_probe.setCurrentIndex(r.plane_probe.findData(a['id']))
        wait_until(app,final_ready)
        assert all([site['rgba'] for site in c.probe_sites]==[site['rgba'] for site in source_sites] for c in r.probe_plane_canvases)
        assert any(hasattr(mesh,'color') and np.allclose(mesh.color.rgba,source_rgba) for mesh in r._visuals)
        ids=set(np.unique(r._cache[0][3]))-{0}
        assert [v['id'] in ids for v in r.regions.regions]==sorted([v['id'] in ids for v in r.regions.regions],reverse=True)
        r.view_choice.setCurrentIndex(0); r.scene_mode.setCurrentIndex(3)
        wait_until(app,lambda:final_ready() and len(r.volume_view.label_visuals)==2)
        assert not r.volume_view.label_visuals[1].visible
        row=next(i for i,v in enumerate(r.regions.regions) if v['id']==2)
        r.regions.table.item(row,0).setCheckState(Qt.CheckState.Unchecked)
        assert not r.volume_view.label_visuals[2].visible
        r.view_choice.setCurrentIndex(3); r.atlas_display.setCurrentIndex(r.atlas_display.findData('labels'))
        wait_until(app,final_ready)
        assert r.probe_plane_canvases[0].image.isGrayscale()  # Both labels removed from the image.
        # Changing Probes after Reconstruction was already opened must refresh
        # its previously cached views, not retain an independent appearance.
        w.go_step(5); wait_until(app,ready)
        assert all(not visual.visible for visual in t.volume_view.label_visuals.values())
        table.item(0,4).setText('75%'); table.item(0,0).setCheckState(Qt.CheckState.Unchecked)
        w.go_step(6); wait_until(app,final_ready)
        assert all(not c.probe_sites for c in r.probe_plane_canvases)
        recording_row=next(i for i in range(r.recording_filter.rowCount()) if r.recording_filter.item(i,2).data(Qt.ItemDataRole.UserRole)==identifier)
        r.recording_filter.item(recording_row,2).setCheckState(Qt.CheckState.Checked)
        assert all(len(c.probe_sites)==384 and c.probe_sites[0]['rgba'][3]==.75 for c in r.probe_plane_canvases)
        w.go_step(5); wait_until(app,ready)
        assert table.item(0,0).checkState()==Qt.CheckState.Checked and table.item(0,4).text()=='75%'
        w.save_project(); w.load_project_path(path); wait_until(app,ready)
        assert table.item(0,4).text()=='75%' and not t.session['regions']['1']['visible'] and not t.session['regions']['2']['visible']
        assert not errors,errors
    finally:
        wait_until(app,lambda:not w.busy); w.close()

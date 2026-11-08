import copy
import json

import numpy as np
import pytest
from numpy.testing import assert_allclose

from probemap.registration.insertion_model import new_insertion
from probemap.registration.neuropixels_model import probe_settings,probe_coordinates
from probemap.registration.reconstruction import (orthogonal_geometry,reconstruction_report,
    recording_bands,shanks,slice_reconstruction,write_reconstruction_report)
from probemap.registration.track_model import LabelAtlas,plane_world
from test_surface_entries import box_surface


def sample_probe():
    session={'insertions':[],'regions':{}}; item=new_insertion(session); s=probe_settings(item)
    s['pose']=dict(tip0_source_zyx_um=[400,0,-375],down_source_zyx=[-1,0,0],
        lateral_source_zyx=[0,0,1],normal_source_zyx=[0,-1,0])
    item['recordings']=[dict(id='r',name='a',visible=True,opacity=1,depth_offset_um=30,
        channels=[dict(channel=i,shank=0,row=r,column=i%2,recorded=True) for i,r in enumerate((0,1,5,6))])]
    matrix=np.array([[.7,.15,.12,200],[.1,1.2,.1,-50],[.05,.1,.8,100],[0,0,0,1.]])
    return session,item,matrix


def test_native_depth_after_affine_and_report_does_not_depend_on_tracing_points(tmp_path):
    session,item,matrix=sample_probe(); original=copy.deepcopy(item); surface=box_surface()
    labels=LabelAtlas(np.ones((80,80,80),np.uint16),(100,100,100),[dict(id=1,name='Brain',color_rgb=[255,0,0])],source='fixture')
    report=reconstruction_report(session,matrix,surface,labels)
    probe=report['probes'][0]
    segments=shanks(item,matrix)
    for index,shank in enumerate(probe['shanks']):
        expected=(3000-segments[index,0,0])/.7
        assert_allclose(shank['depth_source_um'],expected,atol=1e-5)
        atlas_distance=np.linalg.norm(np.array(shank['entry_atlas_zyx_um'])-segments[index,0])
        assert abs(expected-atlas_distance)>100
        assert shank['length_source_um']==10000
    assert probe['valid_depth_shanks']==4
    assert_allclose(probe['mean_depth_source_um'],np.mean([s['depth_source_um'] for s in probe['shanks']]))
    c=probe['channels'][0]
    assert_allclose(c['depth_source_um'],probe['shanks'][0]['depth_source_um']-(217-30),atol=1e-5)
    assert item==original
    item['points']=[{'atlas_zyx_um':[1e8,1e8,1e8]}]*50
    assert reconstruction_report(session,matrix,surface,labels)==report
    write_reconstruction_report(report,tmp_path)
    assert json.loads((tmp_path/'reconstruction.json').read_text())==report
    assert len((tmp_path/'reconstruction_shanks.csv').read_text().splitlines())==5
    assert len((tmp_path/'reconstruction_channels.csv').read_text().splitlines())==5
    item['probe']['pose']['tip0_source_zyx_um'][0]=5000
    assert reconstruction_report(session,matrix,surface)['probes'][0]['mean_depth_source_um'] is None
    assert reconstruction_report(session,matrix,None)['probes'][0]['valid_depth_shanks']==0


def test_recording_bands_preserve_gaps_offsets_and_finite_slice_intersections():
    session,item,matrix=sample_probe(); rec=item['recordings'][0]
    bands=recording_bands(item,rec,np.eye(4))
    assert len(bands)==2
    assert_allclose(bands[0]['heights_um'],[179.5,209.5])
    assert_allclose(bands[1]['heights_um'],[254.5,284.5])
    bounds=[[-1000,-2000,-2000],[12000,2000,2000]]
    geometry=orthogonal_geometry(bounds,0,600)
    reconstruction=slice_reconstruction([item],np.eye(4),geometry,25)
    assert len(reconstruction['lines'])==4
    assert len(reconstruction['sites'])==1
    assert len(reconstruction['bands'])==1
    assert all(abs(np.array(line['points'])[:,0]-600).max()<=12.5+1e-6 for line in reconstruction['lines'])
    far=orthogonal_geometry(bounds,0,11500)
    assert slice_reconstruction([item],np.eye(4),far,25)==dict(lines=[],sites=[],bands=[])
    rec['visible']=False
    assert not slice_reconstruction([item],np.eye(4),geometry,25)['bands']
    # Geometry is exactly the shared atlas-aligned sampling grid for all views.
    for axis in range(3):
        g=orthogonal_geometry(bounds,axis,123)
        assert_allclose(plane_world(g)[:,axis],123)


@pytest.mark.gui
def test_final_workflow_views_intersections_reports_and_resume(tmp_path):
    from probemap.registration.macos_qt import initialize_qt
    app=initialize_qt()
    from qtpy.QtCore import Qt
    from probemap.registration.desktop_app import ProbeMapWindow
    from probemap.registration.desktop_model import AlignmentProject,ImageSpec
    from probemap.registration.insertion_model import insertion_session
    from probemap.registration.landmarks import transform_points
    from probemap.registration.volumes import write_volume
    from test_desktop_tracks import wait_until
    shape=(33,37,41); spacing=(100,100,100)
    z,y,x=np.indices(shape); data=((((z-16)/13)**2+((y-18)/14)**2+((x-20)/16)**2)<1).astype(np.uint16)*50000
    paths=[write_volume(tmp_path/name,data,spacing) for name in ('atlas.tif','brain.tif','electrodes.tif')]
    p=AlignmentProject(ImageSpec(paths[0],shape,spacing),ImageSpec(paths[1],shape,spacing),tmp_path/'out',track=ImageSpec(paths[2],shape,spacing))
    matrix=np.array([[.8,.05,.04,200],[.02,1.05,.04,-50],[.01,.02,.9,100],[0,0,0,1.]])
    raw=np.array([[400,500,600],[2500,500,600],[400,3000,600],[400,500,3500],[2500,3000,3500.]])
    p.pairs=[dict(reference=a.tolist(),moving=b.tolist()) for a,b in zip(transform_points(raw,matrix),raw)]; p.fit()
    labels_dir=tmp_path/'labels'; labels_dir.mkdir()
    write_volume(labels_dir/'1_left.tif',((data>0)&(x<20)).astype(np.uint8),spacing)
    write_volume(labels_dir/'2_right.tif',((data>0)&(x>=20)).astype(np.uint8),spacing)
    p.labels_dir=labels_dir
    session=insertion_session(p); item=new_insertion(session); s=probe_settings(item)
    s['pose']=dict(tip0_source_zyx_um=[1200,1600,1300],down_source_zyx=[-1,0,0],lateral_source_zyx=[0,0,1],normal_source_zyx=[0,-1,0])
    from probemap.registration.neuropixels_model import probe_plane
    item['source_plane']=probe_plane(item,np.eye(4))
    from pathlib import Path
    from probemap.registration.recordings import import_recordings,assign_recording
    recordings,failures=import_recordings(session,[Path(__file__).parent/'fixtures/np2013_example.imro'])
    assert not failures
    assign_recording(session,recordings[0]['id'],item['id'])
    session['reconstruction_view']={'probe_opacity':65}  # Older single slider.
    # Older final-page indexes must migrate to the single final workspace.
    p.display={'step':8}; path=p.save(tmp_path/'out/project.json')
    w=ProbeMapWindow(); errors=[]; w._error=errors.append; w.resize(1500,1000); w.show(); r=w.reconstruction
    def ready(): return not w.busy and not r._running and not r.timer.isActive() and r.report is not None
    try:
        w.load_project_path(path); wait_until(app,ready)
        assert w.step==6 and w.pages.currentWidget() is r
        assert [b.text() for b in w.nav_buttons][-4:]==['Tracks','Probes','Reconstruction','Electrophysiology']
        assert len(w.nav_buttons)==8 and len(r.report['probes'])==1
        assert len(r._visuals)>=2 and r.volume_view.point_records==[]
        original=copy.deepcopy(r.session['insertions'][0]['probe'])
        assert r.probe_opacity_inside.value()==50 and r.probe_opacity_outside.value()==100
        shaft_meshes=lambda:[v for v in r._visuals if v.order==20]
        assert sorted(round(m.color.alpha,2) for m in shaft_meshes())==[.5,1.]
        camera=r.volume_view.camera_record()
        r.surface_entries_check.setChecked(True); app.processEvents()
        assert r.surface_visuals.active and not r.surface_visuals.mesh.visible
        assert len(r.surface_visuals.records)==4 and len(shaft_meshes())==2
        assert r.volume_view.visuals[0].visible and r.volume_view.camera_record()==camera
        r.probe_opacity_inside.setValue(0)
        assert [m.color.alpha for m in shaft_meshes()]==[1.]
        assert len(r.surface_visuals.records)==4
        r.probe_opacity_inside.setValue(30); r.probe_opacity_outside.setValue(80)
        assert_allclose(sorted(m.color.alpha for m in shaft_meshes()),[.3,.8])
        r.space.setCurrentIndex(1); app.processEvents()
        entry_visuals=r.surface_visuals
        if entry_visuals.visible_records:
            expected=transform_points([record['atlas_zyx_um'] for record in entry_visuals.visible_records],r.rigid@np.linalg.inv(w.project.total))[:,::-1]
            assert_allclose(entry_visuals.markers._data['a_position'],expected,atol=.001)
        r.space.setCurrentIndex(0)
        r.surface_entries_check.setChecked(False); assert r.surface_visuals is None
        r.surface_entries_check.setChecked(True)
        r.view_choice.setCurrentIndex(1); wait_until(app,lambda:ready() and len(r._cache)==3)
        assert all(c.image is not None and c.background_color=='#000000' for c in r.canvases)
        r.view_choice.setCurrentIndex(2); wait_until(app,lambda:ready() and r._cache[0][0] is r.compare_canvases[0])
        assert [entry[1] for entry in r._cache]==['atlas','brain','brain']
        assert_allclose(r._cache[1][4],r.rigid); assert_allclose(r._cache[2][4],w.project.total)
        r.compare_canvases[0].restore_view(dict(zoom=1.6,pan=[5,9],rotation=90))
        assert r.compare_canvases[2].view_record()==r.compare_canvases[0].view_record()
        r.compare_channel.setCurrentIndex(1); r.compare_axis.setCurrentIndex(1); wait_until(app,ready)
        assert [entry[1] for entry in r._cache]==['atlas','electrodes','electrodes']
        from probemap.registration.track_model import plane_uv
        report_before=copy.deepcopy(r.report)
        r.compare_axis.setCurrentIndex(r.compare_axis.findData(item['id']))
        wait_until(app,lambda:ready() and 'source_normal_scale' in r._cache[0][5])
        assert r.compare_axis.currentText()=='Probe · A'
        assert r._cache[0][5]==r._cache[2][5]
        for index,mapping in ((0,w.project.total),(1,r.rigid),(2,w.project.total)):
            _,distance=plane_uv(r._cache[index][5],shanks(r.items[0],mapping).reshape(-1,3))
            assert_allclose(distance,0,atol=1e-7)
            assert len(r.compare_canvases[index].reconstruction['lines'])==4
        r.atlas_display.setCurrentIndex(r.atlas_display.findData('labels'))
        r.regions.names.setChecked(True)
        assert r.compare_canvases[0].region_labels
        assert not r.compare_canvases[0].image.isGrayscale()
        ids=r._cache[0][3]
        row,col=np.argwhere(ids==1)[0]
        r.regions.table.item(0,0).setCheckState(Qt.CheckState.Unchecked)
        assert r.compare_canvases[0].image.pixelColor(int(col),int(row)).getRgb()[:3]==(0,0,0)
        assert all(name[0]!='1_left' for name in r.compare_canvases[0].region_labels)
        r.regions._all(0,False)
        assert r.compare_canvases[0].region_labels==[]
        assert all(not style.get('visible',True) for style in r.regions.styles.values())
        r.regions._all(0,True)
        r.atlas_display.setCurrentIndex(r.atlas_display.findData('overlay'))
        r.compare_slider.setValue(1); wait_until(app,ready)
        assert r._probe_offsets[item['id']]==100
        assert all(not canvas.reconstruction['lines'] for canvas in r.compare_canvases)
        r.compare_slider.setValue(0); wait_until(app,ready)
        assert r.report==report_before
        r.view_choice.setCurrentIndex(0); r.scene_mode.setCurrentIndex(1); app.processEvents()
        image=r.volume_view.canvas.render(); assert image.shape[0]>100
        r.scene_mode.setCurrentIndex(2); app.processEvents()
        assert r.surface_visuals is not None and len(r.surface_visuals.records)==4
        r.probe_filter.item(0,1).setCheckState(Qt.CheckState.Unchecked)
        assert r.surface_visuals.records==[]
        r.probe_filter.item(0,1).setCheckState(Qt.CheckState.Checked)
        r.scene_mode.setCurrentIndex(3)
        wait_until(app,lambda:ready() and not r._regions_loading and len(r.volume_view.label_visuals)==2)
        r.regions.table.item(0,0).setCheckState(Qt.CheckState.Unchecked)
        assert not r.volume_view.label_visuals[1].visible
        r.scene_mode.setCurrentIndex(0); r.space.setCurrentIndex(1)
        assert_allclose(r.volume_view.matrix,r.rigid)
        r.space.setCurrentIndex(0); assert_allclose(r.volume_view.matrix,w.project.total)
        assert r.session['insertions'][0]['probe']==original
        r.view_choice.setCurrentIndex(2); wait_until(app,ready); r.slab.setValue(40); r.unit.setCurrentIndex(1)
        w.save_project(); assert (tmp_path/'out/reconstruction_shanks.csv').is_file()
        w.load_project_path(path); wait_until(app,ready)
        assert r.view_choice.currentIndex()==2 and r.slab.value()==40 and r.unit.currentIndex()==1
        assert r.surface_entries_check.isChecked()
        assert r.probe_opacity_inside.value()==30 and r.probe_opacity_outside.value()==80
        assert_allclose(sorted(m.color.alpha for m in shaft_meshes()),[.3,.8])
        assert r.compare_axis.currentData()==item['id'] and r.atlas_display.currentData()=='overlay'
        assert r.regions.names.isChecked() and not r.regions.styles['1']['visible']
        # A separate view reuses the saved Probes plane and complete schematic,
        # rather than the slice intersections used by Comparison.
        from probemap.registration.coarse import transformed_corners
        from probemap.registration.reconstruction import saved_probe_plane_geometries
        from probemap.registration.probe_placement import probe_contours,contact_preview
        from probemap.registration.track_model import sample_world
        r.view_choice.setCurrentIndex(r.view_choice.findText('Probe plane'))
        r.plane_stage.setCurrentIndex(0)
        wait_until(app,lambda:ready() and r._cache[0][0] is r.probe_plane_canvases[0])
        assert [entry[1] for entry in r._cache]==['atlas','brain','electrodes']
        assert r.plane_probe.isVisible() and r.plane_stage.isVisible() and not r.compare_axis.isVisible()
        corners=transformed_corners(shape,spacing,np.eye(4))
        native=saved_probe_plane_geometries(r.items[0],r.rigid,w.project.total,corners)['native']
        expected=sample_world(r.native['electrodes'],np.eye(4),plane_world(native)).reshape(native['shape_hw'])
        assert_allclose(r._cache[2][2],expected,atol=1e-8)
        for index,mapping in ((0,w.project.total),(1,r.rigid),(2,r.rigid)):
            canvas=r.probe_plane_canvases[index]
            assert_allclose([p for p,_ in canvas.probe_polygons],probe_contours(r.items[0],mapping))
            assert_allclose(canvas.placement_contacts,contact_preview(r.items[0],mapping))
            assert len(canvas.probe_sites)==384 and not canvas.insertion_points
            assert canvas.placement_read_only and not canvas.placement_selected
        # Navigation remains linked, and cannot move or delete the saved probe.
        from qtpy.QtCore import QEvent,QPointF
        from qtpy.QtGui import QMouseEvent
        from qtpy.QtTest import QTest
        from qtpy.QtWidgets import QApplication
        canvas=r.probe_plane_canvases[1]; changes=[]
        canvas.probeMoved.connect(lambda *args:changes.append(args))
        canvas.probeDeleted.connect(lambda:changes.append('delete'))
        start=canvas.rect().center(); end=start+type(start)(20,10)
        QTest.mousePress(canvas,Qt.MouseButton.LeftButton,pos=start)
        move=QMouseEvent(QEvent.Type.MouseMove,QPointF(end),QPointF(canvas.mapToGlobal(end)),Qt.MouseButton.NoButton,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier)
        QApplication.sendEvent(canvas,move)
        QTest.mouseRelease(canvas,Qt.MouseButton.LeftButton,pos=end)
        QTest.keyClick(canvas,Qt.Key.Key_Backspace)
        assert not changes and canvas.view_record()['pan']==[20,10]
        assert all(c.view_record()==canvas.view_record() for c in r.probe_plane_canvases)
        canvas.rotate_view(); canvas.zoom_at(QPointF(canvas.rect().center()),1.3)
        r.plane_stage.setCurrentIndex(1); wait_until(app,ready)
        for canvas in r.probe_plane_canvases:
            assert_allclose([p for p,_ in canvas.probe_polygons],probe_contours(r.items[0],w.project.total))
        assert r._cache[0][5]==r._cache[2][5]
        r.recording_filter.item(0,2).setCheckState(Qt.CheckState.Unchecked)
        assert all(not c.probe_sites for c in r.probe_plane_canvases)
        views=[c.view_record() for c in r.probe_plane_canvases]
        w.save_project(); w.load_project_path(path); wait_until(app,ready)
        assert r.view_choice.currentText()=='Probe plane' and r.plane_stage.currentIndex()==1
        assert r.plane_probe.currentData()==item['id']
        assert [c.view_record() for c in r.probe_plane_canvases]==views
        assert r.session['insertions'][0]['probe']==original
        assert not errors
        w.grab().save(str(tmp_path/'reconstruction.png'))
    finally:
        wait_until(app,lambda:not w.busy and not r._running)
        w.close(); app.processEvents()


@pytest.mark.gui
def test_region_priority_keeps_identity_filters_visibility_and_shared_styles():
    from probemap.registration.macos_qt import initialize_qt
    app=initialize_qt()
    from qtpy.QtCore import Qt
    from probemap.registration.desktop_reconstruction_regions import ReconstructionRegions
    regions=[dict(id=i,name=f'Region {i}',hemisphere='left' if i%2 else 'right',color_rgb=[20*i,40,60]) for i in range(1,6)]
    styles={'3':{'visible':False}}
    panel=ReconstructionRegions()
    try:
        panel.initialize(regions,styles,{'styles':{'3':{'visible':True},'5':{'visible':False}}})
        assert panel.styles is styles and not styles['3']['visible'] and not styles['5']['visible']
        panel.prioritize([3,4])
        assert [r['id'] for r in panel.regions]==[3,4,1,2,5]
        assert panel.table.item(0,0).checkState()==Qt.CheckState.Unchecked
        # The checkbox must still target ID 4 after rows have moved.
        panel.table.item(1,0).setCheckState(Qt.CheckState.Unchecked)
        assert not styles['4']['visible'] and '1' not in styles
        panel.side.setCurrentIndex(panel.side.findData('left'))
        assert [r['id'] for i,r in enumerate(panel.regions) if not panel.table.isRowHidden(i)]==[3,1,5]
        panel._all(0,True)
        assert styles['3']['visible'] and styles['5']['visible'] and not styles['4']['visible']
        panel.prioritize([2,5])
        assert [r['id'] for r in panel.regions]==[2,5,1,3,4]
        assert not styles['4']['visible'] and panel.table.item(4,0).checkState()==Qt.CheckState.Unchecked
        assert 'styles' not in panel.state_record()  # Canonical session state is authoritative.
    finally:
        panel.close(); app.processEvents()


def test_reconstruction_export_uses_current_poses_and_native_units(tmp_path):
    import zipfile
    from probemap.registration.desktop_model import AlignmentProject,ImageSpec
    from probemap.registration.insertion_model import insertion_session
    from probemap.registration.volumes import write_volume
    from probemap.registration.exports import export_selection
    shape=(25,27,29); spacing=(100,100,100)
    z,y,x=np.indices(shape)
    data=((((z-12)/10)**2+((y-13)/11)**2+((x-14)/12)**2)<1).astype(np.uint16)*50000
    scan=write_volume(tmp_path/'source.tif',data,spacing)
    p=AlignmentProject(ImageSpec(scan,shape,spacing),ImageSpec(scan,shape,spacing),tmp_path/'project',track=ImageSpec(scan,shape,spacing))
    anchors=[[400,500,600],[2000,500,600],[400,2200,600],[400,500,2400]]
    p.pairs=[dict(reference=a,moving=a) for a in anchors]; p.fit()
    session=insertion_session(p); item=new_insertion(session); settings=probe_settings(item)
    settings['pose']=dict(tip0_source_zyx_um=[1000,1300,1000],down_source_zyx=[-1,0,0],lateral_source_zyx=[0,0,1],normal_source_zyx=[0,-1,0])
    original=copy.deepcopy(session)
    for fmt in ('json','zip'):
        path=tmp_path/f'chosen.{fmt}'
        export_selection(p,dict(path=path,kind='reconstruction',stage='registration',format=fmt))
        if fmt=='zip':
            with zipfile.ZipFile(path) as archive:
                assert set(archive.namelist())=={'reconstruction.json','reconstruction_shanks.csv','reconstruction_channels.csv'}
                report=json.loads(archive.read('reconstruction.json'))
        else: report=json.loads(path.read_text())
        assert report['distance_space']=='source_scan_um' and len(report['probes'])==1
        assert report['probes'][0]['mean_depth_source_um']>1000
    assert session==original and not p.output_dir.exists()


def test_region_lengths_and_channel_depth_use_native_distance_and_label_grid():
    session,item,_=sample_probe(); settings=item['probe']
    settings['pose']['tip0_source_zyx_um']=[400,1000,1000]
    matrix=np.diag([.5,1,1,1.]); matrix[0,3]=100
    data=np.zeros((40,40,40),np.uint16); data[6:]=1
    labels=LabelAtlas(data,(200,100,100),[dict(id=1,name='Deep',color_rgb=[255,0,0])],source='fixture')
    report=reconstruction_report(session,matrix,box_surface(),labels)
    probe=report['probes'][0]; s=probe['shanks'][0]
    assert_allclose(s['depth_source_um'],5400,atol=1e-7)
    # Nearest-neighbour label boundary at atlas Z=1100 -> source h=1600.
    assert abs(s['regions'][0]['length_source_um']-3800)<=25
    assert s['regions'][0]['length_source_um']<=s['depth_source_um']
    # A recording moved above the surface must not be assigned brain tissue.
    item['recordings'][0]['depth_offset_um']=-7000
    channels=reconstruction_report(session,matrix,box_surface(),labels)['probes'][0]['channels']
    assert all(c['depth_source_um']<0 and c['region']=='Above atlas surface' and c['region_id'] is None for c in channels)


def test_probe_comparison_planes_preserve_metric_scale_shear_and_native_offsets():
    from probemap.registration.coarse import transformed_corners
    from probemap.registration.landmarks import transform_points
    from probemap.registration.reconstruction import probe_comparison_geometries
    from probemap.registration.track_model import plane_uv
    session,item,matrix=sample_probe(); original=copy.deepcopy(item)
    corners=transformed_corners((60,60,60),(100,100,100),np.eye(4))
    rigid=np.array([[0,0,1,200],[0,1,0,-100],[-1,0,0,7000],[0,0,0,1.]])
    geometries=probe_comparison_geometries(item,rigid,matrix,[corners],corners)
    assert geometries['atlas']==geometries['after']
    assert geometries['before']['shape_hw']==geometries['after']['shape_hw']
    assert geometries['before']['pixel_size_um']==geometries['after']['pixel_size_um']
    for key,mapping in (('before',rigid),('after',matrix)):
        geometry=geometries[key]
        unit=np.column_stack([geometry[k] for k in ('u_zyx','v_zyx','normal_zyx')])
        assert_allclose(unit.T@unit,np.eye(3),atol=1e-12)
        points=shanks(item,mapping).reshape(-1,3)
        _,distances=plane_uv(geometry,points)
        assert_allclose(distances,0,atol=1e-9)
        assert len(slice_reconstruction([item],mapping,geometry,25)['lines'])==4
        shifted=probe_comparison_geometries(item,rigid,matrix,[corners],corners,offset_um=100)[key]
        _,distance=plane_uv(shifted,points)
        assert_allclose(distance,-100*shifted['source_normal_scale'],atol=1e-9)
        assert not slice_reconstruction([item],mapping,shifted,25)['lines']
    tips_before=shanks(item,rigid)[:,0]; tips_after=shanks(item,matrix)[:,0]
    before_uv,_=plane_uv(geometries['before'],tips_before); after_uv,_=plane_uv(geometries['after'],tips_after)
    step=geometries['before']['pixel_size_um']
    assert_allclose(np.linalg.norm(np.diff(before_uv,axis=0),axis=1)*step,250)
    assert not np.allclose(np.linalg.norm(np.diff(after_uv,axis=0),axis=1)*step,250)
    assert item==original


def test_saved_probe_plane_reuses_probes_sampling_and_maps_full_affine_without_mutation():
    from probemap.registration.coarse import transformed_corners
    from probemap.registration.landmarks import transform_points
    from probemap.registration.reconstruction import saved_probe_plane_geometries
    from probemap.registration.track_model import (plane_geometry,plane_covering_volumes,
        plane_corners,plane_uv,transform_plane)
    _,item,matrix=sample_probe()
    # A deliberately adjusted, oblique saved plane must win over a pose-derived
    # plane or the original tracing points.
    plane=dict(center_um_zyx=[2500,2600,2700],rotation_xyz_deg=[18,24,31],
        width_um=4000,height_um=5000,max_dimension=128)
    item['source_plane']=plane; original=copy.deepcopy(item)
    corners=transformed_corners((60,60,60),(100,100,100),np.eye(4))
    rigid=np.array([[0,0,1,200],[0,1,0,-100],[-1,0,0,7000],[0,0,0,1.]])
    frames=saved_probe_plane_geometries(item,rigid,matrix,corners)
    expected=plane_geometry(plane_covering_volumes(plane,[corners]))
    assert frames['native']==expected
    assert_allclose(plane_world(frames['before']),transform_points(plane_world(expected),rigid))
    after=frames['after']; basis=np.column_stack([after[k] for k in ('u_zyx','v_zyx','normal_zyx')])
    assert_allclose(basis.T@basis,np.eye(3),atol=1e-12)
    mapped=transform_points(plane_corners(expected),matrix)
    uv,distance=plane_uv(after,mapped)
    assert_allclose(distance,0,atol=1e-9)
    assert (uv>=-.5-1e-7).all()
    assert (uv<=np.array(after['shape_hw'][::-1])-.5+1e-7).all()
    assert frames['atlas']==after and item==original
    item['points']=[{'atlas_zyx_um':[1e9,2e9,3e9]}]
    assert saved_probe_plane_geometries(item,rigid,matrix,corners)==frames
    # Older projects can still show a saved atlas plane, or a pose-only probe.
    item.pop('source_plane'); item['plane']=transform_plane(plane,matrix)
    assert saved_probe_plane_geometries(item,rigid,matrix,corners)['native']['shape_hw']
    item.pop('plane')
    assert saved_probe_plane_geometries(item,rigid,matrix,corners)['native']['shape_hw']


def test_depth_inverse_mapping_respects_anisotropic_native_voxel_calibration():
    from probemap.registration.landmarks import transform_points
    session,item,matrix=sample_probe(); settings=item['probe']
    spacing=np.array([40.,25.,10.]); tip_voxel=np.array([10.,20.,30.])
    settings['pose']['tip0_source_zyx_um']=(tip_voxel*spacing).tolist()
    vector=np.array([-3.,0.,-4.]); vector/=np.linalg.norm(vector)
    lateral=np.array([-4.,0.,3.]); lateral/=np.linalg.norm(lateral)
    settings['pose'].update(down_source_zyx=vector.tolist(),lateral_source_zyx=lateral.tolist(),normal_source_zyx=[0,1,0])
    report=reconstruction_report(session,matrix,box_surface())
    for shank in report['probes'][0]['shanks']:
        assert shank['depth_source_um'] is not None
        source_tip=probe_coordinates(settings,shank['shank'],0.)
        entry=np.array(shank['entry_source_zyx_um'])
        voxel_distance=(entry-source_tip)/spacing
        expected=np.linalg.norm(voxel_distance*spacing)
        assert_allclose(shank['depth_source_um'],expected,atol=1e-8)
        atlas_tip=transform_points([source_tip],matrix)[0]
        assert abs(np.linalg.norm(np.array(shank['entry_atlas_zyx_um'])-atlas_tip)-expected)>100
        assert abs(np.linalg.norm(voxel_distance)-expected)>100


def test_label_display_modes_individual_visibility_and_names_leave_labels_unchanged():
    from probemap.registration.reconstruction import label_slice_image
    gray=np.full((4,6),.5); ids=np.tile([1,1,1,2,2,0],(4,1)); original=ids.copy()
    regions=[dict(id=1,name='Left',color_rgb=[255,0,0]),dict(id=2,name='Right',color_rgb=[0,0,255])]
    raw,text=label_slice_image(gray,ids,regions,{},mode='atlas',names=True)
    assert_allclose(raw,gray); assert not text
    overlay,text=label_slice_image(gray,ids,regions,{'2':{'visible':False}},names=True,opacity=.5)
    assert_allclose(overlay[:,0],np.tile([.75,.25,.25],(4,1))); assert_allclose(overlay[:,3],.5)
    assert len(text)==1 and text[0][0]=='Left'
    labels,text=label_slice_image(gray,ids,regions,{'1':{'visible':False}},mode='labels',names=True)
    assert not labels[:,:3].any(); assert_allclose(labels[:,3],np.tile([0,0,1],(4,1)))
    assert len(text)==1 and text[0][0]=='Right'
    assert np.array_equal(ids,original)

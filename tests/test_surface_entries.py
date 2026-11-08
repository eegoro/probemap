import copy
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from probemap.registration.insertion_model import new_insertion
from probemap.registration.landmarks import transform_points
from probemap.registration.neuropixels_model import probe_coordinates,probe_settings
from probemap.registration.surface_entries import AtlasSurface,build_atlas_surface,probe_surface_entries,probe_shank_sections
from probemap.registration.volumes import Volume,write_volume


def box_surface():
    from itertools import product
    vertices=np.array(list(product((-1000,3000),(-1500,1500),(-1500,1500))),float)
    faces=[]
    for axis in range(3):
        others=[i for i in range(3) if i!=axis]
        for side in (vertices[:,axis].min(),vertices[:,axis].max()):
            # Every face consists of its four vertices, ordered around its rim.
            indices=np.flatnonzero(vertices[:,axis]==side)
            uv=vertices[indices][:,others]
            order=np.argsort(np.arctan2(uv[:,1]-uv[:,1].mean(),uv[:,0]-uv[:,0].mean()))
            a,b,c,d=indices[order]
            faces.extend(((a,b,c),(a,c,d)))
    return AtlasSurface(vertices,np.array(faces),tuple((25,25,25)))


def test_surface_entries_use_finite_registered_shanks_and_keep_native_geometry():
    item=new_insertion({'insertions':[]})
    settings=probe_settings(item)
    settings['pose']=dict(tip0_source_zyx_um=[400,0,-375],down_source_zyx=[-1,0,0],
        lateral_source_zyx=[0,0,1],normal_source_zyx=[0,-1,0],surface_depths_um=[111]*4)
    matrix=np.array([[.7,.15,.12,200],[.1,1.2,.1,-50],[.05,.1,.8,100],[0,0,0,1.]])
    original=copy.deepcopy(item); surface=box_surface()
    entries=probe_surface_entries(item,matrix,surface)
    assert len(entries)==4
    tips=transform_points(probe_coordinates(settings,np.arange(4),0),matrix)
    for shank,entry in enumerate(entries):
        height=(3000-tips[shank,0])/.7
        assert_allclose(entry['height_from_tip_um'],height,atol=1e-9)
        expected=transform_points([probe_coordinates(settings,shank,height)],matrix)[0]
        assert_allclose(entry['atlas_zyx_um'],expected,atol=1e-9)
        assert entry['shank']==shank
    assert item==original
    assert not np.allclose(np.linalg.norm(np.diff([e['atlas_zyx_um'] for e in entries],axis=0),axis=1),250)
    # No nearest-surface projection for a miss, a parallel line or a short shank.
    assert surface.entry([0,0,20000],[5000,0,20000]) is None
    assert surface.entry([0,0,0],[100,0,0]) is None
    assert surface.entry([0,0,0],[0,0,0]) is None
    assert_allclose(surface.entry([0,0,0],[5000,0,0])[0],[3000,0,0])
    settings['pose']['tip0_source_zyx_um'][2]=20000
    assert probe_surface_entries(item,matrix,surface)==[]
    assert probe_surface_entries({'probe':{}},matrix,surface)==[]


def test_atlas_envelope_fills_internal_cavities_and_preserves_physical_grid():
    z,y,x=np.indices((45,49,53))
    data=(((z-22)/17)**2+((y-24)/19)**2+((x-26)/21)**2<=1).astype(np.uint16)*100
    data[((z-22)/10)**2+((y-24)/12)**2+((x-26)/14)**2<=1]=1000
    data[18:27,20:29,22:31]=0
    surface=build_atlas_surface(Volume(data,(30,40,50)),max_dimension=32)
    assert_allclose(surface.spacing_um_zyx,(60,80,100))
    center=np.array([22*30,24*40,26*50])
    hit=surface.entry(center,center+[1500,0,0])
    assert hit is not None and hit[0][0]>center[0]+400
    assert_allclose(hit[0][1:],center[1:])
    with pytest.raises(ValueError): build_atlas_surface(Volume(np.zeros((8,8,8)),(1,1,1)))


def test_shaft_sections_follow_registered_envelope_and_preserve_native_shape():
    from test_reconstruction import sample_probe
    from probemap.registration.neuropixels_model import shank_mesh
    surface=box_surface(); _,item,matrix=sample_probe(); original=copy.deepcopy(item)
    sections=probe_shank_sections(item,matrix,surface)
    assert len(sections[True])==len(sections[False])==4
    entries=probe_surface_entries(item,matrix,surface)
    length=item['probe']['geometry']['shank_length_um']
    for shank in range(4):
        boundary=entries[shank]['height_from_tip_um']
        assert_allclose(sections[True][shank],[shank,0,boundary])
        assert_allclose(sections[False][shank],[shank,boundary,length])
        for section in (sections[True][shank],sections[False][shank]):
            vertices,faces=shank_mesh(item,matrix,[section])
            native=transform_points(vertices[:,::-1],np.linalg.inv(matrix))
            pose=item['probe']['pose']; tip=probe_coordinates(item['probe'],shank,0.)
            heights=-(native-tip)@np.array(pose['down_source_zyx'])
            assert_allclose([heights.min(),heights.max()],section[1:],atol=.002)
            assert faces.max()<len(vertices) and np.isfinite(vertices).all()
    assert item==original
    # Both endpoints outside can still pass through the brain. A miss and an
    # entirely internal finite shaft must not be split at an imagined entry.
    assert_allclose(surface.segment_intervals([-2000,0,0],[4000,0,0]),
        [(0,1/6,False),(1/6,5/6,True),(5/6,1,False)])
    assert surface.segment_intervals([0,0,0],[100,0,0])==[(0.,1.,True)]
    assert surface.segment_intervals([0,0,20000],[100,0,20000])==[(0.,1.,False)]


@pytest.mark.gui
def test_surface_entry_mode_visuals_visibility_frames_flip_and_resume(tmp_path):
    from probemap.registration.macos_qt import initialize_qt
    app=initialize_qt()
    from qtpy.QtCore import Qt
    from vispy.util.quaternion import Quaternion
    from probemap.registration.desktop_app import ProbeMapWindow
    from probemap.registration.desktop_model import AlignmentProject,ImageSpec
    from probemap.registration.insertion_model import insertion_session,add_insertion_point,fit_insertion_plane
    from probemap.registration.probe_placement import place_probe
    from test_desktop_tracks import wait_until
    shape=(33,37,41); spacing=(100,100,100)
    z,y,x=np.indices(shape)
    data=((((z-16)/13)**2+((y-18)/14)**2+((x-20)/16)**2)<1).astype(np.uint16)*50000
    atlas=write_volume(tmp_path/'atlas.tif',data,spacing)
    scan=write_volume(tmp_path/'brain.tif',data,spacing)
    electrode=write_volume(tmp_path/'electrodes.tif',data//2,spacing)
    p=AlignmentProject(ImageSpec(atlas,shape,spacing),ImageSpec(scan,shape,spacing),tmp_path/'out',
        track=ImageSpec(electrode,shape,spacing))
    p.coarse=np.diag([.9,.95,1.05,1.])
    matrix=np.array([[.8,.05,.04,200],[.02,1.05,.04,-50],[.01,.02,.9,100],[0,0,0,1.]])
    raw=np.array([[400,500,600],[2500,500,600],[400,3000,600],[400,500,3500],[2500,3000,3500.]])
    p.pairs=[dict(reference=a.tolist(),moving=b.tolist()) for a,b in zip(transform_points(raw,matrix),transform_points(raw,p.coarse))]
    p.fit(); session=insertion_session(p)
    for y0 in (1600,2300):
        item=new_insertion(session)
        points=np.array([[1000,y0,1300],[2000,y0,1300],[1000,y0,2050],[2000,y0,2050.]])
        for point in transform_points(points,p.total): add_insertion_point(p,item,point)
        fit_insertion_plane(item,session['plane'],p.total)
        pose=dict(tip0_source_zyx_um=[1200,y0,1300],down_source_zyx=[-1,0,0],
            lateral_source_zyx=[0,0,1],normal_source_zyx=[0,-1,0])
        place_probe(item,pose,p,session)
    from probemap.registration.recordings import import_recordings,assign_recording
    added,failures=import_recordings(session,[Path(__file__).parent/'fixtures/np2013_example.imro'])
    assert not failures
    assign_recording(session,added[0]['id'],session['insertions'][0]['id'])
    p.display={'step':5}; path=p.save(tmp_path/'out/project.json')
    w=ProbeMapWindow(); errors=[]; w._error=errors.append; w.resize(1500,1000); w.show()
    t=w.track_workspace
    def ready(): return not w.busy and t._cache_current() and not t._running and not t.render_timer.isActive()
    def entries_ready(): return ready() and t._entry_surface_loading is None and t._entry_visuals is not None
    try:
        w.load_project_path(path); wait_until(app,ready)
        t.sidebar_tabs.setCurrentIndex(1)
        original=copy.deepcopy(t.session['insertions'])
        pixels=t._cache['images'][0][0].copy()
        sites=copy.deepcopy(t.track_slice.probe_sites)
        assert len(sites)==384
        layers=copy.deepcopy(t.volume_view.layers.record())
        t.placement_view.setCurrentIndex(1); wait_until(app,entries_ready)
        v=t.volume_view; entries=t._entry_visuals
        assert entries.mesh.visible and len(entries.records)==8
        assert not v.point_markers.visible and not v.plane_outline.visible
        assert all(not mesh.visible for mesh in t._placement_visuals.values())
        assert all(not visual.visible for visual in v.visuals)
        assert len(t.track_slice.probe_polygons)==4
        assert len(t.track_slice.probe_sites)==len(sites)
        assert_allclose([r['atlas_zyx_um'] for r in t.track_slice.probe_sites],[r['atlas_zyx_um'] for r in sites])
        assert [r['label'] for r in t.track_slice.probe_sites]==[r['label'] for r in sites]
        assert_allclose(t._cache['images'][0][0],pixels)
        assert t.session['insertions']==original
        assert v.layers.record()==layers
        v.view.camera.set_state(_quaternion=Quaternion.create_from_euler_angles(0,0,-90,degrees=True))
        v.reset_view(); app.processEvents(); image=v.canvas.render()
        wait_until(app,lambda:len(entries.visible_records)==8)
        assert len(entries.labels.text)==8
        assert np.count_nonzero((image[:,:,0]>image[:,:,2]*1.3)&(image[:,:,1]>image[:,:,2]*1.3))>10
        assert np.count_nonzero((image[:,:,2]>image[:,:,0]*1.3)&(image[:,:,2]>image[:,:,1]*1.1))>10
        points=copy.deepcopy(entries.records)
        v.flip_checks[0].setChecked(True); app.processEvents()
        assert entries.records==points and v.view.camera.flip==(True,False,False)
        assert t.session['insertions']==original
        t.visibility_table.item(0,2).setCheckState(Qt.CheckState.Unchecked)
        assert len(entries.records)==4 and all(r['name']=='B' for r in entries.records)
        t.visibility_table.item(0,2).setCheckState(Qt.CheckState.Checked)
        assert len(entries.records)==8
        t.grab().save(str(tmp_path/'surface-entries.png'))
        w.save_project(); w.load_project_path(path); wait_until(app,entries_ready)
        assert t.placement_view.currentData()=='entries' and t._entry_visuals.mesh.visible
        assert_allclose([r['atlas_zyx_um'] for r in t._entry_visuals.records],[r['atlas_zyx_um'] for r in points])
        assert v.view.camera.flip==(True,False,False)
        t.placement_view.setCurrentIndex(0); app.processEvents()
        assert not t._entry_visuals.mesh.visible and not v.surface_entries_active
        assert v.plane_outline.visible and len(v.point_records)==8
        assert all(mesh.visible for mesh in t._placement_visuals.values())
        assert v.layers.record()==layers
        t.placement_view.setCurrentIndex(1); app.processEvents()
        w.go_step(4); wait_until(app,ready)
        assert not t._entry_visuals.mesh.visible and not v.surface_entries_active
        assert t.tracing_tools['add'].isVisible()
        w.go_step(5); wait_until(app,entries_ready)
        assert t._entry_visuals.mesh.visible and v.surface_entries_active
        assert not errors,errors
    finally:
        wait_until(app,lambda:not w.busy and t._entry_surface_loading is None)
        w.close()

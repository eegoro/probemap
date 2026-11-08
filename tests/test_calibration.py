"""Independent physical grids, voxel anchors and transactional recalculation."""
import copy
import csv
import json
from pathlib import Path

import numpy as np
import pytest
import tifffile
from numpy.testing import assert_allclose

from probemap.registration.calibration import recalibrate_project, prepare_calibration_update
from probemap.registration.desktop_model import AlignmentProject, ImageSpec, export_project, inspect_tiff
from probemap.registration.insertion_model import (add_insertion_point, add_measurement, export_insertions,
    fit_insertion_plane, insertion_session, new_insertion, plane_is_stale, save_manual_plane, sync_tracks)
from probemap.registration.landmarks import transform_points
from probemap.registration.neuropixels_model import add_recording, fit_probe, pose_current, probe_coordinates
from probemap.registration.track_model import active_session, load_label_atlas, plane_geometry, plane_uv
from probemap.registration.volumes import write_volume
from test_track_model import make_track_project
from test_neuropixels import sample_probe, FIXTURE


def annotated_project(tmp_path,model='affine'):
    p=make_track_project(tmp_path)
    p.coarse=np.array([[.8,.13,-.05,9],[-.08,1.2,.16,-7],[.04,.1,.75,3],[0,0,0,1.]])
    raw=np.array([[2,6,8],[12,6,8],[2,24,8],[2,6,40],[12,24,40.]])
    placed=transform_points(raw,p.coarse)
    residual=np.eye(4)
    if model=='affine': residual[:3,:3]=[[1.1,-.1,.06],[.05,.9,.02],[.02,-.04,1.2]]
    residual[:3,3]=[1,2,-3]
    reference=transform_points(placed,residual)
    p.pairs=[dict(reference=r.tolist(),moving=m.tolist()) for r,m in zip(reference,placed)]
    p.fit(model)
    s=insertion_session(p)
    item=new_insertion(s,'Medial')
    native=np.array([[6,12,12],[8,18,12],[10,12,28],[12,18,28.]])
    world=transform_points(native,p.total)
    for point in world: add_insertion_point(p,item,point)
    fit_insertion_plane(item,s['plane'])
    item['tracks']=[dict(id='trace',name='Deep to surface',control_points_atlas_zyx_um=world[:2].tolist(),
                         start_validation=dict(inside=True))]
    sync_tracks(s,p)
    for anchor in ('track','moving','reference'): add_measurement(s,*world[:2],anchor=anchor)
    s['edit_history']=[copy.deepcopy(dict(insertions=s['insertions'],measurements=s['measurements']))]
    return p,s,item,native


def test_four_independent_grids_preserve_image_anchors_and_rebuild_native_measurements(tmp_path):
    p,s,item,native=annotated_project(tmp_path)
    s['insertion_view']['tracing_positions_zyx_um']=[10,20,30]
    original=copy.deepcopy(p.as_record())
    new=recalibrate_project(p,(3,4,5),(4,3,6),labels_spacing=(1,2,3),track_spacing=(5,6,7))
    a=np.diag([1.5,4/3,1.25,1])
    b=np.diag([2,1,1.5,1])
    bt=np.diag([2.5,2,1.75,1])
    assert p.as_record()==original
    assert new.label_spacing==(1,2,3)
    assert new.moving.spacing_um_zyx==(4,3,6) and new.track.spacing_um_zyx==(5,6,7)
    assert_allclose(new.total,a@p.total@np.linalg.inv(b),atol=1e-12)
    for old,pair in zip(p.pairs,new.pairs):
        assert_allclose(np.asarray(pair['reference'])/new.reference.spacing_um_zyx,
                        np.asarray(old['reference'])/p.reference.spacing_um_zyx,atol=1e-12)
        before=transform_points([old['moving']],np.linalg.inv(p.coarse))/p.moving.spacing_um_zyx
        after=transform_points([pair['moving']],np.linalg.inv(new.coarse))/new.moving.spacing_um_zyx
        assert_allclose(after,before,atol=1e-12)
    ns=active_session(new)
    assert 'tracing_positions_zyx_um' not in ns['insertion_view']
    assert ns['insertions'][0]['plane_fit']['measurement_frame']=='source scan physical micrometres'
    assert ns['id']==s['id'] and len(new.track_analysis['sessions'])==1
    ni=ns['insertions'][0]
    assert ni['id']==item['id'] and ni['color']==item['color']
    expected_native=transform_points(native,bt)
    expected_world=transform_points(expected_native,new.total)
    assert_allclose([point['atlas_zyx_um'] for point in ni['points']],expected_world,atol=1e-12)
    assert_allclose([point['source_track_zyx_voxel'] for point in ni['points']],native/p.track.spacing_um_zyx)
    assert not plane_is_stale(ni)
    _,dist=plane_uv(plane_geometry(ni['plane']),expected_world)
    assert_allclose(dist,0,atol=1e-10)
    lengths=[np.linalg.norm(expected_native[1]-expected_native[0]),
             np.linalg.norm((b[:3,:3]@native[:2].T).T[1]-(b[:3,:3]@native[:2].T).T[0])]
    assert_allclose([m['length_um'] for m in ns['measurements'][:2]],lengths)
    assert_allclose(ns['measurements'][2]['points_atlas_zyx_um'],transform_points(s['measurements'][2]['points_atlas_zyx_um'],a))
    assert ns['edit_history'][0]['insertions'][0]['points']==ni['points']
    track=ni['tracks'][0]
    assert track['length_um']==4000
    assert_allclose(np.linalg.norm(np.diff(track['points_source_scan_zyx_um'],axis=0)),4000)
    assert_allclose(track['control_points_atlas_zyx_um'],expected_world[:2])


@pytest.mark.parametrize('model',['rigid','similarity'])
def test_recalibration_keeps_selected_fit_model_and_refits_in_correct_units(tmp_path,model):
    p,*_=annotated_project(tmp_path,model)
    new=recalibrate_project(p,(3,5,8),(3,6,5))
    assert new.fit_is_current and new.fit_result['model']==model
    singular=np.linalg.svd(new.residual[:3,:3],compute_uv=False)
    assert_allclose(singular,np.ones(3) if model=='rigid' else np.repeat(singular.mean(),3),atol=1e-10)
    assert new.track.spacing_um_zyx==p.track.spacing_um_zyx  # No implicit linked change.
    assert new.label_spacing==p.label_spacing


def test_atlas_only_and_manual_plane_keep_native_lengths_and_labels_calibration(tmp_path):
    p,s,item,_=annotated_project(tmp_path)
    save_manual_plane(item,item['plane'])
    old_length=s['measurements'][0]['length_um']
    new=recalibrate_project(p,(4,6,8),p.moving.spacing_um_zyx)
    ns=active_session(new); ni=ns['insertions'][0]
    assert_allclose(ns['measurements'][0]['length_um'],old_length)
    assert ni['plane_fit']['manual'] and not plane_is_stale(ni)
    assert new.label_spacing==(2,3,4)
    labels=load_label_atlas(new.labels_dir,new.reference,tmp_path/'cache',spacing_um_zyx=new.label_spacing)
    assert labels.spacing_um_zyx==(2,3,4)
    assert labels.ids_at([[0,0,26.1]]).tolist()==[20]
    other=load_label_atlas(new.labels_dir,new.reference,tmp_path/'cache',spacing_um_zyx=(2,3,8))
    assert other.ids_at([[0,0,26.1]]).tolist()==[10]
    assert len(list((tmp_path/'cache').glob('*.npy')))==1  # One ID lattice, different physical calibration.


def test_independent_electrodes_export_keeps_entire_native_field_of_view(tmp_path):
    p=make_track_project(tmp_path)
    shape=(15,16,17)
    image=np.full(shape,1234,np.uint16)
    path=write_volume(tmp_path/'other_track.tif',image,(3,4,5))
    p.track=ImageSpec(path,shape,(3,4,5))
    export_project(p,adapted_spacing=(2,3,4),include_ants=False)
    record=json.loads((p.output_dir/'track_adapted.geometry.json').read_text())
    assert record['shape_zyx']!=json.loads((p.output_dir/'scan_adapted.geometry.json').read_text())['shape_zyx']
    assert record['shape_zyx']==list(tifffile.imread(p.output_dir/'track_adapted.tif').shape)
    scale=np.eye(4); scale[:3,:3]=np.diag(p.track.spacing_um_zyx)
    atlas=np.eye(4); atlas[:3,:3]=np.diag(p.reference.spacing_um_zyx)
    assert_allclose(np.load(p.output_dir/'track_to_atlas_zyx_voxel.npy'),np.linalg.inv(atlas)@p.total@scale)


def test_probe_and_channel_maps_rebuilt_with_fixed_hardware_dimensions(tmp_path):
    shape=(70,60,60)
    data=np.zeros(shape,np.uint16); data[20:60,4:55,4:55]=2000
    path=write_volume(tmp_path/'volume.tif',data,(100,100,100))
    spec=ImageSpec(path,shape,(100,100,100))
    p=AlignmentProject(spec,copy.deepcopy(spec),tmp_path/'out',track=copy.deepcopy(spec))
    p.pairs=[dict(reference=v,moving=v) for v in [[2500,1000,1000],[5000,1000,1000],[2500,4000,1000],[2500,1000,4000]]]
    p.fit()
    s=insertion_session(p); item=sample_probe()
    s['insertions']=[item]
    add_recording(item,FIXTURE,'Recording A'); add_recording(item,FIXTURE,'Recording B')
    sync_tracks(s,p)
    # Only source Z doubles; probe pitch, contacts and taper remain hardware dimensions.
    new=recalibrate_project(p,(100,100,100),(200,100,100),track_spacing=(200,100,100))
    ni=active_session(new)['insertions'][0]
    assert pose_current(ni,new.total)
    assert ni['recordings']==item['recordings']
    assert ni['probe']['geometry']==item['probe']['geometry']
    assert_allclose(ni['probe']['pose']['tip0_source_zyx_um'],[8000,1000,1000],atol=1e-8)
    tips=probe_coordinates(ni['probe'],np.arange(4),0)
    assert_allclose(np.linalg.norm(np.diff(tips,axis=0),axis=1),250)
    assert_allclose(ni['probe']['pose']['surface_depths_um'],[4100]*4,atol=70)


def test_apply_refreshes_exports_in_place_and_preserves_source_tags(tmp_path):
    p,s,item,_=annotated_project(tmp_path)
    export_project(p,adapted_spacing=(3,4,5),include_ants=False)
    labels=load_label_atlas(p.labels_dir,p.reference,tmp_path/'cache')
    export_insertions(p,labels)
    files={spec.path:spec.path.read_bytes() for spec in (p.reference,p.moving,p.track)}
    updated,ref,mov=prepare_calibration_update(p,(3,4,5),(4,3,6),p.output_dir/'project.json',
                                              labels_spacing=(1,2,3),track_spacing=(5,6,7))
    loaded=AlignmentProject.load(p.output_dir)
    assert loaded.label_spacing==(1,2,3) and loaded.track.spacing_um_zyx==(5,6,7)
    assert loaded.moving.spacing_um_zyx==(4,3,6) and loaded.reference.spacing_um_zyx==(3,4,5)
    assert active_session(loaded)['id']==s['id']
    assert_allclose(np.load(p.output_dir/'moving_to_atlas_zyx_um.npy'),updated.total)
    assert inspect_tiff(p.output_dir/'scan_on_atlas_grid.tif')['spacing_um_zyx']==[3,4,5]
    assert json.loads((p.output_dir/'insertions.json').read_text())['status'] in ('complete','needs_review')
    with (p.output_dir/'measurements.csv').open(encoding='utf-8-sig') as handle:
        row=next(csv.DictReader(handle))
    assert_allclose(float(row['length_um']),active_session(updated)['measurements'][0]['length_um'])
    assert all(path.read_bytes()==content for path,content in files.items())
    assert not any(path.is_dir() for path in p.output_dir.iterdir())
    assert ref.spacing_um_zyx==(3,4,5) and mov.spacing_um_zyx==(4,3,6)
    assert not any('probemap-calibration-' in path.read_text() for path in p.output_dir.glob('*.json'))


@pytest.mark.parametrize('failure',['cancel','io'])
def test_failed_commit_restores_every_existing_output(tmp_path,monkeypatch,failure):
    p=make_track_project(tmp_path)
    export_project(p,include_ants=False)
    before={path.name:path.read_bytes() for path in p.output_dir.iterdir()}
    original=Path.replace
    count=0
    def replace(source,target):
        nonlocal count
        if Path(target).parent==p.output_dir:
            count+=1
            if count==3 and failure=='io': raise OSError('Injected disk failure')
        return original(source,target)
    monkeypatch.setattr(Path,'replace',replace)
    with pytest.raises((OSError,InterruptedError)):
        prepare_calibration_update(p,(4,6,8),(4,6,8),p.output_dir/'project.json',
                                   cancelled=lambda:failure=='cancel' and count>=2)
    assert {path.name:path.read_bytes() for path in p.output_dir.iterdir()}==before


def test_invalid_calibration_does_not_mutate_and_noop_preserves_state(tmp_path):
    p,*_=annotated_project(tmp_path)
    before=copy.deepcopy(p.as_record())
    for value in [(0,2,3),(1,np.nan,3),(-1,2,3)]:
        with pytest.raises(ValueError):
            recalibrate_project(p,p.reference.spacing_um_zyx,p.moving.spacing_um_zyx,track_spacing=value)
    assert recalibrate_project(p,p.reference.spacing_um_zyx,p.moving.spacing_um_zyx).as_record()==before
    assert p.as_record()==before


def test_labels_only_preserves_confirmed_geometry_and_native_anchors_follow_their_image(tmp_path):
    p,s,item,native=annotated_project(tmp_path)
    old=copy.deepcopy(s)
    new=recalibrate_project(p,p.reference.spacing_um_zyx,p.moving.spacing_um_zyx,labels_spacing=(5,6,7))
    assert active_session(new)==old
    assert new.label_spacing==(5,6,7)
    for anchor in ('moving','reference'):
        add_insertion_point(p,item,transform_points(native[:1],p.total)[0],anchor=anchor)
    changed=recalibrate_project(p,(3,4,5),(4,3,6),track_spacing=(5,6,7))
    points=active_session(changed)['insertions'][0]['points']
    assert_allclose(transform_points([points[-2]['atlas_zyx_um']],np.linalg.inv(changed.total))/changed.moving.spacing_um_zyx,
                    native[:1]/p.moving.spacing_um_zyx)
    assert_allclose(np.asarray(points[-1]['atlas_zyx_um'])/changed.reference.spacing_um_zyx,
                    np.asarray(item['points'][-1]['atlas_zyx_um'])/p.reference.spacing_um_zyx)


def test_pending_landmark_edits_remain_pending_after_recalibration(tmp_path):
    p,*_=annotated_project(tmp_path)
    p.pairs=[]
    new=recalibrate_project(p,(4,6,8),(4,6,8))
    assert not new.fit_is_current and new.pairs==[]
    assert new.fit_result['model']=='affine'
    assert len(new.fitted_pairs)==5
    assert_allclose(new.total,np.diag([2,2,2,1])@p.total@np.diag([.5,.5,.5,1]))


def test_pending_placement_keeps_exports_on_the_accepted_registration(tmp_path):
    p,s,item,_=annotated_project(tmp_path)
    export_project(p,include_ants=False)
    export_insertions(p)
    accepted=p.total.copy()
    p.coarse[:3,3]+=[12,-4,8]
    p.pairs=[]
    p.save(p.output_dir/'project.json')
    updated,*_=prepare_calibration_update(p,(4,6,8),(4,6,8),p.output_dir/'project.json')
    saved=AlignmentProject.load(p.output_dir)
    assert not saved.fit_is_current and saved.pairs==[]
    assert len(saved.track_analysis['sessions'])==1
    calibrated_accepted=np.diag([2,2,2,1])@accepted@np.diag([.5,.5,.5,1])
    assert_allclose(np.load(p.output_dir/'moving_to_atlas_zyx_um.npy'),calibrated_accepted)
    assert_allclose(json.loads((p.output_dir/'insertions.json').read_text())['source_track_to_atlas_zyx_um'],calibrated_accepted)
    assert not np.allclose(saved.total,calibrated_accepted)


def test_calibrated_ants_transforms_roundtrip_native_voxels_after_staging(tmp_path):
    from probemap.registration.ants_workflow import transfer_points
    p,*_=annotated_project(tmp_path)
    export_project(p,include_ants=True)
    new,*_=prepare_calibration_update(p,(3,4,5),(4,3,6),p.output_dir/'project.json',track_spacing=(5,6,7))
    manifest=json.loads((p.output_dir/'export.json').read_text())
    checkpoint=Path(manifest['ants_checkpoint'])
    assert checkpoint.is_file() and checkpoint.parent==p.output_dir
    voxels=np.array([[1,2,3],[4,6,8],[7,9,10.]])
    expected=transform_points(voxels*new.moving.spacing_um_zyx,new.total)/new.reference.spacing_um_zyx
    assert_allclose(transfer_points(voxels,checkpoint),expected,atol=1e-4)
    assert_allclose(transfer_points(expected,checkpoint,direction='reference_to_moving'),voxels,atol=1e-4)


def test_labels_only_update_reuses_scan_exports(tmp_path,monkeypatch):
    p=make_track_project(tmp_path)
    export_project(p,include_ants=False)
    before=(p.output_dir/'scan_on_atlas_grid.tif').read_bytes()
    def unexpected(*args,**kwargs): raise AssertionError('Labels-only calibration must not resample scan TIFFs.')
    monkeypatch.setattr('probemap.registration.desktop_model.export_project',unexpected)
    updated,*_=prepare_calibration_update(p,p.reference.spacing_um_zyx,p.moving.spacing_um_zyx,
                                          p.output_dir/'project.json',labels_spacing=(1,2,3))
    assert updated.label_spacing==(1,2,3)
    assert (p.output_dir/'scan_on_atlas_grid.tif').read_bytes()==before


def test_named_exports_recalibrate_in_place_and_remain_linked_after_relocation(tmp_path):
    from probemap.registration.exports import export_selection
    p=make_track_project(tmp_path)
    target=tmp_path/'chosen'/'brain.ome.tif'
    options=dict(path=str(target),kind='brain',stage='registration',format='ome',extent='full',
                 spacing_um_zyx=list(p.moving.spacing_um_zyx),spacing_policy='source')
    export_selection(p,options)
    p.exports=[options]; p.save(p.output_dir/'project.json')
    original=target.read_bytes()
    new,*_=prepare_calibration_update(p,(4,6,8),(4,6,8),p.output_dir/'project.json')
    assert new.output_dir==p.output_dir
    assert target.read_bytes()!=original
    assert_allclose(inspect_tiff(target)['spacing_um_zyx'],[4,6,8])
    loaded=AlignmentProject.load(p.output_dir)
    assert loaded.exports[0]['path']==str(target)
    assert loaded.exports[0]['spacing_um_zyx']==[4,6,8]
    import shutil
    moved=tmp_path/'moved'
    shutil.copytree(tmp_path/'chosen',moved/'chosen')
    shutil.copytree(p.output_dir,moved/p.output_dir.name)
    relocated=AlignmentProject.load(moved/p.output_dir.name)
    assert relocated.exports[0]['path']==str(moved/'chosen'/'brain.ome.tif')


def test_named_export_commit_failure_rolls_back_external_files_and_project(tmp_path,monkeypatch):
    from probemap.registration.exports import export_selection
    p=make_track_project(tmp_path)
    options=dict(path=str(tmp_path/'chosen.tif'),kind='brain',stage='registration',format='tiff',extent='full',spacing_um_zyx=[2,3,4])
    export_selection(p,options); p.exports=[options]; p.save(p.output_dir/'project.json')
    paths=[Path(options['path']),Path(options['path']).with_suffix('.geometry.json'),p.output_dir/'project.json']
    before=[path.read_bytes() for path in paths]
    replace=Path.replace
    def failing(source,target):
        if Path(target)==p.output_dir/'project.json': raise OSError('Injected commit failure')
        return replace(source,target)
    monkeypatch.setattr(Path,'replace',failing)
    with pytest.raises(OSError,match='commit failure'):
        prepare_calibration_update(p,(4,6,8),(4,6,8),p.output_dir/'project.json')
    assert [path.read_bytes() for path in paths]==before

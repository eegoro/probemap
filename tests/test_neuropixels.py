"""Acquisition metric, IMEC wiring, rigid geometry, signed entry depth and exports."""
import copy
import csv
import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from probemap.registration.insertion_model import add_measurement, sync_measurements, directed_track
from probemap.registration.landmarks import transform_points
from probemap.registration.neuropixels_model import (add_recording, channel_rows, find_surface_depths,
    fit_probe, geometry_for, parse_channelmap, pose_current, probe_coordinates, probe_plane, probe_settings,
    shank_mesh, write_reports, suggest_shank_assignments)
from probemap.registration.track_model import LabelAtlas, plane_geometry, plane_uv

FIXTURE=Path(__file__).parent/'fixtures/np2013_example.imro'


def sample_probe():
    item=dict(id='a',name='Anterior',color='#ff9a3c',tracks=[],points=[])
    s=probe_settings(item)
    for shank in (0,2,3):
        item['tracks'].append(dict(id=str(shank),name=f'Track {shank}',
            control_points_atlas_zyx_um=[[4000,1000,1000+250*shank],[1000,1000,1000+250*shank]]))
        s['assignments'][str(shank)]=shank
    fit_probe(item,np.eye(4))
    return item


def test_probe_tip_is_first_click_and_shanks_follow_track_toward_surface():
    item=sample_probe(); settings=item['probe']
    matrix=np.array([[.8,.1,0,200],[0,1.2,.05,-70],[.05,0,.9,300],[0,0,0,1.]])
    for track in item['tracks']:
        track['control_points_atlas_zyx_um']=transform_points(track['control_points_atlas_zyx_um'],matrix).tolist()
    fit_probe(item,matrix)
    for track in item['tracks']:
        shank=settings['assignments'][track['id']]
        marked=transform_points(track['control_points_atlas_zyx_um'],np.linalg.inv(matrix))
        tip=probe_coordinates(settings,shank,0)
        base=probe_coordinates(settings,shank,settings['geometry']['shank_length_um'])
        assert_allclose(tip,marked[0],atol=1e-9)
        direction=marked[1]-marked[0]
        assert_allclose((base-tip)/np.linalg.norm(base-tip),direction/np.linalg.norm(direction),atol=1e-10)
        assert_allclose(probe_coordinates(settings,shank,3000),marked[1],atol=1e-9)


def test_old_reversed_placement_is_preserved_and_refitted_from_deep_end():
    item=sample_probe(); settings=item['probe']
    settings.pop('direction_convention'); settings['pose'].pop('direction_convention')
    settings['tip_mode']='track_end'; settings['confirmed']=True
    settings['pose']['down_source_zyx']=[-1.,0.,0.]
    old=copy.deepcopy(settings['pose'])
    settings=probe_settings(item)
    assert settings['tip_mode']=='track_start' and not settings['confirmed']
    assert settings['previous_placement']['pose']==old
    assert not pose_current(item,np.eye(4))
    fit_probe(item,np.eye(4))
    assert_allclose(settings['pose']['down_source_zyx'],[1.,0.,0.])
    assert_allclose(settings['pose']['tip0_source_zyx_um'],[4000,1000,1000])
    assert settings['previous_placement']['pose']==old


def test_specimen_ruler_inverts_rotation_scale_shear_and_translation():
    matrix=np.array([[.8,.2,0,120],[0,1.4,.1,-45],[.2,0,1.1,80],[0,0,0,1.]])
    native=np.array([[100,200,300],[400,600,1500.]])
    atlas=transform_points(native,matrix)
    session={'sources':{'matrix':matrix.tolist()}}
    r=add_measurement(session,*atlas)
    assert_allclose(r['length_um'],1300)
    assert not np.isclose(r['length_um'],r['atlas_length_um'])
    assert_allclose(r['points_source_scan_zyx_um'],native)
    legacy={'sources':session['sources'],'measurements':[dict(points_atlas_zyx_um=atlas.tolist(),length_um=999)]}
    sync_measurements(legacy); sync_measurements(legacy)
    assert legacy['measurements'][0]['legacy_atlas_length_um']==999
    assert_allclose(legacy['measurements'][0]['length_um'],1300)
    points=directed_track(atlas,matrix=matrix)
    source=transform_points(points,np.linalg.inv(matrix))
    assert_allclose(np.linalg.norm(source[1]-source[0]),4000)


def test_provisional_order_and_spacing_diagnosis_do_not_rescale_tissue():
    matrix=np.array([[.8,.1,0,400],[0,1.2,.05,120],[.05,0,.9,-200],[0,0,0,1.]])
    item=dict(id='four',name='Four traces',tracks=[],points=[])
    for index,shank in enumerate((0,3,2,1)):
        source=[[2000,1000,1000+shank*250],[3000+index*100,1000,1000+shank*250]]
        item['tracks'].append(dict(id=str(index),name=f'Trace {index}',
            control_points_atlas_zyx_um=transform_points(source,matrix).tolist()))
    settings=probe_settings(item)
    assert suggest_shank_assignments(item,matrix)=={'0':0,'3':1,'2':2,'1':3}
    fit=fit_probe(item,matrix)
    assert_allclose(fit['spacing_check']['observed_pitch_um'],250,atol=1e-9)
    assert not settings['confirmed']
    # A smaller trace separation produces a diagnostic, not a changed voxel
    # size or a smaller piece of hardware.
    for track in item['tracks']:
        raw=transform_points(track['control_points_atlas_zyx_um'],np.linalg.inv(matrix))
        raw[:,2]=1000+(raw[:,2]-1000)*.55
        track['control_points_atlas_zyx_um']=transform_points(raw,matrix).tolist()
    suggest_shank_assignments(item,matrix)
    fit=fit_probe(item,matrix)
    assert_allclose(fit['spacing_check']['observed_pitch_um'],137.5)
    assert_allclose(fit['spacing_check']['observed_outer_centres_um'],412.5)
    assert settings['geometry']['shank_pitch_um']==250
    with pytest.raises(ValueError,match='exactly four'):
        suggest_shank_assignments(sample_probe(),np.eye(4))


def test_real_imro_wiring_and_recording_embedding(tmp_path):
    parsed=parse_channelmap(FIXTURE.read_text())
    assert len(parsed['channels'])==384
    assert parsed['channels'][0]['shank']==2 and parsed['channels'][0]['electrode']==96
    assert parsed['channels'][48]['shank']==3 and parsed['channels'][48]['electrode']==96
    assert sorted(set(c['shank'] for c in parsed['channels']))==[0,1,2,3]
    item=sample_probe()
    a=add_recording(item,FIXTURE,'Baseline'); b=add_recording(item,FIXTURE,'Recovery')
    assert a['id']!=b['id'] and a['opacity']>b['opacity']
    assert a['source_text']==FIXTURE.read_text()
    # Real pasted TXT uses a backslash on a separate line before the next tag.
    assert len(parse_channelmap(FIXTURE.read_text()+'\\\n~muxTbl=(24,16)(0 1)')['channels'])==384
    meta=FIXTURE.read_text()+'\nsnsSaveChanSubset=0:10,20\nimStdby=2:3\n'
    assert sum(c['recorded'] for c in parse_channelmap(meta)['channels'])==10


@pytest.mark.parametrize('change',[lambda s:s.replace('(2013,384)','(0,384)'),
    lambda s:s.replace('(0 2 0 0 96)','(0 2 0 0 97)'),
    lambda s:s.replace('(0 2 0 0 96)','(1 2 0 0 96)'),
    lambda s:s[:s.rfind('(')]])
def test_bad_or_unsupported_maps_are_rejected(change):
    with pytest.raises(ValueError): parse_channelmap(change(FIXTURE.read_text()))


def test_fixed_pitch_tip_offsets_surface_depth_and_atlas_registration():
    item=sample_probe(); s=item['probe']; p=s['pose']
    assert_allclose(p['fit_rms_um'],0,atol=1e-10)
    tips=probe_coordinates(s,np.arange(4),np.zeros(4))
    assert_allclose(np.linalg.norm(np.diff(tips,axis=0),axis=1),[250,250,250])
    assert_allclose(probe_coordinates(s,0,217,0),[3783,1000,992])
    assert_allclose(np.linalg.norm(probe_coordinates(s,0,217,0,face=True)-probe_coordinates(s,0,217,0)),12)
    mask=np.zeros((70,60,60),bool); mask[20:60,5:50,5:50]=True
    depth=find_surface_depths(item,mask,(100,100,100))
    assert_allclose(depth,[2050]*4,atol=15)
    add_recording(item,FIXTURE,'Run A'); add_recording(item,FIXTURE,'Run B')
    item['recordings'][1]['depth_offset_um']=100
    labels=LabelAtlas(np.ones((100,100,100),np.uint16),(100,100,100),[dict(id=1,name='Test region')],'fixture')
    rows=channel_rows(item,np.eye(4),labels)
    assert len(rows)==768 and rows[0]['region']=='Test region'
    assert rows[0]['depth_from_surface_um']<0
    assert_allclose(rows[384]['depth_from_surface_um'],rows[0]['depth_from_surface_um']-100)
    native=copy.deepcopy(rows[0])
    matrix=np.array([[.8,.1,0,200],[0,1.1,0,0],[0,0,.9,-100],[0,0,0,1]])
    # Reuse source-space hardware: registration must not change the native depth.
    transformed=channel_rows(item,matrix,labels)[0]
    assert transformed['depth_from_surface_um']==native['depth_from_surface_um']
    assert transformed['pose_status'].startswith('stale')
    source=np.array([native[f'source_{a}_um'] for a in 'zyx'])
    assert_allclose([transformed[f'atlas_{a}_um'] for a in 'zyx'],transform_points([source],matrix)[0])
    geometry=plane_geometry(probe_plane(item,matrix))
    centerline=transform_points(tips,matrix)
    assert_allclose(plane_uv(geometry,centerline)[1],0,atol=1e-8)
    vertices,faces=shank_mesh(item,matrix)
    assert vertices.shape==(40,3) and faces.max()<40
    s['surface_depths_um']['0']=2400
    find_surface_depths(item,mask,(100,100,100))
    assert s['pose']['surface_depths_um'][0]==2400
    empty=np.zeros_like(mask)
    find_surface_depths(item,empty,(100,100,100))
    assert s['pose']['surface_depths_um'][1] is None
    assert geometry_for('Legacy NP2')['taper_length_um']==175
    assert geometry_for('NP2013')['first_site_center_from_tip_um']==217


def test_report_roundtrip_missing_surface_and_hidden_regions(tmp_path):
    from test_track_model import make_track_project
    project=make_track_project(tmp_path)
    item=sample_probe(); add_recording(item,FIXTURE,'Rest'); add_recording(item,FIXTURE,'Task')
    item['probe']['surface_depths_um']={str(i):3000 for i in range(4)}
    find_surface_depths(item,np.zeros((70,60,60)),(100,100,100))
    session=dict(id='session',sources=dict(matrix=np.eye(4).tolist(),reference=project.reference.record()),
                 insertions=[item],regions={'1':{'visible':False,'name':'Named region'}},measurements=[])
    project.track_analysis={'sessions':[session]}
    labels=LabelAtlas(np.ones((100,100,100),np.uint16),(100,100,100),[dict(id=1,name='Region')],'fixture')
    report=write_reports(project,tmp_path/'reports',labels)
    assert report.exists() and 'Rest' in report.read_text() and 'Named region' in report.read_text()
    with report.with_name('neuropixels_channels.csv').open(encoding='utf-8-sig') as handle: rows=list(csv.DictReader(handle))
    assert len(rows)==768 and rows[0]['channel']=='0' and rows[0]['depth_from_surface_um'].startswith('-')
    assert rows[0]['region']=='Named region'
    path=project.save(tmp_path/'saved.json')
    saved=json.loads(path.read_text())['track_analysis']['sessions'][0]['insertions'][0]
    assert saved['recordings']==item['recordings'] and saved['probe']==item['probe']
    item['probe']['tip_shift_um']=100
    assert not pose_current(item,np.eye(4))
    with pytest.raises(ValueError):
        item['probe']['geometry']['shank_length_um']=5000; fit_probe(item,np.eye(4))

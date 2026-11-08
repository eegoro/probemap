import copy
import csv
import json

import numpy as np
import pytest
import tifffile
from numpy.testing import assert_allclose

from probemap.registration.insertion_model import (TRACK_LENGTH_UM, add_insertion_point, brain_support,
    clip_segment_to_slab, directed_track, export_insertions, fit_insertion_plane, inside_brain,
    insertion_session, move_points, new_insertion, plane_is_stale, save_manual_plane, sync_tracks)
from probemap.registration.track_model import active_session, load_label_atlas, plane_geometry, plane_uv
from probemap.registration.volumes import Volume
from test_track_model import make_track_project


def test_native_plane_fit_survives_anisotropic_registration(tmp_path):
    from probemap.registration.landmarks import transform_points
    project=make_track_project(tmp_path)
    session=insertion_session(project)
    native=new_insertion(session)
    transformed=new_insertion(session)
    points=np.array([[0,0,0],[10,100,0],[-5,0,180],[15,120,190],[0,250,80]],float)
    for point in points: add_insertion_point(project,native,point)
    expected=plane_geometry(fit_insertion_plane(native,session['plane']))
    matrix=np.array([[2.,.8,.1,120],[.2,.5,0,-70],[0,.3,1.4,50],[0,0,0,1]])
    for point in transform_points(points,matrix): add_insertion_point(project,transformed,point)
    fitted=plane_geometry(fit_insertion_plane(transformed,session['plane'],matrix))
    expected_normal=np.linalg.solve(matrix[:3,:3].T,expected['normal_zyx'])
    expected_normal/=np.linalg.norm(expected_normal)
    assert_allclose(abs(np.dot(expected_normal,fitted['normal_zyx'])),1,atol=1e-12)
    assert_allclose(fitted['center_um_zyx'],transform_points([expected['center_um_zyx']],matrix)[0])
    assert_allclose(transformed['plane_fit']['rms_um'],native['plane_fit']['rms_um'])
    assert transformed['plane_fit']['measurement_frame']=='source scan physical micrometres'


def test_insertion_default_names_are_letters_without_changing_existing_names():
    session={}
    names=[new_insertion(session)['name'] for _ in range(28)]
    assert names[:3]==['A','B','C'] and names[25:]==['Z','AA','AB']
    existing=new_insertion(session,'Custom name')
    assert existing['name']=='Custom name'


def test_physical_plane_fit_rank_cache_and_manual_save(tmp_path):
    project=make_track_project(tmp_path)
    session=insertion_session(project)
    item=new_insertion(session)
    # All points on 2*z + 3*y - x = 8, in physical units.
    xy=np.array([[0,0],[0,200],[300,0],[300,200],[175,120]],float)
    points=np.column_stack(((8-3*xy[:,0]+xy[:,1])/2,xy))
    for p in points: add_insertion_point(project,item,p)
    plane=fit_insertion_plane(item,session['plane'])
    assert not plane_is_stale(item)
    _,distance=plane_uv(plane_geometry(plane),points)
    assert_allclose(distance,0,atol=1e-10)
    assert item['plane_fit']['rms_um']<1e-10
    saved=copy.deepcopy(item['plane'])
    add_insertion_point(project,item,[40,60,80])
    assert plane_is_stale(item)
    assert item['plane']==saved  # No implicit re-fit after a point edit.
    manual=copy.deepcopy(saved)
    manual['rotation_xyz_deg'][1]+=5
    save_manual_plane(item,manual)
    assert item['plane_fit']['manual'] and not plane_is_stale(item)
    item['points']=item['points'][:2]
    with pytest.raises(ValueError,match='at least 3'):
        fit_insertion_plane(item,session['plane'])
    item['points']=[]
    for p in [[0,0,0],[10,20,30],[20,40,60]]: add_insertion_point(project,item,p)
    with pytest.raises(ValueError,match='one line'):
        fit_insertion_plane(item,session['plane'])


def test_group_move_single_color_4mm_geometry_and_legacy_migration(tmp_path):
    project=make_track_project(tmp_path)
    session=insertion_session(project)
    a,b=new_insertion(session),new_insertion(session)
    assert a['color']!=b['color']
    p=add_insertion_point(project,a,[4,9,16])
    assert move_points(session,{p['id']},b['id'])==1
    assert not a['points'] and b['points']==[p]
    b['tracks']=[{'id':'t','name':'Channel 1','control_points_atlas_zyx_um':[[4,9,16],[9,12,25]],
                  'start_validation':{'inside':True}}]
    sync_tracks(session,project)
    track=session['tracks'][0]
    assert track['color']==b['color'] and track['insertion_id']==b['id']
    assert_allclose(np.linalg.norm(np.diff(track['points_atlas_zyx_um'],axis=0)),TRACK_LENGTH_UM)
    b['color']='#1188ff'
    sync_tracks(session,project)
    assert track['color']=='#1188ff'
    with pytest.raises(ValueError): directed_track([[1,2,3],[1,2,3]])
    # Migration is reversible and idempotent; the old sampled line is retained.
    session.pop('insertion_schema')
    legacy=copy.deepcopy(session['tracks'])
    converted=insertion_session(project)
    assert converted['legacy_tracks_before_insertions']==legacy
    assert converted['insertions'][0]['tracks'][0]['start_validation']['inside'] is False
    assert insertion_session(project) is converted
    assert len(converted['insertions'])==1


def test_brain_start_mask_and_slice_slab_intersections():
    z,y,x=np.indices((35,40,45))
    data=np.exp(-((z-17)**2+(y-20)**2+(x-22)**2)/80).astype(np.float32)
    volume=Volume(data,(20,25,30))
    mask,threshold=brain_support(volume)
    matrix=np.eye(4)
    matrix[:3,3]=[50,-100,200]
    world=np.array([[17*20,20*25,22*30],[0,0,0]])+matrix[:3,3]
    assert inside_brain(world,mask,volume,matrix).tolist()==[True,False]
    assert 0<threshold<1
    plane=plane_geometry({'center_um_zyx':[0,0,0],'rotation_xyz_deg':[0,0,0],
                          'width_um':100,'height_um':100,'max_dimension':64})
    clipped=clip_segment_to_slab([-20,0,0],[20,0,100],plane,10)
    assert_allclose(clipped,[[-5,0,37.5],[5,0,62.5]])
    assert clip_segment_to_slab([20,0,0],[20,0,100],plane,10) is None


def test_insertion_export_calibrated_matched_channels_fixed_tracks_and_regions(tmp_path):
    project=make_track_project(tmp_path)
    session=insertion_session(project)
    item=new_insertion(session,'Insertion A')
    item['plane']={'center_um_zyx':[4,15,26],'rotation_xyz_deg':[0,0,0],
                   'width_um':52,'height_um':30,'max_dimension':64}
    for p in [[4,9,0],[4,9,52],[4,21,20]]: add_insertion_point(project,item,p)
    item['tracks']=[{'id':'t','name':'Track 1','control_points_atlas_zyx_um':[[4,9,0],[4,9,52]],
                     'start_validation':{'inside':True,'method':'Test mask'}}]
    labels=load_label_atlas(project.labels_dir,project.reference,tmp_path/'cache')
    path=export_insertions(project,labels)
    manifest=json.loads(path.read_text())
    assert manifest['status']=='complete'
    assert_allclose(sum(row['length_um'] for row in manifest['report']['segments']),4000)
    with (path.parent/'tracks_4mm.csv').open() as handle:
        row=next(csv.DictReader(handle))
    assert row['insertion_name']=='Insertion A' and float(row['length_um'])==4000
    for name in ('brain_plane.tif','track_plane.tif','atlas_plane.tif','regions_plane.tif'):
        with tifffile.TiffFile(path.parent/f"insertion_{item['id']}_{name}") as tif:
            a,b=tif.pages[0].tags['XResolution'].value
            assert_allclose(10000*b/a,plane_geometry(item['plane'])['pixel_size_um'],rtol=1e-7)
    assert tifffile.imread(path.parent/f"insertion_{item['id']}_brain_plane.tif").max()>1
    assert export_insertions(project,labels)==path
    assert path.parent==project.output_dir
    assert not any(child.is_dir() for child in path.parent.iterdir())
    item['tracks'][0]['start_validation']['inside']=False
    with pytest.raises(ValueError,match='Validate track starts'):
        export_insertions(project,labels)


def test_xyz_track_points_are_finite_intersections_not_projections():
    from probemap.registration.insertion_model import track_slice_points
    plane={'center_um_zyx':[2000,500,700],'rotation_xyz_deg':[0,0,0],
           'width_um':630,'height_um':630,'max_dimension':64}
    track={'id':'t','name':'T','color':'#11aaff','points_atlas_zyx_um':[[0,0,0],[4000,1000,1400]]}
    hits=track_slice_points(track,plane_geometry(plane))
    assert len(hits)==1
    assert_allclose(hits[0]['atlas_zyx_um'],[2000,500,700])
    plane['center_um_zyx'][0]=5000
    assert not track_slice_points(track,plane_geometry(plane))
    plane['center_um_zyx'][0]=2000
    coplanar=dict(track,points_atlas_zyx_um=[[2000,0,0],[2000,1000,1400]])
    hits=track_slice_points(coplanar,plane_geometry(plane))
    assert len(hits)>2
    assert all(hit['atlas_zyx_um'][0]==2000 for hit in hits)

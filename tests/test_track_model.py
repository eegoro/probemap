import csv
import json

import numpy as np
import pytest
import tifffile
from numpy.testing import assert_allclose

from probemap.registration.desktop_model import AlignmentProject, ImageSpec, export_project
from probemap.registration.landmarks import transform_points
from probemap.registration.track_model import (active_session, export_track_analysis, load_label_atlas,
                                               load_track_data, plane_geometry, plane_uv, plane_world,
                                               plane_covering_volumes, sample_world, track_report)
from probemap.registration.volumes import Volume, write_volume


def make_track_project(tmp_path):
    shape=(10,12,14)
    z,y,x=np.indices(shape)
    data=(100*z+10*y+x).astype(np.uint16)
    brain=write_volume(tmp_path/'brain.tif',data,(2,3,4))
    track=write_volume(tmp_path/'track.tif',data*3,(2,3,4))
    atlas=write_volume(tmp_path/'atlas.tif',data,(2,3,4))
    project=AlignmentProject(ImageSpec(atlas,shape,(2,3,4)),ImageSpec(brain,shape,(2,3,4)),tmp_path/'out',
                             track=ImageSpec(track,shape,(2,3,4)))
    project.coarse[:3,3]=[-2,3,4]
    points=np.array([[2,6,8],[12,6,8],[2,24,8],[2,6,40],[12,24,40.]])
    project.pairs=[{'reference':(point+[2,-3,-4]).tolist(),'moving':point.tolist()} for point in points]
    project.fit()
    directory=tmp_path/'masks'
    directory.mkdir()
    a=np.zeros(shape,np.uint8)
    b=a.copy()
    a[:,:,:7]=1
    b[:,:,7:]=1
    write_volume(directory/'10_left_mask.tif',a,(2,3,4))
    write_volume(directory/'20_right_mask.tif',b,(2,3,4))
    project.labels_dir=directory
    return project


def test_oblique_plane_geometry_round_trip_and_native_sampling():
    plane={'center_um_zyx':[-4,12,30],'rotation_xyz_deg':[23,-37,56],
           'width_um':90,'height_um':60,'max_dimension':128}
    geometry=plane_geometry(plane)
    uv=np.array([[0,0],[10.4,25.8],[127,80.]])
    world=plane_world(geometry,uv)
    actual,distances=plane_uv(geometry,world)
    assert_allclose(actual,uv,atol=1e-12)
    assert_allclose(distances,0,atol=1e-12)
    assert_allclose(np.linalg.norm(np.array(geometry['u_zyx'])),1)
    # Analytic linear intensity field proves interpolation acts in 3D physical
    # coordinates, including inverse placement, unequal spacing and obliquity.
    z,y,x=np.indices((35,40,45))
    volume=Volume((2*z+3*y+4*x).astype(np.float32),(2,3,4))
    matrix=np.eye(4)
    matrix[:3,3]=[-5,4,-9]
    raw=np.array([[8.3,12.4,17.1],[13.8,22.3,20.7]])
    target=transform_points(raw*[2,3,4],matrix)
    assert_allclose(sample_world(volume,matrix,target),raw@[2,3,4],atol=1e-4)


def test_complete_oblique_slice_covers_both_images_after_scaling_and_shear():
    from probemap.registration.coarse import transformed_corners
    shape=(45,65,85); spacing=(90,70,55)
    native=Volume(np.ones(shape,np.float32),spacing)
    matrix=np.array([[.85,.12,0,-1700],[0,1.2,.08,1100],[.06,0,.7,900],[0,0,0,1.]])
    plane=dict(center_um_zyx=[1800,2300,2500],rotation_xyz_deg=[31,-19,24],
               width_um=2200,height_um=6000,max_dimension=512)
    boxes=[transformed_corners(shape,spacing,m) for m in (np.eye(4),matrix)]
    full=plane_covering_volumes(plane,boxes)
    before,after=plane_geometry(plane),plane_geometry(full)
    assert_allclose(after['normal_zyx'],before['normal_zyx'],atol=1e-12)
    assert_allclose(np.dot(np.subtract(after['center_um_zyx'],before['center_um_zyx']),before['normal_zyx']),0,atol=1e-10)
    # Independent, generously sized reslice finds all visible pixels in either
    # image. Every one must lie inside the new shared field of view.
    world=plane_world(plane_geometry(dict(plane,width_um=25000,height_um=25000,max_dimension=768)))
    visible=np.zeros(len(world),bool)
    for m in (np.eye(4),matrix): visible|=sample_world(native,m,world)>0
    assert visible.any()
    uv,_=plane_uv(after,world[visible]); h,w=after['shape_hw']
    assert (uv>=0).all() and (uv[:,0]<w).all() and (uv[:,1]<h).all()
    # Changing the normal offset keeps the requested plane offset intact.
    far=dict(plane,center_um_zyx=(np.array(plane['center_um_zyx'])+30000*np.array(before['normal_zyx'])).tolist())
    outside=plane_geometry(plane_covering_volumes(far,boxes))
    assert_allclose(np.dot(outside['center_um_zyx'],before['normal_zyx']),np.dot(far['center_um_zyx'],before['normal_zyx']),atol=1e-8)


def test_track_transform_export_is_identical_to_brain_grid_and_retains_intensities(tmp_path):
    project=make_track_project(tmp_path)
    path=export_project(project,adapted_spacing=(3,4,5),include_ants=False)
    record=json.loads(path.read_text())
    assert len(record['outputs'])==4
    for brain_name,track_name in (('scan_on_atlas_grid.tif','track_on_atlas_grid.tif'),
                                  ('scan_adapted.tif','track_adapted.tif')):
        brain=tifffile.imread(path.parent/brain_name)
        track=tifffile.imread(path.parent/track_name)
        assert track.dtype==np.uint16 and track.shape==brain.shape
        assert_allclose(track,brain*3,atol=2)
        geometry=json.loads((path.parent/track_name).with_suffix('.geometry.json').read_text())
        assert geometry['source_role']=='track'
        assert_allclose(geometry['source_to_atlas_zyx_um'],project.total)
    assert_allclose(np.load(path.parent/'track_to_atlas_zyx_um.npy'),project.total)
    loaded=AlignmentProject.load(path.parent/'project.json')
    assert loaded.track.record()==project.track.record()
    loaded.track.spacing_um_zyx=(4,3,2)
    export_project(loaded,include_ants=False)
    # Independent channel sampling uses its own voxel map in the common
    # acquisition frame. A different physical origin needs explicit alignment.
    loaded.track.origin_um_zyx=(1,0,0)
    with pytest.raises(ValueError,match='share the same'):
        export_project(loaded,include_ants=False)


def test_native_label_cache_nearest_regions_hidden_styles_and_report_lengths(tmp_path):
    project=make_track_project(tmp_path)
    labels=load_label_atlas(project.labels_dir,project.reference,tmp_path/'cache')
    assert isinstance(labels.data,np.memmap)
    assert_allclose(labels.ids_at([[0,0,0],[0,0,25.9],[0,0,26.1],[-2,0,0]]),[10,10,20,-1])
    session=active_session(project)
    session['regions']={'10':{'visible':False,'name':'Left region'}}
    session['tracks']=[{'id':'electrode','name':'Electrode 1','color':'#ffc966',
                         'points_atlas_zyx_um':[[4,9,0],[4,9,20],[4,9,52]]}]
    report=track_report(session,labels,step_um=1)
    assert [item['region_id'] for item in report['segments']]==[10,20]
    assert report['segments'][0]['region_name']=='Left region'
    assert_allclose([item['length_um'] for item in report['segments']],[26,26])
    assert_allclose(sum(item['length_um'] for item in report['segments']),52)
    assert np.all(np.diff([item['distance_um'] for item in report['samples']])>0)
    cached=load_label_atlas(project.labels_dir,project.reference,tmp_path/'cache')
    assert_allclose(cached.data,labels.data)
    # Display visibility has no effect on the quantitative region assignments.
    assert report['samples'][0]['region_id']==10


def test_track_sessions_preserve_previous_affines_and_analysis_export_round_trip(tmp_path):
    project=make_track_project(tmp_path)
    labels=load_label_atlas(project.labels_dir,project.reference,tmp_path/'cache')
    session=active_session(project)
    session['plane']={'center_um_zyx':[4,15,26],'rotation_xyz_deg':[0,0,0],
                      'width_um':52,'height_um':30,'max_dimension':64}
    session['tracks']=[{'id':'a','name':'Track A','color':'#ffff00',
                        'plane':session['plane'].copy(),'points_atlas_zyx_um':[[4,9,0],[4,9,52]]}]
    assert active_session(project) is session
    path=export_track_analysis(project,labels)
    record=json.loads(path.read_text())
    assert record['status']=='complete'
    with (path.parent/'track_samples.csv').open() as handle:
        rows=list(csv.DictReader(handle))
    for item in (rows[0],rows[-1]):
        raw=np.array([float(item[f'source_track_{axis}_voxel']) for axis in 'zyx'])
        atlas=np.array([float(item[f'atlas_{axis}_um']) for axis in 'zyx'])
        assert_allclose(transform_points((raw*project.track.spacing_um_zyx)[None],project.total)[0],atlas,atol=1e-9)
    native=tifffile.imread(path.parent/'track_plane.tif')
    assert native.dtype==np.float32 and native.max()>1
    for name in ('track_plane.tif','atlas_plane.tif','atlas_regions_plane.tif'):
        with tifffile.TiffFile(path.parent/name) as image:
            numerator,denominator=image.pages[0].tags['XResolution'].value
            assert_allclose(10000*denominator/numerator,record['plane_geometry']['pixel_size_um'],rtol=1e-7)
    saved=AlignmentProject.load(path.parent/'project.json')
    assert saved.track_analysis==project.track_analysis
    assert export_track_analysis(project,labels)==path
    assert path.parent==project.output_dir
    assert not any(child.is_dir() for child in path.parent.iterdir())
    project.pairs[0]['reference'][2]+=1
    project.fit()
    newer=active_session(project)
    assert newer['id']!=session['id'] and not newer['tracks']
    assert len(project.track_analysis['sessions'])==2
    assert project.track_analysis['sessions'][0]['tracks'][0]['name']=='Track A'


def test_mask_geometry_and_overlapping_masks_are_rejected(tmp_path):
    project=make_track_project(tmp_path)
    write_volume(project.labels_dir/'30_overlap_mask.tif',np.ones(project.reference.shape_zyx,np.uint8),(2,3,4))
    with pytest.raises(ValueError,match='overlap'):
        load_label_atlas(project.labels_dir,project.reference,tmp_path/'cache')
    assert not list((tmp_path/'cache').glob('*.partial'))
    wrong=tmp_path/'wrong'
    wrong.mkdir()
    write_volume(wrong/'1_bad.tif',np.ones((5,6,7),np.uint8),(2,3,4))
    with pytest.raises(ValueError,match='shape'):
        load_label_atlas(wrong,project.reference,tmp_path/'cache')


def test_track_preview_preserves_a_thin_off_lattice_peak(tmp_path):
    data=np.zeros((20,22,24),np.uint16)
    data[9,11,13]=50000
    path=write_volume(tmp_path/'sparse.tif',data,(2,3,4))
    native,gpu,limits=load_track_data(ImageSpec(path,data.shape,(2,3,4)),max_dimension=10)
    assert native.data[9,11,13]==50000
    assert gpu.data.max()>0
    assert gpu.data.dtype==np.float32
    assert max(gpu.data.shape)<=10
    assert limits[1]>limits[0]


def test_track_voxel_maps_keep_the_entered_channel_calibration(tmp_path):
    project=make_track_project(tmp_path)
    project.track.spacing_um_zyx=(2.000001,3.000002,4.000003)
    path=export_project(project,include_ants=False)
    physical=np.eye(4)
    physical[:3,:3]=np.diag(project.track.spacing_um_zyx)
    atlas_scale=np.eye(4)
    atlas_scale[:3,:3]=np.diag(project.reference.spacing_um_zyx)
    expected=np.linalg.inv(atlas_scale)@project.total@physical
    assert_allclose(np.load(path.parent/'track_to_atlas_zyx_voxel.npy'),expected,rtol=0,atol=1e-12)
    sidecar=json.loads((path.parent/'track_on_atlas_grid.geometry.json').read_text())
    assert_allclose(sidecar['moving_to_output_voxel_zyx'],expected,rtol=0,atol=1e-12)

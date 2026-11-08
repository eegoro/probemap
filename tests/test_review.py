import numpy as np
from numpy.testing import assert_allclose

from probemap.registration.coarse import pose_matrix
from probemap.registration.desktop_model import AlignmentProject, ImageSpec, orientation_linear
from probemap.registration.landmarks import transform_points
from probemap.registration.review import placement_without_scale, registration_report


def project(tmp_path):
    spec=ImageSpec(tmp_path/'unused.tif',(20,30,40),(7,3,2))
    p=AlignmentProject(spec,spec,tmp_path)
    p.coarse=pose_matrix(spec.shape_zyx,spec.spacing_um_zyx,(11,-23,37),(17,31,-48),(.6,1.4,2))
    p.coarse[:3,:3]@=orientation_linear('YZX',(True,False,False))
    center=(np.asarray(spec.shape_zyx)-1)*spec.spacing_um_zyx/2
    p.coarse[:3,3]=center+[11,-23,37]-p.coarse[:3,:3]@center
    return p


def test_unscaled_placement_preserves_rotation_axes_center_and_native_lengths(tmp_path):
    p=project(tmp_path)
    unscaled=placement_without_scale(p)
    expected=pose_matrix(p.moving.shape_zyx,p.moving.spacing_um_zyx,rotation_xyz_deg=(17,31,-48))[:3,:3]
    expected=expected@orientation_linear('YZX',(True,False,False))
    assert_allclose(unscaled[:3,:3],expected,atol=1e-14)
    center=(np.asarray(p.moving.shape_zyx)-1)*p.moving.spacing_um_zyx/2
    assert_allclose(transform_points(center[None],unscaled),transform_points(center[None],p.coarse))
    points=np.array([[4,9,2],[13,18,25.]])
    assert_allclose(np.linalg.norm(np.diff(transform_points(points,unscaled),axis=0)),np.linalg.norm(np.diff(points,axis=0)))
    assert np.linalg.det(unscaled[:3,:3])<0


def test_report_errors_and_cross_validation_are_physical_and_not_fit_accuracy(tmp_path):
    p=project(tmp_path)
    raw=np.random.default_rng(5).uniform(0,50,(12,3))
    placed=transform_points(raw,p.coarse)
    fine=np.eye(4); fine[:3,:3]=[[1.1,.1,0],[0,.9,0],[0,0,1.2]]; fine[:3,3]=[3,5,-4]
    ref=transform_points(placed,fine)
    p.pairs=[dict(reference=r.tolist(),moving=m.tolist()) for r,m in zip(ref,placed)]
    p.fit()
    result=registration_report(p)
    assert result['pairs']==12 and result['current']
    assert result['before']['rms']>10
    assert result['after']['rms']<1e-11
    assert result['leave_one_out_rms']<1e-11
    assert result['leave_one_out_count']==12
    assert_allclose(result['rms_reduction_percent'],100,atol=1e-10)
    assert_allclose(result['atlas_size_mm_xyz'],np.asarray(p.reference.shape_zyx[::-1])*p.reference.spacing_um_zyx[::-1]/1000)
    assert result['error_frame']=='atlas physical'
    assert 'not independent' in result['accuracy_note']
    p.pairs[0]['reference'][0]+=10
    assert registration_report(p)['after'] is None


def test_voxel_error_uses_each_atlas_axis_and_reduction_uses_the_same_before_baseline(tmp_path):
    p=project(tmp_path)
    raw=np.random.default_rng(3).uniform(0,50,(9,3))
    placed=transform_points(raw,p.coarse)
    ref=placed+np.random.default_rng(4).normal(0,5,placed.shape)
    p.pairs=[dict(reference=r.tolist(),moving=m.tolist()) for r,m in zip(ref,placed)]
    p.fit()
    report=registration_report(p)
    error=transform_points(placed,p.residual)-ref
    expected=np.sqrt(np.mean(np.sum((error/p.reference.spacing_um_zyx)**2,axis=1)))
    assert_allclose(report['rms_atlas_voxels'],expected)
    assert_allclose(report['rms_reduction_percent'],100*(1-report['after']['rms']/report['before']['rms']))


def test_minimum_fit_does_not_invent_leave_one_out_accuracy(tmp_path):
    p=project(tmp_path)
    points=np.array([[0,0,0],[10,0,0],[0,20,0],[0,0,30.]])
    p.pairs=[dict(reference=x.tolist(),moving=x.tolist()) for x in points]
    p.fit()
    assert registration_report(p)['leave_one_out_rms'] is None

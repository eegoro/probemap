import json
from pathlib import Path

import numpy as np
import pytest
import tifffile
from numpy.testing import assert_allclose

from probemap.registration.coarse import pose_matrix
from probemap.registration.desktop_model import (
    AlignmentProject, ImageSpec, export_project, inspect_tiff, orientation_linear,
    full_view_grid,prepare_landmark_views,sample_preview_grid_slice,
)
from probemap.registration.landmarks import transform_points
from probemap.registration.volumes import write_volume
from probemap.registration.volumes import Volume


def make_project(tmp_path):
    shape = (12, 15, 18)
    data = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
    raw = write_volume(tmp_path / "raw.tif", data, (3, 4, 5))
    atlas = write_volume(tmp_path / "atlas.tif", data, (2, 2.5, 3))
    return AlignmentProject(ImageSpec(atlas, shape, (2, 2.5, 3)),
                            ImageSpec(raw, shape, (3, 4, 5)), tmp_path / "outputs")


def test_metadata_detects_spacing_and_marks_unknown_values(tmp_path):
    p = make_project(tmp_path)
    info = inspect_tiff(p.reference.path)
    assert_allclose(info["spacing_um_zyx"], (2, 2.5, 3))
    blank = tmp_path / "uncalibrated.tif"
    tifffile.imwrite(blank, np.zeros((8, 9, 10), np.uint8), photometric="minisblack")
    assert inspect_tiff(blank)["spacing_um_zyx"] == [None, None, None]
    side = blank.with_suffix(".geometry.json")
    side.write_text(json.dumps({"spacing_um_zyx": [1, 2, 3], "origin_um_zyx": [4, 5, 6]}))
    assert inspect_tiff(blank)["spacing_um_zyx"] == [1, 2, 3]
    assert inspect_tiff(blank)["origin_um_zyx"] == [4, 5, 6]


def test_pairs_fit_original_coordinates_and_survive_resume(tmp_path):
    p = make_project(tmp_path)
    p.coarse = pose_matrix(p.moving.shape_zyx, p.moving.spacing_um_zyx, (3, 2, 4), (10, 15, -20), (.8, 1.2, .9))
    raw = np.array([[3, 4, 5], [9, 4, 5], [3, 12, 5], [3, 4, 16], [8, 10, 12.]])
    placed = transform_points(raw * p.moving.spacing_um_zyx, p.coarse)
    fine = pose_matrix(p.reference.shape_zyx, p.reference.spacing_um_zyx, (1, 2, 3), (2, -3, 4), (1.1, .9, 1))
    ref = transform_points(placed, fine)
    p.pairs = [{"reference": r.tolist(), "moving": m.tolist()} for r, m in zip(ref, placed)]
    p.fit()
    assert_allclose(p.total, fine @ p.coarse, atol=1e-10)
    assert p.fit_is_current
    restored = AlignmentProject.load(p.save(tmp_path / "project.json"))
    assert restored.fit_is_current
    assert_allclose(restored.total, p.total)
    restored.pairs[0]["moving"][0] += 2
    assert not restored.fit_is_current
    with pytest.raises(ValueError, match="Fit the current"):
        export_project(restored, include_ants=False)


def test_export_uses_atlas_axes_spacing_and_source_intensities(tmp_path):
    p = make_project(tmp_path)
    p.coarse = pose_matrix(p.moving.shape_zyx, p.moving.spacing_um_zyx, (3, 2, 4), (0, 0, 90), (.8, 1.2, .9))
    manifest_path = export_project(p, "placement", (4, 5, 6), include_ants=False)
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "complete"
    folder = manifest_path.parent
    registered = tifffile.imread(folder / "scan_on_atlas_grid.tif")
    assert registered.shape == p.reference.shape_zyx
    assert registered.dtype == np.uint16
    assert registered.max() > 1  # preview normalization must never leak into export
    assert_allclose(inspect_tiff(folder / "scan_on_atlas_grid.tif")["spacing_um_zyx"], p.reference.spacing_um_zyx)
    assert_allclose(inspect_tiff(folder / "scan_adapted.tif")["spacing_um_zyx"], (4, 5, 6))
    geometry = json.loads((folder / "scan_adapted.geometry.json").read_text())
    points = np.array([[2, 3, 4], [6, 8, 10.]])
    mapped = transform_points(points, geometry["moving_to_output_voxel_zyx"])
    expected = (transform_points(points * p.moving.spacing_um_zyx, p.coarse)-geometry["origin_um_zyx"])/[4,5,6]
    assert_allclose(mapped, expected)
    assert_allclose(transform_points(mapped, geometry["output_voxel_to_moving_zyx"]), points)
    # Re-export updates the same files directly in the selected results folder.
    unrelated=folder/'notes.txt'
    unrelated.write_text('Keep my notes')
    p.coarse[:3,3] += [2,3,4]
    second = export_project(p, "placement", include_ants=False)
    assert second == manifest_path == p.output_dir/'export.json'
    assert_allclose(np.load(folder/'moving_to_atlas_zyx_um.npy'),p.coarse)
    assert not np.array_equal(tifffile.imread(folder/'scan_on_atlas_grid.tif'),registered)
    assert unrelated.read_text()=='Keep my notes'
    assert not any(path.is_dir() for path in folder.iterdir())


def test_orientation_and_cancelled_export(tmp_path):
    assert_allclose(orientation_linear("YXZ", (True, False, False)), [[1,0,0],[0,0,1],[0,-1,0]])
    p = make_project(tmp_path)
    with pytest.raises(InterruptedError):
        export_project(p,"placement",include_ants=False,cancelled=lambda:True)
    manifest = p.output_dir/'export.json'
    assert json.loads(manifest.read_text())["status"] == "incomplete"
    assert not list(p.output_dir.glob("**/*.partial.tif"))
    export_project(p,'placement',include_ants=False)
    previous=(p.output_dir/'scan_on_atlas_grid.tif').read_bytes()
    p.coarse[:3,3] += 50
    with pytest.raises(InterruptedError):
        export_project(p,'placement',include_ants=False,cancelled=lambda:True)
    assert (p.output_dir/'scan_on_atlas_grid.tif').read_bytes()==previous
    assert not list(p.output_dir.glob('*.partial.tif'))


def test_save_updates_one_project_file_and_owns_results_folder(tmp_path):
    p=make_project(tmp_path)
    folder=tmp_path/'chosen outputs'
    folder.mkdir()
    for value in range(5):
        p.display['step']=value
        assert p.save(folder)==folder/'project.json'
    assert p.output_dir==folder
    assert list(folder.iterdir())==[folder/'project.json']
    record=json.loads((folder/'project.json').read_text())
    assert record['display']['step']==4 and record['relative_paths'][str(folder)]=='.'
    # Even a legacy output directory that still exists cannot redirect saves.
    record['output_dir']=str(tmp_path/'obsolete')
    (tmp_path/'obsolete').mkdir()
    (folder/'project.json').write_text(json.dumps(record))
    loaded=AlignmentProject.load(folder)
    assert loaded.output_dir==folder


def test_affine_export_ants_round_trip_uses_original_input_voxels(tmp_path):
    from probemap.registration.ants_workflow import transfer_points
    p=make_project(tmp_path)
    p.coarse=pose_matrix(p.moving.shape_zyx,p.moving.spacing_um_zyx,(2,3,4),(8,-10,15),(.8,1.1,.9))
    raw=np.array([[2,3,4],[7,3,4],[2,10,4],[2,3,12],[7,9,11.]])
    placed=transform_points(raw*p.moving.spacing_um_zyx,p.coarse)
    target=placed+[1,2,3]
    p.pairs=[{"reference":r.tolist(),"moving":m.tolist()} for r,m in zip(target,placed)]
    p.fit()
    path=export_project(p,include_ants=True)
    record=json.loads(path.read_text())
    expected=target/p.reference.spacing_um_zyx
    assert_allclose(transfer_points(raw,record["ants_checkpoint"]),expected,atol=1e-4)
    assert_allclose(transfer_points(expected,record["ants_checkpoint"],direction="reference_to_moving"),raw,atol=1e-4)
    assert path.parent==p.output_dir
    assert Path(record['ants_checkpoint']).parent==p.output_dir
    assert export_project(p,include_ants=True)==path
    assert not any(child.is_dir() for child in p.output_dir.iterdir())


def test_landmark_union_keeps_scan_voxels_outside_all_atlas_faces(tmp_path):
    shape=(9,11,13)
    path=write_volume(tmp_path/'scan.tif',np.ones(shape,np.float32),(2,3,4))
    ref_spec=ImageSpec(path,(5,6,7),(2,3,4))
    mov_spec=ImageSpec(path,shape,(2,3,4))
    data=np.arange(np.prod(shape),dtype=np.float32).reshape(shape)/np.prod(shape)
    source=Volume(data,(2,3,4))
    atlas=Volume(np.zeros(ref_spec.shape_zyx,np.float32),(2,3,4))
    matrix=np.eye(4)
    matrix[:3,3]=[-12,9,-16]
    ref,placed,grid=prepare_landmark_views(atlas,source,ref_spec,mov_spec,matrix)
    origin=np.asarray(grid['origin_um_zyx'])
    assert origin[0]<0 and origin[2]<0
    # Every input voxel centre, including corner voxels outside the atlas, is
    # available on the union display grid after this exact-lattice translation.
    coords=np.indices(shape).reshape(3,-1).T
    world=transform_points(coords*[2,3,4],matrix)
    displayed=np.rint((world-origin)/[2,3,4]).astype(int)
    assert_allclose(placed[tuple(displayed.T)],data.ravel(),atol=1e-6)
    assert ref.shape==placed.shape==tuple(grid['shape_zyx'])
    for axis in range(3):
        index=grid['shape_zyx'][axis]//2
        expected=np.take(placed,index,axis=axis)
        assert_allclose(sample_preview_grid_slice(source,matrix,grid,axis,index),expected,atol=1e-6)


def test_union_contains_rotated_native_cells_and_caps_display_memory(tmp_path):
    from probemap.registration.coarse import transformed_corners
    p=make_project(tmp_path)
    matrix=pose_matrix(p.moving.shape_zyx,p.moving.spacing_um_zyx,(-80,45,100),(25,70,-40),(1.5,2,.8))
    grid=full_view_grid(p.reference,p.moving,matrix,(2,2.5,3),max_voxels=30000,max_dimension=80)
    origin=np.asarray(grid['origin_um_zyx'])
    spacing=np.asarray(grid['spacing_um_zyx'])
    corners=np.vstack([transformed_corners(p.reference.shape_zyx,p.reference.spacing_um_zyx,np.eye(4)),
                       transformed_corners(p.moving.shape_zyx,p.moving.spacing_um_zyx,matrix)])
    assert np.all(corners>=origin-spacing/2-1e-8)
    assert np.all(corners<=origin+(np.asarray(grid['shape_zyx'])-.5)*spacing+1e-8)
    assert np.prod(grid['shape_zyx'])<=30000
    assert max(grid['shape_zyx'])<=80


def test_comparison_grid_keeps_before_and_after_transformed_bounds(tmp_path):
    from probemap.registration.coarse import transformed_corners
    p=make_project(tmp_path)
    before=pose_matrix(p.moving.shape_zyx,p.moving.spacing_um_zyx,(-100,40,-80),(20,-35,70),(1.3,.8,1.2))
    after=pose_matrix(p.moving.shape_zyx,p.moving.spacing_um_zyx,(90,-50,100),(-40,20,10),(.7,1.1,1.5))
    grid=full_view_grid(p.reference,p.moving,after,(2,2.5,3),additional_matrices=(before,),
                        max_voxels=30000,max_dimension=80)
    origin=np.asarray(grid['origin_um_zyx'])
    spacing=np.asarray(grid['spacing_um_zyx'])
    for shape,sizes,matrix in ((p.reference.shape_zyx,p.reference.spacing_um_zyx,np.eye(4)),
                               (p.moving.shape_zyx,p.moving.spacing_um_zyx,before),
                               (p.moving.shape_zyx,p.moving.spacing_um_zyx,after)):
        corners=transformed_corners(shape,sizes,matrix)
        assert np.all(corners>=origin-spacing/2-1e-8)
        assert np.all(corners<=origin+(np.asarray(grid['shape_zyx'])-.5)*spacing+1e-8)
    assert np.prod(grid['shape_zyx'])<=30000
    assert max(grid['shape_zyx'])<=80

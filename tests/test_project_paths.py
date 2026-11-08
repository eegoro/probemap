"""Moving an archive must retain calibrated geometry and its analysis session."""
import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from probemap.registration.desktop_model import AlignmentProject
from probemap.registration.insertion_model import insertion_session,new_insertion,sync_tracks
from test_track_model import make_track_project


@pytest.mark.parametrize('legacy',[False,True])
def test_relocated_data_retains_inputs_calibration_and_existing_insertions(tmp_path,legacy):
    original=tmp_path/'original study'
    project=make_track_project(original/'data')
    session=insertion_session(project)
    item=new_insertion(session,'Anterior')
    item['tracks'].append(dict(id='trace',name='Electrode 1',
        control_points_atlas_zyx_um=[[4,9,10],[4,9,30]],start_validation={'inside':True}))
    sync_tracks(session,project)
    session_id=session['id']
    path=project.save(project.output_dir/'project.json')
    geometry=project.total.copy()
    if legacy:
        record=json.loads(path.read_text())
        record['application']='ProbeMap Align'
        record.pop('relative_paths')
        path.write_text(json.dumps(record))
    destination=tmp_path/'relocated archive'/'old'
    destination.parent.mkdir()
    original.rename(destination)
    moved=destination/'data/out/project.json'
    loaded=AlignmentProject.load(moved)
    assert loaded.moving.path==destination/'data/brain.tif'
    assert loaded.track.path==destination/'data/track.tif'
    assert loaded.reference.path==destination/'data/atlas.tif'
    assert loaded.labels_dir==destination/'data/masks'
    assert loaded.output_dir==destination/'data/out'
    assert loaded.moving.spacing_um_zyx==(2,3,4)
    assert_allclose(loaded.total,geometry)
    restored=insertion_session(loaded)
    assert len(loaded.track_analysis['sessions'])==1
    assert restored['id']==session_id
    assert restored['insertions'][0]['tracks']==item['tracks']
    assert len(restored['tracks'])==1
    loaded.save(moved)
    assert json.loads(moved.read_text())['application']=='ProbeMap'
    assert json.loads(moved.read_text())['relative_paths']
    assert not (moved.parent/'.probemap_history').exists()


def test_legacy_project_prefers_relocated_study_and_current_output_folder(tmp_path):
    old=tmp_path/'software'/'data'
    project=make_track_project(old)
    session=insertion_session(project)
    item=new_insertion(session,'Dorsal')
    expected_id=session['id']
    old_file=project.save(old/'out'/'project.json')
    record=json.loads(old_file.read_text())
    record.pop('relative_paths')
    # The previous copy still exists; use images beside the opened project.
    import shutil
    new=tmp_path/'data'
    shutil.copytree(old,new)
    target=new/'out'/'project.json'
    record['output_dir']=str(old/'out'/'dated_export')
    target.write_text(json.dumps(record))
    loaded=AlignmentProject.load(target.parent)
    assert loaded.reference.path==new/'atlas.tif'
    assert loaded.moving.path==new/'brain.tif'
    assert loaded.track.path==new/'track.tif'
    assert loaded.labels_dir==new/'masks'
    assert loaded.output_dir==target.parent
    restored=insertion_session(loaded)
    assert restored['id']==expected_id
    assert restored['insertions'][0]['id']==item['id']
    assert len(loaded.track_analysis['sessions'])==1


@pytest.mark.parametrize('relative_links',[False,True])
def test_windows_paths_relocate_with_or_without_backslash_links(tmp_path,relative_links):
    from pathlib import PureWindowsPath
    from probemap.project_paths import _paths
    root=tmp_path/'portable study'
    project=make_track_project(root/'data')
    item=new_insertion(insertion_session(project),'Тестовая insertion')
    path=project.save(project.output_dir/'project.json')
    record=json.loads(path.read_text())
    names={}
    for owner,key in _paths(record):
        old=owner[key]
        try: relative=Path(old).relative_to(root)
        except ValueError: continue
        foreign=str(PureWindowsPath('D:/archive/portable study').joinpath(*relative.parts))
        names[old]=foreign; owner[key]=foreign
    if relative_links:
        record['relative_paths']={names.get(k,k):v.replace('/','\\') for k,v in record['relative_paths'].items()}
    else: record.pop('relative_paths')
    path.write_text(json.dumps(record),encoding='utf-8')
    loaded=AlignmentProject.load(path)
    assert loaded.reference.path==root/'data/atlas.tif'
    assert loaded.moving.path==root/'data/brain.tif'
    assert loaded.track.path==root/'data/track.tif'
    assert insertion_session(loaded)['insertions'][0]['id']==item['id']
    assert insertion_session(loaded)['insertions'][0]['name']=='Тестовая insertion'


def test_ants_checkpoint_repairs_windows_transform_paths(tmp_path):
    from probemap.registration.ants_workflow import load_registration
    run=tmp_path/'study/runs/manual'; run.mkdir(parents=True)
    for name in ('forward.h5','inverse.h5'): (run/name).write_bytes(b'placeholder')
    inputs=tmp_path/'study/brain.tif'; inputs.write_bytes(b'placeholder')
    record=dict(schema_version=1,fwd_composite=r'D:\old\study\runs\manual\forward.h5',
                inv_composite=r'D:\old\study\runs\manual\inverse.h5',
                inputs={'moving':{'path':r'D:\old\study\brain.tif'}})
    path=run/'registration.json'; path.write_text(json.dumps(record),encoding='utf-8')
    restored=load_registration(path)
    assert restored['fwd_composite']==str(run/'forward.h5')
    assert restored['inv_composite']==str(run/'inverse.h5')
    assert restored['inputs']['moving']['path']==str(inputs)

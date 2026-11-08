"""Portable library geometry and native placement across different probe families."""
import copy
import json
import numpy as np
import pytest
from numpy.testing import assert_allclose

from probemap.probe_library import (ProbeLibrary,builtin_probes,contacts_for,grid_geometry,
    validate_geometry,contacts_csv,parse_contacts_csv)
from probemap.registration.neuropixels_model import probe_settings,contact_coordinates,contact_mesh,recording_sites
from probemap.registration.probe_placement import pose_on_plane,place_probe,flip_contact_face,contact_preview
from probemap.registration.reconstruction import shanks


def test_builtin_positions_and_custom_library_are_independent(tmp_path):
    library=ProbeLibrary(tmp_path/'probes_library')
    assert not library.path.exists()
    entries=library.entries()
    for entry in entries: validate_geometry(entry['geometry'])
    np1=next(e['geometry'] for e in entries if e['id']=='NP1000')
    assert len(contacts_for(np1))==960
    assert [c['x_um'] for c in np1['contacts'][:4]]==[-8,24,-24,8]
    g=grid_geometry('My tetrode',rows=1,columns=2,layers=2,width=100)
    saved=library.save(dict(geometry=g))
    project_geometry=copy.deepcopy(saved['geometry'])
    saved['geometry']['shank_length_um']=7000
    updated=library.save(saved)
    assert project_geometry['shank_length_um']==5000
    assert len(list(library.path.glob('*.json')))==1
    assert next(e for e in library.entries() if e['id']==saved['id'])['geometry']==updated['geometry']
    library.delete(updated)
    assert project_geometry['model']=='My tetrode'
    with pytest.raises(ValueError): library.delete(entries[0])


def test_validation_and_csv_never_silently_lose_geometry(tmp_path):
    g=grid_geometry('3D array',shanks=3,rows=4,columns=2,layers=2,width=100)
    validate_geometry(g)
    assert parse_contacts_csv(contacts_csv(g))==g['contacts']
    g['contacts'][1]['x_um']=float('nan')
    with pytest.raises(ValueError,match='finite'): validate_geometry(g)
    g=grid_geometry();g['contacts'][1]=copy.deepcopy(g['contacts'][0])
    with pytest.raises(ValueError,match='unique'): validate_geometry(g)
    with pytest.raises(ValueError,match='at most'): grid_geometry(rows=5000,columns=32)
    broken=tmp_path/'broken.json';broken.write_text('{')
    library=ProbeLibrary(tmp_path);assert library.entries();assert library.errors
    unsafe=tmp_path/'unsafe.json';unsafe.write_text(json.dumps(dict(schema_version=1,id='custom-../../escape',geometry=grid_geometry())))
    with pytest.raises(ValueError,match='ID'): library.read(unsafe)


@pytest.mark.parametrize('count',[1,2,6])
def test_native_placement_tracks_face_flip_and_recording_sites(count,tmp_path):
    from probemap.registration.insertion_model import insertion_session,new_insertion
    from test_track_model import make_track_project
    project=make_track_project(tmp_path)
    session=insertion_session(project);item=new_insertion(session)
    settings=probe_settings(item)
    settings['geometry']=grid_geometry('Custom',shanks=count,rows=4,columns=2,layers=2,width=100,site_width=10)
    plane=dict(center_um_zyx=[1000,1000,1000],rotation_xyz_deg=[0,0,0],width_um=4000,height_um=6000,max_dimension=384)
    item['source_plane']=plane
    pose=pose_on_plane(item,plane,[1000,1000,1000],23)
    place_probe(item,pose,project,session)
    assert len(item['tracks'])==count
    assert len(settings['pose']['surface_depths_um'])==count
    before=contact_coordinates(settings,settings['geometry']['contacts'])
    assert len(contact_preview(item,np.eye(4)))==count*16
    segments=shanks(item,np.eye(4));assert segments.shape==(count,2,3)
    assert_allclose(segments[:,0].mean(0),[1000,1000,1000])
    # Imported recording order selects sites by physical identity, not row/column.
    record=dict(channels=[dict(channel=i,shank=c['shank'],electrode=c['electrode']) for i,c in enumerate(settings['geometry']['contacts'][::-1])],depth_offset_um=25)
    active,heights,sources=recording_sites(item,record)
    assert_allclose(sources,before[::-1]+25*np.asarray(pose['down_source_zyx']))
    assert_allclose(heights,[c['y_um'] for c in settings['geometry']['contacts'][::-1]])
    vertices,faces=contact_mesh(item,sources,np.eye(4));assert len(vertices)==len(sources)*4
    settings['pose']=flip_contact_face(item,plane)
    settings['pose']=flip_contact_face(item,plane)
    assert_allclose(contact_coordinates(settings,settings['geometry']['contacts']),before,atol=1e-8)
    from probemap.registration.desktop_model import AlignmentProject
    saved_path=project.save(tmp_path/'out/project.json')
    restored=AlignmentProject.load(saved_path)
    restored_item=restored.track_analysis['sessions'][0]['insertions'][0]
    assert restored_item['probe']['geometry']==settings['geometry']
    assert_allclose(contact_coordinates(restored_item['probe'],restored_item['probe']['geometry']['contacts']),before,atol=1e-8)


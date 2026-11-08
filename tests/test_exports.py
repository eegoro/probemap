import json
import numpy as np
import pytest
import tifffile
from numpy.testing import assert_allclose,assert_array_equal

from probemap.registration.desktop_model import AlignmentProject,ImageSpec,inspect_tiff
from probemap.registration.exports import export_selection
from probemap.registration.volumes import write_volume


def project(tmp_path):
    data=np.arange(120,dtype=np.uint16).reshape(4,5,6)
    brain=write_volume(tmp_path/'brain.tif',data,(2,3,4))
    electrodes=write_volume(tmp_path/'electrodes.tif',data+1000,(2,3,4))
    p=AlignmentProject(ImageSpec(brain,data.shape,(2,3,4)),ImageSpec(brain,data.shape,(2,3,4)),tmp_path/'project',
                       track=ImageSpec(electrodes,data.shape,(2,3,4)))
    points=[[0,0,0],[6,0,0],[0,9,0],[0,0,16],[6,9,16]]
    p.pairs=[dict(reference=(np.asarray(x)+[2,3,4]).tolist(),moving=x) for x in points]
    p.fit()
    return p,data


@pytest.mark.parametrize('stage',['placement','registration'])
@pytest.mark.parametrize('kind',['brain','electrodes'])
@pytest.mark.parametrize('fmt',['tiff','ome'])
def test_selected_volume_retains_native_values_extent_calibration_and_transform(tmp_path,stage,kind,fmt):
    p,data=project(tmp_path)
    path=tmp_path/'chosen'/f'{kind}-{stage}.tif'
    export_selection(p,dict(path=path,kind=kind,stage=stage,format=fmt,spacing_um_zyx=[2,3,4],extent='full'))
    assert_array_equal(tifffile.imread(path),data+(1000 if kind=='electrodes' else 0))
    assert_allclose(inspect_tiff(path)['spacing_um_zyx'],[2,3,4])
    geometry=json.loads(path.with_suffix('.geometry.json').read_text())
    assert_allclose(geometry['source_to_atlas_zyx_um'],p.coarse if stage=='placement' else p.total,atol=1e-12)
    assert_allclose(geometry['moving_to_output_voxel_zyx'],np.eye(4),atol=1e-12)
    assert not p.output_dir.exists() # Export does not create/copy/move projects.
    if fmt=='ome':
        with tifffile.TiffFile(path) as tif: assert tif.is_ome


def test_export_protects_inputs_and_stale_registration_and_cleans_cancelled_file(tmp_path):
    p,_=project(tmp_path)
    options=dict(path=p.moving.path,kind='brain',stage='registration',format='tiff',spacing_um_zyx=[2,3,4],extent='full')
    with pytest.raises(ValueError,match='overwrite'): export_selection(p,options)
    path=tmp_path/'cancelled.tif'; options['path']=path
    with pytest.raises(InterruptedError): export_selection(p,options,cancelled=lambda:True)
    assert not path.exists()
    p.pairs[0]['reference'][0]+=1
    with pytest.raises(ValueError,match='Apply'): export_selection(p,options)


def test_named_transform_and_report_exports_overwrite_same_destination(tmp_path):
    p,_=project(tmp_path)
    for kind,fmt in [('transforms','json'),('transforms','npz'),('report','json'),('report','csv')]:
        path=tmp_path/f'{kind}.{fmt}'
        options=dict(path=path,kind=kind,stage='registration',format=fmt)
        export_selection(p,options); first=path.read_bytes(); export_selection(p,options)
        assert path.read_bytes()==first
    assert_allclose(np.load(tmp_path/'transforms.npz')['total_zyx_um'],p.total)
    assert json.loads((tmp_path/'report.json').read_text())['pairs']==5

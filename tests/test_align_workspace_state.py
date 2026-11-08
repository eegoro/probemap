"""Save-file identity, layer combinations and stage ownership."""

import pytest

pytestmark = pytest.mark.gui
import copy
import hashlib
import json
import time

import numpy as np
from numpy.testing import assert_allclose

from probemap.registration.desktop_model import AlignmentProject,ImageSpec
from probemap.registration.insertion_model import insertion_session,new_insertion,add_insertion_point,save_manual_plane,sync_tracks
from probemap.registration.volumes import write_volume
from test_desktop_tracks import wait_until


def test_complete_workspace_roundtrip_layers_labels_zoom_and_in_place_save(tmp_path):
    from probemap.registration.macos_qt import initialize_qt
    app=initialize_qt()
    from qtpy.QtCore import QPointF,Qt
    from qtpy.QtWidgets import QPushButton
    from probemap.registration.desktop_app import AlignWindow
    shape=(25,30,35)
    spacing=(200,200,200)
    z,y,x=np.indices(shape)
    data=(50000*np.exp(-((z-12)**2+(y-15)**2+(x-17)**2)/50)).astype(np.uint16)
    brain=write_volume(tmp_path/'brain.tif',data,spacing)
    track=write_volume(tmp_path/'electrodes.tif',data//2,spacing)
    spec=ImageSpec(brain,shape,spacing)
    project=AlignmentProject(spec,spec,tmp_path/'exports',track=ImageSpec(track,shape,spacing))
    anchors=[[400,600,800],[4000,600,800],[400,5000,800],[400,600,6000]]
    project.pairs=[dict(reference=p,moving=p) for p in anchors]
    project.fit()
    masks=tmp_path/'labels'
    masks.mkdir()
    write_volume(masks/'1_left.tif',(x<17).astype(np.uint8),spacing)
    write_volume(masks/'2_right.tif',(x>=17).astype(np.uint8),spacing)
    project.labels_dir=masks
    session=insertion_session(project)
    item=new_insertion(session)
    for p in ([2400,3000,3400],[2400,3400,4000],[2400,2200,4000]): add_insertion_point(project,item,p)
    save_manual_plane(item,dict(center_um_zyx=[2400,3000,3400],rotation_xyz_deg=[0,0,0],width_um=6000,height_um=5000,max_dimension=384))
    item['tracks']=[dict(id='electrode',name='Electrode',control_points_atlas_zyx_um=[[2400,3000,3400],[3000,3000,3600]],start_validation={'inside':True})]
    sync_tracks(session,project)
    project.display={'step':3}
    canonical=project.save(tmp_path/'exports/project.json')
    canonical_hash=hashlib.sha256(canonical.read_bytes()).hexdigest()
    opened=project.save(tmp_path/'separate/project.json')
    errors=[]
    w=AlignWindow()
    w._error=errors.append
    w.show()
    try:
        w.load_project_path(opened.parent)
        wait_until(app,lambda:not w.busy and w.reference is not None)
        assert w.project_path==opened
        # Independent combinations, including categorical atlas surfaces on review.
        v=w.volume_view
        assert v.layer_titles==('Atlas','Brain scan','Track scan')
        v.layers.checks[2].setChecked(True)
        v.layers.labels_check.setChecked(True)
        wait_until(app,lambda:not w.busy and len(v.visuals)==3 and len(v.label_visuals)==2)
        v.layers.checks[0].setChecked(False)
        v.layers.checks[1].setChecked(False)
        assert not v.visuals[0].visible and not v.visuals[1].visible and v.visuals[2].visible
        assert all(m.visible for m in v.label_visuals.values())
        v.layers.labels_check.setChecked(False)
        assert all(not m.visible for m in v.label_visuals.values())
        w.go_step(4)
        t=w.track_workspace
        def ready(): return not w.busy and t._cache_current() and not t._running and not t.render_timer.isActive()
        wait_until(app,ready)
        assert len(t.current_insertion()['points'])==3
        assert not any(line.visible for line in t.volume_view.track_visuals.values())
        assert all(not canvas.insertion_tracks for canvas in t.slices)
        # Zoom anchors the same physical point and does not change a transform.
        c=t.track_slice
        pixel=QPointF(c.width()*.55,c.height()*.45)
        world=c._world(pixel)
        matrix=w.project.total.copy()
        c.zoom_at(pixel,3)
        assert_allclose(c._world(pixel),world,atol=1e-8)
        assert c.zoom==3
        t.add_point(c._world(QPointF(c.width()/2,c.height()/2)))
        from probemap.registration.neuropixels_model import probe_settings
        pose=probe_settings(t.current_insertion())
        pose['pose']=dict(tip0_source_zyx_um=[2400,3000,3400],down_source_zyx=[-1,0,0],
            lateral_source_zyx=[0,0,1],normal_source_zyx=[0,-1,0])
        native=copy.deepcopy(t.session['insertions'])
        w.go_step(6)
        r=w.reconstruction
        def final_ready(): return not w.busy and r.report is not None and not r._running and not r.timer.isActive()
        wait_until(app,final_ready)
        assert len(r.report['probes'])==1 and r.volume_view.point_records==[]
        r.view_choice.setCurrentIndex(2); r.compare_axis.setCurrentIndex(1); wait_until(app,final_ready)
        r.compare_canvases[0].restore_view(dict(zoom=2,pan=[10,15],rotation=90))
        r.volume_view.layers.checks[0].setChecked(False)
        r.volume_view.layers.checks[2].setChecked(True)
        r.volume_view.show_axes.setChecked(False)
        r.volume_view.flip_checks[0].setChecked(True)
        r.slab.setValue(40); r.unit.setCurrentIndex(1)
        w.save_project()
        saved=AlignmentProject.load(opened)
        assert saved.display['step']==6 and saved.display['workflow_version']==2
        assert saved.track_analysis['sessions'][0]['insertions']==native
        assert hashlib.sha256(canonical.read_bytes()).hexdigest()==canonical_hash
        assert not (opened.parent/'.probemap_history').exists()
        assert w.project.output_dir==opened.parent and not w.save_button.isVisible()
        assert (opened.parent/'reconstruction.json').is_file()
        w.load_project_path(opened); wait_until(app,final_ready)
        assert w.step==6 and r.view_choice.currentIndex()==2 and r.compare_axis.currentIndex()==1
        assert not r.volume_view.layers.checks[0].isChecked() and r.volume_view.layers.checks[2].isChecked()
        assert r.volume_view.view.camera.flip==(True,False,False)
        assert r.compare_canvases[2].view_record()==dict(zoom=2.,pan=[10.,15.],rotation=90)
        assert r.slab.value()==40 and r.unit.currentIndex()==1
        assert r.session['insertions']==native
        w.go_step(4); wait_until(app,ready)
        assert len(t.current_insertion()['points'])==4 and t.track_slice.zoom==3
        assert_allclose(w.project.total,matrix)
        assert not errors,errors
    finally:
        wait_until(app,lambda:not w.busy)
        w.close()

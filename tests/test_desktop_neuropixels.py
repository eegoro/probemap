from legacy_workspace import open_legacy
"""Native macOS step-9 integration on synthetic calibrated scans."""

import pytest

pytestmark = pytest.mark.gui
import copy
import csv
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

from probemap.registration.desktop_model import AlignmentProject,ImageSpec
from probemap.registration.insertion_model import insertion_session,new_insertion,sync_tracks
from probemap.registration.landmarks import transform_points
from probemap.registration.volumes import write_volume
from test_desktop_tracks import wait_until


def test_step9_maps_geometry_plane_measurements_save_resume_export(tmp_path):
    from probemap.registration.macos_qt import initialize_qt
    app=initialize_qt()
    from probemap.registration.desktop_app import AlignWindow
    shape=(70,35,35); spacing=(100,100,100)
    data=np.zeros(shape,np.uint16); data[20:62,5:30,5:30]=30000
    brain=write_volume(tmp_path/'brain.tif',data,spacing)
    track=write_volume(tmp_path/'track.tif',data//2,spacing)
    atlas=write_volume(tmp_path/'atlas.tif',data,spacing)
    p=AlignmentProject(ImageSpec(atlas,shape,spacing),ImageSpec(brain,shape,spacing),tmp_path/'out',track=ImageSpec(track,shape,spacing))
    matrix=np.array([[.8,.05,0,500],[0,1.1,0,0],[0,0,.9,200],[0,0,0,1.]])
    p.coarse=matrix.copy()
    anchors=transform_points([[2000,800,800],[5000,800,800],[2000,2500,800],[2000,800,2500]],matrix)
    p.pairs=[dict(reference=a.tolist(),moving=a.tolist()) for a in anchors]; p.fit()
    labels=tmp_path/'labels'; labels.mkdir(); write_volume(labels/'1_brain.tif',(data>0).astype(np.uint8),spacing); p.labels_dir=labels
    session=insertion_session(p); item=new_insertion(session,'Left anterior')
    for shank in (0,2,3):
        points=transform_points([[4000,1500,1000+250*shank],[2500,1500,1000+250*shank]],p.total)
        item['tracks'].append(dict(id=f't{shank}',name=f'Trace {shank}',control_points_atlas_zyx_um=points.tolist(),start_validation={'inside':True}))
    sync_tracks(session,p)
    p.display={'step':8}; path=p.save(tmp_path/'out/project.json')
    errors=[]; w=AlignWindow(); w._error=errors.append; w.show(); t=w.track_workspace
    def ready(): return not w.busy and t._cache_current() and not t._running and not t.render_timer.isActive()
    try:
        w.load_project_path(path); open_legacy(w,8); wait_until(app,ready)
        assert w.step==5 and len(w.nav_buttons)==7 and t._mode==4
        assert not t.track_draw_button.isVisible() and not t.select_3d.isVisible()
        # Missing shank IDs must produce actionable inline feedback, not an
        # apparently inert Fit button and an easily missed status-bar message.
        t.probe_fit_button.click(); app.processEvents()
        assert 'Probe was not fitted' in t.probe_fit_info.text()
        assert 'Assign at least one track' in t.probe_fit_info.text()
        for row,shank in enumerate((0,2,3)):
            combo=t.probe_assignment.cellWidget(row,1); combo.setCurrentIndex(combo.findData(shank))
        assert t.probe_tip_mode.currentData()=='track_start'
        t.probe_fit_button.click(); wait_until(app,ready)
        s=t.current_insertion()['probe']
        assert s.get('pose') and s['pose']['fit_rms_um']<1e-6
        fixture=Path(__file__).parent/'fixtures/np2013_example.imro'
        t.add_probe_recordings(paths=[fixture,fixture]); wait_until(app,ready)
        assert len(t.current_insertion()['recordings'])==2
        assert len(t.track_slice.probe_sites)==768
        assert all(c.geometry_record==t.slices[0].geometry_record for c in t.slices)
        # The complete brain slice, not just a 2.2 mm strip around the probe,
        # fits the common field despite the preceding registration scale/shear.
        from probemap.registration.track_model import plane_uv
        edge=transform_points([[0,1500,0],[6900,1500,3400]],p.total)
        uv,_=plane_uv(t.slices[0].geometry_record,edge)
        h,wid=t.slices[0].geometry_record['shape_hw']
        assert (uv>=0).all() and (uv[:,0]<wid).all() and (uv[:,1]<h).all()
        assert t.probe_plane_auto.isChecked()
        assert not t.probe_plane_controls['plane_width_um'].isEnabled()
        full_geometry=copy.deepcopy(t.slices[0].geometry_record)
        t.probe_plane_auto.setChecked(False)
        t.probe_plane_controls['plane_width_um'].setValue(1500)
        wait_until(app,ready)
        assert t.probe_plane_controls['plane_width_um'].isEnabled()
        assert t.slices[0].geometry_record!=full_geometry
        t.track_slice.zoom=3
        t._probe_view_plane(); wait_until(app,ready)
        assert t.slices[0].geometry_record==full_geometry
        assert all(canvas.zoom==1 for canvas in t.slices)
        assert all(c.tools.buttons['add'].isHidden() and c.tools.buttons['draw'].isHidden() for c in t.slices)
        assert t.control_content.isAncestorOf(t.rendering_panel)
        for index in range(t.probe_tabs.count()):
            t.probe_tabs.setCurrentIndex(index); app.processEvents()
            assert t.sidebar_tabs.widget(0).horizontalScrollBar().maximum()==0
        t.probe_tabs.setCurrentIndex(0)
        image=t.volume_view.canvas.render(size=(600,450)); assert image.std()>0
        t.grab().save(str(tmp_path/'step9.png'))
        # Each recording has independent visibility, opacity and a slice selector.
        t.recording_table.cellWidget(1,1).setChecked(False); wait_until(app,ready)
        assert len(t.track_slice.probe_sites)==384
        t.recording_table.cellWidget(1,1).setChecked(True)
        t.probe_recording_filter.setCurrentIndex(1); wait_until(app,ready)
        assert len(t.track_slice.probe_sites)==384
        # Native-space ruler is invariant to the nonuniform/shearing registration.
        source=np.array([[2500,1500,1000],[3500,1500,1000]])
        world=transform_points(source,p.total); t.measure_distance(*world)
        assert_allclose(t.session['measurements'][0]['length_um'],1000)
        for box in t.probe_surface_boxes: box.setValue(2050)
        t.probe_confirm.setChecked(True); w.save_project()
        assert t.current_insertion()['probe']['confirmed']
        report=tmp_path/'out/neuropixels_channels.csv'
        with report.open(encoding='utf-8-sig') as handle: rows=list(csv.DictReader(handle))
        assert len(rows)==768 and rows[0]['pose_status']=='confirmed'
        assert rows[0]['region']!='Labels not loaded'
        assert float(rows[0]['depth_from_surface_um'])<0
        snapshot=copy.deepcopy(t.current_insertion())
        w.go_step(4); wait_until(app,ready)
        assert not t.track_slice.probe_sites
        assert all(not mesh.visible and not sites.visible for mesh,sites in t._probe_visuals.values())
        open_legacy(w,8); wait_until(app,ready)
        assert t.current_insertion()['recordings']==snapshot['recordings']
        t.export_probe_reports(); assert t.last_export.is_file()
        w.load_project_path(path); open_legacy(w,8); wait_until(app,ready)
        assert w.step==5 and t.current_insertion()['probe']==snapshot['probe']
        assert len(t.current_insertion()['recordings'])==2
        # Four unassigned real traces can be fitted immediately; numbering is
        # explicitly provisional because histology cannot identify shank 0.
        item=t.current_insertion()
        points=transform_points([[4000,1500,1250],[2500,1500,1250]],w.project.total)
        item['tracks'].append(dict(id='t1',name='Fourth trace',control_points_atlas_zyx_um=points.tolist(),start_validation={'inside':True}))
        item['probe']['assignments']={}; item['probe'].pop('pose',None)
        t._refresh_probe_controls(); t.probe_fit_button.click(); wait_until(app,ready)
        assert len(item['probe']['assignments'])==4 and item['probe'].get('pose')
        assert 'Provisional' in t.probe_order_info.text()
        assert not item['probe']['confirmed']
        assert_allclose(item['probe']['pose']['spacing_check']['observed_pitch_um'],250,atol=1e-8)
        before=dict(item['probe']['assignments']); t.reverse_probe_order()
        assert item['probe']['assignments']=={key:3-value for key,value in before.items()}
        assert not errors,errors
    finally:
        wait_until(app,lambda:not w.busy)
        w.close()

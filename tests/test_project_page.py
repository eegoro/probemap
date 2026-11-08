"""Project entry, automatic calibration and live display settings on native Qt."""
import copy
import re

import numpy as np
import pytest
from numpy.testing import assert_allclose

from probemap.registration.desktop_model import AlignmentProject
from probemap.registration.insertion_model import add_insertion_point, insertion_session
from probemap.registration.volumes import write_volume
from test_desktop_tracks import wait_until

pytestmark=pytest.mark.gui


def test_project_entry_setup_overview_and_live_quality_preserve_physical_state(tmp_path,monkeypatch):
    from probemap.registration.macos_qt import initialize_qt
    from probemap.registration.desktop_app import ProbeMapWindow
    from qtpy.QtWidgets import QFileDialog,QPushButton
    app=initialize_qt()
    shape=(260,24,28)
    z,y,x=np.indices(shape)
    data=(40000*np.exp(-((z-130)**2/5000+(y-12)**2/40+(x-14)**2/45))).astype(np.uint16)
    template=tmp_path/'template'
    template.mkdir()
    atlas=write_volume(template/'atlas.tif',data,(25,25,25))
    brain=write_volume(tmp_path/'PV233_Ch0_brain.tif',data,(25,25,25))
    electrodes=write_volume(tmp_path/'PV233_Ch1_electrodes.tif',data//2,(25,25,25))
    masks=template/'region_masks'
    masks.mkdir()
    write_volume(masks/'1_brain.tif',(data>1000).astype(np.uint8),(25,25,25))
    folder=tmp_path/'outputs'
    folder.mkdir()
    window=ProbeMapWindow()
    errors=[]
    window._error=errors.append
    window.show()
    t=window.track_workspace
    def ready():
        return not window.busy and not window._render_running and not window.render_timer.isActive()
    def tracks_ready():
        return ready() and t._cache_current() and not t._running and not t.render_timer.isActive()
    try:
        page=window.project_page
        assert page.mode=='welcome' and page.open_button.isVisible() and page.new_button.isVisible()
        assert not window.atlas_input.isVisible() and not window.save_button.isEnabled()
        assert not any(re.match(r'^\d+\s',item.text()) for item in window.nav_buttons)
        assert window.nav_buttons[0].text()=='Project'
        forbidden={'Read metadata','Save project','Project history…','Workflow help'}
        assert not any(b.isVisible() and b.text() in forbidden for b in window.findChildren(QPushButton))
        window.grab().save(str(tmp_path/'welcome.png'))

        monkeypatch.setattr(QFileDialog,'getExistingDirectory',lambda *args,**kwargs:str(folder))
        page.new_button.click()
        assert page.mode=='setup' and page.output_path.text()==str(folder)
        assert not page.continue_button.isEnabled()
        for widget,path in ((window.atlas_input,atlas),(window.scan_input,brain)):
            monkeypatch.setattr(QFileDialog,'getOpenFileName',lambda *args,path=path,**kwargs:(str(path),''))
            widget.browse_button.click()
            assert widget.spec().spacing_um_zyx==(25,25,25)
        assert window.track_input.spec().path==electrodes
        assert window.labels_path.text()==str(masks)
        assert page.continue_button.isEnabled()
        page.already_placed.setChecked(True)
        assert page.continue_button.text()=='Continue to registration'
        page.already_placed.setChecked(False)
        # Reading an unchanged path cannot reset a user's physical calibration.
        window.scan_input.spacing[0].setValue(27)
        window.scan_input.path.editingFinished.emit()
        assert window.scan_input.spec().spacing_um_zyx==(25,25,27)
        window.scan_input.spacing[0].setValue(25)
        window.grab().save(str(tmp_path/'setup.png'))
        page.continue_button.click()
        wait_until(app,ready)
        assert window.step==1 and window.project_path==folder/'project.json'
        assert window.project.track.path==electrodes
        assert window.project.labels_dir==masks
        assert window.reference.data.shape[0]==130
        window.go_step(0)
        assert page.mode=='summary' and not window.atlas_input.isVisible()
        assert page.summary_paths['moving'].value==str(brain)
        window.grab().save(str(tmp_path/'summary.png'))

        p=window.project
        anchors=[[500,150,200],[4500,150,200],[500,500,200],[500,150,600]]
        p.pairs=[dict(reference=point,moving=point) for point in anchors]
        p.fit()
        window.go_step(4)
        wait_until(app,tracks_ready)
        item=t.current_insertion()
        add_insertion_point(p,item,[3200,300,350])
        window.save_project()
        session_id=t.session['id']
        geometry=p.total.copy()
        annotations=copy.deepcopy(item['points'])
        mask=t.brain_mask.copy()
        mask_spacing=t.brain_preview.spacing_um_zyx
        window.open_settings()
        window.quality.setCurrentIndex(window.quality.findData(512))
        window.settings_dialog.grab().save(str(tmp_path/'settings.png'))
        window.settings_dialog.apply_button.click()
        wait_until(app,tracks_ready)
        assert window.step==4 and p.preview_dimension==512
        assert window.reference.data.shape[0]==260
        assert t.track_gpu.data.shape[0]==260
        assert t.session['id']==session_id
        assert t.current_insertion()['points']==annotations
        assert t.brain_preview.spacing_um_zyx==mask_spacing
        assert np.array_equal(t.brain_mask,mask)
        assert_allclose(p.total,geometry)
        assert p.reference.spacing_um_zyx==p.moving.spacing_um_zyx==p.track.spacing_um_zyx==(25,25,25)
        saved=AlignmentProject.load(folder)
        assert saved.preview_dimension==512
        assert insertion_session(saved)['insertions'][0]['points']==annotations
        assert not (folder/'.probemap_history').exists()

        # Cancelling a new project restores the existing document and directory.
        another=tmp_path/'another project'
        another.mkdir()
        monkeypatch.setattr(QFileDialog,'getExistingDirectory',lambda *args,**kwargs:str(another))
        window.new_project()
        assert page.mode=='setup' and p.output_dir==folder
        assert not window.save_button.isEnabled()
        window.cancel_new_project()
        wait_until(app,tracks_ready)
        assert window.project is p and p.output_dir==folder
        assert window.project_path==folder/'project.json'
        assert not list(another.iterdir())
        add_insertion_point(p,t.current_insertion(),[3250,300,350])
        window.load_project_path(folder)
        wait_until(app,tracks_ready)
        assert len(t.current_insertion()['points'])==2
        assert not errors,errors
    finally:
        wait_until(app,lambda:not window.busy)
        window.close()

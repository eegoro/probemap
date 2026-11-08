"""Four editable volumes, apply, reload and independent native label geometry."""
import copy
import pytest
from numpy.testing import assert_allclose

from probemap.registration.desktop_model import AlignmentProject
from probemap.registration.insertion_model import insertion_session
from test_calibration import annotated_project
from test_desktop_tracks import wait_until

pytestmark=pytest.mark.gui


def test_four_volume_calibration_apply_and_resume(tmp_path):
    from probemap.registration.macos_qt import initialize_qt
    from probemap.registration.desktop_app import ProbeMapWindow
    from probemap.registration.desktop_project import CalibrationDialog
    from qtpy.QtCore import QTimer
    from qtpy.QtWidgets import QApplication
    app=initialize_qt()
    p,s,item,_=annotated_project(tmp_path)
    p.display['step']=0
    path=p.save(p.output_dir/'project.json')
    w=ProbeMapWindow()
    errors=[]
    w._error=errors.append
    w.show()
    def ready(): return not w.busy and not w._render_running and not w.render_timer.isActive()
    try:
        w.load_project_path(path)
        wait_until(app,ready)
        assert w.project_page.mode=='summary'
        values={'reference':(3,4,5),'labels':(1,2,3),'moving':(4,3,6),'track':(5,6,7)}
        def edit():
            dialog=QApplication.activeModalWidget()
            assert isinstance(dialog,CalibrationDialog)
            assert set(dialog.fields)==set(values)
            assert not dialog.apply_button.isEnabled()
            for key,spacing in values.items():
                assert all(field.isEnabled() for field in dialog.fields[key])
                for field,value in zip(dialog.fields[key],spacing[::-1]): field.setValue(value)
            assert dialog.values()==values and dialog.apply_button.isEnabled()
            dialog.grab().save(str(tmp_path/'calibration.png'))
            dialog.apply_button.click()
        QTimer.singleShot(80,edit)
        w.project_page.calibration_button.click()
        wait_until(app,ready)
        assert w.step==0 and w.project.fit_is_current
        for key in ('reference','moving','track'):
            assert getattr(w.project,key).spacing_um_zyx==values[key]
            assert ' / '.join(str(v) for v in values[key][::-1]) in w.project_page.summary_details[key].text()
        assert w.project.label_spacing==values['labels']
        assert '3 / 2 / 1' in w.project_page.summary_details['labels'].text()
        saved=AlignmentProject.load(path)
        assert saved.label_spacing==values['labels']
        assert insertion_session(saved)['id']==s['id']
        assert len(insertion_session(saved)['insertions'][0]['points'])==len(item['points'])
        w.go_step(4)
        t=w.track_workspace
        def workspace_ready(): return ready() and t._cache_current() and not t._running and not t.render_timer.isActive()
        wait_until(app,workspace_ready)
        assert t.session['id']==s['id']
        assert t.track_native.spacing_um_zyx==values['track']
        assert t.brain_native.spacing_um_zyx==values['moving']
        t._ensure_labels()
        wait_until(app,lambda:workspace_ready() and t.labels is not None)
        assert_allclose(t.labels.spacing_um_zyx,values['labels'])
        # Optional missing inputs are visible but cannot be calibrated.
        missing=AlignmentProject(p.reference,p.moving,tmp_path/'empty')
        dialog=CalibrationDialog(missing)
        assert all(not field.isEnabled() for key in ('labels','track') for field in dialog.fields[key])
        dialog.close()
        metadata=copy.deepcopy(p)
        metadata.labels_spacing_um_zyx=(2.000000001,3,4)
        dialog=CalibrationDialog(metadata)
        assert not dialog.apply_button.isEnabled()
        dialog.close()
        assert not errors,errors
    finally:
        wait_until(app,lambda:not w.busy)
        w.close()

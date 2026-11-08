"""Native welcome layout, one-click reopening and local preference persistence."""
import numpy as np
import pytest

from probemap.recent_projects import RecentProjects
from probemap.registration.desktop_model import AlignmentProject,ImageSpec
from probemap.registration.volumes import write_volume
from test_desktop_tracks import wait_until

pytestmark=pytest.mark.gui


def test_recent_projects_open_resume_and_handle_removed_folders(tmp_path):
    from probemap.registration.macos_qt import initialize_qt
    app=initialize_qt()
    from probemap.registration.desktop_app import ProbeMapWindow
    from qtpy.QtCore import QPoint,QSettings,Qt
    from qtpy.QtTest import QTest
    settings=QSettings(str(tmp_path/'preferences.ini'),QSettings.Format.IniFormat)
    recent=RecentProjects(settings)
    data=np.random.default_rng(6).integers(0,50000,(12,14,16),dtype=np.uint16)
    source=write_volume(tmp_path/'brain.tif',data,(25,25,25))
    paths=[]
    for name in ('PV233','Study B','PV234','Study C','A project with a longer name'):
        folder=tmp_path/name/'outputs'
        folder.mkdir(parents=True)
        p=AlignmentProject(ImageSpec(source,data.shape,(25,25,25)),ImageSpec(source,data.shape,(25,25,25)),folder)
        p.display={'step':0}
        path=p.save(folder)
        recent.remember(path)
        paths.insert(0,path)
    errors=[]
    w=ProbeMapWindow(recent_projects=recent);w.resize(1080,720);w.show();w._error=errors.append
    def ready(): return not w.busy and not w._render_running and not w.render_timer.isActive()
    try:
        app.processEvents()
        page=w.project_page
        assert page.mode=='welcome' and page.recent_section.isVisible()
        assert len(page.recent_buttons)==5 and w.height()==720
        assert page.open_button.isVisible() and page.new_button.isVisible()
        for button,path in zip(page.recent_buttons,paths):
            assert button.project_path==path
            assert button.path_label.value==str(path.parent)
            corner=button.mapTo(w,QPoint(button.width(),button.height()))
            assert corner.x()<=w.width() and corner.y()<w.height()-20
            assert button.isVisible() and button.width()>300
        w.grab().save(str(tmp_path/'recent-projects-welcome.png'))
        chosen=paths[2]
        QTest.mouseClick(page.recent_buttons[2],Qt.MouseButton.LeftButton)
        wait_until(app,lambda:ready() and w.project is not None)
        assert w.project_path==chosen and page.mode=='summary'
        assert page.summary_recent_section.isVisible() and len(page.summary_recent_buttons)==5
        assert not w.settings_button.isVisible() and not w.export_button.isVisible() and not w.save_state.isVisible()
        w._refresh_file_menu()
        assert len(w.recent_menu.actions())==5 and w.export_action.isEnabled()
        assert recent.entries()[0]==chosen and len(recent.entries())==5
        w.go_step(2);wait_until(app,lambda:ready() and w.placed is not None)
        assert w.workspace_title.text()==w.nav_buttons[2].text()=='Registration'
        w.footer_placement_save.click()
        assert RecentProjects(settings).entries()[0]==chosen
        assert not errors,errors
    finally:
        wait_until(app,lambda:not w.busy)
        w.close()
    reopened_settings=QSettings(str(tmp_path/'preferences.ini'),QSettings.Format.IniFormat)
    again=ProbeMapWindow(recent_projects=RecentProjects(reopened_settings));again.show();again._error=errors.append
    try:
        app.processEvents()
        page=again.project_page
        assert page.mode=='welcome' and page.recent_buttons[0].project_path==chosen
        # A folder may disappear between displaying the list and clicking it.
        missing=page.recent_buttons[-1].project_path
        missing.unlink()
        QTest.mouseClick(page.recent_buttons[-1],Qt.MouseButton.LeftButton)
        assert page.mode=='welcome' and again.project is None
        assert len(page.recent_buttons)==4 and missing not in again.recent_projects.entries()
        bad=tmp_path/'bad.json';bad.write_text('{}')
        before=again.recent_projects.entries()
        again.load_project_path(bad)
        assert errors and again.recent_projects.entries()==before
        errors.clear()
        page.recent_buttons[0].setFocus()
        QTest.keyClick(page.recent_buttons[0],Qt.Key.Key_Space)
        wait_until(app,lambda:not again.busy and again.placed is not None)
        assert again.project_path==chosen and again.step==2
        assert again.workspace_title.text()=='Registration'
        assert not errors,errors
    finally:
        wait_until(app,lambda:not again.busy)
        again.close()

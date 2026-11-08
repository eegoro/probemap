"""View-only rotations, reversible edits and accessible review diagnostics."""
import copy

import numpy as np
import pytest
from numpy.testing import assert_allclose

from probemap.registration.desktop_model import AlignmentProject
from probemap.registration.landmarks import transform_points
from test_desktop_app import wait_until
from test_landmarks_ui import make_project

pytestmark=pytest.mark.gui


def test_workspace_rotations_residuals_clear_undo_panels_and_resume(tmp_path):
    from probemap.registration.macos_qt import initialize_qt
    from probemap.registration.desktop_app import ProbeMapWindow
    from qtpy.QtCore import QPointF,Qt
    from qtpy.QtWidgets import QLabel,QTabWidget
    app=initialize_qt()
    p=make_project(tmp_path)
    raw=np.array([[300,400,500],[1200,400,500],[300,1600,500],[300,400,2100],
                  [1000,1700,2300],[1400,1000,1700],[800,1200,1300.]])
    placed=transform_points(raw,p.coarse)
    noise=np.array([[10,-20,30],[20,40,-10],[50,10,10],[20,-40,30],[-60,20,10],[-20,-30,40],[10,30,-40]])
    p.pairs=[dict(reference=(x+noise[i]+[20,-30,40]).tolist(),moving=x.tolist()) for i,x in enumerate(placed)]
    p.fit()
    original=copy.deepcopy(p.as_record())
    w=ProbeMapWindow(); w.resize(1450,960); w.show()
    errors=[]; w._error=errors.append
    def ready(): return not w.busy and not w._render_running and not w.render_timer.isActive()
    try:
        w._load_project_volumes(p,2); wait_until(app,lambda:ready() and w.placed is not None)
        w.find_landmark(0)
        for axis in range(3):
            ref=w.landmark_panes['reference'][axis]
            mov=w.landmark_panes['moving'][axis]
            for angle in (90,180,270,0):
                mov.rotate_button.click()
                assert ref.canvas.view_rotation==mov.canvas.view_rotation==angle
                for name,pane in [('reference',ref),('moving',mov)]:
                    canvas=pane.canvas
                    pixel=canvas._pixel(p.pairs[0][name])
                    assert_allclose([pixel.x(),pixel.y()],[canvas.width()/2,canvas.height()/2],atol=1)
                    world=canvas._world(pixel)
                    others=[a for a in range(3) if a!=axis]
                    assert_allclose(world[others],np.asarray(p.pairs[0][name])[others],atol=1e-9)
                app.processEvents()
            mov.rotate_button.click()
            mov.fit_button.click()
            assert ref.canvas.view_record()==mov.canvas.view_record()==dict(zoom=1.,pan=[0.,0.],rotation=0)
        w.landmark_panes['moving'][0].rotate_button.click()
        # New annotations use unrotated physical coordinates even while the
        # images are rotated. Undo restores the accepted fit exactly.
        canvas=w.landmark_panes['moving'][0].canvas
        center=QPointF(canvas.width()/2,canvas.height()/2)
        world=canvas._world(center)
        canvas.pointClicked.emit(world)
        assert_allclose(p.pairs[-1]['moving'],world)
        w.undo_points(); assert p.fit_is_current
        w.landmark_tools.clear_button.click()
        assert p.pairs==[] and not p.fit_is_current
        w.undo_points(); assert p.pairs==original['pairs'] and p.fit_is_current
        width=w.landmark_views.width()
        w.panel_toggle.click(); app.processEvents()
        assert not w.controls_stack.isVisible() and w.landmark_views.width()>width
        w.go_step(1); wait_until(app,ready)
        assert w.controls_stack.isVisible()
        pane=w.overlay_panes[0]
        pane.rotate_button.click()
        assert all(c.view_rotation==90 for c in pane.canvases)
        pane.fit_button.click()
        assert all(c.view_rotation==0 for c in pane.canvases)
        w.panel_toggle.click(); app.processEvents()
        w.go_step(3); wait_until(app,ready)
        assert w.controls_stack.isVisible()
        assert all('independent accuracy' not in label.text() for label in w.review_tools.findChildren(QLabel))
        assert 'atlas voxels' in w.review_tools.scale_context.text()
        assert w.review_tools.report['rms_reduction_percent'] is not None
        w.review_tools.more_button.click(); app.processEvents()
        dialog=w.review_tools.residual_dialog
        values=[dialog.table.item(i,2).data(Qt.ItemDataRole.UserRole) for i in range(dialog.table.rowCount())]
        assert values==sorted(values,reverse=True)
        pair=int(dialog.table.item(0,0).text())-1
        assert pair+1==w.review_tools.report['largest_pair']
        dialog.grab().save(str(tmp_path/'residual-table.png'))
        dialog.open_pair(0,2); wait_until(app,lambda:ready() and w.placed is not None)
        assert w.step==2 and w._selected_landmark==pair
        assert not w.controls_stack.isVisible() # This step remembers its collapsed panel.
        w.go_step(3); wait_until(app,ready)
        w.preview_tabs.setCurrentIndex(1); wait_until(app,ready)
        w.slice_comparison.rotate_buttons[1].click()
        assert all(c.view_rotation==90 for c in w.slice_comparison.canvases)
        w.volume_view.show_axes.setChecked(False)
        assert not w.volume_view.axis_line.visible and not w.volume_view.axis_text.visible
        w.volume_view.flip_checks[0].setChecked(True)
        assert w.volume_view.view.camera.flip==(True,False,False)
        w.volume_view.zoom_volume(2)
        before=w.volume_view.view.camera.scale_factor
        w.volume_view.fit_button.click()
        assert w.volume_view.view.camera.scale_factor!=before
        assert w.volume_view.view.camera.flip==(True,False,False)
        w.panel_toggle.click(); app.processEvents()
        saved=w.save_project()
        again=AlignmentProject.load(saved)
        assert again.display['volume_axes_3d'] is False
        assert again.display['volume_flip_xyz']==[True,False,False]
        assert again.display['workspace_panels']=={'1':False,'2':False,'3':False}
        assert again.display['slice_comparison']['views'][0]['rotation']==90
        for name in ('pairs','coarse','residual','fit_result'):
            assert again.as_record()[name]==original[name]
        resumed=ProbeMapWindow();resumed.show();resumed._error=errors.append
        try:
            resumed.load_project_path(saved);wait_until(app,lambda:not resumed.busy and resumed.reference is not None
                and not resumed._render_running and not resumed.render_timer.isActive())
            assert resumed.step==3 and not resumed.controls_stack.isVisible()
            assert all(c.view_rotation==90 for c in resumed.slice_comparison.canvases)
            assert not resumed.volume_view.axis_line.visible
            assert resumed.volume_view.view.camera.flip==(True,False,False)
            resumed.panel_toggle.click();app.processEvents()
            assert resumed.controls_stack.isVisible()
            resumed.resize(1080,720);app.processEvents()
            for step in (1,2,3):
                resumed.go_step(step);wait_until(app,lambda:not resumed.busy and not resumed._render_running and not resumed.render_timer.isActive())
                if not resumed.controls_stack.isVisible(): resumed.panel_toggle.click()
                tabs=resumed.controls_stack.currentWidget()
                for tab in range(QTabWidget.count(tabs)):
                    tabs.setCurrentIndex(tab);app.processEvents()
                    assert tabs.widget(tab).horizontalScrollBar().maximum()==0,(step,tab)
            resumed.grab().save(str(tmp_path/'flat-review-minimum.png'))
        finally:
            wait_until(app,lambda:not resumed.busy);resumed.close()
        assert not errors,errors
    finally:
        wait_until(app,lambda:not w.busy);w.close()

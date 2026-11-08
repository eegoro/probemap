"""Minimal placement layout, deliberate physical edits and view-only navigation."""
import copy
import numpy as np
import pytest
from numpy.testing import assert_allclose

from probemap.registration.desktop_model import AlignmentProject, ImageSpec
from probemap.registration.landmarks import transform_points
from probemap.registration.volumes import write_volume
from test_desktop_tracks import wait_until

pytestmark=pytest.mark.gui


def test_placement_tools_navigation_apply_rendering_and_resume(tmp_path):
    from probemap.registration.macos_qt import initialize_qt
    from probemap.registration.desktop_app import ProbeMapWindow
    from qtpy.QtCore import QEvent, QPoint, QPointF, Qt
    from qtpy.QtGui import QMouseEvent,QWheelEvent
    from qtpy.QtTest import QTest
    from qtpy.QtWidgets import QPushButton,QStyleOptionViewItem
    app=initialize_qt()
    shape=(25,30,35); spacing=(200,180,160)
    z,y,x=np.indices(shape)
    data=(50000*np.exp(-((z-12)**2+(y-15)**2+(x-17)**2)/50)).astype(np.uint16)
    brain=write_volume(tmp_path/'brain.tif',data,spacing)
    track=write_volume(tmp_path/'electrodes.tif',data//2,spacing)
    atlas=write_volume(tmp_path/'atlas.tif',data,(160,180,220))
    p=AlignmentProject(ImageSpec(atlas,shape,(160,180,220)),ImageSpec(brain,shape,spacing),tmp_path/'out',
                       track=ImageSpec(track,shape,spacing))
    anchors=[[400,600,800],[4000,600,800],[400,4500,800],[400,600,4500]]
    p.pairs=[dict(reference=a,moving=a) for a in anchors]; p.fit()
    masks=tmp_path/'labels'; masks.mkdir()
    write_volume(masks/'1_brain.tif',(data>1000).astype(np.uint8),(160,180,220))
    p.labels_dir=masks
    w=ProbeMapWindow(); w.resize(1450,960); w.show()
    errors=[]; w._error=errors.append
    def ready(): return not w.busy and not w._render_running and not w.render_timer.isActive() and w.overlay_grid is not None
    def wheel(widget):
        pos=QPointF(widget.width()*.6,widget.height()*.5)
        event=QWheelEvent(pos,QPointF(widget.mapToGlobal(pos.toPoint())),QPoint(),QPoint(0,120),
                         Qt.MouseButton.NoButton,Qt.KeyboardModifier.NoModifier,Qt.ScrollPhase.NoScrollPhase,False)
        app.sendEvent(widget,event)
    def popup_fits(combo,snapshot=None):
        before=combo.currentIndex()
        combo.showPopup(); app.processEvents()
        try:
            view=combo.view()
            assert view.isVisible()
            option=QStyleOptionViewItem(); option.initFrom(view)
            for row in range(combo.count()):
                index=combo.model().index(row,combo.modelColumn(),combo.rootModelIndex())
                assert view.visualRect(index).width()>=view.itemDelegate().sizeHint(option,index).width(),combo.itemText(row)
            assert view.horizontalScrollBar().maximum()==0
            if snapshot: view.window().grab().save(str(tmp_path/snapshot))
            QTest.keyClick(view,Qt.Key.Key_Escape)
        finally:
            combo.hidePopup(); app.processEvents()
        assert combo.currentIndex()==before
    def drag(widget,delta,button=Qt.MouseButton.LeftButton,modifiers=Qt.KeyboardModifier.NoModifier):
        start=widget.image_rect().center()
        for kind,pos,b,buttons in ((QEvent.Type.MouseButtonPress,start,button,button),
            (QEvent.Type.MouseMove,start+delta,Qt.MouseButton.NoButton,button),
            (QEvent.Type.MouseButtonRelease,start+delta,button,Qt.MouseButton.NoButton)):
            app.sendEvent(widget,QMouseEvent(kind,pos,QPointF(widget.mapToGlobal(pos.toPoint())),b,buttons,modifiers))
    try:
        w._load_project_volumes(p,1); wait_until(app,ready)
        tabs=w.placement_tools; view=w.volume_view
        assert [tabs.tabText(i) for i in range(tabs.count())]==['Position','3D','Slices']
        assert w.workspace_title.text()==w.nav_buttons[1].text()=='Placement'
        assert not w.source_summary.isVisible() and not w.save_button.isVisible()
        assert [b.text() for b in w.footer.findChildren(QPushButton)]==['Back','Apply','Save','Next']
        assert all(pane.label.text()==title for pane,title in zip(w.overlay_panes,('XY','XZ','YZ')))
        assert view.heading.text()=='3D'
        assert all(not pane.canvas.zoom_controls.isVisible() for pane in w.overlay_panes)
        panels=[*w.overlay_panes,view]
        assert max(v.width() for v in panels)-min(v.width() for v in panels)<=1
        assert max(v.height() for v in panels)-min(v.height() for v in panels)<=1
        assert not any(b.isVisible() and ('Fit' in b.text() or 'Save placed' in b.text()) for b in w.findChildren(QPushButton))
        matrix=p.coarse.copy()
        # macOS's native menu used to clip even the three-letter axis orders.
        # Check the painted popup rows at both the minimum and normal window size.
        for width in (1080,1450):
            w.resize(width,960); app.processEvents()
            popup_fits(w.axis_order,f'axis-order-{width}.png')
            assert tabs.widget(0).horizontalScrollBar().maximum()==0
        assert_allclose(p.coarse,matrix)
        for slider,spin,_ in w.pose_controls.values():
            before=(slider.value(),spin.value()); wheel(slider); wheel(spin)
            assert (slider.value(),spin.value())==before
        assert_allclose(p.coarse,matrix)
        pane=w.overlay_panes[0]; canvas=pane.canvas
        slices=[pane.slider.value() for pane in w.overlay_panes]
        point=QPointF(canvas.width()*.6,canvas.height()*.5)
        anchor=canvas._world(point)
        wheel(canvas)
        assert canvas.zoom>1 and [pane.slider.value() for pane in w.overlay_panes]==slices
        assert_allclose(canvas._world(point),anchor,atol=1e-8)
        wheel(pane.slider)
        assert pane.slider.value()==slices[0]
        pan=QPointF(canvas.pan); drag(canvas,QPointF(24,17))
        assert canvas.pan!=pan
        assert_allclose(p.coarse,matrix)
        drag(canvas,QPointF(0,-20),Qt.MouseButton.RightButton)
        assert_allclose(p.coarse,matrix)
        drag(canvas,QPointF(15,9),Qt.MouseButton.MiddleButton)
        wait_until(app,ready)
        assert not np.allclose(p.coarse,matrix)

        # Only registration channels in placement; auxiliary review selections
        # remain available when returning to review.
        p.pairs=[dict(reference=a,moving=a) for a in anchors]; p.fit()
        w.go_step(3); wait_until(app,ready)
        view.layers.checks[2].setChecked(True); view.layers.labels_check.setChecked(True)
        wait_until(app,lambda:ready() and len(view.visuals)==3 and len(view.label_visuals)==1)
        assert view.visuals[2].visible
        w.go_step(1); wait_until(app,ready)
        tabs.setCurrentIndex(1); app.processEvents()
        assert tabs.isAncestorOf(view.layers) and tabs.isAncestorOf(view.plane_controls)
        assert view.layers.checks[0].isVisible() and view.layers.checks[1].isVisible()
        assert not view.layers.checks[2].isVisible() and not view.layers.labels_check.isVisible()
        assert not view.visuals[2].visible and not any(v.visible for v in view.label_visuals.values())
        assert not view.settings_button.isVisible() and tabs.rendering.tabs.count()==2
        controls=tabs.rendering.controls[0]
        for key in ('rendering','blending','colormap','interpolation'):
            popup_fits(controls[key],f'popup-{key}.png')
        controls['rendering'].setCurrentIndex(controls['rendering'].findData('average'))
        assert view.visuals[0].method=='average'
        view.layers.sliders[0].setValue(42)
        assert_allclose(view.visuals[0].opacity,.42)
        w.grab().save(str(tmp_path/'placement-3d.png'))

        tabs.setCurrentIndex(2); app.processEvents()
        assert tabs.widget(2).horizontalScrollBar().maximum()==0
        display=tabs.slice_display
        for combo in (display.channel,display.interpolation,display.colormap,
                      *(pane.mode_selector for pane in w.overlay_panes)):
            popup_fits(combo)
        before=canvas.image.copy()
        display.gamma.setValue(2); display.colormap.setCurrentIndex(display.colormap.findData('cyan'))
        wait_until(app,ready)
        assert display.records[0]['gamma']==2 and display.records[0]['colormap']=='cyan'
        assert canvas.image!=before
        assert_allclose(p.moving.spacing_um_zyx,spacing)
        for pane in w.overlay_panes: pane.mode_selector.setCurrentIndex(1)
        wait_until(app,ready)
        w.grab().save(str(tmp_path/'placement-slices.png'))
        tabs.setCurrentIndex(0)
        w.pose_controls['rotation',1][1].setValue(32)
        w.pose_controls['scale',2][1].setValue(.8)
        # A partial scan is deliberately offset from the atlas centre. Apply
        # changes neither its physical transform nor the current view/depth.
        w.pose_controls['translation',0][1].setValue(470)
        w.pose_controls['translation',1][1].setValue(-320)
        matrix=p.coarse.copy(); pairs=copy.deepcopy(p.pairs)
        positions=[pane.world_depth for pane in w.overlay_panes]
        views=[[canvas.view_record() for canvas in pane.canvases] for pane in w.overlay_panes]
        camera=view.camera_record()
        w.placement_apply.click(); wait_until(app,ready)
        native_center=(np.array(shape)-1)*spacing/2
        atlas_center=(np.array(shape)-1)*p.reference.spacing_um_zyx/2
        assert not np.allclose(transform_points([native_center],p.coarse)[0],atlas_center)
        assert_allclose(p.coarse,matrix,atol=1e-10)
        assert p.pairs==pairs
        assert_allclose([pane.world_depth for pane in w.overlay_panes],positions)
        assert [[canvas.view_record() for canvas in pane.canvases] for pane in w.overlay_panes]==views
        assert view.camera_record()==camera
        assert_allclose(view.matrix,matrix)
        w.grab().save(str(tmp_path/'placement-position.png'))
        w.footer_placement_save.click()
        saved=AlignmentProject.load(p.output_dir)
        assert saved.display['placement_slice_appearance']==display.records
        w.load_project_path(p.output_dir); wait_until(app,ready)
        assert_allclose(w.project.coarse,p.coarse)
        assert w.placement_tools.slice_display.records==display.records
        assert w.volume_view.layers.sliders[0].value()==42
        assert not errors,errors
    finally:
        wait_until(app,lambda:not w.busy)
        w.close()

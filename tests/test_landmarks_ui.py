"""Native editing, matching, navigation and persistence of landmark pairs."""
import copy

import numpy as np
import pytest
from numpy.testing import assert_allclose

from probemap.registration.coarse import pose_matrix
from probemap.registration.desktop_model import AlignmentProject,ImageSpec
from probemap.registration.landmarks import transform_points
from probemap.registration.volumes import write_volume
from test_desktop_app import wait_until

pytestmark=pytest.mark.gui


def make_project(tmp_path):
    shape=(31,35,39); spacing=(60,80,100)
    z,y,x=np.indices(shape)
    data=(50000*np.exp(-((z-15)**2+(y-17)**2+(x-19)**2)/110)).astype(np.uint16)
    path=write_volume(tmp_path/'scan.tif',data,spacing)
    p=AlignmentProject(ImageSpec(path,shape,spacing),ImageSpec(path,shape,spacing),tmp_path/'out')
    p.coarse=pose_matrix(shape,spacing,(120,-80,100),(5,-7,9),(.9,1.1,.95))
    return p


def test_bulk_landmarks_select_delete_undo_apply_save_and_resume(tmp_path):
    from probemap.registration.macos_qt import initialize_qt
    app=initialize_qt()
    from probemap.registration.desktop_app import ProbeMapWindow
    from qtpy.QtCore import QPoint,QPointF,Qt
    from qtpy.QtGui import QWheelEvent
    from qtpy.QtTest import QTest
    from qtpy.QtWidgets import QPushButton,QTableWidget
    w=ProbeMapWindow();w.resize(1450,960);w.show()
    errors=[];w._error=errors.append
    p=make_project(tmp_path)
    coarse=p.coarse.copy()
    moving=transform_points([[300,400,500],[1200,400,500],[300,1600,500],[300,400,2100],[1000,1700,2300]],coarse)
    fine=pose_matrix(p.reference.shape_zyx,p.reference.spacing_um_zyx,(24,-18,32),(2,-3,4),(1.04,.96,1.02))
    reference=transform_points(moving,fine)
    def ready(): return not w.busy and not w._render_running and not w.render_timer.isActive()
    try:
        w._load_project_volumes(p,2);wait_until(app,lambda:ready() and w.placed is not None)
        assert w.workspace_title.text()==w.nav_buttons[2].text()=='Registration'
        assert not w.source_summary.isVisible() and not w.save_button.isVisible()
        assert [b.text() for b in w.footer.findChildren(QPushButton)]==['Back','Apply','Save','Next']
        assert [w.landmark_tools.tabText(i) for i in range(2)]==['Transformation','Visualization']
        assert not w.landmark_tools.findChildren(QTableWidget)
        assert not w.landmark_views.findChildren(QPushButton)
        assert len(w.landmark_views.tool_buttons)==3
        assert p.pairs==[] and not w.next_button.isEnabled()
        for point in reference: w.add_point('reference',point)
        assert len(p.pairs)==5 and all(pair['moving'] is None for pair in p.pairs)
        assert 'Atlas 5 / Scan 0' in w.landmark_tools.count.text()
        assert not w.landmark_apply.isEnabled()
        w.footer_placement_save.click()
        w.load_project_path(p.output_dir);wait_until(app,lambda:ready() and w.placed is not None)
        p=w.project
        assert len(p.pairs)==5 and all(pair['moving'] is None for pair in p.pairs)
        for point in moving: w.add_point('moving',point)
        assert_allclose([pair['reference'] for pair in p.pairs],reference)
        assert_allclose([pair['moving'] for pair in p.pairs],moving)
        assert w.landmark_tools.count.text()=='5 pairs' and w.landmark_apply.isEnabled()
        pairs=copy.deepcopy(p.pairs)

        # Searching is a real keyboard action and locates both members of a
        # pair in all six views; no registration or image resampling occurs.
        search=w.landmark_tools.search
        search.setFocus();QTest.keyClicks(search,'3');QTest.keyClick(search,Qt.Key.Key_Return)
        for name,panes in w.landmark_panes.items():
            for pane in panes:
                assert any(number==3 and active for number,_,active in pane.canvas.visible_points())
                pixel=pane.canvas._pixel(pairs[2][name])
                assert_allclose([pixel.x(),pixel.y()],[pane.canvas.width()/2,pane.canvas.height()/2])
        assert w._selected_landmark==2 and w._landmark_tool=='select'
        canvas=w.landmark_panes['reference'][0].canvas
        QTest.keyClick(canvas,Qt.Key.Key_Backspace)
        assert p.pairs==pairs[:2]+pairs[3:]
        assert w.landmark_tools.count.text()=='4 pairs'
        assert [number for number,_,_ in canvas.points]==[1,2,3,4]
        QTest.keyClick(canvas,Qt.Key.Key_Z,Qt.KeyboardModifier.ControlModifier)
        assert p.pairs==pairs

        # A genuine click on Scan can start a new pair; selecting that marker
        # and pressing Delete removes the whole pair without altering others.
        w.landmark_views.tool_buttons['add'].click()
        canvas=w.landmark_panes['moving'][0].canvas
        canvas.fit_image()
        rect=canvas.image_rect()
        pixel=QPoint(round(rect.left()+rect.width()*.71),round(rect.top()+rect.height()*.38))
        clicked=canvas._world(QPointF(pixel))
        QTest.mouseClick(canvas,Qt.MouseButton.LeftButton,pos=pixel)
        assert len(p.pairs)==6 and p.pairs[-1]['reference'] is None
        assert_allclose(p.pairs[-1]['moving'],clicked)
        w.landmark_views.tool_buttons['select'].click()
        QTest.mouseClick(canvas,Qt.MouseButton.LeftButton,pos=canvas._pixel(clicked).toPoint())
        assert w._selected_landmark==5
        QTest.keyClick(canvas,Qt.Key.Key_Delete)
        assert p.pairs==pairs

        # View gestures must never add points, change depth or alter placement.
        w.landmark_views.tool_buttons['pan'].click()
        before=canvas.view_record();slices=[q.slider.value() for v in w.landmark_panes.values() for q in v]
        start=QPoint(canvas.width()//2,canvas.height()//2)
        QTest.mousePress(canvas,Qt.MouseButton.LeftButton,pos=start)
        QTest.mouseMove(canvas,start+QPoint(20,15))
        QTest.mouseRelease(canvas,Qt.MouseButton.LeftButton,pos=start+QPoint(20,15))
        assert canvas.view_record()!=before
        pos=QPointF(canvas.width()*.6,canvas.height()*.5)
        anchor=canvas._world(pos);zoom=canvas.zoom
        event=QWheelEvent(pos,QPointF(canvas.mapToGlobal(pos.toPoint())),QPoint(),QPoint(0,120),
            Qt.MouseButton.NoButton,Qt.KeyboardModifier.NoModifier,Qt.ScrollPhase.NoScrollPhase,False)
        app.sendEvent(canvas,event)
        assert canvas.zoom>zoom
        assert_allclose(canvas._world(pos),anchor)
        assert [q.slider.value() for v in w.landmark_panes.values() for q in v]==slices
        assert p.pairs==pairs
        assert_allclose(p.coarse,coarse)
        w.landmark_views.tool_buttons['add'].click()
        QTest.keyPress(canvas,Qt.Key.Key_Space)
        QTest.mousePress(canvas,Qt.MouseButton.LeftButton,pos=start)
        QTest.mouseMove(canvas,start+QPoint(15,10))
        QTest.mouseRelease(canvas,Qt.MouseButton.LeftButton,pos=start+QPoint(15,10))
        QTest.keyRelease(canvas,Qt.Key.Key_Space)
        assert canvas.cursor().shape()==Qt.CursorShape.CrossCursor and p.pairs==pairs
        w.landmark_views.tool_buttons['pan'].click()

        w.landmark_apply.click()
        assert w.step==2 and p.fit_is_current and w.next_button.isEnabled()
        assert_allclose(p.total,fine@coarse,atol=1e-10)
        w.fit_model.setCurrentText('Rigid')
        assert not w.next_button.isEnabled()
        w.fit_model.setCurrentText('Affine')
        assert w.next_button.isEnabled()
        w.grab().save(str(tmp_path/'landmarks-transformation.png'))
        w.footer_placement_save.click()
        w.next_button.click();wait_until(app,ready)
        assert w.step==3
        w.go_step(2);wait_until(app,ready)
        w.footer_placement_save.click()
        saved=AlignmentProject.load(p.output_dir)
        assert saved.pairs==pairs and saved.fit_is_current
        w.load_project_path(p.output_dir);wait_until(app,lambda:ready() and w.placed is not None)
        assert w.project.pairs==pairs and w.project.fit_is_current
        assert w._landmark_tool=='pan'
        assert w.landmark_panes['moving'][0].canvas.view_record()==saved.display['landmark_slice_views']['moving'][0]
        assert not errors,errors
    finally:
        wait_until(app,lambda:not w.busy)
        w.close()


def test_shared_depths_individual_adjustments_and_visualization(tmp_path):
    from probemap.registration.macos_qt import initialize_qt
    app=initialize_qt()
    from probemap.registration.desktop_app import ProbeMapWindow
    from qtpy.QtCore import QPoint,Qt
    from qtpy.QtTest import QTest
    w=ProbeMapWindow();w.resize(1450,960);w.show()
    errors=[];w._error=errors.append
    p=make_project(tmp_path)
    def ready(): return not w.busy and not w._render_running and not w.render_timer.isActive()
    try:
        w._load_project_volumes(p,2);wait_until(app,lambda:ready() and w.placed is not None)
        for axis,shared in enumerate(w.landmark_views.shared_sliders):
            ref=w.landmark_panes['reference'][axis]
            mov=w.landmark_panes['moving'][axis]
            ref.slider.setValue(ref.slider.maximum()//2)
            mov.slider.setValue(ref.slider.value()+3)
            ref_depth,mov_depth=ref.world_depth,mov.world_depth
            QTest.keyClick(shared,Qt.Key.Key_Right)
            assert_allclose([ref.world_depth,mov.world_depth],np.array([ref_depth,mov_depth])+ref.spacing[axis])
            old_ref=ref.slider.value()
            assert mov.adjust_slider.isVisible() and not mov.slider.isVisible()
            assert mov.adjust_slider.mapTo(mov,QPoint()).y()+mov.adjust_slider.height()<=mov.canvas.geometry().top()
            assert mov.adjust_slider.minimum()==max(-15,-mov.slider.value())
            assert mov.adjust_slider.maximum()==min(15,mov.slider.maximum()-mov.slider.value())
            QTest.keyClick(mov.adjust_slider,Qt.Key.Key_Left)
            assert ref.slider.value()==old_ref
            assert mov.slider.value()-ref.slider.value()==2
            shared.setValue(shared.maximum())
            assert mov.slider.value()-ref.slider.value()==2
            assert mov.slider.value()==mov.slider.maximum()
            shared.setValue(shared.minimum())
            assert ref.slider.value()==0 and mov.slider.value()==2
        assert all(s.isVisible() for s in w.landmark_views.shared_sliders)
        w.link_slices.setChecked(False)
        assert all(not s.isVisible() for s in w.landmark_views.shared_sliders)
        w.link_slices.setChecked(True)
        panes=[p for v in w.landmark_panes.values() for p in v]
        assert max(p.height() for p in panes)-min(p.height() for p in panes)<=1
        assert max(p.width() for p in panes)-min(p.width() for p in panes)<=1
        assert [p.label.text() for p in panes]==['Atlas · XY','XZ','YZ','Brain scan','','']

        # The matching images stay grayscale; settings are independent and
        # survive reopening without changing measured coordinates or sizes.
        w.landmark_tools.setCurrentIndex(1);app.processEvents()
        depths=[pane.world_depth for pane in panes]
        w.landmark_tools.adjust_range.setValue(37)
        assert [pane.world_depth for pane in panes]==depths
        assert all(pane.adjust_range==37 and pane.adjust_slider.value()==0 for pane in panes)
        assert all(-37<=pane.adjust_slider.minimum()<=0<=pane.adjust_slider.maximum()<=37 for pane in panes)
        display=w.landmark_tools.display
        before=w.landmark_panes['reference'][0].canvas.image.copy()
        original=copy.deepcopy(p.as_record())
        display.gamma.setValue(.6)
        display.brightness.setValue(.1)
        display.invert.setChecked(True)
        display.interpolation.setCurrentIndex(1)
        assert display.records[1]['gamma']==1 and not display.records[1]['invert']
        assert w.landmark_panes['reference'][0].canvas.image!=before
        assert all(pane.canvas.image.isGrayscale() for pane in panes)
        assert all(not pane.canvas.smooth for pane in w.landmark_panes['reference'])
        assert p.as_record()['coarse']==original['coarse'] and p.pairs==original['pairs']
        assert p.reference.spacing_um_zyx==p.moving.spacing_um_zyx==(60,80,100)
        w.grab().save(str(tmp_path/'landmarks-visualization.png'))
        w.footer_placement_save.click()
        records=copy.deepcopy(display.records)
        w.load_project_path(p.output_dir);wait_until(app,lambda:ready() and w.placed is not None)
        assert w.landmark_tools.currentIndex()==1
        assert w.landmark_tools.display.records==records
        assert w.landmark_tools.adjust_range.value()==37
        assert all(pane.adjust_range==37 for group in w.landmark_panes.values() for pane in group)
        assert not errors,errors
    finally:
        wait_until(app,lambda:not w.busy)
        w.close()


def test_linked_landmark_navigation_preserves_pair_centers_and_slice_adjustments(tmp_path):
    from probemap.registration.macos_qt import initialize_qt
    app=initialize_qt()
    from probemap.registration.desktop_app import ProbeMapWindow
    from qtpy.QtCore import QPoint,QPointF,Qt
    from qtpy.QtGui import QWheelEvent
    from qtpy.QtTest import QTest
    from test_trackpad_zoom import pinch
    w=ProbeMapWindow();w.resize(1450,960);w.show()
    errors=[];w._error=errors.append
    p=make_project(tmp_path)
    # A nonzero atlas/scan displacement makes absolute-camera copying visibly
    # wrong after search, even though identical initial views could mask it.
    p.pairs=[dict(reference=[700,1000,1400],moving=[800,1100,1300])]
    def ready(): return not w.busy and not w._render_running and not w.render_timer.isActive()
    def wheel(canvas,delta):
        pos=QPointF(canvas.width()*.6,canvas.height()*.45)
        event=QWheelEvent(pos,QPointF(canvas.mapToGlobal(pos.toPoint())),QPoint(),QPoint(0,delta),
            Qt.MouseButton.NoButton,Qt.KeyboardModifier.NoModifier,Qt.ScrollPhase.NoScrollPhase,False)
        app.sendEvent(canvas,event)
    def pair_pixels(axis):
        return [w.landmark_panes[name][axis].canvas._pixel(p.pairs[0][name]) for name in ('reference','moving')]
    def same_pair_pixels(axis):
        a,b=pair_pixels(axis)
        assert_allclose([a.x(),a.y()],[b.x(),b.y()],atol=1)
    try:
        w._load_project_volumes(p,2);wait_until(app,lambda:ready() and w.placed is not None)
        original=copy.deepcopy(p.as_record())
        w.find_landmark(0)
        depths=[pane.world_depth for group in w.landmark_panes.values() for pane in group]
        for axis in range(3):
            ref=w.landmark_panes['reference'][axis].canvas
            mov=w.landmark_panes['moving'][axis].canvas
            same_pair_pixels(axis)
            wheel(ref,120)
            assert ref.zoom>1 and ref.zoom==mov.zoom
            same_pair_pixels(axis)
            wheel(mov,-120)
            assert_allclose([ref.zoom,mov.zoom],[1,1])
            same_pair_pixels(axis)
            pinch(app,mov,QPointF(mov.width()*.55,mov.height()*.45),[.08,.04],wrong_local=True)
            assert_allclose(ref.zoom,mov.zoom)
            same_pair_pixels(axis)
            w.landmark_views.tool_buttons['pan'].click()
            start=QPoint(ref.width()//2,ref.height()//2)
            before=pair_pixels(axis)
            QTest.mousePress(ref,Qt.MouseButton.LeftButton,pos=start)
            QTest.mouseMove(ref,start+QPoint(22,14))
            QTest.mouseRelease(ref,Qt.MouseButton.LeftButton,pos=start+QPoint(22,14))
            after=pair_pixels(axis)
            for a,b in zip(before,after): assert_allclose([b.x()-a.x(),b.y()-a.y()],[22,14],atol=1)
            same_pair_pixels(axis)
            QTest.mousePress(mov,Qt.MouseButton.RightButton,pos=start)
            QTest.mouseMove(mov,start+QPoint(0,18))
            QTest.mouseRelease(mov,Qt.MouseButton.RightButton,pos=start+QPoint(0,18))
            assert_allclose(ref.zoom,mov.zoom)
            same_pair_pixels(axis)
        assert [pane.world_depth for group in w.landmark_panes.values() for pane in group]==depths
        assert p.pairs==original['pairs'] and p.as_record()['coarse']==original['coarse']
        w.footer_placement_save.click()
        cameras={name:[pane.canvas.view_record() for pane in panes] for name,panes in w.landmark_panes.items()}
        w.load_project_path(p.output_dir);wait_until(app,lambda:ready() and w.placed is not None)
        assert {name:[pane.canvas.view_record() for pane in panes] for name,panes in w.landmark_panes.items()}==cameras
        ref=w.landmark_panes['reference'][0].canvas
        mov=w.landmark_panes['moving'][0].canvas
        ref.fit_image()
        assert ref.view_record()==mov.view_record()=={'zoom':1.,'pan':[0.,0.],'rotation':0}
        assert not errors,errors
    finally:
        wait_until(app,lambda:not w.busy)
        w.close()

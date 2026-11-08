"""Table editing, physical rulers and camera interaction share no annotation side effects."""
import copy

import numpy as np
import pytest
from numpy.testing import assert_allclose

from probemap.registration.insertion_model import normalize_insertion_names
from probemap.registration.desktop_measurements import distance_text
from probemap.registration.desktop_regions import region_side


def test_letter_migration_side_metadata_and_measurement_units():
    session={'insertions':[{'name':'Insertion 1'},{'name':'B'},{'name':'Medial'}]}
    normalize_insertion_names(session)
    assert [i['name'] for i in session['insertions']]==['A','B','Medial']
    normalize_insertion_names(session)
    assert [i['name'] for i in session['insertions']]==['A','B','Medial']
    assert distance_text(500)=='500.0 µm'
    assert distance_text(500,'mm')=='0.500 mm'
    assert distance_text(500,'nm')=='500000.0 nm'
    assert region_side({'mask_path':'101_mc_l_medial_cortex_mask.tif'})=='left'
    assert region_side({'mask_path':'201_mc_r_medial_cortex_mask.tif'})=='right'
    assert region_side({'name':'Midline'})=='unspecified'


@pytest.mark.gui
def test_annotation_tables_context_units_deselect_headers_and_camera(tmp_path):
    from probemap.registration.macos_qt import initialize_qt
    app=initialize_qt()
    from qtpy.QtCore import QPointF, Qt, QEvent
    from qtpy.QtGui import QMouseEvent
    from qtpy.QtTest import QTest
    from qtpy.QtWidgets import QApplication
    from probemap.registration.desktop_app import ProbeMapWindow
    from probemap.registration.desktop_model import AlignmentProject, ImageSpec
    from probemap.registration.coarse import pose_matrix
    from probemap.registration.landmarks import transform_points
    from probemap.registration.volumes import write_volume
    from probemap.registration.insertion_model import insertion_session,new_insertion
    from test_desktop_tracks import wait_until
    shape=(25,30,35); spacing=(160,200,240)
    z,y,x=np.indices(shape)
    data=(50000*np.exp(-((z-12)**2+(y-15)**2+(x-17)**2)/50)).astype(np.uint16)
    brain=write_volume(tmp_path/'brain.tif',data,spacing)
    track=write_volume(tmp_path/'track.tif',data//2,spacing)
    spec=ImageSpec(brain,shape,spacing)
    p=AlignmentProject(spec,spec,tmp_path/'output',track=ImageSpec(track,shape,spacing))
    p.coarse=pose_matrix(shape,spacing,(-300,500,700),(20,-30,40),(.6,1.4,.8))
    raw=np.array([[400,600,800],[3000,600,800],[400,5000,800],[400,600,6000],[2800,4000,5500.]])
    placed=transform_points(raw,p.coarse)
    affine=np.array([[.8,.1,0,300],[0,1.2,.15,-100],[.05,0,1.1,50],[0,0,0,1]])
    atlas=transform_points(placed,affine)
    p.pairs=[dict(reference=a.tolist(),moving=b.tolist()) for a,b in zip(atlas,placed)]
    p.fit(); p.display={'step':4}
    masks=tmp_path/'labels'; masks.mkdir()
    for identifier,side,mask in ((1,'l',x<12),(2,'r',x>=24),(3,'l',(x>=12)&(x<24))):
        write_volume(masks/f'{identifier}_region_{side}_mask.tif',mask.astype(np.uint8),spacing)
    p.labels_dir=masks
    session=insertion_session(p)
    new_insertion(session,'Insertion 1'); new_insertion(session,'B')
    session['insertion_view']={'point_size_px':6,'ui_version':3}
    path=p.save(tmp_path/'output/project.json')
    w=ProbeMapWindow(); w.resize(1480,980); errors=[]; w._error=errors.append; w.show()
    t=w.track_workspace
    def ready(): return not w.busy and t._cache_current() and not t._running and not t.render_timer.isActive()
    def drag(widget,start,end,modifiers=Qt.KeyboardModifier.NoModifier):
        QTest.mousePress(widget,Qt.MouseButton.LeftButton,modifiers,pos=start.toPoint())
        event=QMouseEvent(QEvent.Type.MouseMove,end,QPointF(widget.mapToGlobal(end.toPoint())),Qt.MouseButton.NoButton,Qt.MouseButton.LeftButton,modifiers)
        QApplication.sendEvent(widget,event)
        QTest.mouseRelease(widget,Qt.MouseButton.LeftButton,modifiers,pos=end.toPoint()); app.processEvents()
    try:
        w.load_project_path(path); wait_until(app,ready)
        assert [i['name'] for i in t.session['insertions']]==['A','B']
        assert t.point_size.value()==6
        # Tools belong to the title row; there is no empty toolbar row over the views.
        assert t.workspace_header.indexOf(t.tracing_toolbar)>=0
        assert t.tracing_toolbar.geometry().center().x()>t.width()*.7
        c=t.slices[0]
        QTest.mouseClick(t.title,Qt.MouseButton.LeftButton)
        assert not t.insertion_table.selectionModel().hasSelection() and t.insertion_id is None
        assert all(c.point_mode=='select' for c in t.slices)
        QTest.mouseClick(t.tracing_tools['add'],Qt.MouseButton.LeftButton)
        assert t.current_insertion()['name']=='A' and t.insertion_table.selectionModel().hasSelection()
        center=QPointF(c.rect().center())
        QTest.mouseClick(c,Qt.MouseButton.LeftButton,pos=center.toPoint())
        assert len(t.session['insertions'][0]['points'])==1
        point=t.session['insertions'][0]['points'][0]
        assert c._context_hit(center)==('point',point['id'])
        menu=t.point_context_menu(point['id'])
        targets=next(a.menu() for a in menu.actions() if a.text()=='Move to insertion')
        next(a for a in targets.actions() if a.text()=='B').trigger()
        assert not t.session['insertions'][0]['points'] and len(t.session['insertions'][1]['points'])==1
        t.clear_points_button.click(); assert not t.volume_view.point_records
        t.undo(); assert len(t.volume_view.point_records)==1
        displayed=c._world(center)
        t.measure_distance(displayed,displayed+[0,300,400])
        ruler=t.session['measurements'][0]
        assert_allclose(ruler['length_um'],500,atol=1e-9)
        assert abs(ruler['atlas_length_um']-500)>10
        assert t.measurement_panel.isVisible() and t.measurement_unit.currentText()=='µm'
        t.measurement_unit.setCurrentText('mm')
        assert t.measurement_table.item(0,1).text()=='0.500'
        assert_allclose(ruler['length_um'],500)
        t.measurement_table.item(0,2).setCheckState(Qt.CheckState.Unchecked)
        assert ruler['show_label'] is False
        t.measurement_table.item(0,0).setCheckState(Qt.CheckState.Unchecked)
        assert ruler['visible'] is False
        t.measurement_table.horizontalHeader().toggled.emit(0,True)
        assert ruler['visible'] is True
        t.measurement_table.selectRow(0); QTest.mouseClick(t.title,Qt.MouseButton.LeftButton)
        assert not t.measurement_table.selectionModel().hasSelection()
        t.measurement_context_menu(ruler['id']).actions()[0].trigger()
        assert not t.session['measurements']
        t.undo(); assert len(t.session['measurements'])==1
        # Regions are editable by ID even after sorting and filtering.
        t.sidebar_tabs.setCurrentIndex(1); t.region_group.setChecked(True)
        wait_until(app,lambda:ready() and t.labels is not None)
        table=t.region_table; header=table.horizontalHeader()
        table.sortItems(2,Qt.SortOrder.DescendingOrder)
        region=t._region_for_row(0)
        table.item(0,0).setCheckState(Qt.CheckState.Unchecked)
        assert t.session['regions'][str(region['id'])]['visible'] is False
        QTest.mouseClick(header.viewport(),Qt.MouseButton.LeftButton,pos=header._check_rect(0).center())
        assert all(t.session['regions'][str(r['id'])]['visible'] for r in t.labels.regions)
        QTest.mouseClick(header.viewport(),Qt.MouseButton.LeftButton,pos=header._check_rect(0).center())
        assert all(not t.session['regions'][str(r['id'])]['visible'] for r in t.labels.regions)
        t.region_side_filter.setCurrentIndex(t.region_side_filter.findData('left'))
        assert sum(not table.isRowHidden(row) for row in range(table.rowCount()))==2
        QTest.mouseClick(header.viewport(),Qt.MouseButton.LeftButton,pos=header._check_rect(0).center())
        assert all(t.session['regions'][str(r['id'])]['visible']==(region_side(r)=='left') for r in t.labels.regions)
        QTest.mouseClick(header.viewport(),Qt.MouseButton.LeftButton,pos=header._check_rect(1).center())
        assert sum(t.session['regions'].get(str(r['id']),{}).get('label',False) for r in t.labels.regions)==2
        assert all(table.item(row,3).background().style()==Qt.BrushStyle.NoBrush for row in range(table.rowCount()))
        table.setColumnWidth(2,115); table.setFixedHeight(180)
        # Select mode still permits camera rotation, pan and wheel zoom.
        t.sidebar_tabs.setCurrentIndex(0); t._tracing_tool_selected('select')
        native=t.volume_view.canvas.native
        before=t.volume_view.view.camera.get_state()['_quaternion']
        start=QPointF(25,25); end=start+QPointF(45,25)
        drag(native,start,end)
        after=t.volume_view.view.camera.get_state()['_quaternion']
        assert tuple(before.get_axis_angle())!=tuple(after.get_axis_angle())
        assert t.volume_view.view.camera.interactive and t.volume_view.selection_enabled
        w.save_project()
        saved=AlignmentProject.load(path).track_analysis['sessions'][0]['insertion_view']
        assert saved['measurement_unit']=='mm' and saved['region_table_height']==180
        assert saved['region_columns'][2]==115
        assert saved['region_side_filter']=='left'
        assert t.measurement_table.horizontalScrollBar().maximum()==0
        assert t.region_table.horizontalScrollBar().maximum()==0
        t.grab().save(str(tmp_path/'annotation-tables.png'))
        assert not errors,errors
    finally:
        wait_until(app,lambda:not w.busy); w.close()

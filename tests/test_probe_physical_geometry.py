import copy
import numpy as np
import pytest
from numpy.testing import assert_allclose
from probemap.probe_library import grid_geometry,validate_geometry,ProbeLibrary
from probemap.probe_geometry import (body_mesh,contact_mesh_local,new_contact,perimeter_contacts,
    polygon_valid,triangulate,polygon_area)
from probemap.registration.neuropixels_model import probe_settings,contact_coordinates,contact_mesh,shank_mesh


def physical(kind='polygon'):
    g=grid_geometry('Triangle',rows=1,first_site=0,width=1000,length=1000,thickness=30,site_width=10)
    g.update(geometry_version=2,body_shape=kind,contacts=[],body_polygon_um=[[0,0],[-500,1000],[500,1000]])
    return g


def test_triangle_perimeter_uses_real_side_faces_and_survives_json(tmp_path):
    g=physical()
    for spec in perimeter_contacts(g,15,face='edge'):
        xyz=spec.pop('xyz');g['contacts'].append(new_contact(g,xyz,width_um=10,height_um=8,**spec))
    validate_geometry(g)
    assert all(c['z_um']==0 for c in g['contacts'])
    for c in g['contacts']:
        vertices,faces=contact_mesh_local(c,g)
        assert np.ptp(vertices[:,2])==8
        assert len(faces)==2
        assert_allclose(np.dot(c['u_xyz'],c['v_xyz']),0)
    v,f=body_mesh(g);assert len(v)==6 and len(f)==8
    lib=ProbeLibrary(tmp_path);saved=lib.save(dict(geometry=g))
    assert lib.read(tmp_path/(saved['id']+'.json'))['geometry']==g
    # Face contacts need an inset; centres alone are not sufficient validation.
    bad=copy.deepcopy(g);bad['contacts']=[]
    for spec in perimeter_contacts(bad,12,face='front',offset=0):
        xyz=spec.pop('xyz');bad['contacts'].append(new_contact(bad,xyz,**spec))
    with pytest.raises(ValueError,match='outside'):validate_geometry(bad)


def test_concave_contour_and_crossing_contours():
    poly=[[0,0],[100,0],[100,100],[50,40],[0,100]]
    assert polygon_valid(poly)
    triangles=triangulate(poly)
    assert len(triangles)==3
    assert_allclose(sum(abs(polygon_area(np.array(poly)[f])) for f in triangles),abs(polygon_area(poly)))
    assert not polygon_valid([[0,0],[100,100],[0,100],[100,0]])
    g=physical();g['body_polygon_um']=poly;g['shank_length_um']=100
    v,f=body_mesh(g,[(0,50,90)])
    assert len(v)>0 and len(f)>0 and v[:,1].min()>=50 and v[:,1].max()<=90


def test_wire_terminal_and_curved_surfaces_preserve_orientation_in_native_space():
    g=grid_geometry('Wire',rows=1,first_site=0,width=100,thickness=100,length=2000,taper=0,body='wire',site_width=100)
    g.update(geometry_version=2,contacts=[])
    tip=new_contact(g,[0,0,0],surface='terminal',shape='circle',width_um=100,height_um=100,u_xyz=[1,0,0],v_xyz=[0,0,1]);g['contacts'].append(tip)
    validate_geometry(g)
    v,f=contact_mesh_local(tip,g);assert_allclose(v[:,1],0);assert np.ptp(v[:,2])==100
    for angle in (0,120,240):g['contacts'].append(new_contact(g,[0,500,0],surface='sector',radius_um=50,angle_deg=angle,span_deg=90,height_um=50))
    g['contacts'].append(new_contact(g,[0,1000,0],surface='ring',radius_um=50,height_um=100))
    validate_geometry(g)
    item={};settings=probe_settings(item);settings['geometry']=g
    settings['pose']=dict(tip0_source_zyx_um=[100,200,300],down_source_zyx=[-1,0,0],lateral_source_zyx=[0,0,1],normal_source_zyx=[0,1,0])
    sources=contact_coordinates(settings,g['contacts'])
    vertices,faces=contact_mesh(item,sources,np.eye(4));assert len(vertices)>100
    # Mesh uses terminal X/Z plane, not the global X/Y contact plane.
    assert_allclose(vertices[:24,2],100)
    body,body_faces=shank_mesh(item,np.eye(4));assert len(body)>0 and len(body_faces)>0
    assert_allclose(np.ptp(body,axis=0),[100,100,2000],atol=.01)


@pytest.mark.gui
def test_designer_body_tools_additive_contacts_and_native_pinch(tmp_path):
    from probemap.registration.macos_qt import initialize_qt
    from probemap.registration.desktop_probe_library import ProbeEditor,ProbesLibraryDialog
    from probemap.registration.desktop_app import STYLE
    from test_trackpad_zoom import pinch
    from qtpy.QtCore import QPointF
    app=initialize_qt();e=ProbeEditor();e.setStyleSheet(STYLE);e.show();app.processEvents()
    try:
        assert e.g['body_shape']=='wire' and e.g['contacts'][0]['surface']=='terminal'
        e.table.selectAll();e.delete_selected();assert len(e.g['contacts'])==0
        e.body.setCurrentIndex(e.body.findData('polygon'));e.fields['width'].setValue(1000);e.fields['length'].setValue(1000)
        from qtpy.QtCore import Qt
        from qtpy.QtGui import QMouseEvent
        e.draw_outline();canvas=e.preview.canvas
        for xy in ([0,0],[-400,800],[400,800]):
            pos=canvas.pixel(xy)
            app.sendEvent(canvas,QMouseEvent(QMouseEvent.Type.MouseButtonPress,pos,pos,Qt.MouseButton.LeftButton,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier))
        canvas.finish_outline()
        assert_allclose(e.g['body_polygon_um'],[[0,0],[-400,800],[400,800]])
        e.undo();assert e.g['shank_length_um']==1000
        e.fields['site_width'].setValue(10);e.fields['site_height'].setValue(8)
        e.pattern.setCurrentIndex(e.pattern.findData('perimeter'));e.surface.setCurrentIndex(e.surface.findData('edge'));e.fields['count'].setValue(12);e.generate_layout()
        validate_geometry(e.g);assert len(e.g['contacts'])==12
        e.undo();assert len(e.g['contacts'])==0
        e.redo();assert len(e.g['contacts'])==12
        e.select_contact(0);e.contact_width.setValue(6);assert e.g['contacts'][0]['width_um']==6
        e.preview.view.setCurrentText('Front · X/Y');app.processEvents()
        c=e.preview.canvas;point=QPointF(c.width()*.61,c.height()*.47);before=c.world(point);zoom=c.zoom
        pinch(app,c,point,[.08,.04,-.02],wrong_local=True)
        assert_allclose(c.zoom/zoom,np.prod([1/.92,1/.96,1/1.02]));assert_allclose(c.world(point),before,atol=1e-8)
        e.name.setText('Perimeter probe');e.finish();assert e.result_probe
        library=ProbeLibrary(tmp_path);saved=library.save(e.result_probe)
        dialog=ProbesLibraryDialog(library=library);dialog.search.setText('Perimeter probe');assert dialog.list.count()==1
        assert dialog.current()['geometry']['contacts'][0]['width_um']==6
        dialog.close()
    finally:e._initial=copy.deepcopy(e.g);e.close()

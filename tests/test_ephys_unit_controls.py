"""Unit visibility is independent from waveform selection and sorting filters."""
from types import SimpleNamespace
import numpy as np
from qtpy.QtCore import Qt


def test_unit_table_filters_visibility_and_waveform_stack():
    from probemap.registration.macos_qt import initialize_qt
    app = initialize_qt()
    from probemap.registration.desktop_ephys import ElectrophysiologyWorkspace
    from probemap.registration.desktop_ephys_plots import render_plot
    from probemap.registration.desktop_ephys_settings import defaults
    from matplotlib.figure import Figure
    page = ElectrophysiologyWorkspace(SimpleNamespace(save_project=lambda:None,
        schedule_save=lambda:None, go_step=lambda n:None))
    page.draw = lambda:None
    record = {'id':'r','selected_sorting':'ks'}
    page.current = lambda:record
    page.shank.addItem('Shank 0',0)
    units = [dict(id=i,height_um=300-i*15,shank=0,rate_hz=1.,
                  label='good' if i<2 else 'mua') for i in range(3)]
    page.sorting = ({'units':units,'sampling_rate':30000,'duration_s':10}, {'waveforms':np.ones((3,32))})
    page._fill_units()
    assert page.units.rowCount() == 2
    page.units.item(0,0).setCheckState(Qt.CheckState.Unchecked)
    assert [u['id'] for _,u in page._visible_units()] == [1]
    page.wave_units.item(1,0).setCheckState(Qt.CheckState.Checked)
    page.good.setCurrentIndex(1)
    assert page.units.rowCount() == 3
    assert [u['id'] for _,u in page._visible_units()] == [1,2]
    page.wave_units.item(2,0).setCheckState(Qt.CheckState.Checked)
    page._fill_units()
    assert [u['id'] for _,u in page._waveform_units()] == [1,2]
    page._indices_heights = lambda:(np.array([],int),np.array([]))
    page.plot_board = SimpleNamespace(column_choices=lambda:[],notebook_mode=SimpleNamespace(isChecked=lambda:False),
                                     plot_options=lambda key:defaults(key), keys=lambda:['raster'], depth_range=lambda:(0,500))
    figure = Figure()
    axes = render_plot(page,figure,'waveform',(.2,.1,.7,.8))
    assert len(axes)==2
    assert axes[0].get_position().y0 > axes[1].get_position().y0
    assert 'Unit 1' in axes[0].get_title() and 'Unit 2' in axes[1].get_title()
    assert len(axes[0].lines)==len(axes[1].lines)==1
    page.anatomy_ax = Figure().add_subplot()
    page.unit_view = dict(key=page._unit_view_key(),raster_times=np.array([1.,2.]),
                          raster_depths=np.array([285.,270.]),raster_ids=np.array([1,2]))
    raster_figure=Figure()
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    canvas=FigureCanvasAgg(raster_figure)
    raster = render_plot(page,raster_figure,'raster',(.2,.1,.7,.8))
    artists=raster[0]._unit_artists
    assert artists[1]._unit_ids.tolist()==[1]
    assert artists[2]._unit_ids.tolist()==[2]
    page.plot_board.panels={'raster':dict(axes=raster,canvas=canvas)}
    canvas.mpl_connect('draw_event',page._raster_draw_event)
    canvas.draw()
    initial_pixels=np.asarray(canvas.buffer_rgba()).copy()
    previous_view=page.unit_view
    page.draw=lambda: (_ for _ in ()).throw(AssertionError('Full workspace redraw'))
    page._task=lambda *args: (_ for _ in ()).throw(AssertionError('Unexpected disk read'))
    page.units.item(1,0).setCheckState(Qt.CheckState.Unchecked)
    assert not artists[1].get_visible() and artists[2].get_visible()
    assert not np.array_equal(np.asarray(canvas.buffer_rgba()),initial_pixels)
    assert page.unit_view is previous_view and raster[0]._unit_artists is artists
    assert [u['id'] for _,u in page._waveform_units()]==[1,2]
    page.units.item(1,0).setCheckState(Qt.CheckState.Checked)
    assert artists[1].get_visible()
    np.testing.assert_array_equal(np.asarray(canvas.buffer_rgba()),initial_pixels)
    assert raster[0].get_xlim()==(0,10)
    page.unit_timer.stop(); page.export_timer.stop(); page.close()


def test_duplicate_depth_lanes_are_distinct_stable_and_keep_source_depth():
    from probemap.registration.ephys_units import raster_positions
    units=[dict(id=3,height_um=100),dict(id=1,height_um=100),dict(id=2,height_um=100),dict(id=4,height_um=200)]
    positions=raster_positions(units)
    assert len(set(positions.values()))==len(units)
    assert positions==raster_positions(list(reversed(units)))
    assert positions[4]==200
    assert [u['height_um'] for u in units]==[100,100,100,200]
    assert positions[1]<positions[2]<positions[3]

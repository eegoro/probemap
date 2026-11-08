"""Cluster colouring is independent from signal values and plot selection."""
from types import SimpleNamespace
import numpy as np
import pytest
from matplotlib.figure import Figure
from matplotlib.colors import to_rgba

from probemap.registration.desktop_ephys_plots import render_plot
from probemap.registration.desktop_ephys_legends import plot_legend
from probemap.registration.desktop_ephys_settings import defaults


def page(interval):
    times=np.arange(8)/1000
    heights=np.array([217.,232.,247.])
    waves=np.array([np.sin(times*500),np.cos(times*400),np.sin(times*800)])
    labels=np.array([0,1,2])
    record=dict(id='r',channels=[dict(column=0) for _ in heights])
    options={}
    board=SimpleNamespace(column_choices=lambda:['0'],depth_range=lambda:(0,300),
        column_label=lambda kind:'X = 27 µm',opacity=lambda kind:1.,
        notebook_mode=SimpleNamespace(isChecked=lambda:interval),
        legends=SimpleNamespace(isChecked=lambda:True),
        analysis_fields={'lowpass':SimpleNamespace(value=lambda:40)},
        plot_options=lambda key:options.setdefault(key,defaults(key)))
    data=dict(fs=np.array(1000),times=times,low40_times=times*1000,
        **{'0_heights':heights,'0_lfp':waves,'0_clusters':labels,
           '0_low40':waves*.2,'0_low40_clusters':labels,'0_low40_fs':np.array(1000),
           '0_boundaries':np.array([])})
    return SimpleNamespace(current=lambda:record,plot_board=board,summary=({'sampling_rate':1000},{}),
        _indices_heights=lambda:(np.arange(3),heights),anatomy_ax=Figure().add_subplot(),
        anatomy=None,boundary=SimpleNamespace(isChecked=lambda:False),
        time=SimpleNamespace(value=lambda:0),notebook=data if interval else None,
        trace=(0,waves,{'0':labels},.008,0))


@pytest.mark.parametrize('interval',[False,True])
def test_lfp_pair_has_identical_values_but_distinct_colouring(interval):
    p=page(interval); figures=[Figure(),Figure()]
    plain=render_plot(p,figures[0],'traces',(.1,.1,.8,.8))[0]
    clustered=render_plot(p,figures[1],'traces_clustered',(.1,.1,.8,.8))[0]
    assert len(plain.lines)==len(clustered.lines)==3
    for a,b in zip(plain.lines,clustered.lines):
        np.testing.assert_array_equal(a.get_xdata(),b.get_xdata())
        np.testing.assert_array_equal(a.get_ydata(),b.get_ydata())
    assert len({to_rgba(line.get_color()) for line in plain.lines})==1
    assert len({to_rgba(line.get_color()) for line in clustered.lines})==3
    assert plain.get_ylim()==clustered.get_ylim()
    plot_legend(p,figures[0],'traces',[plain]); plot_legend(p,figures[1],'traces_clustered',[clustered])
    plain_labels=[t.get_text() for legend in figures[0].legends for t in legend.get_texts()]
    cluster_labels=[t.get_text() for legend in figures[1].legends for t in legend.get_texts()]
    assert not any(t.startswith('Cluster ') for t in plain_labels)
    assert [t for t in cluster_labels if t.startswith('Cluster ')]==['Cluster 1','Cluster 2','Cluster 3']


def test_lowpass_pair_uses_same_processed_signal():
    p=page(True)
    plain=render_plot(p,Figure(),'low40_plain',(.1,.1,.8,.8))[0]
    clustered=render_plot(p,Figure(),'low40',(.1,.1,.8,.8))[0]
    for a,b in zip(plain.lines,clustered.lines): np.testing.assert_array_equal(a.get_ydata(),b.get_ydata())
    assert len({to_rgba(line.get_color()) for line in plain.lines})==1
    assert len({to_rgba(line.get_color()) for line in clustered.lines})==3


def test_trace_pair_can_be_opened_together_and_restored_independently():
    from probemap.registration.macos_qt import initialize_qt
    from probemap.registration.desktop_ephys import ElectrophysiologyWorkspace
    app=initialize_qt()
    p=ElectrophysiologyWorkspace(SimpleNamespace(save_project=lambda:None,schedule_save=lambda:None,go_step=lambda n:None))
    p._ensure_canvas(); p._busy=True
    board=p.plot_board
    board.set_checked(['traces','traces_clustered','low40','low40_plain'])
    assert len(board.keys())==4
    board.plot_options('traces')['trace_scale']=2
    assert board.plot_options('traces_clustered')['trace_scale']==1
    state=board.state(); board.restore(state)
    assert set(board.keys())=={'traces','traces_clustered','low40','low40_plain'}
    assert board.plot_options('traces')['trace_scale']==2
    assert board.plot_options('traces_clustered')['trace_scale']==1
    p.stop(); p.close()


def test_adding_clustered_trace_reuses_already_loaded_interval():
    from probemap.registration.macos_qt import initialize_qt
    from probemap.registration.desktop_ephys import ElectrophysiologyWorkspace
    app=initialize_qt()
    p=ElectrophysiologyWorkspace(SimpleNamespace(save_project=lambda:None,schedule_save=lambda:None,go_step=lambda n:None))
    p._ensure_canvas(); p._busy=True
    p.plot_board.set_checked(['traces','traces_clustered'])
    waves=np.array([[0.,1.,0.,-1.]*250,[1.,0.,-1.,0.]*250,[0.,.5,0.,-.5]*250])
    p.current=lambda:dict(id='r',cache_path='/does-not-exist',channels=[dict(column=0)]*3)
    p.summary=({'sampling_rate':1000},{})
    p._indices_heights=lambda:(np.arange(3),np.array([217.,232.,247.]))
    shank=p.shank.currentData()
    duration=p.window_duration.value()
    p.trace=(p.time.value(),waves,None,duration,shank)
    p._task=lambda work,ready,*args:ready(work(lambda *args:None,lambda:False))
    p._busy=False
    p._load_trace()
    assert p.trace[1] is waves
    assert set(p.trace[2])=={'0','mean'}
    assert p._trace_ready()
    p.stop(); p.close()

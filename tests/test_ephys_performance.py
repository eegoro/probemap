"""Avoid redundant IO/redraws and keep interactive jobs from blocking controls."""
import copy
import json
import time
from threading import Event
from types import SimpleNamespace

import numpy as np


def test_selective_sorting_reader_reuses_and_invalidates_cache(tmp_path,monkeypatch):
    from probemap.registration.ephys_analysis import load_sorting
    from numpy.lib.npyio import NpzFile
    root=tmp_path/'sorting'; root.mkdir()
    (root/'manifest.json').write_text(json.dumps(dict(version=2,units=[dict(id=4)])))
    np.savez(root/'units.npz',waveforms=np.ones((1,32)),raster=np.zeros((4,20,30)))
    read=[]; original=NpzFile.__getitem__
    def getitem(self,key):
        read.append(key); return original(self,key)
    monkeypatch.setattr(NpzFile,'__getitem__',getitem)
    info,data=load_sorting(root,fields=('waveforms',))
    assert read==['waveforms'] and list(data)==['waveforms']
    info['units'][0]['id']=999
    again,values=load_sorting(root,fields=('waveforms',))
    assert read==['waveforms'] and again['units'][0]['id']==4
    assert values['waveforms'] is data['waveforms'] and not values['waveforms'].flags.writeable
    np.savez(root/'units.npz',waveforms=np.zeros((1,33)),raster=np.zeros((4,20,30)))
    assert load_sorting(root,fields=('waveforms',))[1]['waveforms'].shape==(1,33)
    assert read==['waveforms','waveforms']


def test_background_discovery_preserves_concurrent_user_edits():
    from probemap.registration.ephys_portable import merge_scanned_exports
    original=dict(id='a',name='Before',insertion_id='one',cache_path='old',
                  selected_sorting='first',unit_preferences={'first':{'hidden':[4]}},sortings=[])
    baseline=[copy.deepcopy(original)]; actual=[original]
    scanned=copy.deepcopy(baseline)
    scanned[0].update(cache_path='new',selected_sorting='server',bundle_signature=[2])
    scanned.append(dict(id='b',name='New',cache_root='new-root'))
    actual[0].update(name='Edited while scanning',insertion_id='two',selected_sorting='manual')
    actual[0]['unit_preferences']['first']['hidden'].append(5)
    changed=merge_scanned_exports(actual,baseline,scanned)
    assert changed=={'a','b'}
    assert actual[0]['name']=='Edited while scanning' and actual[0]['insertion_id']=='two'
    assert actual[0]['selected_sorting']=='manual' and actual[0]['cache_path']=='new'
    assert actual[0]['unit_preferences']['first']['hidden']==[4,5]
    assert 'bundle_signature' not in actual[0]  # Conflict is retried on the next discovery.


def workspace():
    from probemap.registration.macos_qt import initialize_qt
    app=initialize_qt()
    from probemap.registration.desktop_ephys import ElectrophysiologyWorkspace
    page=ElectrophysiologyWorkspace(SimpleNamespace(save_project=lambda:None,
        schedule_save=lambda:None,go_step=lambda n:None))
    return app,page


def test_plot_setting_burst_rebuilds_only_its_panel(monkeypatch):
    app,page=workspace(); page._ensure_canvas(); page._busy=True
    board=page.plot_board; board.render()
    anatomy=page.anatomy_ax; lfp=board.panels['lfp']['axes'][0]; rms=board.panels['rms']['axes'][0]
    def forbidden(*args): raise AssertionError('Unrelated anatomy/full workspace rebuild')
    monkeypatch.setattr(page,'draw',forbidden)
    import probemap.registration.desktop_ephys_board as module
    monkeypatch.setattr(module,'draw_probe',forbidden)
    calls=[]; render=board.render_panel
    monkeypatch.setattr(board,'render_panel',lambda key:(calls.append(key),render(key)))
    for value in (1.1,1.2,1.3,1.4):
        board.plot_options('rms')['line_width']=value; board.plot_settings_changed('rms')
    assert not calls
    board.settings_timer.stop(); board.flush_panels()
    assert calls==['rms'] and board.panels['lfp']['axes'][0] is lfp
    assert board.panels['rms']['axes'][0] is not rms and page.anatomy_ax is anatomy
    page.stop(); page.close()


def test_spike_job_keeps_controls_enabled_and_rejects_stale_result():
    app,page=workspace(); gate=Event(); received=[]; state=['old']
    page._unit_view_key=lambda:state[0]
    page.current=lambda:{'id':'r'}
    def work(progress,cancel):
        gate.wait(2); return 'old-result'
    page._run_unit_job(work,received.append,'old')
    assert not page._busy and page.splitter.isEnabled()
    state[0]='new'; gate.set()
    deadline=time.monotonic()+5
    while page._unit_job is not None and time.monotonic()<deadline:
        app.processEvents(); time.sleep(.005)
    assert page._unit_job is None and received==[]
    page.stop(); page.close()


def test_recent_interval_cache_is_bounded_and_immutable(monkeypatch):
    import probemap.registration.ephys_notebook as module
    monkeypatch.setattr(module,'_WINDOW_LIMIT',64)
    module._WINDOW_MEMORY.clear()
    module._remember_window('a',{'data':np.ones(6)})
    assert module._recalled_window('a') is not None
    module._remember_window('b',{'data':np.zeros(6)})
    assert module._recalled_window('a') is None
    assert not module._recalled_window('b')['data'].flags.writeable
    module._remember_window('too-large',{'data':np.zeros(100)})
    assert module._recalled_window('too-large') is None
    module._WINDOW_MEMORY.clear()


def test_reordering_keeps_existing_plot_axes_and_options(monkeypatch):
    app,page=workspace(); page._ensure_canvas(); page._busy=True
    board=page.plot_board; board.render()
    before={key:panel['axes'][0] for key,panel in board.panels.items()}
    def forbidden(*args): raise AssertionError('Plot rebuilt just to change order')
    monkeypatch.setattr(page,'draw',forbidden)
    board.active_updating=True
    item=board.active_list.takeItem(0); board.active_list.insertItem(1,item)
    board.active_updating=False
    board.active_reordered()
    assert board.keys()==['rms','lfp']
    assert all(board.panels[key]['axes'][0] is ax for key,ax in before.items())
    page.stop(); page.close()


def test_probe_slider_rebuilds_only_anatomy(monkeypatch):
    app,page=workspace(); page._ensure_canvas(); page._busy=True
    board=page.plot_board; board.render()
    axes={key:panel['axes'][0] for key,panel in board.panels.items()}
    def forbidden(*args): raise AssertionError('Unrelated signal/unit panel rebuilt')
    monkeypatch.setattr(board,'render_panel',forbidden)
    page.contact_size.setValue(12)
    board.settings_timer.stop(); board.flush_panels()
    assert all(board.panels[key]['axes'][0] is ax for key,ax in axes.items())
    page.stop(); page.close()

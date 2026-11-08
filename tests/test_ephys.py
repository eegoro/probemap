"""Ephys isolation, actual binary signal processing, sorting and project relocation."""
import copy
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from probemap.registration.ephys import import_folders, read_recording, saved_channels, read_params
from probemap.registration.ephys_analysis import prepare_recording, load_summary, prepare_sorting, load_sorting

META = Path(__file__).parent/'fixtures/np2014_example.meta'


def recording(tmp_path, seconds=5, n_channels=16):
    folder = tmp_path/'recording'; folder.mkdir(parents=True)
    text = META.read_text() + '\nfileCreateTime=2026-10-20T12:34:56\nimSampRate=30000\nimAiRangeMax=0.5\nimMaxInt=2048\n'
    import re
    text = re.sub(r'^fileCreateTime=.*$', 'fileCreateTime=2026-10-20T12:34:56', text, flags=re.M)
    text = re.sub(r'^imSampRate=.*$', 'imSampRate=30000', text, flags=re.M)
    # An intentionally noncontiguous file subset tests acquisition-to-file mapping.
    text = re.sub(r'^snsSaveChanSubset=.*$', f'snsSaveChanSubset=2,4,8:{n_channels+5},384', text, flags=re.M)
    text = re.sub(r'^nSavedChans=.*$', f'nSavedChans={n_channels+1}', text, flags=re.M)
    path = folder/'test.imec0.ap.meta'; path.write_text(text)
    fs = 30000; t = np.arange(seconds*fs)/fs
    data = np.zeros((len(t),n_channels+1),np.int16)
    for i in range(n_channels): data[:,i] = (np.sin(2*np.pi*10*t)*(i+1)*10 + np.sin(2*np.pi*2000*t)*5).astype(np.int16)
    data[:,-1] = 20000  # sync must never appear among neural channels
    data.tofile(path.with_suffix('.bin'))
    return folder, read_recording(path)


def test_import_times_sortings_and_isolation(tmp_path):
    folder, rec = recording(tmp_path)
    assert len(rec['channels']) == 16 and rec['channels'][-1]['file_index'] == 15
    assert rec['name'] == '2026-10-20 12:34:56'
    for name in ('arbitrary-result-name', 'kilosort4'):
        sorting = folder/name; sorting.mkdir()
        np.save(sorting/'spike_times.npy', [100, 200]); np.save(sorting/'spike_clusters.npy', [0, 0])
    added, errors = import_folders([folder])
    assert not errors and len(added) == 1 and len(added[0]['sortings']) == 2
    assert not import_folders([folder], added)[0]
    assert saved_channels('2,4,8:10',5) == [2,4,8,9,10]
    with pytest.raises(ValueError): saved_channels('2,2',2)
    # Import and signal analysis have no anatomical session argument to mutate.
    assert added[0]['insertion_id'] is None


def test_cpu_processing_cache_cancellation_and_spectrum(tmp_path):
    _, rec = recording(tmp_path)
    progress=[]
    result = prepare_recording(rec, tmp_path/'out', lambda p,m: progress.append(p))
    info, data = load_summary(result)
    assert info['sampling_rate'] == pytest.approx(1000)
    assert np.load(Path(result)/'lfp.npy',mmap_mode='r').shape == (16,5000)
    peak = data['frequencies'][np.argmax(data['power'][0])]
    assert peak == pytest.approx(10,abs=.5)
    assert data['correlation'].shape == (16,16) and progress[-1] == 100
    before = (Path(result)/'profiles.npz').stat().st_mtime_ns
    assert prepare_recording(rec,tmp_path/'out') == result
    assert (Path(result)/'profiles.npz').stat().st_mtime_ns == before
    with pytest.raises(InterruptedError): prepare_recording(rec,tmp_path/'cancelled',cancel=lambda:True)
    assert not list((tmp_path/'cancelled').rglob('manifest.json'))
    assert not list((tmp_path/'cancelled').rglob('*.partial-*'))


def test_curated_nonconsecutive_clusters_and_safe_params(tmp_path):
    folder, rec = recording(tmp_path)
    sort = folder/'result'; sort.mkdir()
    (sort/'params.py').write_text("sample_rate = 30000\ndat_path = 'test.imec0.ap.bin'\nraise RuntimeError('MUST NOT EXECUTE')\n")
    assert read_params(sort)['sample_rate'] == 30000
    np.save(sort/'spike_times.npy', [100,200,300,400])
    np.save(sort/'spike_clusters.npy', [19,19,73,19])
    np.save(sort/'spike_templates.npy', [0,1,1,0])
    np.save(sort/'channel_map.npy', [0,1])
    np.save(sort/'channel_positions.npy', [[27,0],[59,0]])
    templates=np.zeros((2,8,2)); templates[0,3,0]=10; templates[1,3,1]=10
    np.save(sort/'templates.npy',templates)
    (sort/'cluster_KSLabel.tsv').write_text('cluster_id\tKSLabel\n19\tgood\n73\tgood\n')
    (sort/'cluster_group.tsv').write_text('cluster_id\tgroup\n19\tmua\n73\tgood\n')
    result=prepare_sorting(rec,dict(path=str(sort)),tmp_path/'out')
    info,data=load_sorting(result)
    assert [u['id'] for u in info['units']] == [19,73]
    assert [u['label'] for u in info['units']] == ['mua','good']
    assert [u['count'] for u in info['units']] == [3,1]
    assert [u['height_um'] for u in info['units']] == [232,247]
    assert data['waveforms'].shape == (2,8)
    assert data['raster'].sum() == 4


def test_project_roundtrip_and_relocated_paths_keep_anatomical_recordings(tmp_path):
    from test_track_model import make_track_project
    from probemap.registration.desktop_model import AlignmentProject
    from probemap.registration.insertion_model import insertion_session,new_insertion
    from probemap.registration.recordings import import_recordings,assign_recording
    old = tmp_path/'old'; p = make_track_project(old)
    session=insertion_session(p); insertion=new_insertion(session)
    added,_=import_recordings(session,[META]); assign_recording(session,added[0]['id'],insertion['id'])
    before=copy.deepcopy(insertion['recordings'])
    _,rec=recording(old); rec['insertion_id']=insertion['id']
    rec['cache_path']=prepare_recording(rec,old/'out')
    rec['depth_levels']=[dict(id='level-1',name='Cortex',height_um=1200.,shank=1,visible=True,label=False,color='#79e7ff')]
    p.ephys={'recordings':[rec],'view':{'plot_board':{'checked':['lfp','rms'],'columns':['0','1','mean'],'order':['rms','lfp']}}}; p.save(old/'out/project.json')
    assert insertion['recordings'] == before
    new=tmp_path/'new'; shutil.copytree(old,new)
    loaded=AlignmentProject.load(new/'out/project.json')
    test=loaded.ephys['recordings'][0]
    assert Path(test['bin_path']).is_relative_to(new)
    assert Path(test['cache_path']).is_relative_to(new)
    assert test['insertion_id'] == insertion['id']
    assert test['depth_levels'] == rec['depth_levels']
    assert loaded.ephys['view']['plot_board'] == p.ephys['view']['plot_board']
    assert insertion_session(loaded)['insertions'][0]['recordings'] == before


@pytest.mark.gui
def test_workspace_multi_import_anatomy_sorting_and_restore(tmp_path,monkeypatch):
    from probemap.registration.macos_qt import initialize_qt
    from qtpy.QtWidgets import QWidget, QToolButton
    from qtpy.QtCore import Qt
    from test_desktop_app import wait_until
    from test_track_model import make_track_project
    from probemap.registration.desktop_ephys import ElectrophysiologyWorkspace
    from probemap.registration.insertion_model import insertion_session,new_insertion
    from probemap.registration.neuropixels_model import probe_settings
    app = initialize_qt()
    class Owner(QWidget):
        def schedule_save(self): pass
        def save_project(self): pass
        def go_step(self, step): pass
    from probemap.registration.desktop_app import STYLE
    owner = Owner(); owner.setStyleSheet(STYLE); page = ElectrophysiologyWorkspace(owner); page.setStyleSheet(STYLE + page.styleSheet())
    p = make_track_project(tmp_path/'project')
    item = new_insertion(insertion_session(p)); settings=probe_settings(item)
    settings['pose']=dict(tip0_source_zyx_um=[400,400,400],down_source_zyx=[-1,0,0],lateral_source_zyx=[0,0,1],normal_source_zyx=[0,-1,0])
    folder, rec = recording(tmp_path/'input')
    page.resize(1440,900); page.show(); page.open(p)
    try:
        page.add_folders(paths=[folder])
        wait_until(app, lambda:not page._busy)
        assert len(p.ephys['recordings']) == 1
        assert not item.get('recordings')
        assert page.plot_stack.currentWidget() is page.plot_host
        assert page.anatomy_ax.get_ylim()[0] == 0
        page.grab().save(str(tmp_path/'probemap-ephys-empty.png'))
        assert page.table.columnCount() == 6
        stamp = page.table.item(0,2).text()
        page.table.item(0,0).setText('Before')
        assert p.ephys['recordings'][0]['name'] == 'Before'
        assert page.table.item(0,2).text() == stamp
        page.table.item(0,1).setData(Qt.ItemDataRole.UserRole, item['id'])
        wait_until(app, lambda:not page._busy)
        assert page.anatomy is not None
        # The application-wide guard must retain both active record and unit selections.
        from probemap.registration.desktop_tables import TableSelectionGuard
        from qtpy.QtTest import QTest
        guard = TableSelectionGuard(page)
        QTest.mouseClick(page.source, Qt.MouseButton.LeftButton)
        app.processEvents(); page.source.hidePopup()
        assert page.current()['id'] == p.ephys['recordings'][0]['id'] and page.table.selectionModel().hasSelection()
        assert page.anatomy['extent'][2] == 0
        assert page.anatomy_ax.get_ylim()[0] == 0
        prepare_button = page.table.cellWidget(0,4).findChild(QToolButton)
        assert prepare_button.toolTip() == 'Prepare signal'
        prepare_button.click()
        assert page._busy and not page.splitter.isEnabled()
        assert not page.back.isEnabled() and not page.save.isEnabled() and not page.collapse.isEnabled()
        assert owner.isEnabled() and page.cancel.isEnabled()
        wait_until(app, lambda:not page._busy,timeout=30)
        assert page.summary is not None and page.splitter.isEnabled()
        assert page.table.item(0,4).text() == '✓'
        assert page.table.cellWidget(0,4) is None
        from qtpy.QtWidgets import QMessageBox
        calls=[]; answers=[QMessageBox.StandardButton.No,QMessageBox.StandardButton.Cancel,QMessageBox.StandardButton.Yes,QMessageBox.StandardButton.No]
        def answer(*args): calls.append(args); return answers.pop(0)
        monkeypatch.setattr(QMessageBox,'question',answer)
        record=p.ephys['recordings'][0]; cached=Path(record['cache_path']); stamp=(cached/'lfp.npy').stat().st_mtime_ns
        page.prepare_signal(record['id']); wait_until(app,lambda:not page._busy)
        assert len(calls)==1 and (cached/'lfp.npy').stat().st_mtime_ns==stamp
        page.prepare_signal(record['id']); assert not page._busy and len(calls)==2
        page.prepare_signal(record['id']); wait_until(app,lambda:not page._busy)
        assert len(calls)==3 and (cached/'lfp.npy').stat().st_mtime_ns!=stamp
        stamp=(cached/'lfp.npy').stat().st_mtime_ns
        page.add_folders(paths=[folder]); wait_until(app,lambda:len(calls)==4 and not page._busy)
        assert len(p.ephys['recordings'])==1 and (cached/'lfp.npy').stat().st_mtime_ns==stamp
        for key in ('spectrum','correlation','traces','activity','lfp','rms'):
            page.plot_board.set_checked([key]); app.processEvents()
            wait_until(app,lambda:not page._busy); page._canvas.draw()
        fixed_canvas=page.anatomy_canvas
        page.plot_board.set_checked(['lfp','rms','spectrum'])
        assert len(page.plot_board.panels['spectrum']['axes'])==2
        assert page.anatomy_canvas is fixed_canvas and page.anatomy_canvas.isVisible()
        page.plot_board.set_checked(['lfp','rms','raster','waveform'])
        sort = folder/'sorting_any_name'; sort.mkdir()
        np.save(sort/'spike_times.npy', [100,200,300,400])
        np.save(sort/'spike_clusters.npy', [19,19,73,19])
        np.save(sort/'spike_templates.npy', [0,1,1,0])
        np.save(sort/'channel_map.npy', [0,1])
        templates=np.zeros((2,8,2)); templates[0,3,0]=10; templates[1,3,1]=10
        np.save(sort/'templates.npy', templates)
        (sort/'cluster_group.tsv').write_text('cluster_id\tgroup\n19\tgood\n73\tmua\n')
        page.attach_sorting(path=sort); wait_until(app, lambda:not page._busy)
        assert page.sorting is not None  # Connecting a result reads it automatically.
        page.span.setCurrentIndex(1)
        page._load_unit_view(); wait_until(app,lambda:page.unit_view is not None and not page._busy)
        page.wave_units.item(0,0).setCheckState(Qt.CheckState.Checked)
        wait_until(app,lambda:not page.plot_board.settings_timer.isActive())
        assert page.unit_view is not None
        assert len(page.plot_board.panels['waveform']['axes'])==1
        assert 'Unit 19' in page.plot_board.panels['waveform']['axes'][0].get_title()
        QTest.mouseClick(page.plot_tools,Qt.MouseButton.LeftButton)
        assert page.units.currentRow()>=0 and page.table.selectionModel().hasSelection()
        page.unit_view=None; page.draw()
        wait_until(app,lambda:page.unit_view is not None and not page._busy)
        page.settings_tabs.setCurrentIndex(4)
        assert page.units.isVisible()
        for detail in ('isi','rate','amplitude','acg','density','region_units'):
            page.plot_board.set_checked(['lfp','rms','raster',detail]); page._canvas.draw()
        page.plot_board.set_checked(['lfp','rms','raster','spectrum'])
        page.capture_state()
        assert 'raster' in p.ephys['view']['plot_board']['checked']
        # A different result must replace the unit data, not only the dropdown name.
        sort2 = folder/'kilosort4_second'; shutil.copytree(sort, sort2)
        np.save(sort2/'spike_clusters.npy', [91,91,91,91])
        (sort2/'cluster_group.tsv').write_text('cluster_id\tgroup\n91\tgood\n')
        page.attach_sorting(path=sort2); wait_until(app, lambda:not page._busy)
        assert [u['id'] for u in page.sorting[0]['units']] == [91]
        record = p.ephys['recordings'][0]
        first_id, second_id = [s['id'] for s in record['sortings']]
        combo = page.table.cellWidget(0,3)
        assert combo.itemText(combo.count()-1) == 'Choose folder…'
        combo.setCurrentIndex(combo.findData(first_id)); combo._activate(combo.currentIndex())
        wait_until(app, lambda:not page._busy)
        assert [u['id'] for u in page.sorting[0]['units']] == [19,73]
        combo = page.table.cellWidget(0,3); combo._remove(second_id)
        assert sort2.is_dir() and len(record['sortings']) == 1
        page.table.setColumnWidth(0, 155)
        page.toggle_sidebar(); page.capture_state()
        assert not p.ephys['view']['sidebar_visible']
        assert p.ephys['view']['splitter_sizes'][0] > 290
        assert p.ephys['view']['column_widths'][0] == 155
        page.sidebar.show(); page.table.setColumnWidth(0, 90)
        page.open(p); wait_until(app, lambda:not page._busy)
        assert page.sidebar.isHidden() and page.table.columnWidth(0) == 155
        assert set(page.plot_board.keys())=={'lfp','rms','raster','spectrum'}
        assert page.anatomy_canvas is fixed_canvas
        page.toggle_sidebar(); page.capture_state()
        page.plot_board.set_checked(['lfp','rms','spectrum'])
        for col, width in enumerate((100,64,90,95,50)): page.table.setColumnWidth(col,width)
        app.processEvents(); wait_until(app, lambda:not page._busy); page._canvas.draw()
        page.grab().save(str(tmp_path/'probemap-ephys-workspace.png'))
        from types import SimpleNamespace
        y = page.anatomy_ax.get_ylim()[1]/2
        page._plot_scroll(SimpleNamespace(inaxes=page.anatomy_ax,xdata=0,ydata=y,button='up',canvas=page.anatomy_canvas))
        for ax in page._depth_axes: np.testing.assert_allclose(ax.get_ylim(),page.anatomy_ax.get_ylim())
        page.resize(1080,720); app.processEvents(); page.grab().save(str(tmp_path/'probemap-ephys-compact.png'))
        page.resize(1440,900)

        from qtpy.QtCore import QPoint
        from types import SimpleNamespace
        board=page.plot_board
        board.set_checked(['lfp','rms','raster'])
        wait_until(app,lambda:not page._busy)
        def aligned():
            app.processEvents(); board.align()
            for canvas in board.canvases(): canvas.draw()
            reference=None
            for ax in [page.anatomy_ax,*page._depth_axes]:
                canvas=ax.figure.canvas
                ys=np.array([ax.transData.transform((0,d))[1] for d in page.anatomy_ax.get_ylim()])/canvas.device_pixel_ratio
                screen=canvas.mapToGlobal(QPoint(0,0)).y()+canvas.height()-ys
                if reference is None: reference=screen
                else: np.testing.assert_allclose(screen,reference,atol=.6)
                np.testing.assert_allclose(ax.get_ylim(),page.anatomy_ax.get_ylim())
                np.testing.assert_allclose(ax.get_yticks(),page.anatomy_ax.get_yticks())
        aligned()
        board.columns['mean'].setChecked(True)
        assert all(check.isChecked() for check in board.columns.values())
        assert len([l for l in board.panels['lfp']['axes'][0].lines if not hasattr(l,'_ephys_role')])==15
        assert board.columns['0'].text()=='X = 27 µm' and board.columns['1'].text()=='X = 59 µm'
        previous_axes=board.panels['lfp']['axes'][0]; previous_summary=page.summary
        for kind,value in [('0',43),('1',71),('mean',90)]: board.column_opacity[kind].setValue(value)
        assert board.panels['lfp']['axes'][0] is previous_axes and page.summary is previous_summary and not page._busy
        plotted=[l for l in board.panels['lfp']['axes'][0].lines if not hasattr(l,'_ephys_role')]
        assert all(line.get_linestyle()=='-' and line.get_alpha()==board.opacity(line._ephys_column) for line in plotted)
        assert {line.get_alpha() for line in plotted}=={.43,.71,.90}
        assert {line.get_alpha() for line in board.panels['rms']['axes'][0].lines if hasattr(line,'_ephys_column')}=={.43,.71,.90}
        lfp_legends=[l for l in board.panels['lfp']['figure'].legends if l._ephys_group is not None]
        assert [legend.get_title().get_text() for legend in lfp_legends]==['X = 27 µm','X = 59 µm','Mean']
        for kind,legend in zip(('0','1','mean'),lfp_legends):
            assert len(legend.legend_handles)==5
            for band,handle in enumerate(legend.legend_handles):
                line=plotted[band*3+int({'0':0,'1':1,'mean':2}[kind])]
                np.testing.assert_allclose(handle.get_color(),line.get_color())
                assert handle.get_alpha()==line.get_alpha() and handle.get_linewidth()==line.get_linewidth()
        opacity_state=board.state(); assert opacity_state['column_opacities']=={'0':43,'1':71,'mean':90}
        legacy_state=board.state(); legacy_state.pop('column_opacities'); legacy_state['line_opacity']=71
        board.restore(legacy_state); assert all(board.opacity(k)==.71 for k in ('0','1','mean'))
        board.restore(opacity_state); page.draw()
        assert [board.opacity(k) for k in ('0','1','mean')]==[.43,.71,.90]

        assert board.panels['lfp']['figure'].legends
        assert any(text.get_text()=='1.5–4 Hz' for legend in board.panels['lfp']['figure'].legends for text in legend.get_texts())
        page.resize(1260,820); board.splitter.setSizes([210,480,300]); aligned()
        state=board.state(); state['order']=['rms','raster','lfp',*[k for k in state['order'] if k not in ('rms','raster','lfp')]]
        board.restore(state); page.draw(); aligned()
        assert board.keys()==['rms','raster','lfp']
        levels=page.levels; levels.tool.setChecked(True)
        depth=page.anatomy_ax.get_ylim()[1]*.6
        px,py=page.anatomy_ax.transData.transform((0,depth))
        levels.motion(SimpleNamespace(inaxes=page.anatomy_ax,ydata=depth,canvas=page.anatomy_canvas))
        assert levels.preview['height_um']==depth and not levels.records()
        assert len(levels.artists)==2*(len(page._depth_axes)+1)
        preview_sites=levels.site_text(levels.preview)
        levels.motion(SimpleNamespace(inaxes=board.panels['rms']['axes'][0],ydata=depth+30,canvas=board.panels['rms']['canvas']))
        assert levels.site_text(levels.preview)!=preview_sites and not levels.records()
        levels.press(SimpleNamespace(button=1,inaxes=page.anatomy_ax,ydata=depth,y=py,canvas=page.anatomy_canvas))
        assert levels.preview is None
        target=depth+10; rms_ax=board.panels['rms']['axes'][0]
        levels.motion(SimpleNamespace(inaxes=rms_ax,ydata=target)); levels.release(None)
        assert len(levels.records())==1 and levels.records()[0]['height_um']==target
        assert len(levels.artists)==2*(len(page._depth_axes)+1)
        levels.table.item(0,3).setCheckState(Qt.CheckState.Unchecked)
        assert len(levels.artists)==len(page._depth_axes)+1
        levels.table.item(0,0).setCheckState(Qt.CheckState.Unchecked); assert not levels.artists
        levels.table.item(0,0).setCheckState(Qt.CheckState.Checked)
        levels.table.item(0,3).setCheckState(Qt.CheckState.Checked)
        levels.table.item(0,1).setText(str(target+3)); assert levels.records()[0]['height_um']==target+3
        page.capture_state(); page.open(p); wait_until(app,lambda:not page._busy)
        assert board.keys()==['rms','raster','lfp'] and levels.records()[0]['height_um']==target+3
        assert [board.opacity(k) for k in ('0','1','mean')]==[.43,.71,.90]
        wait_until(app,lambda:page.unit_view is not None and not page._busy)
        aligned()
        figure,pdf=board.export_pdf(tmp_path/'exported.pdf')
        assert pdf.read_bytes().startswith(b'%PDF-') and b'/FontFile2' in pdf.read_bytes()
        assert len(figure.axes)==4 and len(figure.axes[0].images)>0
        assert {line.get_alpha() for ax in figure.axes for line in ax.lines if hasattr(line,'_ephys_column')}=={.43,.71,.90}
        for ax in figure.axes:
            np.testing.assert_allclose(ax.get_position().y0,figure.axes[0].get_position().y0)
            np.testing.assert_allclose(ax.get_position().height,figure.axes[0].get_position().height)
        assert all(not line.get_rasterized() for ax in figure.axes for line in ax.lines)
        assert all(not coll.get_rasterized() for coll in figure.axes[2].collections)
        import re,zlib
        streams=[]
        for match in re.finditer(rb'stream\r?\n(.*?)\r?\nendstream',pdf.read_bytes(),re.S):
            try: streams.append(zlib.decompress(match.group(1)))
            except zlib.error: pass
        assert any(b' m' in data and b' l' in data for data in streams)  # native PDF paths, not a canvas screenshot
        from probemap.registration import desktop_ephys_board
        original_picker=desktop_ephys_board.QFileDialog.getSaveFileName
        desktop_ephys_board.QFileDialog.getSaveFileName=lambda *args: (str(tmp_path/'interactive.pdf'),'PDF (*.pdf)')
        try:
            board.export_pdf(); assert page._busy and page.activity.running
            wait_until(app,lambda:not page._busy)
            assert (tmp_path/'interactive.pdf').is_file() and not list(tmp_path.glob('*.partial-*'))
        finally: desktop_ephys_board.QFileDialog.getSaveFileName=original_picker
        page.grab().save(str(tmp_path/'probemap-aligned-board.png'))
        levels.table.selectRow(0); levels.remove_selected(); assert not levels.records()
        board.set_checked(['lfp','rms']); page.capture_state()
        assert not page.activity.running
        assert page.activity.percent.text() == '100%'
        # Every task installs a new cancellation token. The button must cancel
        # the current token, while the owner and other pages stay usable.
        import time
        def cancellable(progress, cancel):
            progress(10.25, 'Preparing signal')
            while not cancel(): time.sleep(.005)
            raise InterruptedError('Cancelled')
        page._task(cancellable, lambda value: None)
        wait_until(app, lambda:page.progress.value() == 10)
        assert not page.panel.isEnabled() and owner.isEnabled()
        assert page.activity.percent.text() == '10%'
        assert page.activity.timing.value == 10.25
        assert page.progress.minimum() == 0 and page.progress.maximum() == 100
        now = page.activity.timing.clock()
        monkeypatch.setattr(page.activity.timing, 'clock', lambda: now)
        page.activity.timing.started = now - 65
        page.activity.tick()
        assert '01:05' in page.activity.elapsed_label.text()
        assert page.activity.remaining_label.text()
        page.grab().save(str(tmp_path/'probemap-ephys-progress.png'))
        page.cancel.click(); wait_until(app,lambda:not page._busy)
        assert page.splitter.isEnabled() and page.message.text().startswith('Cancelled')
    finally:
        page.stop(); QThreadPool = __import__('qtpy.QtCore',fromlist=['QThreadPool']).QThreadPool
        QThreadPool.globalInstance().waitForDone(3000); page.close(); owner.close()


def test_partial_final_window_and_source_change_invalidate_cache(tmp_path):
    from probemap.registration.ephys_analysis import fingerprint, boundary_candidates
    _, rec = recording(tmp_path, seconds=41)
    result = prepare_recording(rec, tmp_path/'out')
    info, data = load_summary(result)
    assert data['band_windows'].shape[0] == 2
    assert data['ap_rms'].shape[1] == 21
    assert info['execution']['source_samples_read'] == rec['n_samples']
    assert info['timings_s']['source_read'] > 0
    assert np.load(Path(result)/'lfp.npy', mmap_mode='r').shape == (16,41000)
    original = fingerprint([rec['meta_path'], rec['bin_path']])
    with Path(rec['bin_path']).open('r+b') as stream:
        stream.write(bytes([3,0]))
    assert fingerprint([rec['meta_path'], rec['bin_path']]) != original
    heights=np.arange(20)*15.
    assert not len(boundary_candidates(heights,np.ones((3,5,20))))
    from threading import Event
    cancelled = Event()
    def stop_after_first_block(percent, message):
        if percent > 0 and message.startswith('Preparing signal'): cancelled.set()
    with pytest.raises(InterruptedError):
        prepare_recording(rec,tmp_path/'cancel-prefetch',stop_after_first_block,cancelled.is_set)
    assert not list((tmp_path/'cancel-prefetch').rglob('manifest.json'))
    assert not list((tmp_path/'cancel-prefetch').rglob('*.partial-*'))


def test_parallel_filtering_matches_serial_and_keeps_two_second_ap_bins(tmp_path):
    from scipy import signal
    from probemap.registration.ephys_analysis import process_channel_group
    _, rec = recording(tmp_path, seconds=8, n_channels=64)
    result = prepare_recording(rec, tmp_path/'out')
    info, metrics = load_summary(result)
    source = np.memmap(rec['bin_path'], dtype='<i2', mode='r', shape=(rec['n_samples'],rec['n_saved_channels']))
    fs = rec['sampling_rate']
    filters = (signal.butter(3,300,fs=fs,btype='highpass',output='sos'),
               signal.tf2sos(*signal.iirnotch(50,30,fs)),
               signal.butter(4,300,fs=fs,btype='lowpass',output='sos'))
    expected_lfp, expected_rms = process_channel_group(source,[c['file_index'] for c in rec['channels']],
        rec['uv_per_count'],filters,(1,30),0,8000,0,rec['n_samples'],60000)
    actual = np.load(Path(result)/'lfp.npy',mmap_mode='r',allow_pickle=False)
    np.testing.assert_allclose(actual,expected_lfp,rtol=1e-6,atol=1e-6)
    np.testing.assert_allclose(metrics['ap_rms'],expected_rms,rtol=1e-6,atol=1e-6)
    np.testing.assert_allclose(metrics['ap_times'],[1,3,5,7])
    assert 1 <= info['execution']['workers'] <= 4
    assert info['execution']['source_samples_read'] == rec['n_samples']
    assert info['timings_s']['total'] > 0


def test_upright_anatomy_uses_registered_native_probe_frame(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from probemap.registration import ephys_anatomy as anatomy
    from probemap.registration.neuropixels_model import geometry_for, probe_coordinates
    from probemap.registration.landmarks import transform_points
    geometry = geometry_for('NP2010')
    insertion = {'probe': {'geometry': geometry, 'pose': dict(
        tip0_source_zyx_um=[5000,200,300], down_source_zyx=[-1,0,0],
        lateral_source_zyx=[0,0,1], normal_source_zyx=[0,-1,0])}}
    original = copy.deepcopy(insertion)
    contacts = [dict(shank=s,row=r,column=r%2) for s in (1,3) for r in (300,301)]
    rec = dict(channels=contacts)
    matrix = np.array([[1.2,0,.2,20],[0,.9,0,10],[0,0,1.1,-30],[0,0,0,1.]])
    project = SimpleNamespace(total=matrix,reference=None,moving=None)
    queries = []
    def sample(volume, transform, points):
        queries.append(np.array(points)); return np.arange(len(points),dtype=float)
    monkeypatch.setattr(anatomy,'sample_world',sample)
    result = anatomy.prepare_anatomy(project,insertion,rec,1,tmp_path,
        resources=dict(atlas=object(),brain=object(),labels=None,surface=None))
    assert insertion == original
    assert result['extent'][2] == 0 and result['shanks'] == [0,1,2,3]
    assert result['extent'][3] == geometry['first_site_center_from_tip_um']+306*geometry['site_pitch_um']
    assert set(result['all_shanks']) == {1,3}
    left,right,low,high = result['extent']
    anchor = probe_coordinates(insertion['probe'],0,0.,face=True)
    expected = probe_coordinates(insertion['probe'],0,high,face=True) + left*np.array([0,0,1])
    np.testing.assert_allclose(queries[0][0],transform_points([expected],matrix)[0])
    expected_tip_row = anchor + right*np.array([0,0,1])
    np.testing.assert_allclose(queries[0][-1],transform_points([expected_tip_row],matrix)[0])
    np.testing.assert_allclose(queries[0],queries[1])
    assert result['images']['Atlas'].shape == (1024,256,3)


def test_buffered_reader_reuses_overlap_and_reports_in_block_progress(tmp_path):
    from probemap.registration.ephys_analysis import RecordingReader
    values = np.arange(100000*17,dtype=np.int16).reshape(-1,17)
    path = tmp_path/'source.bin'; values.tofile(path)
    reports = []
    reader = RecordingReader(path, values.shape, 200, lambda:False, lambda *args:reports.append(args))
    reader.READ_BYTES = 256*1024+1  # reads deliberately do not align to sample frames
    try:
        np.testing.assert_array_equal(reader.read(0,50100),values[:50100])
        np.testing.assert_array_equal(reader.read(49900,100000),values[49900:])
        assert reader.samples_read == len(values)
        assert reader.bytes_read == values.nbytes
        assert reports[-1][0] == len(values) and reports[-1][1] > 0
    finally: reader.close()
    assert reader.stream.closed
    # A truncated stream must fail, rather than silently treating missing data as zeros.
    path.write_bytes(b'\0'*8)
    reader = RecordingReader(path,(100,17),10,lambda:False)
    try:
        with pytest.raises(ValueError,match='ended'): reader.read(0,100)
    finally: reader.close()


def test_buffered_reader_can_cancel_between_large_network_reads(tmp_path):
    from threading import Event
    from probemap.registration.ephys_analysis import RecordingReader
    path = tmp_path/'source.bin'; np.zeros((100000,17),np.int16).tofile(path)
    cancelled = Event()
    reader = RecordingReader(path,(100000,17),200,cancelled.is_set,lambda *args:cancelled.set())
    reader.READ_BYTES = 256*1024
    try:
        with pytest.raises(InterruptedError): reader.read(0,100000)
        assert 0 < reader.bytes_read < path.stat().st_size
    finally: reader.close()


@pytest.mark.gui
def test_whole_probe_colors_region_editor_hover_and_panel_width(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from qtpy.QtCore import Qt, QThreadPool
    from qtpy.QtWidgets import QWidget
    from probemap.registration.macos_qt import initialize_qt
    from probemap.registration.desktop_ephys import ElectrophysiologyWorkspace
    from probemap.registration import desktop_ephys
    from probemap.registration.desktop_app import STYLE
    from probemap.registration.ephys_anatomy import probe_site_layout
    from probemap.registration.desktop_ephys_probe import hover_contact
    from probemap.registration.insertion_model import insertion_session, new_insertion
    from probemap.registration.neuropixels_model import probe_settings, probe_coordinates
    from test_desktop_app import wait_until
    from test_track_model import make_track_project
    app = initialize_qt()
    class Owner(QWidget):
        def schedule_save(self): pass
        def save_project(self): pass
        def go_step(self, step): pass
    owner = Owner(); page = ElectrophysiologyWorkspace(owner); page.setStyleSheet(STYLE+page.styleSheet())
    project = make_track_project(tmp_path/'project'); session = insertion_session(project)
    item = new_insertion(session); item['color'] = '#ee77aa'; settings = probe_settings(item)
    settings['pose'] = dict(tip0_source_zyx_um=[5000,0,0],down_source_zyx=[1,0,0],lateral_source_zyx=[0,0,1],normal_source_zyx=[0,1,0])
    session.setdefault('regions',{})['1'] = dict(label=False,visible=True)
    previous_regions = copy.deepcopy(session['regions'])
    _, rec = recording(tmp_path/'input'); rec['insertion_id'] = item['id']; project.ephys['recordings'] = [rec]
    for contact in rec['channels']: contact['row'] += 160
    geometry = settings['geometry']; sites = probe_site_layout(geometry)
    expected_top = max(geometry['first_site_center_from_tip_um']+c['row']*geometry['site_pitch_um'] for c in rec['channels'])+5*geometry['site_pitch_um']
    # Synthetic anatomy makes visibility and label placement observable without a server.
    yy,xx = np.meshgrid(np.linspace(10000,0,1024),np.linspace(-140,890,256),indexing='ij')
    mask = ((xx-350)/900)**2+((yy-1500)/1900)**2 < 1
    ids = np.where(mask,np.where(yy<1700,1,2),0)
    gray = np.where(mask,.65+.1*np.sin(yy/200),0)
    image = np.repeat(gray[...,None],3,axis=2)
    regions = [dict(id=1,name='Region A',color_rgb=[100,180,220]),dict(id=2,name='Region B',color_rgb=[190,120,210])]
    heights = np.array([geometry['first_site_center_from_tip_um']+c['row']*geometry['site_pitch_um'] for c in rec['channels']])
    anatomy = dict(resources={},geometry=geometry,sites=sites,extent=[-140,890,0,10000],focus_high=3500,
        images=dict(Atlas=image,Scan=image),label_ids=ids,label_regions=regions,region_positions={'1':[300,1000],'2':[350,2400]},
        overlay=np.zeros((1024,256,4)),depths={s:3200+s*60 for s in range(4)},depth_um=3200,
        heights=heights,contact_regions=[1]*len(heights),regions={'1':'Region A','2':'Region B'},
        first_site_um=geometry['first_site_center_from_tip_um'],pitch_um=geometry['site_pitch_um'])
    monkeypatch.setattr(desktop_ephys,'prepare_anatomy',lambda *args,**kwargs:anatomy)
    page.resize(1600,1000); page.show(); page.open(project)
    try:
        wait_until(app,lambda:not page._busy)
        assert page.record_color(rec) == item['color']
        assert page.table.item(0,5).data(Qt.ItemDataRole.UserRole) == item['color']
        assert len(sites['heights']) == 4*1280
        assert len(page.anatomy_ax.patches) == 4
        assert page.anatomy_ax.get_ylim() == (0,expected_top)
        assert page.anatomy_ax.get_title() == 'Atlas' and page.anatomy_caption.text() == 'Atlas'
        # Contact positions use the physical column offset, not a symmetric approximation.
        c = rec['channels'][0]
        height = geometry['first_site_center_from_tip_um']+c['row']*geometry['site_pitch_um']
        x = c['shank']*geometry['shank_pitch_um']+geometry['left_site_center_um']+c['column']*geometry['column_pitch_um']-geometry['shank_width_um']/2
        native = probe_coordinates(settings,c['shank'],height,c['column'],face=True)
        np.testing.assert_allclose(native, probe_coordinates(settings,0,0,face=True)-height*np.array(settings['pose']['down_source_zyx'])+x*np.array(settings['pose']['lateral_source_zyx']))
        page.anatomy_canvas.draw()
        px,py = page.anatomy_ax.transData.transform([x,height])
        hover_contact(page,SimpleNamespace(inaxes=page.anatomy_ax,x=px,y=py))
        assert f'{height:g} µm from tip' in page._hover_text and 'Recording' in page._hover_text
        page.labels.setChecked(True); page.regions_toggle.setChecked(True)
        exported,pdf=page.plot_board.export_pdf(tmp_path/'.hidden-export.pdf')
        assert pdf.name=='hidden-export.pdf' and pdf.is_file()
        region_paths=[patch for patch in exported.axes[0].patches if (patch.get_gid() or '').startswith('region-')]
        assert {patch.get_gid() for patch in region_paths}=={'region-1','region-2'}
        assert len(exported.axes[0].images)==1  # Only the scan is rasterized.
        import stat
        if hasattr(pdf.stat(),'st_flags') and hasattr(stat,'UF_HIDDEN'):
            assert not pdf.stat().st_flags & stat.UF_HIDDEN
        table = page.region_editor.region_table
        row = next(r for r in range(table.rowCount()) if table.item(r,2).data(Qt.ItemDataRole.UserRole)==1)
        table.item(row,1).setCheckState(Qt.CheckState.Checked)
        table.item(row,2).setText('Custom region')
        wait_until(app,lambda:any(t.get_text()=='Custom region' for t in page.anatomy_ax.texts))
        assert any(t.get_text()=='Custom region' for t in page.anatomy_ax.texts)
        table.item(row,0).setCheckState(Qt.CheckState.Unchecked)
        wait_until(app,lambda:not any(t.get_text()=='Custom region' for t in page.anatomy_ax.texts))
        assert not any(t.get_text()=='Custom region' for t in page.anatomy_ax.texts)
        overlay = np.asarray(page.anatomy_ax.images[-1].get_array())
        assert np.all(overlay[ids==1,3] == 0)
        assert session['regions'] == previous_regions  # changes are local to ephys display
        page.splitter.setSizes([650,950]); app.processEvents()
        assert page.splitter.sizes()[0] > 520
        assert page.anatomy_ax.get_ylim() == (0,expected_top)
        page.probe_opacity.setValue(30); page.inactive_contacts.setChecked(False)
        wait_until(app,lambda:not page.plot_board.settings_timer.isActive())
        assert len(page.anatomy_ax.collections) == 1
        board=page.plot_board
        board.columns['1'].setChecked(False)
        active=page.anatomy_ax.collections[-1].get_offsets()
        expected=[c for c in rec['channels'] if c['column']==0]
        assert len(active)==len(expected)
        np.testing.assert_allclose(sorted(active[:,0]),sorted(c['shank']*geometry['shank_pitch_um']+geometry['left_site_center_um']-geometry['shank_width_um']/2 for c in expected))
        assert len(page.anatomy_ax.patches)==4  # all probe bodies remain visible
        board.columns['mean'].setChecked(True)
        assert len(page.anatomy_ax.collections[-1].get_offsets())==len(rec['channels'])
        board.columns['mean'].setChecked(False)
        board.columns['0'].setChecked(False)
        assert len(page.anatomy_ax.collections[-1].get_offsets())==0
        board.columns['0'].setChecked(True); board.columns['1'].setChecked(True)
        page.capture_state(); page.open(project); wait_until(app,lambda:not page._busy)
        assert page.anatomy_ax.get_ylim() == (0,expected_top) and page.probe_opacity.value() == 30
        assert not page.inactive_contacts.isChecked() and page.regions_toggle.isChecked()
        assert project.ephys['anatomy_display']['regions']['1']['name'] == 'Custom region'
        table = page.region_editor.region_table
        row = next(r for r in range(table.rowCount()) if table.item(r,2).data(Qt.ItemDataRole.UserRole)==1)
        table.item(row,0).setCheckState(Qt.CheckState.Checked)
        page.splitter.setSizes([480,1120]); app.processEvents(); page.grab().save(str(tmp_path/'probemap-probe-focus-display.png'))
        page.inactive_contacts.setChecked(True); app.processEvents()
        page.grab().save(str(tmp_path/'probemap-whole-probe-display.png'))
    finally:
        page.stop(); QThreadPool.globalInstance().waitForDone(3000); page.close(); owner.close()


def test_electrode_column_mean_is_linear_and_preserves_missing_rows():
    from probemap.registration.desktop_ephys_plots import column_series
    record=dict(channels=[dict(column=0),dict(column=1),dict(column=0),dict(column=1)])
    groups=dict((k,(h,v)) for k,h,v in column_series(record,np.arange(4),np.array([217,217,247,247]),np.array([1.,9.,4.,16.]),['0','1','mean']))
    np.testing.assert_allclose(groups['0'][1],[1,4]); np.testing.assert_allclose(groups['1'][1],[9,16])
    np.testing.assert_allclose(groups['mean'][1],[5,10])
    np.testing.assert_allclose(groups['mean'][0],[217,247])
    assert not np.allclose(10*np.log10(groups['mean'][1]),(10*np.log10(groups['0'][1])+10*np.log10(groups['1'][1]))/2)


@pytest.mark.gui
def test_notebook_workspace_seven_plots_cache_legends_and_pdf(tmp_path,monkeypatch):
    from probemap.registration.macos_qt import initialize_qt
    from qtpy.QtWidgets import QWidget
    from test_desktop_app import wait_until
    from test_track_model import make_track_project
    from probemap.registration.desktop_ephys import ElectrophysiologyWorkspace
    import probemap.registration.ephys_notebook as notebook
    app=initialize_qt()
    class Owner(QWidget):
        def schedule_save(self): pass
        def save_project(self): pass
        def go_step(self,step): pass
    from probemap.registration.desktop_app import STYLE
    owner=Owner(); owner.setStyleSheet(STYLE); page=ElectrophysiologyWorkspace(owner); page.setStyleSheet(STYLE+page.styleSheet())
    project=make_track_project(tmp_path/'project')
    folder,rec=recording(tmp_path/'input',n_channels=32)
    rec['cache_path']=prepare_recording(rec,tmp_path/'out'); project.ephys={'recordings':[rec]}
    # Numeric pwlf equivalence is tested independently; use inexpensive fixed fits for UI QA.
    monkeypatch.setattr(notebook,'fit_profile',lambda h,v:(np.linspace(h.min(),h.max(),7),np.asarray(v)))
    page.resize(1800,1050); page.show(); page.open(project)
    try:
        wait_until(app,lambda:not page._busy)
        board=page.plot_board; board.notebook_mode.setChecked(True)
        keys=['lfp','spectrum','correlation','correlation_profile','traces','low40','csd']
        board.set_checked(keys); app.processEvents(); wait_until(app,lambda:not page._busy)
        page._load_notebook(); wait_until(app,lambda:not page._busy,timeout=30)
        assert page.notebook is not None
        board.column_opacity['0'].setValue(35); board.column_opacity['1'].setValue(80)
        for key in keys:
            for ax in board.panels[key]['axes']:
                for artist in [*ax.lines,*ax.collections]:
                    if hasattr(artist,'_ephys_column'): assert artist.get_alpha()==board.opacity(artist._ephys_column)
        assert (Path(page.current()['cache_path'])/'cached_window.npz').is_file()
        board.align(); app.processEvents()
        for key in keys:
            panel=board.panels[key]; panel['canvas'].draw()
            assert panel['axes'] and panel['figure'].legends
            assert not any('Calculate the' in text.get_text() for ax in panel['axes'] for text in ax.texts)
            for legend in panel['figure'].legends:
                assert legend.get_window_extent().y1 < panel['axes'][0].bbox.y0-20
        assert len(board.panels['correlation_profile']['axes'][0].lines)>1
        assert board.panels['csd']['axes'][0].collections
        mesh=board.panels['csd']['axes'][0].collections[0]
        scale=board.panels['csd']['figure']._ephys_colorbars[0]['bar']
        assert scale.orientation=='horizontal' and 0 in scale.get_ticks()
        assert np.all(np.diff(scale.get_ticks())>0)
        np.testing.assert_allclose(scale.get_ticks()[[0,-1]],mesh.get_clim())
        assert scale.solids.get_rasterized() is False
        assert scale.ax.get_xlabel()=='CSD · relative units'
        assert len(board.panels['spectrum']['axes'])==2  # colourbar is not a data facet
        heights_before=[ax.bbox.height for ax in [page.anatomy_ax,*page._depth_axes]]
        board.legends.setChecked(False); app.processEvents(); board.align()
        for ax,before in zip([page.anatomy_ax,*page._depth_axes],heights_before): assert ax.bbox.height>before+50
        assert not any(c.figure.legends or getattr(c.figure,'_ephys_colorbars',[]) for c in board.canvases())
        hidden_state=board.state(); assert hidden_state['legends'] is False
        board.legends.setChecked(True); board.restore(hidden_state); page.draw(); app.processEvents()
        assert not board.legends.isChecked()
        hidden_figure,_=board.export_pdf(tmp_path/'without-legends.pdf')
        assert not hidden_figure.legends and not getattr(hidden_figure,'_ephys_colorbars',[])
        board.legends.setChecked(True); app.processEvents(); board.align()
        board.set_checked(['lfp','correlation_profile','csd']); board.columns['1'].setChecked(False)
        page.resize(1650,1000); app.processEvents(); board.align()
        for canvas in board.canvases(): canvas.draw()
        page.grab().save(str(tmp_path/'probemap-notebook-workspace.png'))
        figure,pdf=board.export_pdf(tmp_path/'notebook.pdf'); figure.canvas.draw()
        assert pdf.is_file() and len(figure.legends)>=3
        # A saved project reloads this derived result without another source read.
        stamp=(Path(page.current()['cache_path'])/'cached_window.npz').stat().st_mtime_ns
        monkeypatch.setattr(notebook,'source_window',lambda *args,**kwargs:(_ for _ in ()).throw(AssertionError('must reuse cache')))
        monkeypatch.setattr(notebook,'cached_window',lambda *args,**kwargs:(_ for _ in ()).throw(AssertionError('must reuse cache')))
        page.capture_state(); page.open(project)
        wait_until(app,lambda:page.notebook is not None and not page._busy,timeout=15)
        assert stamp==(Path(page.current()['cache_path'])/'cached_window.npz').stat().st_mtime_ns
    finally:
        page.stop(); page.close(); owner.close(); app.processEvents()


@pytest.mark.gui
def test_contextual_plot_inspectors_time_navigation_and_signal_settings(tmp_path,monkeypatch):
    from qtpy.QtCore import Qt,QThreadPool
    from qtpy.QtWidgets import QWidget,QLabel,QPushButton,QCheckBox,QToolButton
    from probemap.registration.macos_qt import initialize_qt
    from probemap.registration.desktop_ephys import ElectrophysiologyWorkspace
    from probemap.registration.desktop_app import STYLE
    from probemap.registration.desktop_ephys_settings import defaults
    from probemap.registration.desktop_ephys_plots import series_style
    import probemap.registration.ephys_notebook as signal_module
    import probemap.registration.ephys_profiles as profile_module
    from test_track_model import make_track_project
    from test_desktop_app import wait_until
    app=initialize_qt()
    class Owner(QWidget):
        def schedule_save(self): pass
        def save_project(self): pass
        def go_step(self,step): pass
    def simple_fit(h,v,segments=6): return np.linspace(min(h),max(h),segments+1),np.asarray(v)
    monkeypatch.setattr(signal_module,'fit_profile',simple_fit); monkeypatch.setattr(profile_module,'fit_profile',simple_fit)
    owner=Owner(); owner.setStyleSheet(STYLE); page=ElectrophysiologyWorkspace(owner); page.setStyleSheet(STYLE+page.styleSheet())
    project=make_track_project(tmp_path/'project'); folder,rec=recording(tmp_path/'input',seconds=8,n_channels=32)
    rec['cache_path']=prepare_recording(rec,tmp_path/'out'); project.ephys={'recordings':[rec]}
    page.resize(1600,1050); page.show(); page.open(project)
    try:
        wait_until(app,lambda:page.profile_fits is not None and not page._busy,timeout=30)
        board=page.plot_board
        assert [page.settings_tabs.tabText(i) for i in range(4)]==['Recordings','Plots','Anatomy','Marks']
        for cls in (QLabel,QPushButton,QCheckBox,QToolButton):
            assert not any('notebook' in widget.text().lower() or 'notebook' in widget.toolTip().lower() for widget in page.findChildren(cls))
        assert board.active_list.count()==2 and board.list.isHidden()
        page.settings_tabs.setCurrentIndex(1); inspector=board.inspectors['lfp']; inspector.toggle.setChecked(True)
        inspector.fields['auto_range'].setChecked(False); inspector.fields['xmin'].setValue(-15); inspector.fields['xmax'].setValue(35)
        wait_until(app,lambda:not board.settings_timer.isActive())
        np.testing.assert_allclose(board.panels['lfp']['axes'][0].get_xlim(),[-15,35])
        assert any(getattr(line,'_ephys_role',None)=='fit' for line in board.panels['lfp']['axes'][0].lines)
        inspector.fields['transitions'].setChecked(True)
        wait_until(app,lambda:not board.settings_timer.isActive())
        assert any(getattr(line,'_ephys_role',None)=='transition' for line in board.panels['lfp']['axes'][0].lines)
        inspector.bands.item(0,0).setCheckState(Qt.CheckState.Unchecked)
        wait_until(app,lambda:not board.settings_timer.isActive())
        assert not any(getattr(line,'_ephys_band',None)==0 for line in board.panels['lfp']['axes'][0].lines)
        inspector.bands.item(0,0).setCheckState(Qt.CheckState.Checked); inspector.bands.item(0,3).setText('#ff0000')
        wait_until(app,lambda:not board.settings_timer.isActive())
        line=next(l for l in board.panels['lfp']['axes'][0].lines if getattr(l,'_ephys_band',None)==0)
        np.testing.assert_allclose(line.get_color(),series_style('#ff0000',line._ephys_column)['color'])
        board.toggle_plot('lfp',False); assert inspector.isHidden()
        assert board.options['lfp']['xmin']==-15 and board.options['lfp']['bands'][0]['color']=='#ff0000'
        board.toggle_plot('lfp',True); assert not inspector.isHidden() and inspector.fields['xmin'].value()==-15
        inspector.bands.item(0,2).setText('6'); wait_until(app,lambda:page.profile_fits is not None and not page._busy)
        assert board.plot_options('lfp')['bands'][0]['high']==6
        inspector.reset(); wait_until(app,lambda:page.profile_fits is not None and not page._busy)
        assert board.plot_options('lfp')==defaults('lfp')
        board.toggle_plot('lfp_gradient',True)
        assert board.panels['lfp_gradient']['axes'][0].lines
        board.toggle_plot('spectrum',True); spectrum=board.inspectors['spectrum']
        spectrum.fields['scale'].setCurrentText('log'); spectrum.fields['auto_range'].setChecked(False)
        spectrum.fields['xmin'].setValue(1); spectrum.fields['xmax'].setValue(200)
        spectrum.fields['auto_color'].setChecked(False); spectrum.fields['cmin'].setValue(-80); spectrum.fields['cmax'].setValue(40)
        spectrum.fields['cmap'].setCurrentText('cividis')
        wait_until(app,lambda:not board.settings_timer.isActive())
        ax=board.panels['spectrum']['axes'][0]; assert ax.get_xscale()=='log'; np.testing.assert_allclose(ax.get_xlim(),[1,200])
        np.testing.assert_allclose(ax.collections[0].get_clim(),[-80,40]); assert ax.collections[0].cmap.name=='cividis'
        board.toggle_plot('spectrum',False); board.toggle_plot('spectrum',True)
        assert spectrum.fields['cmap'].currentText()=='cividis'
        board.set_checked(['lfp','lfp_gradient','spectrum']); inspector.toggle.setChecked(True); spectrum.toggle.setChecked(False)
        app.processEvents(); page.grab().save(str(tmp_path/'probemap-contextual-plots.png'))
        board.notebook_mode.setChecked(True)
        wait_until(app,lambda:page.notebook is not None and not page._busy,timeout=30)
        page.time_slider.setValue(150)
        wait_until(app,lambda:page.notebook is not None and float(page.notebook['start'])==1.5 and not page._busy,timeout=30)
        assert page.time.value()==1.5
        page.window_duration.setValue(1.)
        board.analysis_fields['nperseg'].setCurrentText('512'); board.analysis_fields['nfft'].setCurrentText('2048')
        board.analysis_fields['c_factor'].setChecked(False); board.analysis_fields['reference'].setChecked(True)
        wait_until(app,lambda:page.notebook is not None and float(page.notebook.get('duration',0))==1 and len(page.notebook['0_frequencies'])==1025 and not page._busy,timeout=30)
        assert len(page.notebook['0_lfp'][0])<1100
        board.column_opacity['0'].setValue(60); page.capture_state(); saved=copy.deepcopy(project.ephys['view']['plot_board'])
        page.open(project); wait_until(app,lambda:page.notebook is not None and not page._busy,timeout=30)
        assert board.state()['plot_settings']==saved['plot_settings'] and page.window_duration.value()==1.
        assert not board.analysis_fields['c_factor'].isChecked() and board.analysis_fields['reference'].isChecked()
        assert board.opacity('0')==.6 and board.inspectors['spectrum'].fields['cmap'].currentText()=='cividis'
    finally:
        page.stop(); QThreadPool.globalInstance().waitForDone(5000); page.close(); owner.close(); app.processEvents()

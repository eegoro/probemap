"""Cache identity, crash recovery, overwrite and migration without signal loss."""
import copy
import json
import os
from pathlib import Path
import shutil
from threading import Event

import pytest
from test_ephys import recording
from probemap.registration.ephys_analysis import prepare_recording,load_summary
from probemap.registration.ephys_cache import (recording_root,cache_lock,migrate_recording_cache,
    complete_signal,publish_cache)


def test_named_cache_reuses_one_result_and_replaces_only_after_success(tmp_path):
    folder,rec=recording(tmp_path/'input'); output=tmp_path/'out'
    target=Path(prepare_recording(rec,output))
    assert target==output/'ephys'/folder.name/'signal'
    assert complete_signal(target)
    stamp=(target/'lfp.npy').stat().st_mtime_ns
    assert prepare_recording(rec,output)==str(target)
    assert (target/'lfp.npy').stat().st_mtime_ns==stamp
    original=(target/'manifest.json').read_bytes(); cancelled=Event()
    def stop(percent,message):
        if message.startswith('Preparing signal') and percent>0: cancelled.set()
    with pytest.raises(InterruptedError): prepare_recording(rec,output,stop,cancelled.is_set,overwrite=True)
    assert (target/'manifest.json').read_bytes()==original and (target/'lfp.npy').stat().st_mtime_ns==stamp
    assert not list(target.parent.glob('.*.partial'))
    prepare_recording(rec,output,overwrite=True)
    assert (target/'lfp.npy').stat().st_mtime_ns!=stamp
    assert not list(target.parent.glob('.*.previous'))
    assert len(list(output.rglob('lfp.npy')))==1


def test_changed_source_requires_explicit_overwrite(tmp_path):
    _,rec=recording(tmp_path/'input'); output=tmp_path/'out'
    target=Path(prepare_recording(rec,output)); old=load_summary(target)[0]['key']
    with Path(rec['bin_path']).open('r+b') as stream: stream.write(b'\x01\x00')
    with pytest.raises(FileExistsError): prepare_recording(rec,output)
    assert load_summary(target)[0]['key']==old
    assert prepare_recording(rec,output,overwrite=True)==str(target)
    assert load_summary(target)[0]['key']!=old
    assert len(list(output.rglob('manifest.json')))==1


def test_legacy_cache_migration_removes_only_abandoned_work(tmp_path,monkeypatch):
    from probemap.registration import ephys_cache
    folder,rec=recording(tmp_path/'input'); output=tmp_path/'out'
    target=Path(prepare_recording(rec,output)); root=target.parent
    key=load_summary(target)[0]['key']; legacy=output/'ephys'/rec['id']; legacy.mkdir()
    source=legacy/key; target.rename(source)
    for p in root.iterdir(): p.unlink()
    root.rmdir(); rec.pop('cache_root'); rec['cache_path']=str(source)
    original=(source/'manifest.json').read_bytes()
    stale=legacy/(key+'.partial-old'); stale.mkdir(); (stale/'lfp.npy').write_bytes(b'old work')
    stamp=(source/'manifest.json').stat().st_mtime-100; os.utime(stale/'lfp.npy',(stamp,stamp))
    active=legacy/(key+'.partial-active'); active.mkdir(); (active/'lfp.npy').write_bytes(b'active work'); os.utime(active/'lfp.npy',(stamp,stamp))
    monkeypatch.setattr(ephys_cache,'_in_use',lambda path:path==active)
    result=migrate_recording_cache(rec,output)
    assert result['removed_bytes']==len(b'old work') and not stale.exists() and active.exists()
    assert Path(rec['cache_path'])==output/'ephys'/folder.name/'signal'
    assert (Path(rec['cache_path'])/'manifest.json').read_bytes()==original
    assert complete_signal(rec['cache_path'])
    monkeypatch.setattr(ephys_cache,'_in_use',lambda path:False)
    (legacy/'.DS_Store').write_bytes(b'Finder metadata')
    migrate_recording_cache(rec,output); assert not legacy.exists()
    # An old project file may still point at the former hash directory: repair it.
    rec['cache_path']=str(source); rec.pop('cache_root')
    migrate_recording_cache(rec,output)
    assert complete_signal(rec['cache_path']) and Path(rec['cache_path']).name=='signal'


def test_recover_interrupted_publish_and_clear_stale_partial(tmp_path):
    _,rec=recording(tmp_path/'input'); output=tmp_path/'out'
    target=Path(prepare_recording(rec,output)); rec['cache_path']=str(target)
    previous=target.with_name('.signal.previous'); target.rename(previous)
    stage=target.with_name('.signal.partial'); stage.mkdir(); (stage/'lfp.npy').write_bytes(b'unfinished')
    migrate_recording_cache(rec,output)
    assert complete_signal(target) and not previous.exists() and not stage.exists()


def test_same_folder_names_from_different_sources_do_not_share_cache(tmp_path):
    _,one=recording(tmp_path/'one'); _,two=recording(tmp_path/'two'); output=tmp_path/'out'
    first=prepare_recording(one,output); second=prepare_recording(two,output)
    assert first!=second and complete_signal(first) and complete_signal(second)
    again=copy.deepcopy(one); again['id']='new-import-id'; again.pop('cache_root',None)
    assert prepare_recording(again,output)==first


def test_cache_lock_rejects_another_writer(tmp_path):
    _,rec=recording(tmp_path/'input'); output=tmp_path/'out'
    with cache_lock(rec,output):
        with pytest.raises(RuntimeError,match='another ProbeMap'): prepare_recording(rec,output)
    assert complete_signal(prepare_recording(rec,output))


def test_publish_failure_restores_previous_cache(tmp_path,monkeypatch):
    target=tmp_path/'signal'; target.mkdir(); (target/'manifest.json').write_text('{}')
    stage=tmp_path/'.signal.partial'; stage.mkdir()
    real_rename=Path.rename
    def fail_stage(path,destination):
        if path==stage: raise OSError('simulated interrupted publish')
        return real_rename(path,destination)
    monkeypatch.setattr(Path,'rename',fail_stage)
    with pytest.raises(OSError): publish_cache(stage,target)
    assert (target/'manifest.json').is_file() and not (tmp_path/'.signal.previous').exists()

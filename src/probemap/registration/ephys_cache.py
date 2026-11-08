"""Readable recording folders, exclusive preparation and recoverable cache replacement."""
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
import hashlib
import json
import os
import re
import shutil
import subprocess


def safe_name(value):
    value=re.sub(r'[\x00-\x1f/\\\\:*?"<>|]','_',str(value)).strip(' .')[:180]
    if value.split('.')[0].upper() in {'CON','PRN','AUX','NUL',*[f'COM{i}' for i in range(1,10)],*[f'LPT{i}' for i in range(1,10)]}: value='_'+value
    return value if value and value not in ('atlas-labels','figures') else (value or 'recording')+'_recording'


def _json(path):
    try: return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError,ValueError): return {}


def recording_root(record,output,create=False):
    base=Path(output)/'ephys'; name=safe_name(Path(record.get('folder') or Path(record['bin_path']).parent).name)
    owner=dict(record_id=record['id'],source_bin=str(record['bin_path']),source_meta=str(record['meta_path']))
    def matches(value): return (value.get('record_id')==record['id'] or value.get('source_bin')==owner['source_bin'] or
                               (record.get('portable') and value.get('source_bin')==record.get('portable_source_bin')))
    candidates=[]
    stored=record.get('cache_root')
    if stored and (Path(stored).parent.resolve()==base.resolve() or
                   (record.get('bundle_signature') and Path(stored).resolve().is_relative_to(Path(output).resolve().parent/'ephys'))):
        candidates.append(Path(stored))
    suffix=hashlib.sha256(owner['source_bin'].encode()).hexdigest()[:8]
    candidates += [base/name,base/(name+'__'+safe_name(Path(record['bin_path']).stem)),base/(name+'__'+suffix)]
    for root in dict.fromkeys(candidates):
        if root.is_symlink(): continue
        existing=_json(root/'recording.json')
        if root.exists() and not matches(existing): continue
        if create:
            root.mkdir(parents=True,exist_ok=True)
            try:
                with (root/'recording.json').open('x',encoding='utf-8') as stream: json.dump(owner,stream,indent=2)
            except FileExistsError:
                if not matches(_json(root/'recording.json')): continue
            record['cache_root']=str(root)
        return root
    raise FileExistsError('Recording cache folder belongs to another source. Choose a different output folder.')


@contextmanager
def cache_lock(record,output):
    from ..platform_io import exclusive_file_lock
    root=recording_root(record,output,create=True)
    with exclusive_file_lock(root/'.prepare.lock'):
        yield root


def locked_preparation(output_index):
    def decorate(function):
        @wraps(function)
        def wrapped(record,*args,**kwargs):
            output=args[output_index] if len(args)>output_index else kwargs['output']
            with cache_lock(record,output): return function(record,*args,**kwargs)
        return wrapped
    return decorate


def sorting_target(record,sorting,output):
    base=recording_root(record,output)/'sortings'; name=safe_name(Path(sorting['path']).name)
    candidate=base/name; info=_json(candidate/'manifest.json')
    if candidate.exists() and info.get('source_path') not in (None,str(sorting['path'])) and not (sorting.get('id') is not None and info.get('sorting_id')==sorting['id']) and sorting.get('cache_path')!=str(candidate):
        candidate=base/(name+'__'+hashlib.sha256(str(sorting['path']).encode()).hexdigest()[:8])
    return candidate


def recover_cache(target):
    target=Path(target); previous=target.with_name('.'+target.name+'.previous')
    if previous.exists():
        if not target.exists(): previous.rename(target)
        elif (target/'manifest.json').is_file(): shutil.rmtree(previous)
        else:
            shutil.rmtree(target); previous.rename(target)


def staging_folder(target):
    target=Path(target); recover_cache(target)
    stage=target.with_name('.'+target.name+'.partial')
    if stage.exists(): shutil.rmtree(stage)
    stage.mkdir(parents=True)
    return stage


def publish_cache(stage,target):
    """Keep the previous complete result until its replacement is complete."""
    stage,target=Path(stage),Path(target); recover_cache(target)
    previous=target.with_name('.'+target.name+'.previous')
    if target.exists(): target.rename(previous)
    try: stage.rename(target)
    except BaseException:
        if previous.exists() and not target.exists(): previous.rename(target)
        raise
    if previous.exists(): shutil.rmtree(previous)


def complete_signal(folder):
    folder=Path(folder); info=_json(folder/'manifest.json')
    if not info or not (folder/'profiles.npz').is_file() or not (folder/'lfp.npy').is_file(): return False
    try:
        import numpy as np
        with np.load(folder/'profiles.npz',allow_pickle=False) as values:
            if 'bands' not in values.files or 'ap_rms' not in values.files: return False
            values['bands']; values['ap_rms']
        lfp=np.load(folder/'lfp.npy',mmap_mode='r',allow_pickle=False)
        return lfp.shape==(len(info['channels']),info['n_samples']) and lfp.dtype==np.float32
    except (OSError,ValueError,KeyError,EOFError): return False


def _in_use(folder):
    command=shutil.which('lsof')
    if not command: return True  # Never remove an old process's work without checking.
    files=[str(p) for p in folder.iterdir() if p.is_file()]
    if not files: return False
    try:
        result=subprocess.run([command,'-t','--',*files],capture_output=True,text=True,timeout=5)
        return bool(result.stdout.strip() or result.stderr.strip() or result.returncode not in (0,1))
    except (OSError,subprocess.TimeoutExpired): return True


def migrate_recording_cache(record,output):
    """Move legacy ID/hash caches without reading or reprocessing the source binary."""
    base=Path(output)/'ephys'; legacy=base/record['id']
    current=Path(record['cache_path']) if record.get('cache_path') else None
    named=recording_root(record,output)
    if not legacy.is_dir() and not named.exists(): return dict(changed=False,removed_bytes=0)
    changed=False; removed=0
    with cache_lock(record,output) as root:
        target=root/'signal'; recover_cache(target)
        for previous in root.rglob('.*.previous'):
            recover_cache(previous.with_name(previous.name[1:-len('.previous')]))
        if not target.exists() and current is not None and current.parent==legacy and complete_signal(current):
            current.rename(target); changed=True
        if complete_signal(target):
            if record.get('cache_path')!=str(target): record['cache_path']=str(target); changed=True
        for sorting in record.get('sortings',[]):
            source=Path(sorting['cache_path']) if sorting.get('cache_path') else None
            destination=sorting_target(record,sorting,output)
            if source and source.parent==legacy and (source/'manifest.json').is_file() and not destination.exists():
                destination.parent.mkdir(parents=True,exist_ok=True); source.rename(destination)
                sorting['cache_path']=str(destination); changed=True
        # The lock also makes crash leftovers from the new writer safe to discard.
        for stage in root.rglob('.*.partial'):
            if stage.is_dir():
                removed+=sum(p.stat().st_size for p in stage.rglob('*') if p.is_file()); shutil.rmtree(stage); changed=True
        if legacy.is_dir() and complete_signal(target):
            completed_time=(target/'manifest.json').stat().st_mtime_ns
            for stage in legacy.glob('*.partial-*'):
                files=list(stage.iterdir()) if stage.is_dir() else []
                if not files or any(not f.is_file() or f.name!='lfp.npy' or f.stat().st_mtime_ns>completed_time for f in files): continue
                if _in_use(stage): continue
                removed+=sum(f.stat().st_size for f in files); shutil.rmtree(stage); changed=True
            remaining=list(legacy.iterdir())
            if remaining and all(p.name=='.DS_Store' and p.is_file() for p in remaining):
                for p in remaining: p.unlink()
            if not any(legacy.iterdir()): legacy.rmdir(); changed=True
    return dict(changed=changed,removed_bytes=removed)

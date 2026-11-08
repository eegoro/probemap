"""Electrophysiology sources, deliberately independent of anatomical recordings.

No function in this module appends to insertion['recordings']. Paths participate
in the project's normal relocation mechanism. Source binaries are never copied.
"""
from __future__ import annotations

import ast
import hashlib
import os
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from .neuropixels_model import parse_channelmap


def metadata_values(path):
    return dict(line.lstrip('~').split('=', 1) for line in
                Path(path).read_text(encoding='utf-8-sig').splitlines() if '=' in line)


def saved_channels(value, count):
    if value.strip() == 'all': return list(range(count))
    result = []
    for item in value.split(','):
        bounds = list(map(int, item.strip().split(':')))
        if len(bounds) == 1: result.append(bounds[0])
        elif len(bounds) == 2 and bounds[1] >= bounds[0]: result.extend(range(bounds[0], bounds[1] + 1))
        else: raise ValueError('Invalid saved channel subset.')
    if len(result) != count or len(set(result)) != count:
        raise ValueError('Saved channel subset does not match nSavedChans.')
    return result


def read_recording(path):
    path = Path(path).resolve()
    text = path.read_text(encoding='utf-8-sig')
    meta = metadata_values(path)
    parsed = parse_channelmap(text)
    binary = path.with_suffix('.bin')
    if not binary.is_file(): raise ValueError(f'{path.name}: matching .bin was not found.')
    fs = float(meta.get('imSampRate', 0)); count = int(meta.get('nSavedChans', 0))
    if fs <= 1000 or count < 1: raise ValueError('Invalid sampling rate or channel count.')
    size = binary.stat().st_size
    if not size or size % (2 * count): raise ValueError(f'{binary.name}: incomplete or empty binary frames.')
    subset = saved_channels(meta.get('snsSaveChanSubset', 'all'), count)
    columns = {ch: i for i, ch in enumerate(subset)}
    channels = [dict(c, file_index=columns[c['channel']]) for c in parsed['channels']
                if c['recorded'] and c['channel'] in columns]
    if not channels: raise ValueError('No active neural channels in this recording.')
    # NP2 uses fixed AP gain; do not silently substitute NP1 gain defaults.
    scale = float(meta['imAiRangeMax']) / int(meta['imMaxInt']) / 80 * 1e6
    stamp = meta.get('fileCreateTime_original') or meta.get('fileCreateTime', '')
    try: name = datetime.fromisoformat(stamp.replace('Z', '+00:00')).strftime('%Y-%m-%d %H:%M:%S')
    except ValueError: name = path.stem.removesuffix('.ap')
    return dict(id=uuid4().hex, name=name, recorded_at=stamp, folder=str(path.parent),
                meta_path=str(path), bin_path=str(binary), metadata=meta,
                channels=channels, sampling_rate=fs, n_saved_channels=count,
                n_samples=size // (2 * count), duration_s=size / (2 * count * fs),
                uv_per_count=scale, first_site_um=float(meta.get('imTipLength', 206))+11, insertion_id=None, sortings=[], selected_sorting=None,
                meta_sha256=hashlib.sha256(text.encode()).hexdigest())


def read_params(folder):
    """Read literal Phy settings without executing params.py or loading pickle."""
    path = Path(folder) / 'params.py'
    if not path.exists(): return {}
    result = {}
    for node in ast.parse(path.read_text(encoding='utf-8')).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try: result[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError): pass
    return result


def discover(folder, cancel=lambda: False):
    metas, sortings = [], []
    for root, dirs, files in os.walk(Path(folder).expanduser().resolve()):
        if cancel(): raise InterruptedError('Cancelled')
        dirs[:] = sorted(d for d in dirs if not d.startswith('.') and d not in ('ephys', '__pycache__'))
        files = set(files)
        if {'spike_times.npy', 'spike_clusters.npy'} <= files:
            sortings.append(Path(root))
            dirs[:] = []
        metas.extend(Path(root) / f for f in sorted(files) if f.endswith('.meta') and not f.endswith(('.lf.meta', '.nidq.meta')))
    return metas, sortings


def sorting_source(path):
    path = Path(path).resolve()
    return dict(id=uuid4().hex, name=path.name, path=str(path))


def import_folders(folders, existing=(), progress=lambda *args: None, cancel=lambda: False):
    known = {str(Path(r['meta_path']).resolve()) for r in existing}
    added, errors = [], []
    for index, folder in enumerate(folders):
        progress(round(index * 100 / max(1, len(folders))), f'Reading folder {index + 1} of {len(folders)}')
        metas, sortings = discover(folder, cancel)
        if not metas: errors.append(f'{Path(folder).name}: no Neuropixels .meta files found.')
        for path in metas:
            if str(path) in known: continue
            try:
                record = read_recording(path)
                for sorting in sortings:
                    params = read_params(sorting)
                    dat = params.get('dat_path', '')
                    if isinstance(dat, (tuple, list)): dat = dat[0] if dat else ''
                    filename = str(dat).replace('\\', '/').split('/')[-1]
                    # Never attach another recording's sorting by proximity alone.
                    if len(metas) == 1:
                        matches = True
                    elif filename:
                        matches = filename == Path(record['bin_path']).name
                    else:
                        matches = len(metas) == 1 or sorting.is_relative_to(path.parent)
                    if matches: record['sortings'].append(sorting_source(sorting))
                if record['sortings']: record['selected_sorting'] = record['sortings'][0]['id']
                added.append(record); known.add(str(path))
            except (OSError, ValueError, KeyError, SyntaxError) as error:
                errors.append(f'{path.name}: {error}')
    return sorted(added, key=lambda r: (r['recorded_at'], r['meta_path'])), errors

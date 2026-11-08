"""Discover completed, relocatable server exports without opening raw binaries."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from uuid import uuid4


def _inside(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Portable export path escapes its recording folder.')
    return path


def sync_exports(output, records):
    """Merge published exports; preserve project names, insertion links and sorting IDs."""
    base = Path(output) / 'ephys'
    changed = sync_bundles(output, records)
    if not base.is_dir():
        return changed
    for root in sorted(base.iterdir()):
        marker = root / 'portable.json'
        if root.name.startswith('.') or root.is_symlink() or not marker.is_file():
            continue
        try:
            info = json.loads(marker.read_text(encoding='utf-8'))
            if info.get('format') != 'probemap-ephys' or info.get('version') != 1:
                raise ValueError('Unsupported portable export version.')
            source = json.loads((root / 'recording.json').read_text(encoding='utf-8'))
            manifest = json.loads((root / 'signal' / 'manifest.json').read_text(encoding='utf-8'))
            if manifest.get('version') != 2 or not source.get('channels'):
                raise ValueError('Incomplete signal metadata.')
            if not all((root / 'signal' / name).is_file() for name in ('lfp.npy', 'profiles.npz')):
                raise ValueError('Incomplete signal files.')
            meta = _inside(root, info['meta_file'])
            if not meta.is_file():
                raise ValueError('Missing SpikeGLX metadata.')
            sortings = []
            for relative in info.get('sortings', []):
                folder = _inside(root, relative)
                if not all((folder / name).is_file() for name in ('spike_times.npy', 'spike_clusters.npy')):
                    raise ValueError('Incomplete Kilosort export.')
                sortings.append(dict(path=str(folder), name=folder.name))
            signature = (marker.stat().st_mtime_ns, marker.stat().st_size, str(root.resolve()))
            existing = next((r for r in records if r['id'] == source['id'] or
                             r.get('bin_path') == source['bin_path'] or
                             r.get('portable_source_bin') == source['bin_path'] or
                             r.get('cache_root') == str(root.resolve())), None)
            if existing and existing.get('portable_signature') == list(signature):
                continue
            record = existing if existing is not None else dict(source)
            for field in ('channels', 'sampling_rate', 'n_saved_channels', 'n_samples',
                          'duration_s', 'uv_per_count', 'first_site_um', 'metadata',
                          'meta_sha256', 'recorded_at'):
                record[field] = source[field]
            previous = {s['path']: s for s in record.get('sortings', [])}
            for sorting in sortings:
                sorting.update(previous.get(sorting['path'], {}))
                sorting.setdefault('id', uuid4().hex)
            # Keep manually attached sortings when a server export is refreshed.
            paths = {s['path'] for s in sortings}
            sortings += [s for s in record.get('sortings', []) if s['path'] not in paths
                         and not s.get('portable_export')]
            for sorting in sortings:
                if sorting['path'] in paths:
                    sorting['portable_export'] = True
            record.update(cache_root=str(root.resolve()), cache_path=str((root / 'signal').resolve()),
                          folder=str(root.resolve()), meta_path=str(meta),
                          portable_source_bin=source['bin_path'], portable_signature=list(signature),
                          portable=True, sortings=sortings)
            if record.get('selected_sorting') not in {s['id'] for s in sortings}:
                record['selected_sorting'] = sortings[0]['id'] if sortings else None
            if existing is None:
                records.append(record)
            changed = True
        except (OSError, ValueError, KeyError, TypeError) as error:
            logging.getLogger(__name__).warning('Cannot import ephys export %s: %s', root, error)
    return changed


def _source_identity(path):
    import re
    return re.sub(r'_t(?:cat|\d+)(?=\.imec)', '_t', Path(str(path).replace('\\','/')).name)


def sync_bundles(output, records):
    """Read standalone server bundles in outputs/ephys or the animal's ephys."""
    import copy
    from .ephys_cache import complete_signal
    changed = False
    output = Path(output).resolve()
    bases = [output / 'ephys']
    if output.name == 'outputs':
        bases.append(output.parent / 'ephys')
    for base in bases:
        if not base.is_dir():
            continue
        for marker in sorted(base.glob('*/bundle.json')):
            if marker.parent.name.startswith('.') or marker.parent.is_symlink():
                continue
            try:
                bundle = json.loads(marker.read_text(encoding='utf-8'))
                if bundle.get('bundle_version') != 1 or bundle.get('cache_version') != 2:
                    raise ValueError('Unsupported server bundle version')
                for item in bundle['records']:
                    root = _inside(marker.parent, item['relative_root'])
                    source = item['record']
                    signature = [str(root), marker.stat().st_mtime_ns, marker.stat().st_size]
                    matches = [r for r in records if r['id'] == source['id'] or
                               r.get('cache_root') == str(root) or
                               _source_identity(r.get('portable_source_bin', r.get('bin_path', ''))) ==
                               _source_identity(source['bin_path'])]
                    if len(matches) > 1:
                        raise ValueError('Ambiguous existing recording match')
                    existing = matches[0] if matches else None
                    if existing and existing.get('bundle_signature') == signature:
                        continue
                    if not complete_signal(root / 'signal'):
                        raise ValueError('Incomplete prepared signal')
                    manifest = json.loads((root / 'signal/manifest.json').read_text(encoding='utf-8'))
                    if manifest.get('version') != 2 or manifest['channels'] != source['channels']:
                        raise ValueError('Signal channel metadata mismatch')
                    meta = _inside(root, 'source/' + Path(source['meta_path'].replace('\\','/')).name)
                    if not meta.is_file():
                        raise ValueError('Missing exported metadata')
                    rec = copy.deepcopy(existing if existing is not None else source)
                    for key in ('recorded_at', 'metadata', 'channels', 'sampling_rate',
                                'n_saved_channels', 'n_samples', 'duration_s', 'uv_per_count',
                                'first_site_um', 'meta_sha256'):
                        rec[key] = source[key]
                    previous = rec.get('sortings', [])
                    sortings = []
                    for exported in source.get('sortings', []):
                        folder = _inside(root, 'kilosort/' + exported['bundle_name'])
                        cache = _inside(root, 'sortings/' + exported['bundle_name'])
                        info = json.loads((cache / 'manifest.json').read_text(encoding='utf-8'))
                        if info.get('version') != 2 or not all((cache / f).is_file() for f in
                                ('times.npy', 'amplitudes.npy', 'units.npz')):
                            raise ValueError('Incomplete prepared sorting')
                        old = next((s for s in previous if s.get('name') == exported['name']), {})
                        sortings.append(dict(id=old.get('id', exported['id']), name=exported['name'],
                                             path=str(folder), cache_path=str(cache), portable_export=True))
                    paths = {s['path'] for s in sortings}
                    sortings.extend(s for s in previous if s['path'] not in paths and
                                    not s.get('portable_export') and Path(s['path']).is_dir() and
                                    s.get('name') not in {v['name'] for v in sortings})
                    rec.update(folder=str(root), meta_path=str(meta),
                               bin_path=str(meta.with_suffix('.bin')),
                               portable=True, portable_source_bin=source['bin_path'],
                               cache_root=str(root), cache_path=str(root / 'signal'),
                               bundle_signature=signature, sortings=sortings)
                    if rec.get('selected_sorting') not in {s['id'] for s in sortings}:
                        rec['selected_sorting'] = sortings[0]['id'] if sortings else None
                    if existing is None:
                        records.append(rec)
                    else:
                        existing.clear(); existing.update(rec)
                    changed = True
            except (OSError, ValueError, KeyError, TypeError) as error:
                logging.getLogger(__name__).warning('Cannot import server bundle %s: %s', marker, error)
    return changed


def merge_scanned_exports(records, baseline, scanned):
    """Apply background discovery without reverting edits made while it ran."""
    source_fields = {'channels','sampling_rate','n_saved_channels','n_samples','duration_s',
                     'uv_per_count','first_site_um','metadata','meta_sha256','recorded_at',
                     'folder','meta_path','bin_path','cache_root','cache_path','portable',
                     'portable_source_bin','portable_signature','bundle_signature','sortings','selected_sorting'}
    changed_ids=set()
    previous={r['id']:r for r in baseline}
    current={r['id']:r for r in records}
    missing=object()
    for exported in scanned:
        before=previous.get(exported['id'])
        actual=current.get(exported['id'])
        if before is None:
            if actual is None and not any((exported.get('cache_root') and r.get('cache_root')==exported['cache_root']) or
                    (exported.get('portable_source_bin',exported.get('bin_path')) and _source_identity(r.get('portable_source_bin',r.get('bin_path',''))) ==
                    _source_identity(exported.get('portable_source_bin',exported.get('bin_path','')))) for r in records):
                records.append(exported); current[exported['id']]=exported; changed_ids.add(exported['id'])
            continue
        if actual is None: continue  # The user removed this recording during discovery.
        conflict=False
        updates={}
        for key in source_fields-{'portable_signature','bundle_signature'}:
            value=exported.get(key,missing)
            old=before.get(key,missing)
            if value is missing or value==old: continue
            if actual.get(key,missing)!=old:
                conflict=True; continue
            updates[key]=value
        if not conflict:
            for key in ('portable_signature','bundle_signature'):
                if key in exported and exported[key]!=actual.get(key): updates[key]=exported[key]
        if updates:
            actual.update(updates); changed_ids.add(actual['id'])
    return changed_ids

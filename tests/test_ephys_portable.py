"""Server exports remain usable after their original filesystem disappears."""
import json
import shutil
from pathlib import Path


import numpy as np

from probemap.registration.ephys_portable import sync_exports
from probemap.registration.ephys_analysis import load_summary, prepare_recording, prepare_sorting, load_sorting
from probemap.registration.ephys_cache import migrate_recording_cache
from test_ephys import recording


def export(tmp_path):
    source, record = recording(tmp_path / 'server')
    output = tmp_path / 'local'
    root = output / 'ephys' / 'recording'
    root.mkdir(parents=True)
    prepared = prepare_recording(record, tmp_path / 'prepared')
    shutil.copytree(prepared, root / 'signal')
    shutil.copy2(record['meta_path'], root / 'recording.meta')
    (root / 'recording.json').write_text(json.dumps(record))
    sorting = root / 'kilosort' / 'kilosort4'; sorting.mkdir(parents=True)
    np.save(sorting / 'spike_times.npy', [100, 200])
    np.save(sorting / 'spike_clusters.npy', [0, 0])
    np.save(sorting / 'channel_map.npy', [c['file_index'] for c in record['channels']])
    template = np.zeros((1, 32, len(record['channels'])), dtype=np.float32)
    template[0, 10, 0] = -5
    np.save(sorting / 'templates.npy', template)
    np.save(sorting / 'spike_templates.npy', [0, 0])
    (root / 'portable.json').write_text(json.dumps(dict(
        format='probemap-ephys', version=1, meta_file='recording.meta',
        sortings=['kilosort/kilosort4'])))
    shutil.rmtree(source)
    return output, root


def test_discovery_offline_deduplication_and_project_choices(tmp_path):
    output, root = export(tmp_path)
    records = []
    assert sync_exports(output, records)
    record = records[0]
    assert len(records) == 1 and record['portable']
    assert record['sortings'][0]['path'] == str(root / 'kilosort' / 'kilosort4')
    assert record['selected_sorting'] == record['sortings'][0]['id']
    load_summary(record['cache_path'])
    migrate_recording_cache(record, output)
    cache = prepare_sorting(record, record['sortings'][0], output)
    assert load_sorting(cache)[0]['units'][0]['count'] == 2
    record['insertion_id'] = 'insertion-3'; record['name'] = 'Before'
    assert not sync_exports(output, records)
    marker = root / 'portable.json'; marker.write_text(marker.read_text() + '\n')
    sorting_id = record['selected_sorting']
    assert sync_exports(output, records)
    assert len(records) == 1 and record['name'] == 'Before'
    assert record['insertion_id'] == 'insertion-3'
    assert record['selected_sorting'] == sorting_id


def test_incomplete_or_escaping_exports_are_ignored(tmp_path):
    output, root = export(tmp_path)
    marker = root / 'portable.json'
    info = json.loads(marker.read_text()); info['meta_file'] = '../../outside.meta'
    marker.write_text(json.dumps(info))
    records = []
    assert not sync_exports(output, records) and not records
    info['meta_file'] = 'recording.meta'; marker.write_text(json.dumps(info))
    (root / 'signal' / 'lfp.npy').unlink()
    assert not sync_exports(output, records) and not records


def test_standalone_bundle_merges_catgt_source_and_keeps_anatomy(tmp_path):
    output, root = export(tmp_path)
    source = json.loads((root / 'recording.json').read_text())
    folder = root / 'kilosort/kilosort4'
    sorting = dict(id='server-sorting', name='kilosort4', path=str(folder))
    source['meta_path'] = str(root / 'recording.meta')
    cache = prepare_sorting(source, sorting, tmp_path / 'sorting-output')
    destination = root / 'sortings/kilosort4'
    destination.parent.mkdir()
    shutil.copytree(cache, destination)
    source['sortings'] = [dict(sorting, bundle_name='kilosort4')]
    source['bin_path'] = '/server/session_tcat.imec0.ap.bin'
    source['meta_path'] = '/server/session_tcat.imec0.ap.meta'
    (root / 'source').mkdir()
    shutil.copyfile(root / 'recording.meta', root / 'source/session_tcat.imec0.ap.meta')
    (root / 'portable.json').unlink()
    (root / 'recording.json').write_text(json.dumps(dict(
        record_id=source['id'], source_bin=source['bin_path'], source_meta=source['meta_path'])))
    (root / 'bundle.json').write_text(json.dumps(dict(
        bundle_version=1, cache_version=2,
        records=[dict(relative_root='.', record=source)])))
    existing = dict(source, id='local-id', name='Before', insertion_id='anatomy-id',
                    bin_path='/old/session_t0.imec0.ap.bin', sortings=[], selected_sorting=None)
    records = [existing]
    assert sync_exports(output, records)
    assert len(records) == 1
    assert existing['id'] == 'local-id' and existing['name'] == 'Before'
    assert existing['insertion_id'] == 'anatomy-id'
    assert not Path(existing['bin_path']).exists()
    load_summary(existing['cache_path'])
    load_sorting(existing['sortings'][0]['cache_path'])
    migrate_recording_cache(existing, output)
    assert existing['cache_root'] == str(root.resolve())
    assert not sync_exports(output, records)

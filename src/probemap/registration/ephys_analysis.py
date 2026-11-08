"""Bounded-memory CPU processing and disposable, versioned ephys caches."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
import shutil
from pathlib import Path
from uuid import uuid4

import numpy as np
from scipy import signal

from ..platform_io import close_memmap
from .ephys import read_params
from .ephys_cache import (recording_root, sorting_target, locked_preparation, staging_folder, publish_cache, recover_cache, complete_signal)

VERSION = 2
BANDS = ((1.5, 4), (4, 10), (10, 30), (30, 80), (80, 200))


def check_cancel(cancel):
    if cancel(): raise InterruptedError('Cancelled')


def fingerprint(paths):
    """Metadata + bounded edge reads; no full network-file hash at startup."""
    digest = hashlib.sha256(str(VERSION).encode())
    for path in paths:
        path = Path(path)
        stat = path.stat()
        digest.update(f'{path.name}:{stat.st_size}:{stat.st_mtime_ns}'.encode())
        with path.open('rb') as stream:
            digest.update(stream.read(65536))
            if stat.st_size > 65536:
                stream.seek(-65536, 2); digest.update(stream.read(65536))
    return digest.hexdigest()[:24]


def _cache_stamp(folder, archive):
    folder = Path(folder).resolve()
    return str(folder), tuple((p.stat().st_mtime_ns,p.stat().st_size)
                             for p in (folder/'manifest.json',folder/archive))


from functools import lru_cache
from copy import deepcopy


@lru_cache(maxsize=4)
def _load_arrays(folder, archive, stamp, fields):
    folder=Path(folder)
    info=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    if info.get('version') != VERSION:
        raise ValueError('Cache version changed; prepare the recording again.')
    with np.load(folder/archive,allow_pickle=False) as saved:
        data={key:saved[key] for key in (saved.files if fields is None else fields)}
    for value in data.values(): value.flags.writeable=False
    return info,data


def _cached_arrays(folder,archive,fields=None):
    path,stamp=_cache_stamp(folder,archive)
    info,data=_load_arrays(path,archive,stamp,None if fields is None else tuple(fields))
    # UI annotations must not modify the shared reader cache.
    return deepcopy(info),dict(data)


def load_summary(folder):
    return _cached_arrays(folder,'profiles.npz')


class RecordingReader:
    """Sequential buffered IO, including on SMB; retain overlap without rereading it."""
    READ_BYTES = 4*1024**2

    def __init__(self, path, shape, overlap, cancel, on_read=lambda *args: None):
        self.shape = shape; self.overlap = overlap; self.cancel = cancel; self.on_read = on_read
        self.stream = open(path, 'rb', buffering=0)
        self.tail = None; self.tail_start = 0; self.until = 0
        self.bytes_read = self.samples_read = 0; self.seconds_reading = 0.; self.last_report = 0.

    def close(self): self.stream.close()

    def read(self, start, stop):
        check_cancel(self.cancel)
        started = time.perf_counter(); count = self.shape[1]; frame_bytes = count*2
        result = np.empty((stop-start, count), dtype='<i2')
        reused = 0
        if self.tail is not None and self.tail_start <= start < self.until:
            reused = min(stop, self.until)-start
            result[:reused] = self.tail[start-self.tail_start:start-self.tail_start+reused]
        if start+reused < stop:
            position = (start+reused)*frame_bytes
            if self.stream.tell() != position: self.stream.seek(position)
            destination = memoryview(result[reused:]).cast('B'); offset = 0
            while offset < len(destination):
                check_cancel(self.cancel)
                amount = self.stream.readinto(destination[offset:offset+self.READ_BYTES])
                if not amount: raise ValueError('Recording binary ended before its declared sample count.')
                offset += amount; self.bytes_read += amount
                self.samples_read = self.bytes_read//frame_bytes
                now = time.perf_counter()
                if now-self.last_report >= .25 or offset == len(destination):
                    elapsed = self.seconds_reading+now-started
                    self.on_read((position+offset)/frame_bytes, self.bytes_read/max(elapsed, 1e-9))
                    self.last_report = now
        keep = min(len(result), self.overlap)
        self.tail = result[-keep:].copy() if keep else None; self.tail_start = stop-keep; self.until = stop
        self.seconds_reading += time.perf_counter()-started
        check_cancel(self.cancel)
        return result


def process_channel_group(source, file_indices, scale, filters, ratio, offset, length, core_start, core_length, rms_block):
    """Independent channel batches release the GIL inside SciPy's filters."""
    high, notch, low = filters
    raw = np.asarray(source[:, file_indices].T, dtype=np.float32)
    raw *= scale
    ap = signal.sosfiltfilt(high, raw, axis=1)
    core = ap[:, core_start:core_start+core_length]
    rms = []
    for start in range(0, core_length, rms_block):
        segment = core[:, start:start+rms_block]
        rms.append(np.sqrt(np.einsum('ij,ij->i', segment, segment)/segment.shape[1]))
    rms = np.asarray(rms, dtype=np.float32).T
    del ap, core
    filtered = signal.sosfiltfilt(low, signal.sosfiltfilt(notch, raw, axis=1), axis=1)
    del raw
    reduced = signal.resample_poly(filtered, *ratio, axis=1)
    return np.array(reduced[:, offset:offset+length], dtype=np.float32), rms


@locked_preparation(0)
def prepare_recording(record, output, progress=lambda *args: None, cancel=lambda: False, overwrite=False):
    started = time.perf_counter()
    progress(0, 'Checking recording source')
    check_cancel(cancel)
    if hashlib.sha256(Path(record['meta_path']).read_text(encoding='utf-8-sig').encode()).hexdigest() != record['meta_sha256']:
        raise ValueError('Recording metadata changed. Remove this analysis entry and import the folder again.')
    key = fingerprint([record['meta_path'], record['bin_path']])
    target = recording_root(record, output)/'signal'; recover_cache(target)
    if not overwrite and complete_signal(target):
        info,_ = load_summary(target)
        if info['key'] == key:
            progress(100, 'Using cached signal profiles'); return str(target)
    if target.exists() and not overwrite:
        raise FileExistsError('Prepared files already exist. Confirm overwrite to replace them.')
    temporary = staging_folder(target)
    lfp=None
    try:
        progress(0, 'Opening recording')
        channels = record['channels']; n = len(channels); fs = record['sampling_rate']
        # Rational resampling also supports measured, non-integer sample rates.
        from fractions import Fraction
        ratio = Fraction(1000 / fs).limit_denominator(10000)
        up, down = ratio.numerator, ratio.denominator
        new_fs = fs * up / down
        total = record['n_samples']; count = record['n_saved_channels']
        nout = math.ceil(total * up / down)
        if nout < 4096: raise ValueError('At least 4.1 seconds are required for low-frequency profiling.')
        lfp = np.lib.format.open_memmap(temporary / 'lfp.npy', mode='w+', dtype='float32', shape=(n, nout))
        columns = [c['file_index'] for c in channels]
        # Chunk boundaries are exact multiples of the rational resampling grid.
        rms_block = max(down, int(fs * 2) // down * down)
        # Cap each newly read time block at ~96 MiB of int16 source data.
        # The single prefetched block and small channel batches bound RAM use.
        multiplier = max(1, min(4, int(96*1024**2 / (rms_block*count*2))))
        block = rms_block*multiplier
        workers = min(4, max(1, (os.cpu_count() or 1)//3), max(1, math.ceil(n/16)))
        channel_batch = min(32, math.ceil(n/workers))
        groups = [list(range(i, min(n, i+channel_batch))) for i in range(0, n, channel_batch)]
        pad = max(down, math.ceil(fs / down) * down)
        notch = signal.tf2sos(*signal.iirnotch(50, 30, fs))
        low = signal.butter(4, 300, fs=fs, btype='lowpass', output='sos')
        high = signal.butter(3, 300, fs=fs, btype='highpass', output='sos')
        rms, rms_times = [], []
        timings = dict(source_check=time.perf_counter()-started, source_wait=0., signal_compute=0.)
        def reading(position, speed):
            progress(min(74., 75*position/total), f'Reading recording · {position/fs:.0f} / {total/fs:.0f} s · {speed/1024**2:.1f} MiB/s')
        reader = RecordingReader(record['bin_path'], (total, count), 2*pad, cancel, reading)
        reader_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='ephys-read')
        compute_pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='ephys-filter')
        try:
            pending = reader_pool.submit(reader.read, 0, min(total, block+pad))
            for start in range(0, total, block):
                check_cancel(cancel)
                stop = min(total, start+block)
                left, right = max(0, start-pad), min(total, stop+pad)
                tick = time.perf_counter()
                progress(round(75*start/total), f'Reading recording · {start/fs:.0f} / {total/fs:.0f} s')
                source = pending.result(); timings['source_wait'] += time.perf_counter()-tick
                # Exactly one block ahead; disk/network IO overlaps CPU filtering.
                if stop < total:
                    pending = reader_pool.submit(reader.read, max(0, stop-pad), min(total, stop+block+pad))
                tick = time.perf_counter()
                dest = start*up//down; length = math.ceil(stop*up/down)-dest; offset = (start-left)*up//down
                progress(round(75*start/total), f'Preparing signal · {start/fs:.0f} / {total/fs:.0f} s')
                futures = [(group, compute_pool.submit(process_channel_group, source,
                    [columns[i] for i in group], record['uv_per_count'], (high, notch, low),
                    (up, down), offset, length, start-left, stop-start, rms_block)) for group in groups]
                chunk_rms = np.empty((n, math.ceil((stop-start)/rms_block)), np.float32)
                for group, future in futures:
                    check_cancel(cancel)
                    reduced, group_rms = future.result()
                    lfp[group, dest:dest+length] = reduced
                    chunk_rms[group] = group_rms
                rms.append(chunk_rms)
                for rms_start in range(start, stop, rms_block):
                    rms_times.append((rms_start+min(stop, rms_start+rms_block))/(2*fs))
                timings['signal_compute'] += time.perf_counter()-tick
                progress(round(75*stop/total), f'Preparing signal · {stop/fs:.0f} / {total/fs:.0f} s')
        finally:
            reader_pool.shutdown(wait=True, cancel_futures=True)
            compute_pool.shutdown(wait=True, cancel_futures=True)
            reader.close()
        source_samples_read = reader.samples_read
        timings['source_read'] = reader.seconds_reading
        del reader, source, pending, futures
        rms = np.concatenate(rms, axis=1)
        progress(75, 'Writing LFP cache')
        lfp.flush()
        profile_started = time.perf_counter()
        # Quantify non-overlapping windows. Keep all temporal profiles for comparison.
        power_sum = correlation_sum = None
        band_windows, window_times = [], []
        window_count = 0
        window = max(1, round(new_fs * 20))
        bad = np.zeros(n, bool)
        for start in range(0, nout, window):
            check_cancel(cancel)
            data = np.asarray(lfp[:, start:min(nout, start+window)], dtype=np.float64)
            if data.shape[1] < 4096: continue
            variance = np.std(data, axis=1); bad |= ~np.isfinite(variance) | (variance < 1e-6)
            freq, power = signal.welch(data, fs=new_fs, nperseg=4096, nfft=4096, axis=1)
            power_sum = power if power_sum is None else power_sum + power
            centered = data - data.mean(axis=1, keepdims=True)
            norm = np.linalg.norm(centered, axis=1)
            normalized = np.divide(centered, norm[:, None], out=np.zeros_like(centered), where=norm[:, None] > 0)
            correlation = normalized @ normalized.T
            correlation_sum = correlation if correlation_sum is None else correlation_sum + correlation
            window_count += 1
            band_windows.append(np.array([np.mean(power[:, (freq >= lo) & (freq <= hi)], axis=1) for lo, hi in BANDS]))
            window_times.append(start / new_fs)
            progress(75 + round(20 * min(nout, start+window) / nout), 'Calculating depth profiles')
        if not window_count: raise ValueError('Recording is too short for signal profiling.')
        band_windows = np.asarray(band_windows, dtype=np.float32)
        timings['profiles'] = time.perf_counter()-profile_started
        save_started = time.perf_counter()
        progress(95, 'Saving depth profiles')
        np.savez_compressed(temporary / 'profiles.npz', frequencies=freq.astype('float32'),
            power=(power_sum/window_count).astype('float32'),
            correlation=(correlation_sum/window_count).astype('float32'),
            bands=np.mean(band_windows, axis=0), band_windows=band_windows,
            window_times=window_times, ap_rms=rms,
            ap_times=rms_times, bad_channels=bad)
        close_memmap(lfp); lfp=None
        timings['cache_write'] = time.perf_counter()-save_started
        timings['total'] = time.perf_counter()-started
        info = dict(version=VERSION, key=key, timings_s=timings,
                    execution=dict(source_io='buffered-sequential', read_bytes=RecordingReader.READ_BYTES, workers=workers, block_s=block/fs, channel_batch=channel_batch, source_samples_read=source_samples_read), sampling_rate=new_fs, n_samples=nout,
                    source_duration_s=record['duration_s'], channels=channels,
                    preprocessing='50 Hz notch; 300 Hz lowpass; polyphase resampling to ~1000 Hz; no C-factor or common reference')
        (temporary / 'manifest.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
        check_cancel(cancel)
        target.parent.mkdir(parents=True, exist_ok=True)
        publish_cache(temporary,target)
        progress(100, 'Signal profiles ready')
        return str(target)
    except BaseException:
        close_memmap(lfp)
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def boundary_candidates(heights, bands, bad=None):
    """Exploratory strongest joint transition; not a validated brain classifier.

    Return per-window candidates. Never infer an edge outside measured contacts.
    Duplicate column depths are averaged before evaluating spatial gradients.
    """
    heights = np.asarray(heights)
    good = np.ones(len(heights), bool) if bad is None else ~np.asarray(bad)
    levels = np.unique(heights[good])
    if len(levels) < 12: return np.array([])
    profiles = np.array([np.mean(bands[:, :, good][:, :, heights[good] == h], axis=2) for h in levels]).transpose(1, 2, 0)
    logpower = 10 * np.log10(np.maximum(profiles, 1e-20))
    from scipy.ndimage import gaussian_filter1d
    smooth = gaussian_filter1d(logpower, 1, axis=2)
    changes = np.abs(np.diff(smooth, axis=2)) / np.diff(levels)
    changes /= np.maximum(np.median(changes, axis=2, keepdims=True), 1e-9)
    score = np.median(changes, axis=1)
    score[:, :3] = 0; score[:, -3:] = 0
    # Gaps in channel coverage must not become artificial transition candidates.
    score[:, np.diff(levels) > 1.6 * np.median(np.diff(levels))] = 0
    valid = np.max(score, axis=1) >= 3
    return (levels[:-1] + levels[1:])[np.argmax(score[valid], axis=1)] / 2


def labels_from_sorting(folder):
    labels = {}
    source = None
    for name, column in [('cluster_KSLabel.tsv', 'KSLabel'), ('cluster_group.tsv', 'group')]:
        path = Path(folder) / name
        if path.exists():
            with path.open(encoding='utf-8-sig') as stream:
                for row in csv.DictReader(stream, delimiter='\t'):
                    if row.get(column): labels[int(row['cluster_id'])] = row[column].lower()
            source = name
    return labels, source


@locked_preparation(1)
def prepare_sorting(record, sorting, output, progress=lambda *args: None, cancel=lambda: False):
    progress(0, 'Checking sorting files')
    check_cancel(cancel)
    folder = Path(sorting['path'])
    paths = sorted(folder.glob('*.npy')) + sorted(folder.glob('*.tsv'))
    if (folder / 'params.py').exists(): paths.append(folder / 'params.py')
    key = fingerprint(paths + [record['meta_path']])
    target = sorting_target(record, sorting, output); recover_cache(target)
    if (target / 'manifest.json').exists():
        try:
            if json.loads((target/'manifest.json').read_text(encoding='utf-8'))['source_key'] == key:
                load_sorting(target); return str(target)
        except (OSError,ValueError,KeyError): pass
    temp = staging_folder(target)
    grouped=grouped_amps=None
    def array(name): return np.load(folder / (name + '.npy'), mmap_mode='r', allow_pickle=False)
    try:
        times = array('spike_times').reshape(-1); clusters = array('spike_clusters').reshape(-1)
        if len(times) != len(clusters): raise ValueError('Spike times and clusters have different lengths.')
        fs = float(read_params(folder).get('sample_rate', record['sampling_rate']))
        if fs <= 0: raise ValueError('Invalid Kilosort sample rate.')
        if len(times) and (np.min(times) < 0 or np.max(times)/fs > record['duration_s'] + 1):
            raise ValueError('Kilosort timestamps exceed this recording. Attach the matching recording or sorting.')
        channel_map = array('channel_map').reshape(-1)
        channels = {c['file_index']: c for c in record['channels']}
        if any(int(c) not in channels for c in channel_map):
            raise ValueError('Kilosort channel map does not match the active channels in this recording.')
        mapped = [channels[int(c)] for c in channel_map]
        templates = array('templates') if (folder / 'templates.npy').exists() else None
        template_indices = array('spike_templates') if (folder / 'spike_templates.npy').exists() else None
        if template_indices is not None and template_indices.ndim == 2: template_indices = template_indices[:, 0]
        if template_indices is not None and len(template_indices) != len(times): raise ValueError('Invalid spike_templates length.')
        inverse_whitening = array('whitening_mat_inv') if (folder / 'whitening_mat_inv.npy').exists() else None
        if inverse_whitening is not None and inverse_whitening.shape != (len(channel_map),len(channel_map)):
            raise ValueError('Invalid inverse whitening matrix shape.')
        sparse_indices = array('templates_ind') if (folder / 'templates_ind.npy').exists() else None
        positions = array('spike_positions') if (folder / 'spike_positions.npy').exists() else None
        if positions is not None and len(positions) != len(times): raise ValueError('Invalid spike_positions length.')
        amps = array('amplitudes').reshape(-1) if (folder / 'amplitudes.npy').exists() else None
        if amps is not None and len(amps) != len(times): raise ValueError('Invalid amplitudes length.')
        labels, label_source = labels_from_sorting(folder)
        progress(10, 'Indexing sorted spikes')
        check_cancel(cancel)
        order = np.argsort(clusters, kind='stable')
        sorted_clusters = np.asarray(clusters[order])
        ids, starts, counts = np.unique(sorted_clusters, return_index=True, return_counts=True)
        # Persist grouped spike times once, for O(unit spikes) detail access.
        grouped = np.lib.format.open_memmap(temp / 'times.npy', mode='w+', dtype='float64', shape=(len(times),))
        grouped_amps = np.lib.format.open_memmap(temp / 'amplitudes.npy', mode='w+', dtype='float32', shape=(len(times),))
        units, waves = [], []
        raster = np.zeros((4, 640, 1600), np.float32)
        raster_good = np.zeros_like(raster)
        g = dict(first=record.get('first_site_um', 217.), pitch=15.)
        waveform_fs = fs
        for index, (unit, start, count) in enumerate(zip(ids, starts, counts)):
            check_cancel(cancel)
            indices = order[start:start+count]
            unit_times = np.asarray(times[indices], dtype=np.float64) / fs
            chronological = np.argsort(unit_times); indices = indices[chronological]; unit_times = unit_times[chronological]
            grouped[start:start+count] = unit_times
            grouped_amps[start:start+count] = amps[indices] if amps is not None else np.nan
            # Curated cluster IDs need not equal template indices.
            if template_indices is not None and templates is not None:
                tids, weights = np.unique(template_indices[indices], return_counts=True)
                if np.any(tids < 0) or np.any(tids >= len(templates)): raise ValueError('Invalid template index.')
                selected_templates = np.asarray(templates[tids], dtype=np.float32)
                if selected_templates.shape[2] != len(channel_map):
                    if sparse_indices is None: raise ValueError('Sparse templates require templates_ind.npy.')
                    expanded = np.zeros((len(tids), selected_templates.shape[1], len(channel_map)), np.float32)
                    for slot, tid in enumerate(tids):
                        indices = np.asarray(sparse_indices[int(tid)],dtype=int)
                        if np.any(indices < 0) or np.any(indices >= len(channel_map)): raise ValueError('Invalid sparse template channels.')
                        expanded[slot][:, indices] = selected_templates[slot]
                    selected_templates = expanded
                template = np.average(selected_templates, axis=0, weights=weights)
            elif templates is not None and not (folder / 'cluster_group.tsv').exists() and int(unit) < len(templates):
                template = np.asarray(templates[int(unit)], dtype=np.float32)
            else: template = None
            if template is not None:
                if template.shape[1] != len(channel_map): raise ValueError('Sparse templates are not supported by this importer.')
                if inverse_whitening is not None: template = template @ inverse_whitening
                peak = int(np.argmax(np.ptp(template, axis=0)))
                contact = mapped[peak]; wave = template[:, peak]
                # Keep waveform in exported template units, not falsely labelled uV.
                height = g['first'] + contact['row'] * g['pitch']; shank = contact['shank']
            elif positions is not None:
                positions_ch = np.load(folder / 'channel_positions.npy', mmap_mode='r', allow_pickle=False)
                location = np.median(positions[indices], axis=0)
                peak = int(np.argmin(np.linalg.norm(positions_ch-location, axis=1)))
                contact = mapped[peak]; height = g['first'] + contact['row'] * g['pitch']; shank = contact['shank']; wave = np.array([])
            else: raise ValueError('Unit depth requires spike_positions or a usable spike-to-template map.')
            if positions is not None:
                # Kilosort depth origin may differ. Anchor its Y coordinate to known contacts.
                positions_ch = np.load(folder / 'channel_positions.npy', mmap_mode='r', allow_pickle=False)
                height += float(np.median(positions[indices, 1])) - float(positions_ch[peak, 1])
            bins = np.clip((unit_times / record['duration_s'] * 1600).astype(int), 0, 1599)
            depthbin = int(np.clip(height / 10000 * 640, 0, 639))
            np.add.at(raster[shank, depthbin], bins, 1)
            if labels.get(int(unit)) == 'good': np.add.at(raster_good[shank, depthbin], bins, 1)
            isi = np.diff(unit_times)
            hist, _ = np.histogram(isi * 1000, np.linspace(0, 100, 51))
            rate, _ = np.histogram(unit_times, np.linspace(0, record['duration_s'], 61))
            units.append(dict(id=int(unit), start=int(start), count=int(count), height_um=float(height), shank=int(shank),
                label=labels.get(int(unit), 'unlabelled'), rate_hz=float(count/record['duration_s']),
                isi_violation_fraction=float(np.mean(isi < .002)) if len(isi) else None,
                isi_hist=hist.tolist(), rate_bins=(rate * 60 / record['duration_s']).tolist()))
            waves.append(wave.astype('float32'))
            progress(10 + round(85 * (index+1) / max(1, len(ids))), f'Reading units · {index+1} / {len(ids)}')
        grouped.flush(); grouped_amps.flush()
        close_memmap(grouped); close_memmap(grouped_amps); grouped=grouped_amps=None
        max_wave = max((len(w) for w in waves), default=0)
        padded = np.full((len(waves), max_wave), np.nan, np.float32)
        for i, wave in enumerate(waves): padded[i, :len(wave)] = wave
        np.savez_compressed(temp / 'units.npz', raster=raster, raster_good=raster_good, waveforms=padded)
        info = dict(version=VERSION, source_path=str(sorting['path']), sorting_id=sorting.get('id'), units=units, label_source=label_source, sampling_rate=waveform_fs,
                    duration_s=record['duration_s'], source_key=key)
        (temp / 'manifest.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
        check_cancel(cancel); publish_cache(temp,target)
        return str(target)
    except BaseException:
        close_memmap(grouped); close_memmap(grouped_amps)
        shutil.rmtree(temp, ignore_errors=True); raise


def load_sorting(folder, fields=None):
    """Load only requested arrays; the workspace does not use binned raster maps."""
    return _cached_arrays(folder,'units.npz',fields)

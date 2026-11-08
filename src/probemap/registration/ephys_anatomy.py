"""Probe-aligned physical section for the ephys page; no annotation mutations."""
import copy
from pathlib import Path

import numpy as np

from .landmarks import transform_points
from .neuropixels_model import probe_coordinates
from .surface_entries import probe_surface_entries
from .surface_entries import build_atlas_surface
from .track_model import load_label_atlas, raw_volume, sample_world


def probe_site_layout(geometry):
    """All physical sites, using exactly the same offsets as probe_coordinates."""
    shanks, rows, columns = np.meshgrid(np.arange(geometry['shanks']),
        np.arange(geometry['sites_per_shank']//2), np.arange(2), indexing='ij')
    shanks, rows, columns = shanks.ravel(), rows.ravel(), columns.ravel()
    heights = geometry['first_site_center_from_tip_um']+rows*geometry['site_pitch_um']
    xs = shanks*geometry['shank_pitch_um']+geometry['left_site_center_um']+columns*geometry['column_pitch_um']-geometry['shank_width_um']/2
    return dict(shanks=shanks, rows=rows, columns=columns, heights=heights, xs=xs)


def recording_depth_limit(record, geometry):
    """Show five additional physical rows beyond the highest recorded contact."""
    first, pitch = geometry['first_site_center_from_tip_um'], geometry['site_pitch_um']
    return max((first+c['row']*pitch for c in record.get('channels', [])), default=first)+5*pitch


def prepare_anatomy(project, insertion, record, shank, output, progress=lambda *args: None, resources=None, cancel=lambda: False):
    if insertion is None or not insertion.get('probe', {}).get('pose'): return None
    insertion = copy.deepcopy(insertion)
    settings = insertion['probe']; g = settings['geometry']; pose = settings['pose']
    contacts = [c for c in record['channels'] if c['shank'] == shank]
    heights = np.array([g['first_site_center_from_tip_um'] + c['row']*g['site_pitch_um'] for c in contacts])
    all_contacts = record['channels']
    all_heights = np.array([g['first_site_center_from_tip_um'] + c['row']*g['site_pitch_um'] for c in all_contacts])
    shanks = list(range(g['shanks']))
    low, high = 0., recording_depth_limit(record, g)
    left, right = min(shanks)*g['shank_pitch_um']-140, max(shanks)*g['shank_pitch_um']+140
    progress(10, 'Reading probe-aligned anatomy')
    if resources is None:
        atlas = raw_volume(project.reference)
        if cancel(): raise InterruptedError('Cancelled')
        brain = raw_volume(project.moving)
        resources = dict(atlas=atlas, brain=brain, labels=None, surface=None)
        progress(25, 'Reading atlas labels')
        if project.labels_dir:
            resources['labels'] = load_label_atlas(project.labels_dir, project.reference, Path(output)/'ephys'/'atlas-labels', spacing_um_zyx=project.label_spacing)
        if cancel(): raise InterruptedError('Cancelled')
        progress(65, 'Finding anatomical surface')
        resources['surface'] = build_atlas_surface(atlas)
    entries = probe_surface_entries(insertion, project.total, resources['surface']) if resources['surface'] is not None else []
    depths = {}
    for entry in entries:
        if entry['shank'] not in shanks: continue
        tip = probe_coordinates(settings, entry['shank'], 0.)
        source_entry = transform_points([entry['atlas_zyx_um']], np.linalg.inv(project.total))[0]
        depth = float(np.dot(source_entry-tip, -np.asarray(pose['down_source_zyx'])))
        if depth > 0:
            depths[entry['shank']] = depth
    yy = np.linspace(high, low, 1024); xx = np.linspace(left, right, 256)
    anchor = probe_coordinates(settings, 0, 0., face=True)
    world = anchor - yy[:, None, None]*np.asarray(pose['down_source_zyx']) + xx[None, :, None]*np.asarray(pose['lateral_source_zyx'])
    atlas_world = transform_points(world.reshape(-1, 3), project.total)
    atlas = resources['atlas']
    images = {}
    for name, spec, matrix in [('Atlas', project.reference, np.eye(4)), ('Scan', project.moving, project.total)]:
        values = sample_world(atlas if name == 'Atlas' else resources['brain'], matrix, atlas_world).reshape(1024, 256)
        lo, hi = np.percentile(values, (1, 99.5)); values = np.clip((values-lo)/max(hi-lo, 1e-9), 0, 1)
        images[name] = np.repeat(values[..., None], 3, axis=2)
    labels = resources['labels']; region_names = []; contact_regions = []; label_regions = []; region_positions = {}
    ids = None
    if labels is not None:
        ids = labels.ids_at(atlas_world).reshape(1024, 256)
        colors = np.zeros((1024, 256, 4), np.float32)
        present = set(np.unique(ids).tolist())
        label_regions = [r for r in labels.regions if r['id'] in present]
        for region in label_regions:
            mask = ids == region['id']
            indices = np.argwhere(mask)
            center = indices.mean(axis=0)
            row, col = indices[np.argmin(np.sum((indices-center)**2,axis=1))]
            region_positions[str(region['id'])] = [float(xx[col]), float(yy[row])]
            colors[mask, :3] = np.asarray(region['color_rgb'])/255
            colors[mask, 3] = .38
        center_ids = ids[:, int(np.argmin(abs(xx-shank*g['shank_pitch_um']))) ]
        for identifier in np.unique(center_ids):
            region = next((r for r in labels.regions if r['id'] == identifier), None)
            if region:
                locs = np.flatnonzero(center_ids == identifier)
                region_names.append((region['name'], float(yy[locs[len(locs)//2]])))
        coords = probe_coordinates(settings, np.full(len(contacts), shank), heights,
                                   np.array([c['column'] for c in contacts]), face=True)
        contact_regions = labels.ids_at(transform_points(coords, project.total)).tolist()
    else: colors = None
    if cancel(): raise InterruptedError('Cancelled')
    surface = resources['surface']; depth = None
    if surface is not None:
        entry = next((e for e in entries if e['shank'] == shank), None)
        tip = probe_coordinates(settings, shank, 0.)
        if entry is not None and surface.contains(transform_points([tip], project.total)[0]):
            source_entry = transform_points([entry['atlas_zyx_um']], np.linalg.inv(project.total))[0]
            depth = float(np.linalg.norm(source_entry-tip))
    return dict(resources=resources, geometry=g, sites=probe_site_layout(g), label_ids=ids, label_regions=label_regions,
                region_positions=region_positions, images=images, overlay=colors, extent=[left, right, low, high], shanks=shanks, shank_pitch_um=g['shank_pitch_um'],
                shank_width_um=g['shank_width_um'], taper_length_um=g['taper_length_um'], depths=depths,
                all_heights=all_heights, all_columns=[c['column'] for c in all_contacts],
                all_shanks=[c['shank'] for c in all_contacts],
                region_names=region_names, regions={str(r['id']):r['name'] for r in labels.regions} if labels else {},
                depth_um=depth, heights=heights,
                columns=[c['column'] for c in contacts], contact_regions=contact_regions,
                first_site_um=g['first_site_center_from_tip_um'], pitch_um=g['site_pitch_um'])

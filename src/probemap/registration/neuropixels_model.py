"""Calibrated NP2 four-shank geometry, SpikeGLX maps and auditable channel reports.

Probe poses and lengths live in source TIFF-local physical ZYX micrometres.
The registration maps geometry to atlas space only for rendering and region lookup.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import html
import json
import re
import tempfile
from pathlib import Path
from uuid import uuid4

import numpy as np
from scipy import ndimage
from scipy.spatial.transform import Rotation

from .landmarks import transform_points
from .track_model import plane_geometry, region_name

PROBE_SOURCES = {
    'hardware': 'https://github.com/billkarsh/ProbeTable/blob/main/Tables/probe_features.json',
    'imro': 'https://github.com/billkarsh/SpikeGLX/blob/master/Markdown/Metadata.md',
    'contact_origin': 'https://github.com/SpikeInterface/probeinterface/blob/main/src/probeinterface/neuropixels_tools.py',
    'legacy': 'Neuropixels 2.0 User Manual V1.0.6, pp. 14, 41-42',
}

# Electrode block starts for channel groups 0:48:384, verified against every
# bank of the supplied IMEC Electrode-Channel mapping workbook (four shanks).
WIRING_BLOCKS=((0,288,48,336,96,192,144,240),(288,0,336,48,192,96,240,144),
               (96,192,144,240,0,288,48,336),(192,96,240,144,288,0,336,48))


def geometry_for(model='NP2013'):
    if model not in ('NP2010','NP2013','NP2014','Legacy NP2'):
        from ..probe_library import ProbeLibrary
        entry=next((e for e in ProbeLibrary().entries() if e['geometry']['model']==model),None)
        if entry is None: raise ValueError(f'Unknown probe model: {model}')
        return copy.deepcopy(entry['geometry'])
    legacy=model=='Legacy NP2'
    return dict(model=model,shanks=4,sites_per_shank=1280,shank_pitch_um=250.,
                shank_width_um=70.,shank_thickness_um=24.,site_pitch_um=15.,
                column_pitch_um=32.,site_size_um=12.,left_site_center_um=27.,
                shank_length_um=10250. if legacy else 10000.,
                taper_length_um=175. if legacy else 206.,
                first_site_center_from_tip_um=186. if legacy else 217.,
                contact_origin_note='First site centre = taper length + 11 um (ProbeInterface convention); verify for the physical probe.',
                sources=dict(PROBE_SOURCES))


def parse_channelmap(text):
    """Accept .imro, exported TXT or a complete SpikeGLX .meta; IDs are zero based."""
    found=re.search(r'(?:~?imroTbl\s*=\s*)?(\(\s*(2013|2014|24)\s*,\s*(\d+)\s*\))',text)
    if not found:
        raise ValueError('Expected an NP2 four-shank IMRO table (type 2013, 2014 or 24).')
    probe_type,count=int(found[2]),int(found[3])
    if count!=384:
        raise ValueError('A four-shank NP2 recording map must contain 384 acquisition channels.')
    body=re.split(r'\n\s*~?\w+\s*=',text[found.end():],maxsplit=1)[0].strip()
    body=re.sub(r'\\\s*(?:\n|$)','',body).strip()
    entries=re.findall(r'\(([^()]*)\)',body)
    if len(entries)!=count or re.sub(r'\([^()]*\)','',body).strip():
        raise ValueError('IMRO table is truncated or contains unexpected entries.')
    channels=[]
    for entry in entries:
        try:
            values=[int(x) for x in entry.split()]
        except ValueError as e:
            raise ValueError('IMRO entries must contain integer values.') from e
        if len(values)!=5:
            raise ValueError('Each IMRO entry needs channel, shank, bank, reference and electrode IDs.')
        ch,shank,bank,ref,electrode=values
        if not (0<=ch<384 and 0<=shank<4 and 0<=electrode<1280 and bank==electrode//384
                and 0<=ref<=(20 if probe_type==24 else 5)):
            raise ValueError(f'Invalid channel/shank/bank/reference/electrode in channel {ch}.')
        expected=WIRING_BLOCKS[shank][ch//48]+ch%48
        if electrode%384!=expected:
            raise ValueError(f'Channel {ch}: electrode {electrode} on shank {shank} conflicts with NP2 wiring.')
        channels.append(dict(channel=ch,shank=shank,bank=bank,reference=ref,electrode=electrode,
                             row=electrode//2,column=electrode%2,recorded=True))
    channels.sort(key=lambda c:c['channel'])
    if [c['channel'] for c in channels]!=list(range(384)):
        raise ValueError('Each acquisition channel 0-383 must occur exactly once.')
    if len({(c['shank'],c['electrode']) for c in channels})!=384:
        raise ValueError('Two channels select the same physical electrode.')
    # Geometry maps corroborate wiring; never confuse geometric coordinates with tip offsets.
    gm=re.search(r'~?snsGeomMap\s*=([^\n]+)',text)
    if gm:
        rows=re.findall(r'\(([^()]+)\)',gm[1])[1:]
        if len(rows)!=384: raise ValueError('snsGeomMap must contain 384 site entries.')
        for c,row in zip(channels,rows):
            parts=row.split(':')
            if len(parts)!=4: raise ValueError('Malformed snsGeomMap site.')
            s,x,y,used=map(float,parts)
            if not np.allclose([s,x,y],[c['shank'],27+32*c['column'],15*c['row']]):
                raise ValueError(f"snsGeomMap disagrees with IMRO for channel {c['channel']}.")
            c['recorded']=bool(used)
    subset=re.search(r'^snsSaveChanSubset[ \t]*=[ \t]*([^\r\n]*)',text,re.M)
    if subset and subset[1].strip()!='all':
        saved=set()
        for item in subset[1].strip().split(','):
            bounds=[int(x) for x in item.split(':')]
            if len(bounds)==1: saved.add(bounds[0])
            elif len(bounds)==2: saved.update(range(bounds[0],bounds[1]+1))
            else: raise ValueError('Invalid saved channel subset.')
        for c in channels: c['recorded'] &= c['channel'] in saved
    standby=re.search(r'^imStdby[ \t]*=[ \t]*([^\r\n]*)',text,re.M)
    if standby and standby[1].strip():
        for part in standby[1].strip().split(','):
            bounds=list(map(int,part.split(':')))
            for ch in range(bounds[0],bounds[-1]+1):
                if 0<=ch<384: channels[ch]['recorded']=False
    model=re.search(r'^imDatPrb_pn[ \t]*=[ \t]*([^\r\n]*)',text,re.M)
    return dict(probe_type=probe_type,probe_model=model[1].strip() if model else None,channels=channels,index_base=0,
                imro_sha256=hashlib.sha256((found[1]+body).encode()).hexdigest())


def add_recording(insertion,path,name=None):
    path=Path(path)
    text=path.read_text(encoding='utf-8-sig')
    parsed=parse_channelmap(text)
    if name is None:
        name=path.stem
        existing={r['name'] for r in insertion.get('recordings',[])}
        count=2
        while name in existing: name=f'{path.stem} ({count})'; count+=1
    record=dict(id=uuid4().hex[:12],name=name,path=str(path.resolve()),
                source_text=text,source_sha256=hashlib.sha256(text.encode()).hexdigest(),
                visible=True,opacity=1/(1+.3*len(insertion.get('recordings',[]))),
                depth_offset_um=0.,**parsed)
    insertion.setdefault('recordings',[]).append(record)
    return record


def probe_settings(insertion):
    settings=insertion.setdefault('probe',dict(geometry=geometry_for(),assignments={},
        direction_convention='deep_tip_to_surface',
        tip_mode='track_start',tip_shift_um=0.,lateral_shift_um=0.,normal_shift_um=0.,
        roll_deg=0.,flip_direction=False,confirmed=False,surface_depths_um={},
        display=dict(shanks=True,sites=True,shank_opacity=.18,site_opacity=1.,slice_recording='all',
                     plane_auto_fit=True,plane_offset_um=0.,plane_width_um=2200.,plane_height_um=6000.)))
    if settings.get('direction_convention')!='deep_tip_to_surface':
        # Preserve the previous placement before changing the old entry-first
        # assumption. Existing channel reports will be stale until refitted.
        settings['previous_placement']=copy.deepcopy({key:settings[key] for key in
            ('pose','tip_mode','flip_direction','tip_shift_um','lateral_shift_um',
             'normal_shift_um','roll_deg','surface_depths_um','confirmed') if key in settings})
        settings.update(direction_convention='deep_tip_to_surface',tip_mode='track_start',
                        flip_direction=False,confirmed=False,surface_depths_um={})
    settings['display'].setdefault('plane_auto_fit',True)
    return settings


def suggest_shank_assignments(insertion,matrix):
    """A provisional spatial order for FOUR traces; the mirror ambiguity remains."""
    tracks=insertion.get('tracks',[])
    if len(tracks)!=probe_settings(insertion)['geometry']['shanks']:
        raise ValueError('Automatic ordering needs exactly four traces for NP2, or one trace per shank for other probes. With fewer traces, assign their known shank IDs manually.')
    if len(tracks)==1:
        settings=probe_settings(insertion); settings['assignments']={tracks[0]['id']:0}; settings['confirmed']=False
        return settings['assignments']
    controls=np.array([transform_points(t['control_points_atlas_zyx_um'],np.linalg.inv(matrix)) for t in tracks])
    vectors=controls[:,1]-controls[:,0]
    lengths=np.linalg.norm(vectors,axis=1)
    if np.any(lengths<1e-6): raise ValueError('Each trace needs two different points.')
    directions=vectors/lengths[:,None]
    if np.any(directions@directions[0]<.5):
        raise ValueError('Trace directions disagree. Draw all tracks from the deep tip toward the brain surface on step 6.')
    down=directions.mean(0); down/=np.linalg.norm(down)
    starts=controls[:,0]-controls[:,0].mean(0)
    lateral=starts-(starts@down)[:,None]*down
    _,singular,basis=np.linalg.svd(lateral,full_matrices=False)
    if singular[0]<1e-6: raise ValueError('The traces do not have a measurable lateral separation.')
    axis=basis[0]
    if axis[np.argmax(np.abs(axis))]<0: axis=-axis
    order=np.argsort(lateral@axis)
    assignments={tracks[int(index)]['id']:shank for shank,index in enumerate(order)}
    settings=probe_settings(insertion)
    settings['assignments']=assignments
    settings['assignment_method']='Provisional spatial ordering; verify the shank-0 side or reverse the order.'
    settings['confirmed']=False
    return assignments


def fit_probe(insertion,matrix):
    """Fit parallel shanks at fixed hardware pitch; no affine scaling of hardware."""
    settings=probe_settings(insertion)
    g=settings['geometry']
    if 'contacts' in g:
        from ..probe_library import validate_geometry
        validate_geometry(g)
    else:
        dimensions=[g[k] for k in ('shank_pitch_um','shank_width_um','shank_thickness_um','shank_length_um',
                                   'taper_length_um','first_site_center_from_tip_um','site_pitch_um','site_size_um')]
        if not np.isfinite(dimensions).all() or min(dimensions)<=0 or g['taper_length_um']>=g['shank_length_um']:
            raise ValueError('Probe dimensions must be finite, positive and consistent with the shank length.')
        last_site=g['first_site_center_from_tip_um']+(g['sites_per_shank']//2-1)*g['site_pitch_um']
        if last_site+g['site_size_um']/2>g['shank_length_um']:
            raise ValueError('The recording-site array extends beyond the probe base; check geometry.')
    tracks=insertion.get('tracks',[])
    if not tracks: raise ValueError('Draw at least one track on step 6 before reconstructing this probe.')
    assignments=settings['assignments']
    selected=[(t,assignments[t['id']]) for t in tracks if t['id'] in assignments and assignments[t['id']] is not None]
    if not selected: raise ValueError('Assign at least one track to a shank (0-3), then fit the probe.')
    numbers=[int(n) for _,n in selected]
    if len(numbers)!=len(set(numbers)) or any(n not in range(g['shanks']) for n in numbers):
        raise ValueError('Assign each shank ID to at most one track.')
    inverse=np.linalg.inv(matrix)
    controls=[transform_points(t['control_points_atlas_zyx_um'],inverse) for t,_ in selected]
    vectors=np.array([p[1]-p[0] for p in controls])
    lengths=np.linalg.norm(vectors,axis=1)
    if np.any(lengths<1e-6): raise ValueError('Each assigned trace needs two different points.')
    directions=vectors/lengths[:,None]
    if np.any(directions@directions[0]<.5):
        raise ValueError('Tracks disagree on direction. Draw them all from the deep tip toward the brain surface.')
    # Tracks point from deep tip towards entry/base. Hardware height grows in
    # that direction, while down is the physical insertion direction (deeper).
    down=-directions.mean(0); down/=np.linalg.norm(down)
    if settings['flip_direction']: down=-down
    tips=np.array([p[0] if settings['tip_mode']=='track_start' else
                   p[1] if settings['tip_mode']=='direction_point' else
                   p[0]+directions[i]*float(t.get('length_um',4000))
                   for i,((t,_),p) in enumerate(zip(selected,controls))])
    if len(selected)>1:
        centered=np.array(numbers)-np.mean(numbers)
        lateral=np.sum((tips-tips.mean(0))*centered[:,None],axis=0)
        lateral-=down*np.dot(lateral,down)
    else:
        # A line leaves probe roll undetermined; use the saved insertion plane as an explicit hint.
        plane=insertion.get('plane')
        if plane is None: raise ValueError('A single track needs a saved insertion plane to define probe roll.')
        normal_atlas=np.asarray(plane_geometry(plane)['normal_zyx'])
        normal_source=np.asarray(matrix)[:3,:3].T@normal_atlas
        lateral=np.cross(normal_source,down)
    if np.linalg.norm(lateral)<1e-6: raise ValueError('Shank tracks do not determine a lateral direction.')
    lateral/=np.linalg.norm(lateral)
    normal=np.cross(down,lateral); normal/=np.linalg.norm(normal)
    angle=np.deg2rad(settings['roll_deg'])
    lateral=np.cos(angle)*lateral+np.sin(angle)*normal
    normal=np.cross(down,lateral)
    tip0=np.mean(tips-np.array(numbers)[:,None]*g['shank_pitch_um']*lateral,axis=0)
    tip0+=down*settings['tip_shift_um']+lateral*settings['lateral_shift_um']+normal*settings['normal_shift_um']
    residuals=np.linalg.norm(tips-(tip0+np.array(numbers)[:,None]*g['shank_pitch_um']*lateral),axis=1)
    pose=dict(direction_convention='deep_tip_to_surface',
              tip0_source_zyx_um=tip0.tolist(),down_source_zyx=down.tolist(),
              lateral_source_zyx=lateral.tolist(),normal_source_zyx=normal.tolist(),
              fit_rms_um=float(np.sqrt(np.mean(residuals**2))),fit_max_error_um=float(max(residuals)),
              assigned_tracks=[t['id'] for t,_ in selected],assigned_shanks=numbers,
              fit_input_sha256=probe_signature(insertion,matrix),surface_depths_um=[None]*g['shanks'],
              surface_method=['not determined']*g['shanks'])
    # Independently diagnose lateral scale from the traced centre lines. Never
    # alter TIFF calibration or hardware pitch to force agreement with the data.
    cross_section=np.array([points[0] for points in controls])
    cross_section-=(cross_section@down)[:,None]*down
    order=np.argsort(numbers)
    pitches=[float(np.linalg.norm(cross_section[j]-cross_section[i])/(numbers[j]-numbers[i]))
             for i,j in zip(order[:-1],order[1:])]
    if pitches:
        observed=float(np.median(pitches))
        pose['spacing_check']=dict(observed_pitch_um=observed,nominal_pitch_um=g['shank_pitch_um'],
            observed_to_nominal_ratio=observed/g['shank_pitch_um'],
            observed_outer_centres_um=float(np.linalg.norm(cross_section[order[-1]]-cross_section[order[0]])),
            expected_outer_centres_um=(max(numbers)-min(numbers))*g['shank_pitch_um'],
            note='Source-scan trace spacing; a discrepancy does not identify calibration error, tissue shrinkage or incorrect trace selection by itself.')
    settings['pose']=pose
    settings['confirmed']=False
    return pose


def probe_signature(insertion,matrix):
    settings=probe_settings(insertion)
    payload=dict(settings={k:v for k,v in settings.items() if k not in ('pose','confirmed','display','surface_depths_um')},
                 tracks=[(t['id'],t['control_points_atlas_zyx_um'],t.get('length_um',4000)) for t in insertion.get('tracks',[])],
                 matrix=np.asarray(matrix).tolist())
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()


def pose_current(insertion,matrix):
    pose=probe_settings(insertion).get('pose')
    return bool(pose and pose.get('fit_input_sha256')==probe_signature(insertion,matrix))


def probe_coordinates(settings,shanks,heights,columns=None,face=False):
    """Height grows from tip towards the base, in source scan micrometres."""
    g,p=settings['geometry'],settings['pose']
    lateral=np.asarray(shanks)*g['shank_pitch_um']
    if columns is not None: lateral=lateral+g['left_site_center_um']+np.asarray(columns)*g['column_pitch_um']-g['shank_width_um']/2
    result=(np.asarray(p['tip0_source_zyx_um'])+np.asarray(lateral)[...,None]*p['lateral_source_zyx']
            -np.asarray(heights)[...,None]*p['down_source_zyx'])
    if face: result=result+np.asarray(p['normal_source_zyx'])*g['shank_thickness_um']/2
    return result


def find_surface_depths(insertion,mask,spacing):
    """First outside-to-brain boundary approached from the probe base; preview precision."""
    s=probe_settings(insertion); p=s['pose']; g=s['geometry']
    if g.get('body_shape')=='none':
        p['surface_depths_um']=[None]*g['shanks']; p['surface_method']=['body geometry unknown']*g['shanks']
        return p['surface_depths_um']
    step=max(1.,min(spacing)/4)
    heights=np.linspace(0,g['shank_length_um'],int(np.ceil(g['shank_length_um']/step))+1)
    depths,methods=[],[]
    for shank in range(g['shanks']):
        override=s['surface_depths_um'].get(str(shank))
        if override is not None:
            depths.append(float(override)); methods.append('manual tip-to-surface depth in source scan um'); continue
        source=probe_coordinates(s,np.full(len(heights),shank),heights)
        coords=source/np.asarray(spacing)
        occupancy=ndimage.map_coordinates(np.asarray(mask,dtype=np.uint8),coords.T,order=0,mode='constant',cval=0,prefilter=False)>0
        # Require brain at the tip, and a real brain->background transition, not a cropped volume edge.
        crossings=np.flatnonzero(occupancy[:-1]&~occupancy[1:])
        if not occupancy[0] or not len(crossings):
            depths.append(None); methods.append('surface unavailable: tip outside mask or no boundary'); continue
        i=int(crossings[-1])
        boundary=(coords[i]+coords[i+1])/2
        if np.any(boundary<.5) or np.any(boundary>np.asarray(mask.shape)-1.5):
            depths.append(None); methods.append('surface unavailable: scan boundary'); continue
        depths.append(float((heights[i]+heights[i+1])/2)); methods.append('brain-mask boundary in source scan; verify manually')
    p['surface_depths_um']=depths; p['surface_method']=methods
    p['surface_spacing_um_zyx']=list(spacing)
    return depths


def channel_rows(insertion,matrix,labels=None,styles=None):
    s=probe_settings(insertion); p=s.get('pose')
    if p is None: return []
    current=pose_current(insertion,matrix)
    rows=[]
    for rec in insertion.get('recordings',[]):
        active,heights,sources=recording_sites(insertion,rec)
        if not active: continue
        atlas_points=transform_points(sources,matrix)
        identifiers=labels.ids_at(atlas_points) if labels is not None else [None]*len(active)
        for c,height,source,atlas,identifier in zip(active,heights,sources,atlas_points,identifiers):
            identifier=int(identifier) if identifier is not None else None
            height=float(height)
            surface=p.get('surface_depths_um',[None]*s['geometry']['shanks'])[c['shank']]
            signed=None if surface is None else height-surface-rec.get('depth_offset_um',0.)
            atlas_name=region_name(identifier,labels.regions,styles) if labels is not None else 'Labels not loaded'
            row=dict(insertion_id=insertion['id'],insertion=insertion['name'],recording_id=rec['id'],recording=rec['name'],
                     probe_model=s['geometry']['model'],channel=c['channel'],channel_index_base=0,shank=c['shank'],
                     electrode=c['electrode'],bank=c.get('bank'),reference=c.get('reference'),
                     height_from_tip_um=height,depth_from_surface_um=signed,
                     inside_surface=None if signed is None else signed<=0,
                     atlas_region_id=identifier,atlas_region=atlas_name,
                     region_id=None if signed is not None and signed>0 else identifier,
                     region='Outside brain (above entry surface)' if signed is not None and signed>0 else atlas_name,
                     surface_method=p.get('surface_method',['not determined']*s['geometry']['shanks'])[c['shank']],
                     pose_status='stale - refit required' if not current else 'confirmed' if s.get('confirmed') else 'estimated - verify tip and shank order',
                     measurement_frame='source scan physical micrometres; no tissue-shrinkage correction',
                     channelmap_sha256=rec['source_sha256'])
            row.update({f'source_{a}_um':float(v) for a,v in zip('zyx',source)})
            row.update({f'atlas_{a}_um':float(v) for a,v in zip('zyx',atlas)})
            rows.append(row)
    return rows


def recording_sites(insertion,recording):
    """Actual selected contact centres, in authoritative native scan micrometres."""
    s=probe_settings(insertion)
    active=[c for c in recording['channels'] if c.get('recorded',True)]
    if not active: return [],np.empty(0),np.empty((0,3))
    if 'contacts' in s['geometry']:
        lookup={(c['shank'],c['electrode']):c for c in s['geometry']['contacts']}
        try: contacts=[lookup[(c['shank'],c['electrode'])] for c in active]
        except KeyError as error: raise ValueError('Recording map does not match this probe geometry.') from error
        heights=np.array([c['y_um'] for c in contacts])
        sources=contact_coordinates(s,contacts)
    else:
        heights=s['geometry']['first_site_center_from_tip_um']+np.array([c['row'] for c in active])*s['geometry']['site_pitch_um']
        sources=probe_coordinates(s,np.array([c['shank'] for c in active]),heights,
            np.array([c['column'] for c in active]),face=True)
    sources+=np.asarray(s['pose']['down_source_zyx'])*recording.get('depth_offset_um',0.)
    return active,heights,sources


def contact_coordinates(settings,contacts):
    """Explicit local XYZ positions, transformed once by the rigid native pose."""
    p=settings['pose']; g=settings['geometry']
    xyz=np.array([[c['x_um']+c['shank']*g['shank_pitch_um'],c['y_um'],c['z_um']] for c in contacts],float).reshape(-1,3)
    return (np.asarray(p['tip0_source_zyx_um'])+xyz[:,0,None]*p['lateral_source_zyx']
            -xyz[:,1,None]*p['down_source_zyx']+xyz[:,2,None]*p['normal_source_zyx'])


def contact_mesh(insertion,sources,matrix):
    """Transform every corner of the physical contact squares, including shear."""
    s=probe_settings(insertion); pose=s['pose']; half=s['geometry']['site_size_um']/2
    sources=np.asarray(sources,float).reshape(-1,3)
    g=s['geometry']
    if g.get('geometry_version',1)>=2:
        from ..probe_geometry import contact_mesh_local
        contacts=g['contacts']
        all_centres=contact_coordinates(s,contacts)
        if len(sources)==len(contacts) and np.allclose(sources,all_centres): selected=contacts
        elif len(sources):
            from scipy.spatial import cKDTree
            selected=[contacts[int(i)] for i in cKDTree(all_centres).query(sources)[1]]
        else:selected=[]
        vertices=[]; faces=[]; basis=np.column_stack([pose['lateral_source_zyx'],-np.asarray(pose['down_source_zyx']),pose['normal_source_zyx']])
        for c,source in zip(selected,sources):
            local,triangles=contact_mesh_local(c,g)
            centre=np.array([c['x_um']+c['shank']*g['shank_pitch_um'],c['y_um'],c['z_um']])
            mapped=source+(local-centre)@basis.T
            faces.extend((triangles+len(vertices)).tolist());vertices.extend(mapped)
        return transform_points(np.asarray(vertices).reshape(-1,3),matrix)[:,::-1].astype(np.float32),np.asarray(faces,np.uint32).reshape(-1,3)
    height=g.get('site_height_um',g['site_size_um'])/2
    if g.get('contact_shape')=='circle':
        angles=np.linspace(0,2*np.pi,16,endpoint=False)
        offsets=np.column_stack((np.cos(angles),np.sin(angles)))
    else: offsets=np.array([[-1,-1],[1,-1],[1,1],[-1,1]])
    quads=sources[:,None,:]+half*offsets[:,0,None]*np.asarray(pose['lateral_source_zyx'])+height*offsets[:,1,None]*np.asarray(pose['down_source_zyx'])
    vertices=transform_points(quads.reshape(-1,3),matrix)[:,::-1].astype(np.float32)
    n=len(offsets); triangles=np.array([[0,i,i+1] for i in range(1,n-1)])
    faces=(triangles[None,:,:]+np.arange(len(sources))[:,None,None]*n).reshape(-1,3).astype(np.uint32)
    return vertices,faces


def probe_plane(insertion,matrix):
    """Atlas display plane of the transformed rigid source-space probe."""
    s=probe_settings(insertion); p=s['pose']; d=s['display']
    linear=np.asarray(matrix)[:3,:3]
    u=linear@p['lateral_source_zyx']; u/=np.linalg.norm(u)
    v=-linear@p['down_source_zyx']; v-=u*np.dot(u,v); v/=np.linalg.norm(v)
    normal=np.cross(u[::-1],v[::-1])[::-1]
    depths=[value for value in p.get('surface_depths_um',[]) if value is not None]
    center=probe_coordinates(s,(s['geometry']['shanks']-1)/2,float(np.mean(depths) if depths else 4000)/2)
    center=transform_points(center[None],matrix)[0]+normal*d.get('plane_offset_um',0)
    rotation=np.column_stack((u[::-1],v[::-1],normal[::-1]))
    return dict(center_um_zyx=center.tolist(),rotation_xyz_deg=Rotation.from_matrix(rotation).as_euler('xyz',degrees=True).tolist(),
                width_um=d.get('plane_width_um',2200),height_um=d.get('plane_height_um',6000),max_dimension=768)


def shank_mesh(insertion,matrix,sections=None):
    """Tapered shaft sections in native height units, then registered together."""
    s=probe_settings(insertion); g=s['geometry']; p=s['pose']
    if g.get('geometry_version',1)>=2:
        from ..probe_geometry import body_mesh
        local,faces=body_mesh(g,sections)
        basis=np.column_stack([p['lateral_source_zyx'],-np.asarray(p['down_source_zyx']),p['normal_source_zyx']])
        source=np.asarray(p['tip0_source_zyx_um'])+local@basis.T
        return transform_points(source,matrix)[:,::-1].astype(np.float32),faces.astype(np.uint32)
    vertices,faces=[],[]
    if sections is None: sections=[(shank,0.,g['shank_length_um']) for shank in range(g['shanks'])]
    for shank,start,stop in sections:
        start,stop=np.clip([start,stop],0,g['shank_length_um'])
        if stop<=start: continue
        if g.get('body_shape')=='wire':
            offset=len(vertices); count=16
            angles=np.linspace(0,2*np.pi,count,endpoint=False)
            for h in (start,stop):
                center=probe_coordinates(s,shank,h)
                vertices.extend(center+np.cos(a)*g['shank_width_um']/2*np.asarray(p['lateral_source_zyx'])+np.sin(a)*g['shank_thickness_um']/2*np.asarray(p['normal_source_zyx']) for a in angles)
            for i in range(count):
                j=(i+1)%count
                faces.extend([[offset+i,offset+j,offset+count+j],[offset+i,offset+count+j,offset+count+i]])
            for i in range(1,count-1): faces.extend([[offset,offset+i+1,offset+i],[offset+count,offset+count+i,offset+count+i+1]])
            continue
        heights=[start]+([g['taper_length_um']] if start<g['taper_length_um']<stop else [])+[stop]
        half_width=lambda h:g['shank_width_um']/2*(min(1.,h/g['taper_length_um']) if g.get('body_shape','tapered')=='tapered' and g['taper_length_um']>0 else 1.)
        contour=[(-half_width(h),h) for h in heights]+[(half_width(h),h) for h in heights[::-1]]
        contour=[point for i,point in enumerate(contour) if i==0 or point!=contour[i-1]]
        if contour[-1]==contour[0]: contour.pop()
        n=len(contour)
        offset=len(vertices)
        for side in (-1,1):
            for x,h in contour:
                point=probe_coordinates(s,shank,h)+x*np.asarray(p['lateral_source_zyx'])+side*g['shank_thickness_um']/2*np.asarray(p['normal_source_zyx'])
                vertices.append(point)
        for b in range(1,n-1):
            faces.extend([[offset,offset+b,offset+b+1],[offset+n+b+1,offset+n+b,offset+n]])
        for i in range(n):
            j=(i+1)%n; faces.extend([[offset+i,offset+j,offset+n+j],[offset+i,offset+n+j,offset+n+i]])
    return transform_points(np.asarray(vertices).reshape(-1,3),matrix)[:,::-1].astype(np.float32),np.asarray(faces,np.uint32).reshape(-1,3)


def write_reports(project,folder,labels=None):
    """One row per saved recording channel. Missing surface/labels stay unavailable."""
    folder=Path(folder); folder.mkdir(parents=True,exist_ok=True)
    if labels is None and project.labels_dir and any(i.get('recordings') and i.get('probe',{}).get('pose')
            for session in project.track_analysis.get('sessions',[]) for i in session.get('insertions',[])):
        from .track_model import load_label_atlas
        labels=load_label_atlas(project.labels_dir,project.reference,Path(tempfile.gettempdir())/'probemap'/'native-labels',
                                spacing_um_zyx=project.label_spacing)
    rows=[]; probes=[]; measurements=[]
    for session in project.track_analysis.get('sessions',[]):
        matrix=np.asarray(session['sources']['matrix'])
        from .insertion_model import sync_measurements
        sync_measurements(session)
        for m in session.get('measurements',[]):
            measurements.append(dict(session=session['id'],measurement_id=m['id'],length_um=m['length_um'],
                                     atlas_length_um=m['atlas_length_um'],measurement_frame=m['measurement_frame'],
                                     **{f'{frame}_{axis}{end}_um':point[i] for frame,key in [('source','points_source_scan_zyx_um'),('atlas','points_atlas_zyx_um')]
                                        for end,point in enumerate(m[key]) for i,axis in enumerate('zyx')}))
        for insertion in session.get('insertions',[]):
            if not insertion.get('probe',{}).get('pose'): continue
            # Historic sessions may use another atlas: do not assign current labels to them.
            matching=session['sources']['reference']==project.reference.record()
            result=channel_rows(insertion,matrix,labels if matching else None,session.get('regions'))
            for row in result: row['session_id']=session['id']
            rows.extend(result)
            probes.append(dict(session_id=session['id'],insertion=insertion['name'],probe=insertion['probe'],
                               recordings=insertion.get('recordings',[])))
    def csv_file(name,records,default):
        target=folder/name; temp=target.with_suffix(target.suffix+'.tmp')
        with temp.open('w',newline='',encoding='utf-8-sig') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(records[0]) if records else default)
            writer.writeheader(); writer.writerows(records)
        temp.replace(target)
    csv_file('neuropixels_channels.csv',rows,['insertion','recording','channel','region','depth_from_surface_um','pose_status'])
    csv_file('measurements.csv',measurements,['session','measurement_id','length_um','measurement_frame'])
    (folder/'neuropixels_reconstruction.json').write_text(json.dumps(dict(schema_version=1,probes=probes,
        source_scan_spacing_um_zyx=list(project.moving.spacing_um_zyx),sources=PROBE_SOURCES),indent=2)+'\n', encoding='utf-8')
    columns=['insertion','recording','channel','shank','electrode','region','depth_from_surface_um','pose_status']
    table=''.join('<tr>'+''.join('<td>'+html.escape('unavailable' if r.get(k) is None else
        f'{r[k]:.1f}' if isinstance(r[k],float) else str(r.get(k,'')))+'</td>' for k in columns)+'</tr>' for r in rows)
    (folder/'neuropixels_report.html').write_text('<!doctype html><meta charset="utf-8"><title>Neuropixels channel reconstruction</title>'
        '<style>body{font:15px system-ui;margin:30px;color:#162332}table{border-collapse:collapse}td,th{border:1px solid #cad4df;padding:7px}th{position:sticky;top:0;background:#e4edf5}tr:nth-child(even){background:#f4f7fa}</style>'
        '<h1>Neuropixels channel reconstruction</h1><p>Source-scan physical micrometres. Surface = 0; negative values are below the entry surface along the shank. '
        'No tissue-shrinkage correction. Channel, electrode and shank IDs are zero based. Unavailable values are not zero. '
        'Estimated or stale poses require checking tip placement and shank order. Channel maps describe selected hardware sites; recording quality is not inferred.</p>'
        '<table><thead><tr>'+''.join('<th>'+html.escape(k.replace('_',' '))+'</th>' for k in columns)+'</tr></thead><tbody>'+table+'</tbody></table>', encoding='utf-8')
    return folder/'neuropixels_report.html'

"""Native scan planes and rigid, interactive four-shank probe placement."""
import copy
from uuid import uuid4

import numpy as np

from .landmarks import transform_points
from .insertion_model import points_signature, sync_tracks
from .neuropixels_model import probe_settings, probe_signature, probe_coordinates, contact_coordinates
from .track_model import plane_geometry, transform_plane


def source_plane(insertion,template,matrix):
    """Migrate a saved atlas plane once; thereafter retain native coordinates."""
    if insertion.get('source_plane') is None:
        atlas=copy.deepcopy(insertion.get('plane') or template)
        if insertion.get('plane') is None and insertion['points']:
            atlas['center_um_zyx']=np.mean([p['atlas_zyx_um'] for p in insertion['points']],axis=0).tolist()
        insertion['source_plane']=transform_plane(atlas,np.linalg.inv(matrix))
        insertion['source_plane']['coordinate_frame']='source scan physical micrometres'
    return insertion['source_plane']


def save_source_plane(insertion,plane,matrix,*,manual=True):
    plane_geometry(plane)
    insertion['source_plane']=copy.deepcopy(plane)
    insertion['source_plane']['coordinate_frame']='source scan physical micrometres'
    insertion['plane']=transform_plane(plane,matrix)
    insertion['plane']['coordinate_frame']='atlas physical micrometres'
    insertion.pop('plane_draft',None)
    if manual:
        points=np.array([p['atlas_zyx_um'] for p in insertion['points']]).reshape(-1,3)
        g=plane_geometry(plane)
        distances=(transform_points(points,np.linalg.inv(matrix))-g['center_um_zyx'])@g['normal_zyx'] if len(points) else np.array([])
        insertion['plane_fit']=dict(method='Manual plane',manual=True,point_signature=points_signature(insertion),
            measurement_frame='source scan physical micrometres',n_points=len(points),
            **(dict(rms_um=float(np.sqrt(np.mean(distances**2))),max_error_um=float(np.max(np.abs(distances)))) if len(distances) else {}))


def pose_on_plane(insertion,plane,tip_center,angle_deg):
    """Pointer anchors the centre of the four deep tips; angle points to base."""
    s=probe_settings(insertion); g=plane_geometry(plane)
    angle=np.deg2rad(angle_deg)
    u,v=np.array(g['u_zyx']),np.array(g['v_zyx'])
    lateral=(np.cos(angle)*u-np.sin(angle)*v)*contact_side(insertion,plane)
    base=np.sin(angle)*u+np.cos(angle)*v
    down=-base
    return dict(direction_convention='deep_tip_to_surface',
        tip0_source_zyx_um=(np.asarray(tip_center)-((s['geometry']['shanks']-1)/2)*s['geometry']['shank_pitch_um']*lateral).tolist(),
        down_source_zyx=down.tolist(),lateral_source_zyx=lateral.tolist(),
        normal_source_zyx=np.cross(down,lateral).tolist(),
        fit_rms_um=0.,fit_max_error_um=0.,placement_method='manual native plane',
        surface_depths_um=[None]*s['geometry']['shanks'],surface_method=['not determined']*s['geometry']['shanks'])


def contact_side(insertion,plane):
    """Recording face relative to the saved native plane, independent of yaw."""
    s=probe_settings(insertion); p=s.get('pose')
    if p is None: return int(s.get('placement_contact_side',1))
    g=plane_geometry(plane)
    front=np.cross(-np.asarray(g['v_zyx']),g['u_zyx'])
    return 1 if np.dot(p['normal_source_zyx'],front)>=0 else -1


def flip_contact_face(insertion,plane):
    """Roll 180 degrees about the shaft centre: swap face and shank order."""
    s=probe_settings(insertion); p=s.get('pose')
    s['placement_contact_side']=-contact_side(insertion,plane)
    if p is None: return None
    center=probe_coordinates(s,(s['geometry']['shanks']-1)/2,0.)
    result=copy.deepcopy(p)
    result['lateral_source_zyx']=(-np.asarray(p['lateral_source_zyx'])).tolist()
    result['normal_source_zyx']=(-np.asarray(p['normal_source_zyx'])).tolist()
    result['tip0_source_zyx_um']=(center-((s['geometry']['shanks']-1)/2)*s['geometry']['shank_pitch_um']*np.asarray(result['lateral_source_zyx'])).tolist()
    return result


def contact_preview(insertion,matrix):
    """Sparse real site centres; only an orientation cue, not a channel map."""
    s=probe_settings(insertion); g=s['geometry']
    if 'contacts' in g:
        return transform_points(contact_coordinates(s,g['contacts']),matrix)
    rows=np.arange(0,g['sites_per_shank']//2,32)
    shanks,rows,columns=np.meshgrid(np.arange(g['shanks']),rows,np.arange(2),indexing='ij')
    height=g['first_site_center_from_tip_um']+rows.ravel()*g['site_pitch_um']
    source=probe_coordinates(s,shanks.ravel(),height,columns.ravel(),face=True)
    return transform_points(source,matrix)


def place_probe(insertion,pose,project,session):
    """Persist one authoritative native pose and compatible shank centre lines."""
    s=probe_settings(insertion)
    if insertion.get('plane') is None:
        save_source_plane(insertion,insertion['source_plane'],project.total)
    if any(not t.get('probe_placement_generated') for t in insertion['tracks']):
        insertion.setdefault('tracks_before_probe_placement',copy.deepcopy(insertion['tracks']))
        insertion['tracks']=[]
    tracks=insertion['tracks']
    s['pose']=copy.deepcopy(pose)
    s.update(tip_mode='track_start',tip_shift_um=0.,lateral_shift_um=0.,normal_shift_um=0.,
             roll_deg=0.,flip_direction=False,confirmed=False,surface_depths_um={})
    s['assignments']={}
    for shank in range(s['geometry']['shanks']):
        tip=probe_coordinates(s,shank,0.)
        control=transform_points([tip,tip-np.asarray(pose['down_source_zyx'])*1000],project.total)
        if shank>=len(tracks): tracks.append(dict(id=uuid4().hex[:12],name=f'Shank {shank}'))
        track=tracks[shank]
        track.update(control_points_atlas_zyx_um=control.tolist(),probe_placement_generated=True,
                     plane=copy.deepcopy(insertion['plane']),start_validation={'inside':False,'method':'Needs brain-mask validation'})
        s['assignments'][track['id']]=shank
    del tracks[s['geometry']['shanks']:]
    s['assignment_method']='Manual probe placement; shank 0 follows the displayed order.'
    s['pose'].update(assigned_tracks=[t['id'] for t in tracks],assigned_shanks=list(range(s['geometry']['shanks'])))
    sync_tracks(session,project)
    s['pose']['fit_input_sha256']=probe_signature(insertion,project.total)
    return s['pose']


def probe_contours(insertion,matrix):
    s=probe_settings(insertion); g=s['geometry']; p=s['pose']
    result=[]
    if g.get('geometry_version',1)>=2:
        from ..probe_geometry import body_polygon
        if g.get('body_shape')=='none':return result
        if g.get('body_shape')=='prism':
            poly=np.asarray(g['body_polygon_um']); w=max(abs(poly[:,0]))
            outline=[[-w,0],[w,0],[w,g['shank_length_um']],[-w,g['shank_length_um']]]
        elif g.get('body_shape') in ('wire','cylinder'):
            w=g['shank_width_um']/2;outline=[[-w,0],[-w,g['shank_length_um']],[w,g['shank_length_um']],[w,0]]
        else:outline=body_polygon(g)
        for shank in range(g['shanks']):
            poly=[probe_coordinates(s,shank,h)+x*np.asarray(p['lateral_source_zyx']) for x,h in outline]
            result.append(transform_points(poly,matrix))
        return result
    for shank in range(s['geometry']['shanks']):
        poly=[probe_coordinates(s,shank,h)+x*np.asarray(p['lateral_source_zyx'])
              for x,h in ((0 if g.get('body_shape','tapered')=='tapered' else -g['shank_width_um']/2,0),(-g['shank_width_um']/2,g['taper_length_um']),
                  (-g['shank_width_um']/2,g['shank_length_um']),
                  (g['shank_width_um']/2,g['shank_length_um']),(g['shank_width_um']/2,g['taper_length_um']))]
        result.append(transform_points(poly,matrix))
    return result

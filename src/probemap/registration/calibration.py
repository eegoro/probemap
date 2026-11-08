"""Recalibrate physical coordinates without moving image-anchored annotations."""
from __future__ import annotations

import copy
import hashlib
import json
import warnings
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from .desktop_model import load_preview
from .insertion_model import (brain_support, fit_insertion_plane, inside_brain,
    plane_is_stale, save_manual_plane, sync_measurements, sync_tracks)
from .landmarks import fit_landmarks, transform_points
from .neuropixels_model import find_surface_depths, fit_probe
from .track_model import plane_geometry
from .volumes import read_volume, spacing_zyx


def _scale(old,new):
    result=np.eye(4)
    result[:3,:3]=np.diag(spacing_zyx(new)/spacing_zyx(old))
    return result


def _calibrated(spec,spacing):
    spacing=tuple(spacing_zyx(spacing))
    if np.array_equal(spec.spacing_um_zyx,spacing): return copy.deepcopy(spec)
    return replace(spec,spacing_um_zyx=spacing,
                   initial_spacing_um_zyx=spec.initial_spacing_um_zyx or spec.spacing_um_zyx)


def _same_image(record,spec):
    return record.get('path')==str(spec.path) and tuple(record.get('shape_zyx',()))==spec.shape_zyx


def _point(value,matrix):
    return transform_points(np.asarray(value,float).reshape(-1,3),matrix)[0].tolist()


def _plane(plane,matrix):
    """Map an oblique plane and enclose its transformed rectangular field."""
    if plane is None: return None
    if np.allclose(matrix,np.eye(4),rtol=0,atol=1e-12): return copy.deepcopy(plane)
    geometry=plane_geometry(plane)
    linear=matrix[:3,:3]
    u=linear@geometry['u_zyx']; u/=np.linalg.norm(u)
    normal=np.linalg.solve(linear.T,geometry['normal_zyx']); normal/=np.linalg.norm(normal)
    v=np.cross(normal[::-1],u[::-1])[::-1]
    center=transform_points([geometry['center_um_zyx']],matrix)[0]
    old_center=np.asarray(geometry['center_um_zyx'])
    old_u=np.asarray(geometry['u_zyx'])*plane['width_um']/2
    old_v=np.asarray(geometry['v_zyx'])*plane['height_um']/2
    corners=transform_points([old_center+i*old_u+j*old_v for i in (-1,1) for j in (-1,1)],matrix)-center
    basis=np.column_stack((u[::-1],v[::-1],normal[::-1]))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',UserWarning)
        angles=Rotation.from_matrix(basis).as_euler('xyz',degrees=True)
    return dict(plane,center_um_zyx=center.tolist(),rotation_xyz_deg=angles.tolist(),
                width_um=float(np.ptp(corners@u)),height_um=float(np.ptp(corners@v)))


def _payload(payload,matrix,sources,mask,mask_volume,measurement_maps):
    """Rebuild a session or an undo snapshot in its new calibrated frame."""
    was_session='sources' in payload
    payload['sources']=sources
    if payload.get('plane') is not None: payload['plane']=_plane(payload['plane'],matrix)
    for measure in payload.get('measurements',[]):
        anchor=measurement_maps.get(measure.get('anchor_image','track'),matrix)
        measure['points_atlas_zyx_um']=transform_points(measure['points_atlas_zyx_um'],anchor).tolist()
    for item in payload.get('insertions',[]):
        current_plane=not plane_is_stale(item)
        for point in item['points']:
            anchor=measurement_maps.get(point.get('anchor_image','track'),matrix)
            point['atlas_zyx_um']=_point(point['atlas_zyx_um'],anchor)
            native=transform_points([point['atlas_zyx_um']],np.linalg.inv(sources['matrix']))[0]
            point['source_track_zyx_voxel']=(native/sources['track']['spacing_um_zyx']).tolist()
        for key in ('plane','plane_draft'):
            if item.get(key) is not None: item[key]=_plane(item[key],matrix)
        # Source planes must follow voxel recalibration, rather than retaining
        # dimensions from the previous scan calibration.
        item.pop('source_plane',None)
        fit=item.get('plane_fit') or {}
        if item.get('plane') is not None and current_plane:
            if fit.get('manual',False): save_manual_plane(item,item['plane'])
            else: fit_insertion_plane(item,item['plane'],np.asarray(sources['matrix']))
        if item.get('plane') is not None and item.get('source_plane') is None:
            from .probe_placement import source_plane
            source_plane(item,item['plane'],np.asarray(sources['matrix']))
        draft=item.get('drawing_draft')
        if draft:
            for key in ('start','direction'):
                if draft.get(key) is not None: draft[key]=_point(draft[key],matrix)
            if draft.get('plane') is not None: draft['plane']=_plane(draft['plane'],matrix)
        for track in item['tracks']:
            track['control_points_atlas_zyx_um']=transform_points(track['control_points_atlas_zyx_um'],matrix).tolist()
            if track.get('plane') is not None: track['plane']=_plane(track['plane'],matrix)
            if mask is not None:
                inside=bool(inside_brain(track['control_points_atlas_zyx_um'][:1],mask,mask_volume,sources['matrix'])[0])
                track['start_validation']=dict(inside=inside,method='Recalibrated brain foreground envelope',
                    preview_spacing_um_zyx=list(mask_volume.spacing_um_zyx))
    view=SimpleNamespace(total=np.asarray(sources['matrix']),track=SimpleNamespace(**sources['track']))
    if 'insertions' in payload:
        sync_tracks(payload,view)
    else:
        # Older projects stored arbitrary polylines before insertion groups.
        # Keep their vertices and native lengths until the normal migration.
        for track in payload.get('tracks',[]):
            points=transform_points(track['points_atlas_zyx_um'],matrix)
            source=transform_points(points,np.linalg.inv(sources['matrix']))
            track.update(points_atlas_zyx_um=points.tolist(),points_source_scan_zyx_um=source.tolist(),
                         points_source_track_zyx_voxel=(source/sources['track']['spacing_um_zyx']).tolist(),
                         length_um=float(np.linalg.norm(np.diff(source,axis=0),axis=1).sum()))
            if track.get('plane') is not None: track['plane']=_plane(track['plane'],matrix)
        sync_measurements(payload)
    for item in payload.get('insertions',[]):
        if item.get('probe',{}).get('pose'):
            fit_probe(item,sources['matrix'])
            if mask is not None: find_surface_depths(item,mask,mask_volume.spacing_um_zyx)
    if not was_session:
        payload.pop('sources')
        payload.pop('tracks',None)


def recalibrate_project(project,reference_spacing,scan_spacing,progress=None,*,labels_spacing=None,track_spacing=None):
    """Return a complete new project; the input object and TIFFs stay intact.

    Atlas, labels, brain and electrodes have independent voxel calibrations.
    A maps old atlas micrometres to new ones; B does the same for the brain.
    Placement becomes A C B^-1. Landmarks remain on their image voxels and any
    existing fit is recalculated using its selected model. Insertion geometry
    follows the resulting map; finite 4 mm tracks and rigid hardware are rebuilt
    in source micrometres rather than scaled as pieces of the tissue.
    """
    reference_spacing=spacing_zyx(reference_spacing)
    scan_spacing=spacing_zyx(scan_spacing)
    labels_spacing=spacing_zyx(project.label_spacing if labels_spacing is None else labels_spacing)
    track_spacing=spacing_zyx((project.track.spacing_um_zyx if project.track else scan_spacing)
                             if track_spacing is None else track_spacing)
    result=copy.deepcopy(project)
    geometry_unchanged=(np.array_equal(project.reference.spacing_um_zyx,reference_spacing)
            and np.array_equal(project.moving.spacing_um_zyx,scan_spacing)
            and (project.track is None or np.array_equal(project.track.spacing_um_zyx,track_spacing)))
    if geometry_unchanged:
        if project.labels_dir is not None: result.labels_spacing_um_zyx=tuple(labels_spacing)
        return result
    atlas_scale=_scale(project.reference.spacing_um_zyx,reference_spacing)
    source_scale=_scale(project.moving.spacing_um_zyx,scan_spacing)
    result.reference=_calibrated(project.reference,reference_spacing)
    result.moving=_calibrated(project.moving,scan_spacing)
    if project.track is not None: result.track=_calibrated(project.track,track_spacing)
    if project.labels_dir is not None: result.labels_spacing_um_zyx=tuple(labels_spacing)
    result.validate_track()
    result.coarse=atlas_scale@project.coarse@np.linalg.inv(source_scale)
    result.residual=atlas_scale@project.residual@np.linalg.inv(atlas_scale)
    for pair in result.pairs:
        for key in ('reference','moving'):
            if pair.get(key) is not None: pair[key]=_point(pair[key],atlas_scale)
    if project.fit_is_current:
        result.fit(project.fit_result['model'])
    elif result.fit_result is not None:
        # Pending landmark edits must stay pending. Refit the last accepted
        # pairs, retaining the distinction from the user's current edits.
        for pair in result.fitted_pairs:
            for key in ('reference','moving'):
                if pair.get(key) is not None: pair[key]=_point(pair[key],atlas_scale)
        complete=[p for p in result.fitted_pairs if p.get('reference') is not None and p.get('moving') is not None]
        result.fit_result=fit_landmarks([p['reference'] for p in complete],
                                       [p['moving'] for p in complete],model=project.fit_result['model'])
        result.residual=np.array(result.fit_result['moving_to_reference_zyx_um'])
        result.fitted_coarse=(atlas_scale@np.asarray(project.fitted_coarse)@np.linalg.inv(source_scale)).tolist()
    if progress: progress(15,'Recalculating landmarks and physical coordinates')

    active=result.track_analysis.get('active_session')
    for session in result.track_analysis.get('sessions',[]):
        old=copy.deepcopy(session['sources'])
        if 'track' not in old or 'reference' not in old: continue
        same_track=project.track is not None and _same_image(old['track'],project.track)
        same_atlas=_same_image(old['reference'],project.reference)
        old_brain=old.get('moving',project.moving.record())
        same_brain=_same_image(old_brain,project.moving) and (same_track or 'moving' in old)
        b=_scale(old_brain['spacing_um_zyx'],scan_spacing) if same_brain else np.eye(4)
        bt=_scale(old['track']['spacing_um_zyx'],track_spacing) if same_track else np.eye(4)
        a=_scale(old['reference']['spacing_um_zyx'],reference_spacing) if same_atlas else np.eye(4)
        sources=copy.deepcopy(old)
        if same_track: sources['track']=result.track.record()
        if same_brain: sources['moving']=result.moving.record()
        if same_atlas: sources['reference']=result.reference.record()
        new_matrix=a@np.asarray(old['matrix'])@np.linalg.inv(b)
        if session['id']==active and same_track and same_brain and same_atlas and result.fit_is_current:
            new_matrix=result.total
        sources['matrix']=new_matrix.tolist()
        inverse=np.linalg.inv(old['matrix'])
        annotation_map=new_matrix@bt@inverse
        measurement_maps={'track':annotation_map,'moving':new_matrix@b@inverse,'reference':a}
        session['sources']=sources
        signature={key:sources[key] for key in ('track','reference','matrix')}
        session['signature']=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
        mask=mask_volume=None
        if same_brain:
            previous=session.get('brain_mask',{})
            saved=previous.get('preview_spacing_um_zyx')
            if saved is not None:
                factors=np.maximum(1,np.rint(np.asarray(saved)/old_brain['spacing_um_zyx']).astype(int))
                mask_volume=read_volume(result.moving.path,scan_spacing,factors,normalize=True)
            else: mask_volume=load_preview(result.moving,256)
            mask,threshold=brain_support(mask_volume,previous.get('threshold'))
            session['brain_mask']=dict(previous,threshold=threshold,
                                      preview_spacing_um_zyx=list(mask_volume.spacing_um_zyx))
        _payload(session,annotation_map,sources,mask,mask_volume,measurement_maps)
        for snapshot in session.get('edit_history',[]):
            _payload({'insertions':snapshot} if isinstance(snapshot,list) else snapshot,
                     annotation_map,sources,mask,mask_volume,measurement_maps)
        view=session.get('insertion_view',{})
        if view.get('orthogonal_positions_zyx_um') is not None:
            view['orthogonal_positions_zyx_um']=_point(view['orthogonal_positions_zyx_um'],annotation_map)
        for key in ('camera','slice_views','tracing_positions_zyx_um'): view.pop(key,None)
        if progress: progress(40,'Rebuilding tracks, insertion planes and probe surfaces')

    display=result.display
    if display.get('overlay_position_um_zyx') is not None:
        display['overlay_position_um_zyx']=_point(display['overlay_position_um_zyx'],atlas_scale)
    for key,value in display.get('landmark_positions_um_zyx',{}).items():
        display['landmark_positions_um_zyx'][key]=_point(value,atlas_scale)
    # Export sampling is expressed in atlas axes/units, independently of the
    # source acquisition calibration used for quantitative measurements.
    if 'adapted_spacing_um_xyz' in display:
        ratio=(reference_spacing/project.reference.spacing_um_zyx)[::-1]
        display['adapted_spacing_um_xyz']=(np.asarray(display['adapted_spacing_um_xyz'])*ratio).tolist()
    for key in ('placement_controls','placement_baseline_zyx','placement_slice_views','landmark_slice_views','camera_3d','slice_comparison'):
        display.pop(key,None)
    return result


def prepare_calibration_update(project,reference_spacing,scan_spacing,project_file,
                               progress=None,cancelled=lambda:False,*,labels_spacing=None,track_spacing=None):
    """Stage derived files, then replace the current results with rollback.

    Temporary data are kept outside the results folder. A failed or cancelled
    recalculation retains both the original project and its existing exports.
    No versions or permanent backup folders are created.
    """
    import shutil
    import tempfile
    from importlib.util import find_spec
    from pathlib import Path
    from ..project_paths import relative_path_links
    from .desktop_model import export_project
    from .insertion_model import export_insertions
    from .neuropixels_model import write_reports
    from .track_model import export_track_analysis,load_label_atlas

    def check():
        if cancelled(): raise InterruptedError('Calibration cancelled.')
    check()
    updated=recalibrate_project(project,reference_spacing,scan_spacing,progress,
                                labels_spacing=labels_spacing,track_spacing=track_spacing)
    reference=load_preview(updated.reference,updated.preview_dimension)
    moving=load_preview(updated.moving,updated.preview_dimension)
    check()
    project_file=Path(project_file).resolve()
    folder=project_file.parent
    folder.mkdir(parents=True,exist_ok=True)
    old_export=json.loads((folder/'export.json').read_text(encoding='utf-8')) if (folder/'export.json').is_file() else None
    with tempfile.TemporaryDirectory(prefix='probemap-calibration-') as temporary:
        scratch=Path(temporary).resolve()
        stage=scratch/'results'; stage.mkdir()
        rollback=scratch/'rollback'; rollback.mkdir()
        updated.output_dir=stage
        labels=None
        accepted=updated
        if updated.fit_result is not None and not updated.fit_is_current:
            accepted=copy.deepcopy(updated)
            accepted.coarse=np.asarray(updated.fitted_coarse)
            accepted.pairs=copy.deepcopy(updated.fitted_pairs)
        def export_progress(value,message):
            check()
            if progress: progress(45+int(value*.4),message)
        geometry_changed=any(getattr(updated,key).record()!=getattr(project,key).record()
                             for key in ('reference','moving')) or (
                             updated.track is not None and updated.track.record()!=project.track.record())
        if old_export and geometry_changed:
            adapted=next((item['spacing_um_zyx'] for item in old_export.get('outputs',[])
                          if item['filename']=='scan_adapted.tif'),None)
            if adapted is not None:
                adapted=(np.asarray(adapted)*np.asarray(updated.reference.spacing_um_zyx)/project.reference.spacing_um_zyx).tolist()
            ants=bool(old_export.get('ants_checkpoint'))
            if ants and find_spec('ants') is None:
                raise ValueError('Install antspyx to update the existing ANTs transforms.')
            exported=accepted if old_export['stage']=='affine' else updated
            export_project(exported,old_export['stage'],adapted,ants,export_progress,cancelled)
        check()
        if updated.labels_dir and ((folder/'insertions.json').exists() or (folder/'tracks.json').exists()):
            labels=load_label_atlas(updated.labels_dir,updated.reference,
                                   Path(tempfile.gettempdir())/'probemap/native-labels',export_progress,
                                   spacing_um_zyx=updated.label_spacing)
        # Saved analysis belongs to its accepted registration, even when the
        # user is currently editing placement or landmark pairs. Refresh those
        # results without silently accepting the pending edits.
        analysis=accepted
        session=next((s for s in updated.track_analysis.get('sessions',[])
                      if s['id']==updated.track_analysis.get('active_session')),None)
        if session is not None and not np.allclose(accepted.total,session['sources']['matrix'],rtol=0,atol=1e-10):
            analysis=copy.deepcopy(accepted)
            analysis.coarse=np.asarray(session['sources']['matrix'])
            analysis.residual=np.eye(4)
            analysis.fitted_coarse=analysis.coarse.tolist()
            analysis.fitted_pairs=copy.deepcopy(analysis.pairs)
        if (folder/'insertions.json').is_file(): export_insertions(analysis,labels,export_progress,allow_invalid=True)
        if (folder/'tracks.json').is_file() and labels is not None:
            export_track_analysis(analysis,labels,export_progress)
        if any(s.get('measurements') or any(i.get('probe',{}).get('pose') for i in s.get('insertions',[]))
               for s in updated.track_analysis.get('sessions',[])):
            write_reports(updated,stage,labels)
        check()
        # Named exports share the same transaction, including destinations
        # outside the project folder. Deleted exports are never resurrected.
        from .exports import export_selection
        named_files=[]
        for index,options in enumerate(updated.exports):
            if not geometry_changed and options['kind']!='analysis': continue
            target=Path(options['path']).resolve()
            if not target.is_file(): continue
            current=copy.deepcopy(options)
            exported=accepted if options['stage']=='registration' else updated
            if options['kind'] in ('brain','electrodes'):
                source=exported.track if options['kind']=='electrodes' else exported.moving
                if options.get('spacing_policy')=='atlas': current['spacing_um_zyx']=list(exported.reference.spacing_um_zyx)
                elif options.get('spacing_policy')=='source': current['spacing_um_zyx']=list(source.spacing_um_zyx)
                options['spacing_um_zyx']=current['spacing_um_zyx']
            named=scratch/'named'/str(index)/target.name
            current['path']=str(named)
            export_selection(exported,current,export_progress,cancelled)
            named_files.append((named,target))
            if options['kind'] in ('brain','electrodes'):
                named_files.append((named.with_suffix('.geometry.json'),target.with_suffix('.geometry.json')))
            check()
        updated.output_dir=folder
        # Rewrite ANTs checkpoint/config paths after staging. Image links keep
        # pointing to their original TIFFs; only generated-file locations move.
        def unstage(value):
            if isinstance(value,str) and (value==str(stage) or value.startswith(str(stage)+'/')):
                return str(folder)+value[len(str(stage)):]
            if isinstance(value,list): return [unstage(item) for item in value]
            if isinstance(value,dict): return {unstage(key):unstage(item) for key,item in value.items()}
            return value
        for path in stage.glob('*.json'):
            if path.name=='project.json': continue
            path.write_text(json.dumps(unstage(json.loads(path.read_text(encoding='utf-8'))),indent=2)+'\n', encoding='utf-8')
        (stage/'project.json').unlink(missing_ok=True)
        record=updated.as_record()
        record['relative_paths']=relative_path_links(record,project_file)
        (stage/project_file.name).write_text(json.dumps(record,indent=2)+'\n', encoding='utf-8')
        files=sorted(path for path in stage.iterdir() if path.is_file() and path.name!=project_file.name)
        files.append(stage/project_file.name)  # Commit project metadata last.
        transfers=[(path,folder/path.name) for path in files[:-1]]+named_files+[(files[-1],project_file)]
        protected={spec.path for spec in (project.reference,project.moving,project.track) if spec is not None}
        if project.labels_dir:
            protected.update(path.resolve() for path in Path(project.labels_dir).iterdir() if path.is_file())
        if any(target.resolve() in protected for _,target in transfers) or any(target==project_file for _,target in named_files):
            raise ValueError('A generated result would overwrite an input image. Choose a separate project folder.')
        committed=[]
        pending=None
        try:
            for index,(source,target) in enumerate(transfers):
                check()
                backup=rollback/str(index)
                if target.exists(): shutil.copy2(target,backup)
                pending=target.with_name('.'+target.name+'.tmp')
                shutil.copy2(source,pending)
                pending.replace(target)
                pending=None
                committed.append((target,backup))
                if progress: progress(85+int(15*(index+1)/len(transfers)),'Updating calibrated project files')
        except BaseException:
            if pending is not None: pending.unlink(missing_ok=True)
            for target,backup in reversed(committed):
                if backup.is_file(): shutil.copy2(backup,target)
                else: target.unlink(missing_ok=True)
            raise
    return updated,reference,moving

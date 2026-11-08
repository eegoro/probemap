"""Read-only reconstruction from native probe poses, never tracing annotations."""
from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import numpy as np

from .insertion_model import clip_segment_to_slab
from .landmarks import transform_points
from .neuropixels_model import probe_coordinates, recording_sites
from .surface_entries import probe_surface_entries
from .track_model import plane_uv, region_name


def fitted_probes(session):
    return [item for item in session.get('insertions',[]) if (item.get('probe') or {}).get('pose')]


def shanks(item,matrix):
    settings=item['probe']; length=settings['geometry']['shank_length_um']
    if settings['geometry'].get('body_shape')=='none': return np.empty((0,2,3))
    tips=probe_coordinates(settings,np.arange(settings['geometry']['shanks']),0.)
    bases=probe_coordinates(settings,np.arange(settings['geometry']['shanks']),length)
    return np.stack((transform_points(tips,matrix),transform_points(bases,matrix)),axis=1)


def recording_bands(item,recording,matrix):
    """Continuous occupied row ranges, preserving gaps and native offsets."""
    settings=item['probe']; g=settings['geometry']
    active=[c for c in recording.get('channels',[]) if c.get('recorded',True)]
    result=[]
    for shank in range(g['shanks']):
        rows=sorted({c['row'] for c in active if c['shank']==shank})
        groups=np.split(rows,np.flatnonzero(np.diff(rows)>1)+1) if rows else []
        for group in groups:
            heights=g['first_site_center_from_tip_um']+np.array([group[0]-.5,group[-1]+.5])*g['site_pitch_um']-recording.get('depth_offset_um',0.)
            heights=np.clip(heights,0,g['shank_length_um'])
            if heights[1]<=heights[0]: continue
            points=transform_points(probe_coordinates(settings,shank,heights),matrix)
            result.append(dict(shank=shank,points=points,heights_um=heights.tolist()))
    return result


def slice_reconstruction(items,matrix,geometry,thickness_um,*,recordings=True):
    """Finite geometry clipped to the actual physical slab, not projected points."""
    lines=[]; sites=[]; bands=[]
    for item in items:
        if not item.get('probe_visible',True): continue
        for index,segment in enumerate(shanks(item,matrix)):
            clipped=clip_segment_to_slab(*segment,geometry,thickness_um)
            if clipped is not None: lines.append(dict(points=clipped,color=item['color'],name=f"{item['name']}{index}"))
        if not recordings: continue
        for rec in item.get('recordings',[]):
            if not rec.get('visible',True) or rec.get('opacity',1)<=0: continue
            color=rec.get('color') or item['color']; opacity=rec.get('opacity',1)
            channels,_,native=recording_sites(copy.deepcopy(item),rec)
            mapped=transform_points(native,matrix)
            _,distance=plane_uv(geometry,mapped)
            for channel,point,offset in zip(channels,mapped,distance):
                if abs(offset)<=thickness_um/2:
                    settings=item['probe']; g=settings['geometry']
                    shaft_scale=float(np.linalg.norm(np.asarray(matrix)[:3,:3]@settings['pose']['down_source_zyx']))
                    sites.append(dict(point=point,color=color,opacity=opacity,name=f"{rec['name']} · {channel['channel']}",
                        pitch_um=g['site_pitch_um']*shaft_scale,size_um=g['site_size_um']*shaft_scale))
            for band in recording_bands(item,rec,matrix):
                clipped=clip_segment_to_slab(*band['points'],geometry,thickness_um)
                if clipped is not None: bands.append(dict(points=clipped,color=color,opacity=opacity,name=rec['name'],
                    width_um=item['probe']['geometry']['shank_width_um']*float(np.linalg.norm(np.asarray(matrix)[:3,:3]@item['probe']['pose']['lateral_source_zyx']))))
    return dict(lines=lines,sites=sites,bands=bands)


def orthogonal_geometry(bounds,axis,depth,max_dimension=512):
    lower,upper=np.asarray(bounds,float); axes=((2,1),(2,0),(1,0))
    u_axis,v_axis=axes[axis]; u=np.eye(3)[u_axis]; v=np.eye(3)[v_axis]; normal=np.eye(3)[axis]
    step=float(max(upper[u_axis]-lower[u_axis],upper[v_axis]-lower[v_axis])/(max_dimension-1))
    step=max(step,1e-6)
    width,height=np.ceil((upper-lower)[[u_axis,v_axis]]/step).astype(int)+1
    origin=lower.copy(); origin[axis]=depth
    forward=np.eye(4); forward[:3,:3]=np.column_stack((normal,v,u))*step; forward[:3,3]=origin
    center=origin+u*(width-1)*step/2+v*(height-1)*step/2
    return dict(shape_hw=[int(height),int(width)],pixel_size_um=step,u_zyx=u.tolist(),v_zyx=v.tolist(),
        normal_zyx=normal.tolist(),origin_um_zyx=origin.tolist(),center_um_zyx=center.tolist(),
        plane_voxel_drc_to_atlas_zyx_um=forward.tolist(),atlas_zyx_um_to_plane_voxel_drc=np.linalg.inv(forward).tolist())


def _metric_plane(center,u,v,normal,width,height,step):
    shape=np.maximum(2,np.ceil(np.array([height,width])/step).astype(int)+1)
    origin=center-u*(shape[1]-1)*step/2-v*(shape[0]-1)*step/2
    forward=np.eye(4); forward[:3,:3]=np.column_stack((normal,v,u))*step; forward[:3,3]=origin
    return dict(shape_hw=shape.tolist(),pixel_size_um=float(step),u_zyx=u.tolist(),v_zyx=v.tolist(),
        normal_zyx=normal.tolist(),origin_um_zyx=origin.tolist(),center_um_zyx=center.tolist(),
        plane_voxel_drc_to_atlas_zyx_um=forward.tolist(),atlas_zyx_um_to_plane_voxel_drc=np.linalg.inv(forward).tolist())


def probe_comparison_geometries(item,rigid,total,source_corners,atlas_corners,offset_um=0.,max_dimension=512):
    """Corresponding probe planes with a shared physical pixel size and field.

    Before uses rigid placement; Atlas/After use the full registration. Each
    grid is orthonormal in its display space, so scale/shear remain visible.
    The plane offset is in native scan micrometres, never atlas voxels.
    """
    settings=item['probe']; pose=settings['pose']
    u=np.asarray(pose['lateral_source_zyx'],float); u=u/np.linalg.norm(u)
    v=np.asarray(pose['down_source_zyx'],float); v=v-u*np.dot(v,u); v=v/np.linalg.norm(v)
    normal=np.asarray(pose['normal_source_zyx'],float); normal=normal/np.linalg.norm(normal)
    anchor=probe_coordinates(settings,(settings['geometry']['shanks']-1)/2,0.)
    centroid=np.vstack(source_corners).mean(0)
    center=anchor+u*np.dot(centroid-anchor,u)+v*np.dot(centroid-anchor,v)+normal*offset_um
    native_tips=probe_coordinates(settings,np.arange(settings['geometry']['shanks']),0.)
    frames=[]; extents=[]; normal_scales=[]
    for matrix,with_atlas in ((rigid,False),(total,True)):
        matrix=np.asarray(matrix,float); linear=matrix[:3,:3]
        c=transform_points([center],matrix)[0]
        n=np.linalg.solve(linear.T,normal); normal_scales.append(1./np.linalg.norm(n)); n/=np.linalg.norm(n)
        a=linear@u; a/=np.linalg.norm(a)
        b=linear@v; b-=a*np.dot(a,b); b/=np.linalg.norm(b)
        boxes=[transform_points(corners,matrix) for corners in source_corners]
        if with_atlas: boxes.append(np.asarray(atlas_corners,float))
        intersections=[]
        for corners in boxes:
            distance=(corners-c)@n
            intersections.extend(corners[np.abs(distance)<1e-7])
            for i in range(8):
                for bit in (1,2,4):
                    j=i^bit
                    if j>i and distance[i]*distance[j]<0:
                        t=distance[i]/(distance[i]-distance[j])
                        intersections.append(corners[i]+t*(corners[j]-corners[i]))
        points=np.asarray(intersections) if intersections else np.vstack(boxes)
        points=np.vstack((points,transform_points(native_tips,matrix)))
        uv=(points-c)@np.column_stack((a,b))
        extents.append(np.maximum(2*np.max(np.abs(uv),axis=0),1)*1.08)
        frames.append((c,a,b,n))
    width,height=np.max(extents,axis=0)
    step=float(max(width,height)/(max_dimension-1))
    before,after=[_metric_plane(*frame,width,height,step) for frame in frames]
    for geometry,scale in zip((before,after),normal_scales): geometry['source_normal_scale']=float(scale)
    return dict(atlas=copy.deepcopy(after),before=before,after=after)


def saved_probe_plane_geometries(item,rigid,total,source_corners):
    """The saved Probes plane and field, reused without fitting or annotation input."""
    from .neuropixels_model import probe_plane
    from .track_model import plane_geometry,plane_covering_volumes,plane_corners,transform_plane,transform_plane_geometry
    if item.get('source_plane') is not None:
        plane=copy.deepcopy(item['source_plane'])
    elif item.get('plane') is not None:
        plane=transform_plane(item['plane'],np.linalg.inv(total))
    else:
        plane=probe_plane(copy.deepcopy(item),np.eye(4))
    native=plane_geometry(plane_covering_volumes(plane,[source_corners]))
    before=transform_plane_geometry(native,rigid)
    # Rectify the registered plane to physical, orthogonal display pixels;
    # transform every probe contour/contact with the same full registration.
    linear=np.asarray(total)[:3,:3]
    u=linear@native['u_zyx']; u/=np.linalg.norm(u)
    normal=np.linalg.solve(linear.T,native['normal_zyx']); normal/=np.linalg.norm(normal)
    v=linear@native['v_zyx']; v-=u*np.dot(u,v); v/=np.linalg.norm(v)
    center=transform_points([native['center_um_zyx']],total)[0]
    corners=transform_points(plane_corners(native),total)
    extent=2*np.max(np.abs((corners-center)@np.column_stack((u,v))),axis=0)
    step=float(max(extent)/(max(native['shape_hw'])-1))
    after=_metric_plane(center,u,v,normal,*extent,step)
    return dict(native=native,before=before,after=after,atlas=copy.deepcopy(after))


def label_slice_image(gray,ids,regions,styles,mode='overlay',opacity=.4,names=False):
    """Display-only label colors and names; visibility never changes analysis."""
    gray=np.asarray(gray); rgb=gray.ndim==3
    result=gray.copy() if rgb else gray if mode=='atlas' else np.repeat(gray[...,None],3,axis=2)
    if mode=='atlas' or ids is None: return result,[]
    if mode=='labels': result=np.zeros(gray.shape if rgb else (*gray.shape,3),float)
    alpha=1. if mode=='labels' else float(opacity)
    text=[]
    for region in regions:
        style=styles.get(str(region['id']),{})
        if not style.get('visible',True): continue
        mask=ids==region['id']
        if not mask.any(): continue
        color=style.get('color_rgb',region['color_rgb'])
        result[mask]=result[mask]*(1-alpha)+np.asarray(color)/255*alpha
        if names:
            rows,cols=np.nonzero(mask)
            index=np.argmin((rows-rows.mean())**2+(cols-cols.mean())**2)
            text.append((style.get('name',region['name']),[int(cols[index]),int(rows[index])],color))
    return result,text


def reconstruction_report(session,matrix,surface=None,labels=None):
    """Atlas entry location; native probe arc length and native physical dimensions.

    Depth is unavailable when the tip is outside the atlas envelope or the
    finite shank has no entry. Atlas image bounds are never a brain boundary.
    Region lengths use <=25 um native samples, labels use their own grid.
    """
    probes=[]
    for original in fitted_probes(session):
        item=copy.deepcopy(original); settings=item['probe']; g=settings['geometry']
        length=g['shank_length_um']
        entries={e['shank']:e for e in probe_surface_entries(item,matrix,surface)} if surface else {}
        atlas_segments=shanks(item,matrix)
        direction=atlas_segments[0,1]-atlas_segments[0,0]
        tilt=float(np.degrees(np.arccos(np.clip(abs(direction[1])/np.linalg.norm(direction),0,1))))
        shank_rows=[]; channel_rows=[]
        for index,(tip,base) in enumerate(atlas_segments):
            entry=entries.get(index); depth=None; entry_source=None
            if entry is not None:
                entry_source=transform_points([entry['atlas_zyx_um']],np.linalg.inv(matrix))[0]
                if surface.contains(tip):
                    native_tip=probe_coordinates(settings,index,0.)
                    depth=float(np.linalg.norm(entry_source-native_tip))
            tissue_length=length if depth is None else min(length,depth)
            n=max(2,int(np.ceil(tissue_length/25))+1); heights=np.linspace(0,tissue_length,n)
            points=transform_points(probe_coordinates(settings,index,heights),matrix)
            regions=[]
            if labels is not None:
                # Midpoints yield lengths, without double-counting interval ends.
                ids=labels.ids_at((points[:-1]+points[1:])/2)
                for identifier in dict.fromkeys(int(i) for i in ids):
                    if identifier<=0: continue
                    regions.append(dict(id=identifier,name=region_name(identifier,labels.regions,session.get('regions',{})),
                        length_source_um=float(np.count_nonzero(ids==identifier)*tissue_length/(n-1))))
            shank_rows.append(dict(shank=index,depth_source_um=depth,length_source_um=length,
                width_source_um=g['shank_width_um'],thickness_source_um=g['shank_thickness_um'],
                entry_atlas_zyx_um=entry['atlas_zyx_um'] if entry else None,
                entry_source_zyx_um=None if entry_source is None else entry_source.tolist(),regions=regions))
        for rec in item.get('recordings',[]):
            channels,_,sources=recording_sites(item,rec)
            atlas=transform_points(sources,matrix)
            identifiers=labels.ids_at(atlas) if labels is not None else [None]*len(channels)
            for channel,source,point,identifier in zip(channels,sources,atlas,identifiers):
                shank=channel['shank']; depth=shank_rows[shank]['depth_source_um']
                height=float(np.dot(np.asarray(settings['pose']['tip0_source_zyx_um'])+
                    shank*g['shank_pitch_um']*np.asarray(settings['pose']['lateral_source_zyx'])-source,
                    settings['pose']['down_source_zyx']))
                contact_depth=None if depth is None else depth-height
                above=contact_depth is not None and contact_depth<0
                channel_rows.append(dict(recording=rec['name'],recording_id=rec['id'],channel=channel['channel'],shank=shank,
                    depth_source_um=contact_depth,
                    source_zyx_um=source.tolist(),atlas_zyx_um=point.tolist(),atlas_region_id=None if identifier is None else int(identifier),region_id=None if identifier is None or above else int(identifier),
                    region='Above atlas surface' if above else 'Unavailable' if identifier is None else region_name(int(identifier),labels.regions,session.get('regions',{}))))
        depths=[s['depth_source_um'] for s in shank_rows if s['depth_source_um'] is not None]
        probes.append(dict(id=item['id'],name=item['name'],color=item['color'],model=g['model'],
            tilt_atlas_y_deg=tilt,
            mean_depth_source_um=float(np.mean(depths)) if depths else None,valid_depth_shanks=len(depths),
            shank_pitch_source_um=g['shank_pitch_um'],shanks=shank_rows,channels=channel_rows))
    return dict(distance_space='source_scan_um',boundary='estimated_atlas_envelope',
        region_sampling_source_um=25,contact_depth_positive='inside_atlas_surface',
        surface_spacing_atlas_um_zyx=None if surface is None else list(surface.spacing_um_zyx),labels_available=labels is not None,probes=probes)


def project_reconstruction_report(project,surface=None,labels=None):
    from .insertion_model import insertion_session
    session=insertion_session(project)
    report=reconstruction_report(session,project.total,surface,labels)
    report.update(session_id=session['id'],registration_current=project.fit_is_current,
        source_to_atlas_zyx_um=project.total.tolist(),
        sources={name:dict(path=str(spec.path),shape_zyx=list(spec.shape_zyx),spacing_um_zyx=list(spec.spacing_um_zyx))
                 for name,spec in (('atlas',project.reference),('brain',project.moving),('electrodes',project.track)) if spec is not None})
    return report


def write_reconstruction_report(report,folder):
    folder=Path(folder); folder.mkdir(parents=True,exist_ok=True)
    (folder/'reconstruction.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n', encoding='utf-8')
    with (folder/'reconstruction_shanks.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=['probe','shank','depth_source_um','length_source_um','width_source_um','thickness_source_um','tilt_atlas_y_deg','regions'])
        writer.writeheader()
        for probe in report['probes']:
            for row in probe['shanks']:
                writer.writerow(dict(probe=probe['name'],tilt_atlas_y_deg=probe['tilt_atlas_y_deg'],**{k:row[k] for k in writer.fieldnames if k not in ('probe','regions','tilt_atlas_y_deg')},
                    regions='; '.join(r['name'] for r in row['regions'])))
    with (folder/'reconstruction_channels.csv').open('w',newline='',encoding='utf-8') as stream:
        fields=['probe','recording','channel','shank','depth_source_um','region_id','region','source_z_um','source_y_um','source_x_um','atlas_z_um','atlas_y_um','atlas_x_um']
        writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader()
        for probe in report['probes']:
            for row in probe['channels']:
                values={k:row[k] for k in fields if k in row}; values['probe']=probe['name']
                for space in ('source','atlas'):
                    values.update({f'{space}_{axis}_um':value for axis,value in zip('zyx',row[f'{space}_zyx_um'])})
                writer.writerow(values)

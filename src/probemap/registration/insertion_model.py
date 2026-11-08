"""Insertion points, cached physical plane fits and directed 4 mm tracks.

Annotations use atlas TIFF-local Z/Y/X micrometres for display; distances and
directed lengths use the inverse complete registration in source scan units.
"""
from __future__ import annotations

import colorsys
import copy
import csv
import hashlib
import json
import re
import warnings
from pathlib import Path
from uuid import uuid4

import numpy as np
import tifffile
from scipy import ndimage
from scipy.spatial.transform import Rotation

from .landmarks import transform_points
from .track_model import (active_session, plane_geometry, plane_world, raw_volume,
                          sample_world, track_report)

TRACK_LENGTH_UM = 4000.


def alphabetic_insertion_name(number):
    result=''
    while True:
        number,remainder=divmod(number,26)
        result=chr(65+remainder)+result
        if not number: return result
        number-=1


def normalize_insertion_names(session):
    """Upgrade old generated names, retaining descriptive names and unique letters."""
    items=session.get('insertions',[])
    legacy=[item for item in items if re.fullmatch(r'Insertion\s+\d+',item['name'])]
    reserved={item['name'].casefold() for item in items if item not in legacy}
    for item in legacy:
        index=max(0,int(item['name'].split()[-1])-1)
        while alphabetic_insertion_name(index).casefold() in reserved: index+=1
        item['name']=alphabetic_insertion_name(index)
        reserved.add(item['name'].casefold())


def new_insertion(session, name=None):
    items=session.setdefault("insertions",[])
    index=len(items)
    # Use the largest gap between existing hues, including after deletion.
    hues=sorted(colorsys.rgb_to_hsv(*(int(item["color"][i:i+2],16)/255 for i in (1,3,5)))[0]
                for item in items)
    hue=.12
    if hues:
        gaps=[((hues[(i+1)%len(hues)]-h)%1 if len(hues)>1 else 1,h) for i,h in enumerate(hues)]
        gap,start=max(gaps)
        hue=(start+gap/2)%1
    rgb=colorsys.hsv_to_rgb(hue,.72,1.)
    existing={item['name'].casefold() for item in items}
    while alphabetic_insertion_name(index).casefold() in existing: index+=1
    insertion={"id":uuid4().hex[:12],"name":name or alphabetic_insertion_name(index),
               "color":"#"+"".join(f"{round(c*255):02x}" for c in rgb),"points":[],
               "plane":None,"plane_fit":None,"tracks":[]}
    items.append(insertion)
    return insertion


def insertion_session(project):
    session=active_session(project)
    if session.get("insertion_schema")==1:
        normalize_insertion_names(session)
        for snapshot in session.get('edit_history',[]):
            normalize_insertion_names({'insertions':snapshot} if isinstance(snapshot,list) else snapshot)
        return session
    session["insertion_schema"]=1
    session["insertions"]=[]
    session["insertion_view"]={}
    # Keep the unmodified annotation data for audit/recovery during migration.
    legacy=copy.deepcopy(session.get("tracks",[]))
    if legacy:
        session["legacy_tracks_before_insertions"]=legacy
        for old in legacy:
            item=new_insertion(session,old.get("name","Imported insertion"))
            item["color"]=old.get("color",item["color"])
            item["plane"]=copy.deepcopy(old.get("plane",session["plane"]))
            for position in old.get("points_atlas_zyx_um",[]):
                add_insertion_point(project,item,position)
            points=old.get("points_atlas_zyx_um",[])
            if len(points)>=2 and np.linalg.norm(np.subtract(points[-1],points[0]))>1e-6:
                item["tracks"].append({"id":old["id"],"name":old["name"],
                    "control_points_atlas_zyx_um":[points[0],points[-1]],
                    "start_validation":{"inside":False,"method":"Imported; validate start on step 6"}})
            item["plane_fit"]={"method":"imported","point_signature":points_signature(item),"manual":True}
    sync_tracks(session,project)
    return session


def points_signature(insertion):
    payload=sorted((p["id"],p["atlas_zyx_um"]) for p in insertion["points"])
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()


def plane_is_stale(insertion):
    fit=insertion.get("plane_fit") or {}
    return insertion.get("plane") is None or fit.get("point_signature")!=points_signature(insertion)


def add_insertion_point(project,insertion,world,*,anchor='track'):
    world=np.asarray(world,dtype=float)
    if world.shape!=(3,) or not np.isfinite(world).all():
        raise ValueError("Point coordinates must be finite 3D positions.")
    if anchor not in ('track','moving','reference'):
        raise ValueError('Unknown point image.')
    source=transform_points(world[None],np.linalg.inv(project.total))[0]/project.track.spacing_um_zyx
    point={"id":uuid4().hex[:12],"atlas_zyx_um":world.tolist(),"source_track_zyx_voxel":source.tolist(),
           "anchor_image":anchor}
    insertion["points"].append(point)
    return point


def move_points(session,point_ids,target_id):
    target=next(item for item in session["insertions"] if item["id"]==target_id)
    moved=[]
    for item in session["insertions"]:
        if item is target: continue
        moved.extend(p for p in item["points"] if p["id"] in point_ids)
        item["points"]=[p for p in item["points"] if p["id"] not in point_ids]
    target["points"].extend(moved)
    return len(moved)


def move_tracks(session,track_ids,target_id):
    """Transfer ownership without changing any physical geometry or plane."""
    target=next(item for item in session['insertions'] if item['id']==target_id)
    moved=[]
    for item in session['insertions']:
        if item is target: continue
        moved.extend(track for track in item['tracks'] if track['id'] in track_ids)
        item['tracks']=[track for track in item['tracks'] if track['id'] not in track_ids]
    target['tracks'].extend(moved)
    return len(moved)


def sync_measurements(session):
    """Keep atlas display endpoints; measure in the calibrated acquisition frame."""
    matrix=np.asarray(session['sources']['matrix'],float)
    for record in session.get('measurements',[]):
        points=np.asarray(record['points_atlas_zyx_um'],float)
        source=transform_points(points,np.linalg.inv(matrix))
        record.setdefault('legacy_atlas_length_um',record.get('length_um'))
        record.update(points_source_scan_zyx_um=source.tolist(),
                      atlas_length_um=float(np.linalg.norm(points[1]-points[0])),
                      length_um=float(np.linalg.norm(source[1]-source[0])),
                      measurement_frame='source scan physical micrometres')


def add_measurement(session,start,end,*,anchor='track'):
    """Specimen distance after inverting both placement and affine."""
    points=np.asarray([start,end],float)
    if points.shape!=(2,3) or not np.isfinite(points).all():
        raise ValueError('Measurement endpoints must be finite 3D positions.')
    length=float(np.linalg.norm(points[1]-points[0]))
    if length<1e-6: raise ValueError('Choose two different points.')
    if anchor not in ('track','moving','reference'):
        raise ValueError('Unknown measurement image.')
    result={'id':uuid4().hex[:12],'points_atlas_zyx_um':points.tolist(),'length_um':length,'anchor_image':anchor}
    session.setdefault('measurements',[]).append(result)
    sync_measurements(session)
    return result


def fit_insertion_plane(insertion,template,matrix=None):
    """Least orthogonal squared-distance plane, in physical units (SVD).

    At least 3 noncollinear points are needed. Two points or a single straight
    electrode do not determine a plane; never invent a normal in that case.
    """
    points=np.asarray([p["atlas_zyx_um"] for p in insertion["points"]],dtype=float).reshape(-1,3)
    if matrix is not None:
        points=transform_points(points,np.linalg.inv(matrix))
    if len(points)<3:
        raise ValueError("Mark at least 3 noncollinear points, or set and save the plane manually.")
    center=points.mean(0)
    _,singular,basis=np.linalg.svd(points-center,full_matrices=False)
    if singular[0]<1e-6 or singular[1]<max(1e-6,singular[0]*1e-4):
        raise ValueError("These points lie on one line. Add points on another track, or save a manual plane.")
    normal=basis[-1]
    if normal[np.argmax(np.abs(normal))]<0: normal=-normal
    # Stable in-plane axes follow the closest atlas axis, not unstable PCA
    # eigenvectors when two planar singular values are almost equal.
    candidates=np.eye(3)
    seed=candidates[np.argmin(np.abs(candidates@normal))]
    u=seed-normal*np.dot(seed,normal)
    u/=np.linalg.norm(u)
    distances=(points-center)@normal
    if matrix is not None:
        native_v=np.cross(normal[::-1],u[::-1])[::-1]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',UserWarning)
            native_angles=Rotation.from_matrix(np.column_stack((u[::-1],native_v[::-1],normal[::-1]))).as_euler('xyz',degrees=True)
        insertion['source_plane']=dict(copy.deepcopy(template),center_um_zyx=center.tolist(),
            rotation_xyz_deg=native_angles.tolist(),coordinate_frame='source scan physical micrometres')
        center=transform_points(center[None],matrix)[0]
        normal=np.linalg.solve(np.asarray(matrix)[:3,:3].T,normal)
        normal/=np.linalg.norm(normal)
        u=np.asarray(matrix)[:3,:3]@u
        u-=normal*np.dot(u,normal)
        u/=np.linalg.norm(u)
    v=np.cross(normal[::-1],u[::-1])[::-1]
    matrix=np.column_stack((u[::-1],v[::-1],normal[::-1]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",UserWarning)  # Euler gimbal lock retains the same physical plane.
        angles=Rotation.from_matrix(matrix).as_euler("xyz",degrees=True)
    plane=copy.deepcopy(insertion.get("plane") or template)
    plane.update(center_um_zyx=center.tolist(),rotation_xyz_deg=angles.tolist(),
                 coordinate_frame="atlas physical micrometres")
    insertion["plane"]=plane
    insertion.pop("plane_draft",None)
    insertion["plane_fit"]={"method":"SVD orthogonal least squares","point_signature":points_signature(insertion),
        "n_points":len(points),"rms_um":float(np.sqrt(np.mean(distances**2))),
        "max_error_um":float(np.max(np.abs(distances))),"singular_values_um":singular.tolist(),"manual":False,
        "measurement_frame":"source scan physical micrometres" if matrix is not None else "atlas physical micrometres"}
    return plane


def save_manual_plane(insertion,plane):
    plane_geometry(plane)
    insertion["plane"]=copy.deepcopy(plane)
    insertion.pop("plane_draft",None)
    previous=copy.deepcopy(insertion.get("plane_fit") or {})
    previous.update(method="Manual plane",manual=True,point_signature=points_signature(insertion))
    points=np.asarray([p["atlas_zyx_um"] for p in insertion["points"]],float).reshape(-1,3)
    if len(points):
        geometry=plane_geometry(plane)
        distances=(points-np.array(geometry["center_um_zyx"]))@geometry["normal_zyx"]
        previous.update(n_points=len(points),rms_um=float(np.sqrt(np.mean(distances**2))),
                        max_error_um=float(np.max(np.abs(distances))))
    insertion["plane_fit"]=previous


def directed_track(control_points,length_um=TRACK_LENGTH_UM,matrix=None):
    points=np.asarray(control_points,dtype=float)
    if points.shape!=(2,3) or not np.isfinite(points).all():
        raise ValueError("A straight track needs a start and a direction point.")
    if matrix is not None:
        points=transform_points(points,np.linalg.inv(matrix))
    vector=points[1]-points[0]
    norm=float(np.linalg.norm(vector))
    if norm<1e-6:
        raise ValueError("The direction point must differ from the start.")
    result=np.vstack((points[0],points[0]+vector/norm*length_um))
    return transform_points(result,matrix) if matrix is not None else result


def sync_tracks(session,project):
    tracks=[]
    for insertion in session.get("insertions",[]):
        for track in insertion["tracks"]:
            points=directed_track(track["control_points_atlas_zyx_um"],matrix=project.total)
            source=transform_points(points,np.linalg.inv(project.total))
            track.update(insertion_id=insertion["id"],insertion_name=insertion["name"],color=insertion["color"],
                         length_um=TRACK_LENGTH_UM,points_atlas_zyx_um=points.tolist(),
                         atlas_length_um=float(np.linalg.norm(points[1]-points[0])),
                         measurement_frame='source scan physical micrometres',
                         points_source_scan_zyx_um=source.tolist(),
                         points_source_track_zyx_voxel=(source/project.track.spacing_um_zyx).tolist())
            tracks.append(track)
    session["tracks"]=tracks
    sync_measurements(session)
    return tracks


def brain_support(volume,threshold=None):
    """Editable foreground envelope on the normalized brain preview only."""
    from skimage.filters import threshold_otsu
    data=ndimage.gaussian_filter(np.asarray(volume.data,dtype=np.float32),.8)
    finite=data[np.isfinite(data)]
    automatic=float(threshold_otsu(finite)) if finite.size and finite.max()>finite.min() else .1
    value=automatic if threshold is None else float(threshold)
    mask=data>value
    mask=ndimage.binary_closing(mask,iterations=1)
    labels,count=ndimage.label(mask)
    if count:
        sizes=np.bincount(labels.ravel())
        sizes[0]=0
        mask=labels==np.argmax(sizes)
        mask=ndimage.binary_fill_holes(mask)
    return mask,value


def inside_brain(world,mask,volume,matrix):
    raw=transform_points(np.asarray(world,dtype=float).reshape(-1,3),np.linalg.inv(matrix))/volume.spacing_um_zyx
    coords=np.floor(raw+.5).astype(int)
    valid=np.all((coords>=0)&(coords<mask.shape),axis=1)
    result=np.zeros(len(raw),bool)
    result[valid]=mask[tuple(coords[valid].T)]
    return result


def clip_segment_to_slab(a,b,geometry,thickness):
    """Return the true segment within a finite slice slab (or None)."""
    a,b=np.asarray(a,float),np.asarray(b,float)
    normal=np.asarray(geometry["normal_zyx"])
    da=float((a-np.array(geometry["center_um_zyx"]))@normal)
    delta=float((b-a)@normal)
    half=max(float(thickness)/2,1e-6)
    if abs(delta)<1e-12:
        return np.vstack((a,b)) if abs(da)<=half else None
    lo,hi=sorted(((-half-da)/delta,(half-da)/delta))
    lo,hi=max(0,lo),min(1,hi)
    return np.vstack((a+(b-a)*lo,a+(b-a)*hi)) if hi>=lo else None


def track_slice_points(track,geometry):
    """Reconstruct finite-track hits in one slice; never project distant tracks.

    A crossing produces one exact intersection. A track lying in the slice is
    represented by sampled points along that in-slice portion, rather than a
    dashed projected line. Endpoints within the slice voxel are retained.
    """
    points=np.asarray(track["points_atlas_zyx_um"],float).reshape(-1,3)
    normal=np.asarray(geometry["normal_zyx"],float)
    center=np.asarray(geometry["center_um_zyx"],float)
    spacing=float(geometry["pixel_size_um"])
    result=[]
    distance=0.
    source_points=track.get('points_source_scan_zyx_um')
    for segment_index,(a,b) in enumerate(zip(points[:-1],points[1:])):
        vector=b-a
        length=float(np.linalg.norm(vector))
        if length<1e-9: continue
        native_length=(float(np.linalg.norm(np.subtract(source_points[segment_index+1],source_points[segment_index])))
                       if source_points is not None and len(source_points)==len(points) else length)
        da=float((a-center)@normal)
        delta=float(vector@normal)
        if abs(delta)<1e-8:
            if -.5<=da/spacing<.5:
                fractions=np.linspace(0,1,min(512,max(2,int(np.ceil(length/spacing))+1)))
            else: fractions=[]
        else:
            crossing=-da/delta
            if 0<=crossing<=1: fractions=[crossing]
            elif abs(da)<spacing/2: fractions=[0.]
            elif abs(da+delta)<spacing/2: fractions=[1.]
            else: fractions=[]
        for t in fractions:
            position=a+t*vector
            if result and np.linalg.norm(position-np.array(result[-1]["atlas_zyx_um"]))<1e-8: continue
            result.append({"atlas_zyx_um":position.tolist(),"distance_um":distance+t*native_length,
                           "track_id":track["id"],"name":track["name"],"color":track["color"]})
        distance+=native_length
    return result


def export_insertions(project,labels=None,progress=None,*,allow_invalid=False):
    session=insertion_session(project)
    sync_tracks(session,project)
    tracks=session["tracks"]
    if not tracks:
        raise ValueError("Create at least one 4 mm track before exporting.")
    invalid=[f"{t['insertion_name']} / {t['name']}" for t in tracks if not t.get("start_validation",{}).get("inside",False)]
    if invalid and not allow_invalid:
        raise ValueError("Validate track starts inside the brain on step 6: "+", ".join(invalid))
    folder=Path(project.output_dir).resolve()
    folder.mkdir(parents=True,exist_ok=True)
    project.save(folder/"project.json")
    from .neuropixels_model import write_reports
    write_reports(project,folder,labels)
    manifest={"status":"in_progress","coordinate_order":"ZYX","units":"um","track_length_um":TRACK_LENGTH_UM,
              "measurement_frame":"source scan physical micrometres",
              "session":session,"source_track_to_atlas_zyx_um":project.total.tolist()}
    try:
        with (folder/"insertion_points.csv").open("w",newline="",encoding="utf-8") as handle:
            writer=csv.writer(handle)
            writer.writerow(["insertion_id","insertion_name","color","point_id","atlas_z_um","atlas_y_um","atlas_x_um",
                             "source_track_z_voxel","source_track_y_voxel","source_track_x_voxel"])
            for item in session["insertions"]:
                for point in item["points"]:
                    writer.writerow([item["id"],item["name"],item["color"],point["id"],*point["atlas_zyx_um"],*point["source_track_zyx_voxel"]])
        with (folder/"tracks_4mm.csv").open("w",newline="",encoding="utf-8") as handle:
            writer=csv.writer(handle)
            writer.writerow(["insertion_id","insertion_name","track_id","track_name","color","length_um",
                             "source_start_z_um","source_start_y_um","source_start_x_um","source_end_z_um","source_end_y_um","source_end_x_um",
                             "atlas_start_z_um","atlas_start_y_um","atlas_start_x_um","atlas_end_z_um","atlas_end_y_um","atlas_end_x_um"])
            for t in tracks:
                writer.writerow([t["insertion_id"],t["insertion_name"],t["id"],t["name"],t["color"],TRACK_LENGTH_UM,
                                 *t["points_source_scan_zyx_um"][0],*t["points_source_scan_zyx_um"][1],
                                 *t["points_atlas_zyx_um"][0],*t["points_atlas_zyx_um"][1]])
        report=track_report(session,labels) if labels is not None else None
        if report:
            by_id={t["id"]:t for t in tracks}
            for filename,key in (("track_samples.csv","samples"),("track_region_segments.csv","segments")):
                rows=[]
                for row in report[key]:
                    t=by_id[row["track_id"]]
                    record=dict(row,insertion_id=t["insertion_id"],insertion_name=t["insertion_name"])
                    if "atlas_zyx_um" in record:
                        position=record.pop("atlas_zyx_um")
                        raw=transform_points(np.asarray(position)[None],np.linalg.inv(project.total))[0]/project.track.spacing_um_zyx
                        record.update(zip((f"atlas_{axis}_um" for axis in "zyx"),position))
                        record.update(zip((f"source_track_{axis}_voxel" for axis in "zyx"),raw.tolist()))
                    rows.append(record)
                with (folder/filename).open("w",newline="",encoding="utf-8") as handle:
                    if rows:
                        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
                        writer.writeheader()
                        writer.writerows(rows)
            manifest.update(report=report,region_ontology=labels.regions,atlas_labels_source=labels.source)
        sources=None
        for index,item in enumerate(session["insertions"]):
            if item.get("plane") is None: continue
            if progress: progress(int(10+85*index/max(1,len(session["insertions"]))),f"Saving {item['name']} matched planes")
            if sources is None:
                sources=[("brain",raw_volume(project.moving),project.total),
                         ("track",raw_volume(project.track),project.total),("atlas",raw_volume(project.reference),np.eye(4))]
            geometry=plane_geometry(item["plane"])
            world=plane_world(geometry)
            prefix=f"insertion_{item['id']}_"
            (folder/f"{prefix}plane_geometry.json").write_text(json.dumps(geometry,indent=2)+"\n", encoding='utf-8')
            for name,volume,matrix in sources:
                data=sample_world(volume,matrix,world).reshape(geometry["shape_hw"])
                tifffile.imwrite(folder/f"{prefix}{name}_plane.tif",data.astype(np.float32),photometric="minisblack",
                    resolution=(10000/geometry["pixel_size_um"],)*2,resolutionunit="CENTIMETER",
                    metadata={"axes":"YX","unit":"um","coordinate_frame":"oblique plane U/V"})
            if labels is not None:
                ids=np.maximum(0,labels.ids_at(world)).reshape(geometry["shape_hw"]).astype(np.uint16)
                tifffile.imwrite(folder/f"{prefix}regions_plane.tif",ids,photometric="minisblack",
                    resolution=(10000/geometry["pixel_size_um"],)*2,resolutionunit="CENTIMETER",metadata={"axes":"YX","unit":"um"})
        manifest["status"]="needs_review" if invalid else "complete"
        if invalid: manifest['invalid_track_starts']=invalid
    except BaseException as error:
        manifest.update(status="incomplete",error=str(error))
        raise
    finally:
        (folder/"insertions.json").write_text(json.dumps(manifest,indent=2)+"\n", encoding='utf-8')
    return folder/"insertions.json"

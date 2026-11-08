"""Physical oblique planes, native atlas labels and electrode track exports."""
from __future__ import annotations

import colorsys
import copy
import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import numpy as np
import tifffile
from scipy import ndimage
from scipy.spatial.transform import Rotation

from .desktop_model import AlignmentProject, inspect_tiff
from .landmarks import transform_points, validate_affine
from .volumes import Volume, spacing_zyx


def plane_geometry(plane):
    center=np.asarray(plane["center_um_zyx"],dtype=float)
    angles=np.asarray(plane["rotation_xyz_deg"],dtype=float)
    extent=np.asarray([plane["width_um"],plane["height_um"]],dtype=float)
    dimension=int(plane.get("max_dimension",512))
    if center.shape!=(3,) or angles.shape!=(3,) or not np.isfinite(np.r_[center,angles,extent]).all():
        raise ValueError("Plane position, angles and field of view must be finite.")
    if min(extent)<=0 or not 64<=dimension<=2048:
        raise ValueError("Choose a positive field of view and 64–2048 preview pixels.")
    rotation=Rotation.from_euler("xyz",angles,degrees=True).as_matrix()
    u,v,normal=rotation[::-1,0],rotation[::-1,1],rotation[::-1,2]
    spacing=float(max(extent)/(dimension-1))
    width,height=np.maximum(2,np.ceil(extent/spacing-1e-10).astype(int)+1)
    origin=center-u*(width-1)*spacing/2-v*(height-1)*spacing/2
    forward=np.eye(4)
    forward[:3,:3]=np.column_stack((normal,v,u))*spacing
    forward[:3,3]=origin
    return {"shape_hw":[int(height),int(width)],"pixel_size_um":spacing,
            "u_zyx":u.tolist(),"v_zyx":v.tolist(),"normal_zyx":normal.tolist(),
            "origin_um_zyx":origin.tolist(),"center_um_zyx":center.tolist(),
            "plane_voxel_drc_to_atlas_zyx_um":forward.tolist(),
            "atlas_zyx_um_to_plane_voxel_drc":np.linalg.inv(forward).tolist()}


def plane_world(geometry, uv=None):
    """Plane pixel U/V centres to physical atlas ZYX coordinates."""
    if uv is None:
        row,col=np.indices(geometry["shape_hw"],dtype=float)
        uv=np.column_stack((col.ravel(),row.ravel()))
    uv=np.asarray(uv,dtype=float).reshape(-1,2)
    matrix=np.asarray(geometry["plane_voxel_drc_to_atlas_zyx_um"])
    return uv[:,0,None]*matrix[:3,2]+uv[:,1,None]*matrix[:3,1]+matrix[:3,3]


def plane_uv(geometry, world):
    coords=transform_points(np.asarray(world).reshape(-1,3),geometry["atlas_zyx_um_to_plane_voxel_drc"])
    return coords[:,[2,1]],coords[:,0]*geometry["pixel_size_um"]


def plane_corners(geometry):
    h,w=geometry["shape_hw"]
    return plane_world(geometry,[[-.5,-.5],[w-.5,-.5],[w-.5,h-.5],[-.5,h-.5]])


def transform_plane(plane,matrix):
    """Map a physical plane, enclosing its transformed rectangular field."""
    import warnings
    geometry=plane_geometry(plane)
    linear=np.asarray(matrix)[:3,:3]
    u=linear@geometry['u_zyx']; u/=np.linalg.norm(u)
    normal=np.linalg.solve(linear.T,geometry['normal_zyx']); normal/=np.linalg.norm(normal)
    v=np.cross(normal[::-1],u[::-1])[::-1]
    center=transform_points([geometry['center_um_zyx']],matrix)[0]
    old_center=np.asarray(geometry['center_um_zyx'])
    old_u=np.asarray(geometry['u_zyx'])*plane['width_um']/2
    old_v=np.asarray(geometry['v_zyx'])*plane['height_um']/2
    corners=transform_points([old_center+i*old_u+j*old_v for i in (-1,1) for j in (-1,1)],matrix)-center
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',UserWarning)
        angles=Rotation.from_matrix(np.column_stack((u[::-1],v[::-1],normal[::-1]))).as_euler('xyz',degrees=True)
    return dict(copy.deepcopy(plane),center_um_zyx=center.tolist(),rotation_xyz_deg=angles.tolist(),
                width_um=float(np.ptp(corners@u)),height_um=float(np.ptp(corners@v)))


def transform_plane_geometry(geometry,matrix):
    """Map the exact sampling grid, including affine shear, for 3D display."""
    result=copy.deepcopy(geometry)
    forward=np.asarray(matrix)@np.asarray(geometry['plane_voxel_drc_to_atlas_zyx_um'])
    result['plane_voxel_drc_to_atlas_zyx_um']=forward.tolist()
    result['atlas_zyx_um_to_plane_voxel_drc']=np.linalg.inv(forward).tolist()
    result['center_um_zyx']=transform_points([geometry['center_um_zyx']],matrix)[0].tolist()
    return result


def plane_covering_volumes(plane,corner_sets,padding=.04):
    """Enclose complete plane/volume intersections without changing the plane.

    Each set contains the eight affine-transformed voxel-cell corners in the
    product order used by transformed_corners. Only the display field changes;
    the normal, normal offset and underlying image calibration stay intact.
    """
    geometry=plane_geometry(plane)
    center=np.asarray(geometry['center_um_zyx'])
    u,v,normal=(np.asarray(geometry[key]) for key in ('u_zyx','v_zyx','normal_zyx'))
    intersections=[]
    boxes=[np.asarray(corners,float).reshape(8,3) for corners in corner_sets]
    for corners in boxes:
        distance=(corners-center)@normal
        intersections.extend(corners[np.abs(distance)<1e-7])
        for i in range(8):
            for bit in (1,2,4):
                j=i^bit
                if j>i and distance[i]*distance[j]<0:
                    fraction=distance[i]/(distance[i]-distance[j])
                    intersections.append(corners[i]+fraction*(corners[j]-corners[i]))
    if not boxes: return dict(plane)
    # A plane wholly outside the images remains blank, with a useful field of
    # view; never clamp its normal position back into the specimen.
    points=np.asarray(intersections) if intersections else np.vstack(boxes)
    uv=(points-center)@np.column_stack((u,v))
    lower,upper=uv.min(0),uv.max(0)
    midpoint=(lower+upper)/2
    extent=np.maximum(upper-lower,1.)*(1+2*padding)
    return dict(plane,center_um_zyx=(center+midpoint[0]*u+midpoint[1]*v).tolist(),
                width_um=float(extent[0]),height_um=float(extent[1]))


def sample_world(volume, matrix, world, order=1):
    source=transform_points(world,np.linalg.inv(validate_affine(matrix)))/volume.spacing_um_zyx
    return ndimage.map_coordinates(volume.data,source.T,order=order,output=np.float32,
                                   mode="constant",cval=0,prefilter=False)


def raw_volume(spec):
    with tifffile.TiffFile(spec.path) as tif:
        if tuple(tif.series[0].shape)!=spec.shape_zyx:
            raise ValueError(f"Image shape changed: {spec.path}")
        data=tif.asarray(out="memmap",maxworkers=2)
    return Volume(data,spec.spacing_um_zyx,spec.path)


def load_track_data(spec,max_dimension=192):
    """Native data for precise reslicing; a peak-preserving GPU preview only."""
    native=raw_volume(spec)
    sample=np.asarray(native.data).ravel()[::max(1,native.data.size//500000)]
    sample=sample[np.isfinite(sample)]
    if not sample.size:
        raise ValueError("Track scan contains no finite intensities.")
    low,high=np.percentile(sample,(1,99.9))
    if high<=low:
        low,high=float(sample.min()),float(sample.max())
    if high<=low:
        low,high=0.,max(1.,float(high))
    factors=np.maximum(1,np.ceil(np.asarray(native.data.shape)/max_dimension).astype(int))
    while np.prod(np.ceil(np.asarray(native.data.shape)/factors))>32_000_000:
        axis=int(np.argmax(np.asarray(native.data.shape)/factors))
        factors[axis]+=1
    # Centred maxima preserve thin electrode signals in the 3D overview. The
    # displayed slice and all quantitative exports still sample native data.
    peak=ndimage.maximum_filter(native.data,size=tuple(factors),mode="nearest") if max(factors)>1 else native.data
    preview=np.array(peak[tuple(slice(None,None,int(f)) for f in factors)],dtype=np.float32,copy=True)
    np.nan_to_num(preview,copy=False,nan=float(low),posinf=float(high),neginf=float(low))
    preview-=float(low)
    preview/=float(high-low)
    np.clip(preview,0,1,out=preview)
    gpu=Volume(preview,tuple(np.asarray(spec.spacing_um_zyx)*factors),spec.path,(float(low),float(high)))
    return native,gpu,[float(low),float(high)]


@dataclass
class LabelAtlas:
    data: np.ndarray
    spacing_um_zyx: tuple
    regions: list
    source: str

    def ids_at(self, world):
        coords=np.floor(np.asarray(world).reshape(-1,3)/self.spacing_um_zyx+.5).astype(np.int64)
        valid=np.all((coords>=0)&(coords<np.asarray(self.data.shape)),axis=1)
        labels=np.full(len(coords),-1,dtype=np.int32)
        labels[valid]=self.data[tuple(coords[valid].T)]
        return labels


def load_label_atlas(directory, reference, cache_dir, progress=None, *, spacing_um_zyx=None):
    """Cache one native integer label lattice; never interpolate region IDs."""
    directory=Path(directory).resolve()
    files=sorted(p for p in directory.iterdir() if p.suffix.lower() in (".tif",".tiff"))
    if not files:
        raise ValueError("The labels folder must contain atlas-aligned binary TIFF masks.")
    first=inspect_tiff(files[0])
    independent=spacing_um_zyx is not None
    shape=tuple(first['shape_zyx']) if independent else reference.shape_zyx
    spacing=tuple(spacing_zyx(spacing_um_zyx)) if independent else reference.spacing_um_zyx
    regions=[]
    used=set()
    for index,path in enumerate(files):
        match=re.match(r"(\d+)[_-]",path.stem)
        identifier=int(match.group(1)) if match else index+1
        if identifier in used or not 1<=identifier<65535:
            raise ValueError("Mask filenames must define unique positive region IDs below 65535.")
        used.add(identifier)
        info=inspect_tiff(path)
        if tuple(info["shape_zyx"])!=shape:
            raise ValueError(f"Mask {path.name} does not match the labels grid shape.")
        # A project override changes physical calibration, never source tags.
        # Masks within the folder must still describe one consistent lattice.
        candidates=[first['spacing_um_zyx']] if independent else [reference.spacing_um_zyx]
        if not independent and reference.initial_spacing_um_zyx is not None:
            candidates.append(reference.initial_spacing_um_zyx)
        if not any(all(actual is None or expected is None or np.isclose(actual,expected,atol=1e-5,rtol=0)
                       for actual,expected in zip(info['spacing_um_zyx'],spacing)) for spacing in candidates):
            raise ValueError(f"Mask {path.name} voxel sizes do not match the labels grid.")
        if not np.allclose(np.asarray(info["origin_um_zyx"])-reference.origin_um_zyx,0,atol=1e-5):
            # Plain TIFF masks have no origin; they explicitly share the atlas
            # voxel grid. A nonzero calibrated origin must agree with it.
            if np.any(np.asarray(info["origin_um_zyx"])!=0):
                raise ValueError(f"Mask {path.name} has a different calibrated origin.")
        hue=(index*.61803398875)%1
        color=[round(c*255) for c in colorsys.hsv_to_rgb(hue,.65,.95)]
        name=re.sub(r"^\d+[_-]?|_mask$","",path.stem).replace("_"," ")
        regions.append({"id":identifier,"name":name,"color_rgb":color,"mask_path":str(path)})
    # ProbeMap atlas bundles document exactly which combined label file and
    # ontology generated their masks. Reuse that native lattice only while all
    # source masks are unchanged since that manifest, otherwise rebuild them.
    combined=None
    ontology=[]
    for summary_path in sorted(directory.parent.glob("*summary.json")):
        try:
            summary=json.loads(summary_path.read_text(encoding='utf-8'))
            bundle=summary.get("files",{})
            if (Path(bundle.get("region_masks_dir","/")).resolve()==directory
                    and set(summary.get("region_mask_files",[]))=={p.name for p in files}
                    and max(p.stat().st_mtime_ns for p in files)<=summary_path.stat().st_mtime_ns):
                candidate=Path(bundle["labels_tiff"])
                if candidate.is_file():
                    with tifffile.TiffFile(candidate) as tif:
                        if tuple(tif.series[0].shape)==shape:
                            combined=candidate
                    if bundle.get("ontology_json"):
                        ontology=json.loads(Path(bundle["ontology_json"]).read_text(encoding='utf-8'))
                    break
        except (OSError,ValueError,KeyError,TypeError):
            continue
    by_id={int(item["id"]):item for item in ontology}
    for region in regions:
        source=by_id.get(region["id"],{})
        region["name"]=source.get("display_name",source.get("name",region["name"]))
        region["color_rgb"]=source.get("color_rgb",region["color_rgb"])
    signature={"version":2,"shape":shape,
               "files":[[str(p),p.stat().st_size,p.stat().st_mtime_ns] for p in files]}
    if combined:
        signature["combined"]=[str(combined),combined.stat().st_size,combined.stat().st_mtime_ns]
    key=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()[:24]
    cache=Path(cache_dir)/f"labels_{key}.npy"
    cache.parent.mkdir(parents=True,exist_ok=True)
    if not cache.is_file():
        temporary=cache.with_name(cache.name+f".{uuid4().hex[:8]}.partial")
        data=np.lib.format.open_memmap(temporary,mode="w+",dtype=np.uint16,shape=shape)
        try:
            if combined:
                if progress: progress(40,"Preparing native atlas labels for fast plane slicing")
                with tifffile.TiffFile(combined) as tif:
                    tif.asarray(out=data,maxworkers=2)
                for start in range(0,shape[0],16):
                    if not set(np.unique(data[start:start+16])).issubset(used|{0}):
                        raise ValueError("Combined atlas IDs do not match the selected masks.")
            else:
                data[:]=0
                for number,(path,region) in enumerate(zip(files,regions)):
                    if progress: progress(int(10+80*number/len(files)),f"Loading region {number+1}/{len(files)}: {region['name']}")
                    with tifffile.TiffFile(path) as tif:
                        mapped=None if len(tif.pages)==shape[0] else tif.asarray(out="memmap",maxworkers=2)
                        for start in range(0,shape[0],8):
                            stop=min(start+8,shape[0])
                            block=(tif.asarray(key=range(start,stop),maxworkers=2) if mapped is None else mapped[start:stop])
                            mask=np.asarray(block).reshape((stop-start,*shape[1:]))!=0
                            target=data[start:stop]
                            if np.any(mask&(target!=0)):
                                raise ValueError("Region masks overlap. Supply a non-overlapping atlas label set; "
                                                 "overlapping voxels cannot receive one unambiguous region ID.")
                            target[mask]=region["id"]
            data.flush()
            from ..platform_io import close_memmap
            close_memmap(data)
            del data
            data=None
            temporary.replace(cache)
        except BaseException:
            if data is not None:
                from ..platform_io import close_memmap
                close_memmap(data); del data
            temporary.unlink(missing_ok=True)
            raise
    data=np.load(cache,mmap_mode="r")
    palette_path=cache.with_suffix('.palette-v1.json')
    if palette_path.is_file(): palette=json.loads(palette_path.read_text(encoding='utf-8'))
    else:
        from .region_colors import region_palette
        palette=region_palette(data,regions)
        palette_path.write_text(json.dumps(palette,sort_keys=True), encoding='utf-8')
    for region in regions: region['color_rgb']=palette[str(region['id'])]
    return LabelAtlas(data,spacing,regions,str(directory))


def active_session(project):
    """Keep older fits/sources as separate sessions instead of moving old lines silently."""
    project.validate_track()
    if project.track is None or not project.fit_is_current:
        raise ValueError("A matching track scan and a current fitted transform are required.")
    analysis=project.track_analysis
    sessions=analysis.setdefault("sessions",[])
    signature={"track":project.track.record(),"reference":project.reference.record(),"matrix":project.total.tolist()}
    key=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
    for session in sessions:
        if session.get("signature")==key:
            analysis["active_session"]=session["id"]
            return session
    from .coarse import transformed_corners
    corners=np.vstack((transformed_corners(project.reference.shape_zyx,project.reference.spacing_um_zyx,np.eye(4)),
                       transformed_corners(project.track.shape_zyx,project.track.spacing_um_zyx,project.total)))
    center=(corners.max(0)+corners.min(0))/2
    span=float(np.max(corners.max(0)-corners.min(0)))
    session={"id":uuid4().hex,"signature":key,"sources":dict(signature,moving=project.moving.record()),"tracks":[],"regions":{},
             "plane":{"center_um_zyx":center.tolist(),"rotation_xyz_deg":[0.,0.,0.],
                      "width_um":span,"height_um":span,"max_dimension":512},"view":{}}
    sessions.append(session)
    analysis["active_session"]=session["id"]
    return session


def region_name(identifier, regions, styles=None):
    if identifier<0: return "Outside atlas"
    if identifier==0: return "Unlabeled atlas"
    source=next((item for item in regions if item["id"]==identifier),{})
    return (styles or {}).get(str(identifier),{}).get("name",source.get("name",f"Region {identifier}"))


def track_report(session, atlas, step_um=None):
    """Native nearest-neighbour membership along uniformly sampled polylines."""
    step=float(step_um or min(atlas.spacing_um_zyx)/2)
    if not np.isfinite(step) or step<=0:
        raise ValueError("Track sampling step must be positive.")
    samples,segments=[],[]
    styles=session.get("regions",{})
    inverse=np.linalg.inv(np.asarray(session.get('sources',{}).get('matrix',np.eye(4)),float))
    for track in session["tracks"]:
        points=np.asarray(track["points_atlas_zyx_um"],dtype=float).reshape(-1,3)
        distance=0.
        runs=[]
        for a,b in zip(points[:-1],points[1:]):
            length=float(np.linalg.norm(inverse[:3,:3]@(b-a)))
            if length<1e-9: continue
            intervals=max(1,int(np.ceil(length/step-1e-10)))
            edges=np.linspace(0,length,intervals+1)
            mids=(edges[:-1]+edges[1:])/2
            ids=atlas.ids_at(a+(b-a)*mids[:,None]/length)
            for i,identifier in enumerate(ids):
                start,end=distance+edges[i],distance+edges[i+1]
                if runs and runs[-1][0]==int(identifier): runs[-1][2]=float(end)
                else: runs.append([int(identifier),float(start),float(end)])
            sample_edges=edges[1:] if distance>0 else edges
            world=a+(b-a)*sample_edges[:,None]/length
            for d,point,identifier in zip(distance+sample_edges,world,atlas.ids_at(world)):
                samples.append({"track_id":track["id"],"track_name":track["name"],"distance_um":float(d),
                                "atlas_zyx_um":point.tolist(),"region_id":int(identifier),
                                "region_name":region_name(int(identifier),atlas.regions,styles)})
            distance+=length
        for identifier,start,end in runs:
            segments.append({"track_id":track["id"],"track_name":track["name"],"region_id":identifier,
                             "region_name":region_name(identifier,atlas.regions,styles),
                             "entry_um":start,"exit_um":end,"length_um":end-start})
    return {"sampling_step_um":step,"samples":samples,"segments":segments,
            "measurement_frame":"source scan physical micrometres",
            "distance_convention":"Source-scan arc length from the first vertex; atlas labels assign regions only."}


def export_track_analysis(project, atlas, progress=None):
    session=active_session(project)
    report=track_report(session,atlas)
    folder=Path(project.output_dir).resolve()
    folder.mkdir(parents=True,exist_ok=True)
    project.save(folder/"project.json")
    geometry=plane_geometry(session["plane"])
    manifest={"status":"in_progress","units":"um","coordinate_order":"ZYX",
              "session":session,"plane_geometry":geometry,"report":report,
              "region_ontology":atlas.regions,
              "atlas_labels_source":atlas.source,"native_atlas_spacing_um_zyx":atlas.spacing_um_zyx,
              "source_track_to_atlas_zyx_um":project.total.tolist()}
    try:
        if progress: progress(10,"Saving electrode coordinates and region intervals")
        points_path=folder/"track_samples.csv"
        with points_path.open("w",newline="",encoding="utf-8") as handle:
            writer=csv.writer(handle)
            writer.writerow(["track_id","track_name","distance_um","atlas_z_um","atlas_y_um","atlas_x_um",
                             "source_track_z_voxel","source_track_y_voxel","source_track_x_voxel","region_id","region_name"])
            for item in report["samples"]:
                raw=transform_points(np.asarray(item["atlas_zyx_um"])[None],np.linalg.inv(project.total))[0]/project.track.spacing_um_zyx
                writer.writerow([item["track_id"],item["track_name"],item["distance_um"],*item["atlas_zyx_um"],*raw,
                                 item["region_id"],item["region_name"]])
        with (folder/"track_region_segments.csv").open("w",newline="",encoding="utf-8") as handle:
            columns=["track_id","track_name","region_id","region_name","entry_um","exit_um","length_um"]
            writer=csv.DictWriter(handle,fieldnames=columns)
            writer.writeheader()
            writer.writerows(report["segments"])
        world=plane_world(geometry)
        for index,(name,spec,matrix) in enumerate((("track_plane.tif",project.track,project.total),
                                                  ("atlas_plane.tif",project.reference,np.eye(4)))):
            if progress: progress(30+index*25,f"Saving {name} from native intensities")
            data=sample_world(raw_volume(spec),matrix,world).reshape(geometry["shape_hw"])
            tifffile.imwrite(folder/name,data.astype(np.float32),photometric="minisblack",
                             resolution=(10000/geometry["pixel_size_um"],)*2,resolutionunit="CENTIMETER",
                             metadata={"axes":"YX","unit":"um","coordinate_frame":"oblique plane U/V"})
        labels=atlas.ids_at(world).reshape(geometry["shape_hw"])
        tifffile.imwrite(folder/"atlas_regions_plane.tif",np.maximum(labels,0).astype(np.uint16),photometric="minisblack",
                         resolution=(10000/geometry["pixel_size_um"],)*2,resolutionunit="CENTIMETER",
                         metadata={"axes":"YX","unit":"um","coordinate_frame":"oblique plane U/V"})
        (folder/"plane_geometry.json").write_text(json.dumps(geometry,indent=2)+"\n", encoding='utf-8')
        palette=[]
        for region in atlas.regions:
            style=session.get("regions",{}).get(str(region["id"]),{})
            palette.append({"id":region["id"],"name":style.get("name",region["name"]),
                            "color_rgb":style.get("color_rgb",region["color_rgb"]),
                            "visible":style.get("visible",True),"label":style.get("label",False)})
        (folder/"region_palette.json").write_text(json.dumps(palette,indent=2)+"\n", encoding='utf-8')
        manifest["status"]="complete"
        if progress: progress(100,"Track analysis saved")
    except BaseException as error:
        manifest.update(status="incomplete",error=str(error))
        raise
    finally:
        (folder/"tracks.json").write_text(json.dumps(manifest,indent=2)+"\n", encoding='utf-8')
    return folder/"tracks.json"

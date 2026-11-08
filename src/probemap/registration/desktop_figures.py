"""Native-data figure rendering, independent of interactive preview resolution."""
import copy,json
from pathlib import Path
import numpy as np
from qtpy.QtCore import QPoint,QPointF,QRectF,Qt
from qtpy.QtGui import QImage,QPainter,QRegion
from qtpy.QtWidgets import QWidget


def slice_sampler(sources,matrices,settings,*,visible_pixels=None,normalizations=None,opacities=None,labels=None,label_matrix=None,label_mode='atlas',styles=None,label_opacity=.4):
    """Lazy native sampling with the current per-layer display parameters."""
    from .track_model import sample_world,raw_volume
    from .desktop_slice_display import slice_grayscale,slice_rgb,blend_rgb
    from .landmarks import transform_points
    settings=copy.deepcopy(settings); styles=copy.deepcopy(styles or {})
    if visible_pixels is not None:
        from .desktop_slice_display import frozen_slice_settings
        settings=[frozen_slice_settings(pixels,record) for pixels,record in zip(visible_pixels,settings)]
    for record in settings: record['auto_contrast']=False  # Freeze the visible full-slice limits across render tiles.
    matrices=[np.asarray(m).copy() for m in matrices]; native=[None]*len(sources)
    def sample(world,include_labels=True):
        result=np.zeros((len(world),3),np.float32)
        for i,(source,matrix,record) in enumerate(zip(sources,matrices,settings)):
            if native[i] is None: native[i]=source if hasattr(source,'data') else raw_volume(source)
            pixels=sample_world(native[i],matrix,world,order=0 if record.get('interpolation')=='nearest' else 1)
            if normalizations and normalizations[i] is not None:
                low,high=normalizations[i]; pixels=np.clip((pixels-low)/(high-low),0,1)
            gray=slice_grayscale(pixels,record); rgb=slice_rgb(gray,record.get('colormap','gray'),record.get('colormap_colors'))
            alpha=record.get('opacity',1)*(opacities[i] if opacities else 1)
            blend=record.get('blending','additive' if len(sources)>1 else 'translucent')
            result=blend_rgb(result,rgb,alpha,blend)
        if include_labels and labels is not None and label_mode!='atlas':
            ids=labels.ids_at(world if label_matrix is None else transform_points(world,label_matrix))
            if label_mode=='labels': result[:]=0
            alpha=1 if label_mode=='labels' else label_opacity
            for region in labels.regions:
                style=styles.get(str(region['id']),{})
                if not style.get('visible',True): continue
                mask=ids==region['id']; color=np.asarray(style.get('color_rgb',region['color_rgb']))/255
                result[mask]=(1-alpha)*result[mask]+alpha*color
        return np.clip(result,0,1)
    def vector_regions(world):
        if labels is None or label_mode=='atlas': return []
        ids=labels.ids_at(world if label_matrix is None else transform_points(world,label_matrix))
        return [(region['id'],ids==region['id'],styles.get(str(region['id']),{}).get('color_rgb',region['color_rgb']),
                 1 if label_mode=='labels' else label_opacity)
                for region in labels.regions if styles.get(str(region['id']),{}).get('visible',True)]
    sample.vector_regions=vector_regions
    sample.label_mode=label_mode
    sample.resolution_um=min(float(np.min(np.linalg.svd(matrix[:3,:3]@np.diag(source.spacing_um_zyx),compute_uv=False)))
        for source,matrix in zip(sources,matrices))
    return sample


def figure_button(canvas):
    from .desktop_widgets import view_button
    return view_button('save','Save figure to project figs (format in Settings)',lambda:canvas.window().save_figure(canvas))


def next_figure_path(folder,suffix='.png'):
    folder=Path(folder)/'figs'; folder.mkdir(parents=True,exist_ok=True)
    index=max((int(p.stem) for p in folder.iterdir() if p.stem.isdigit()),default=0)+1
    while True:
        path=folder/f'{index}{suffix}'
        try:
            with path.open('xb'): pass
            return path
        except FileExistsError: index+=1


def write_png(path,image,metadata):
    image.setDotsPerMeterX(11811); image.setDotsPerMeterY(11811)
    image.setText('Software','ProbeMap'); image.setText('Description',json.dumps(metadata,ensure_ascii=False))
    if not image.save(str(path),'PNG'): raise OSError('Could not save figure PNG.')


def array_image(data):
    data=np.ascontiguousarray(data,dtype=np.uint8)
    fmt=QImage.Format.Format_RGBA8888 if data.shape[-1]==4 else QImage.Format.Format_RGB888
    return QImage(data.data,data.shape[1],data.shape[0],data.strides[0],fmt).copy()


def export_slice(owner,canvas,path,long_edge=4096):
    if canvas.image is None or not hasattr(canvas,'figure_sampler'):
        path.unlink(missing_ok=True); owner.statusBar().showMessage('Wait for the slice to finish loading.',3500); return
    width,height=canvas.width(),canvas.height(); scale=max(2.,long_edge/max(width,height))
    size=(round(width*scale),round(height*scale)); sampler=canvas.figure_sampler
    vector=path.suffix.lower()=='.svg'
    origin=canvas._world(QPointF(0,0)); dx=canvas._world(QPointF(1,0))-origin; dy=canvas._world(QPointF(0,1))-origin
    rect=canvas.image_rect(); state=copy.deepcopy(canvas.view_record())
    interpolation=getattr(canvas,'display_interpolation','linear')
    filtered=interpolation not in ('nearest','linear')
    raster_scale=min(scale,max(32/width,32/height,2*max(np.linalg.norm(dx),np.linalg.norm(dy))/sampler.resolution_um)) if filtered else scale
    raster=(round(width*raster_scale),round(height*raster_scale))
    def prepare(progress):
        result=np.zeros((raster[1],raster[0],3),np.float32 if filtered else np.uint8)
        columns=(np.arange(raster[0])+.5)/raster_scale
        for start in range(0,raster[1],32):
            rows=(np.arange(start,min(start+32,raster[1]))+.5)/raster_scale
            world=origin+rows[:,None,None]*dy+columns[None,:,None]*dx
            rgb=sampler(world.reshape(-1,3),include_labels=not vector).reshape(len(rows),raster[0],3)
            if vector and getattr(sampler,'label_mode','atlas')=='labels': rgb[:]=0
            inside=(columns[None]>=rect.left())&(columns[None]<rect.right())&(rows[:,None]>=rect.top())&(rows[:,None]<rect.bottom())
            rgb[~inside]=0; result[start:start+len(rows)]=rgb if filtered else np.clip(rgb*255,0,255).astype(np.uint8)
            progress(round(100*(start+len(rows))/raster[1]),'Rendering native slice figure…')
        regions=[]
        if vector and hasattr(sampler,'vector_regions'):
            # Match the displayed slice lattice; retain a separate path per region.
            from .figure_regions import qt_region
            region_scale=min(scale,2.)
            rw,rh=round(width*region_scale),round(height*region_scale)
            yy,xx=np.mgrid[:rh,:rw]; points=origin+(yy.ravel()[:,None]+.5)/region_scale*dy+(xx.ravel()[:,None]+.5)/region_scale*dx
            for identifier,mask,color,alpha in sampler.vector_regions(points):
                regions.append((qt_region(mask.reshape(rh,rw) & ((xx+.5)/region_scale>=rect.left()) & ((xx+.5)/region_scale<rect.right()) & ((yy+.5)/region_scale>=rect.top()) & ((yy+.5)/region_scale<rect.bottom()),region_scale),color,alpha))
        return result,regions
    def ready(prepared):
        rgb,regions=prepared
        if filtered:
            from .desktop_image_renderer import render_image
            rgb=render_image(rgb,interpolation,size,QRectF(0,0,*size),0)
        image=QImage(*size,QImage.Format.Format_ARGB32); image.fill(Qt.GlobalColor.black)
        canvas._figure_background=array_image(rgb)
        canvas._figure_region_paths=regions
        canvas._vector_export=vector
        try:
            if vector:
                from qtpy.QtSvg import QSvgGenerator
                from qtpy.QtCore import QSize
                generator=QSvgGenerator(); generator.setFileName(str(path)); generator.setSize(QSize(*size))
                generator.setViewBox(QRectF(0,0,*size)); generator.setTitle('ProbeMap slice')
                painter=QPainter(generator)
            else: painter=QPainter(image)
            painter.scale(scale,scale)
            canvas.render(painter,QPoint(),QRegion(),QWidget.RenderFlag.DrawWindowBackground)
            painter.end()
            if not vector: write_png(path,image,dict(kind='slice',source='original TIFF samples',view=state,pixels=size,interpolation=interpolation))
            owner.statusBar().showMessage(f'Saved figs/{path.name}',5000)
        finally:
            del canvas._figure_background; del canvas._figure_region_paths; del canvas._vector_export; canvas.update()
    owner._start_job(prepare,lambda value:_finish(owner,path,lambda:ready(value)),'Saving native slice figure…',on_error=lambda error:path.unlink(missing_ok=True))


def _finish(owner,path,callback):
    try: callback()
    except Exception as error:
        path.unlink(missing_ok=True); owner._error(str(error))


def export_volume(owner,view,path,long_edge=4096):
    if view.canvas is None:
        path.unlink(missing_ok=True); return
    if path.suffix.lower()=='.svg':
        from .figure_svg3d import export_scene_svg
        export_scene_svg(view,path,owner)
        return
    from vispy.gloo import gl
    view.canvas.set_current()
    limit=int(gl.glGetParameter(0x8073))
    textures=[i for i,v in enumerate(view.visuals) if v.visible and v.opacity>0]
    specs=([owner.project.reference,owner.project.moving,owner.project.track] if view is owner.volume_view else
        [owner.project.reference,owner.project.track,owner.project.moving])
    previews=list(view.render_volumes); settings=copy.deepcopy(view.render_settings)
    camera=view.camera_record(); matrix=view.matrix.copy()
    width,height=view.canvas.size; scale=max(2.,long_edge/max(width,height)); size=(round(width*scale),round(height*scale))
    context=next((w for w in (owner.track_workspace,owner.reconstruction) if w.volume_view is view),None)
    entries=(getattr(context,'surface_visuals',None) or getattr(context,'_entry_visuals',None)) if context else None
    if entries is not None and (not entries.active or not entries.mesh.visible): entries=None
    visible_regions=[i for i,v in view.label_visuals.items() if v.visible]
    labels=getattr(context,'labels',None)
    def prepare(progress):
        import psutil
        from .track_model import raw_volume,load_label_atlas
        from .volumes import intensity_limits
        budget=max(128_000_000,psutil.virtual_memory().available*.4)
        strides={i:np.maximum(1,np.ceil(np.asarray(specs[i].shape_zyx)/limit).astype(int)) for i in textures}
        sizes=lambda:{i:np.prod(np.ceil(np.asarray(specs[i].shape_zyx)/s))*4 for i,s in strides.items()}
        while sum(sizes().values())>budget:
            largest=max(sizes(),key=sizes().get)
            stride=strides[largest]; stride[np.argmax(np.asarray(specs[largest].shape_zyx)/stride)]+=1
        volumes={}; factors={}
        for i in textures:
            spec=specs[i]; native=raw_volume(spec); stride=strides[i]
            data=np.asarray(native.data[tuple(slice(None,None,int(s)) for s in stride)],dtype=np.float32).copy()
            low,high=previews[i].intensity_limits or intensity_limits(native.data)
            np.nan_to_num(data,copy=False,nan=low,posinf=high,neginf=low); data-=low; data/=high-low; np.clip(data,0,1,out=data)
            volumes[i]=(data,np.asarray(spec.spacing_um_zyx)*stride); factors[i]=stride.tolist()
            progress(round(50*(len(volumes)/max(1,len(textures)))),'Preparing original 3D volumes…')
        region_meshes=[]; surface=None
        if visible_regions:
            atlas=labels
            if atlas is None:
                import tempfile
                atlas=load_label_atlas(owner.project.labels_dir,owner.project.reference,Path(tempfile.gettempdir())/'probemap'/'native-labels',spacing_um_zyx=owner.project.label_spacing)
            from scipy.ndimage import find_objects
            from skimage.measure import marching_cubes
            boxes=find_objects(atlas.data)
            for identifier in visible_regions:
                box=boxes[identifier-1]
                if box is None: continue
                mask=np.pad(atlas.data[box]==identifier,1)
                vertices,faces,_,_=marching_cubes(mask.astype(np.float32),.5,spacing=atlas.spacing_um_zyx,allow_degenerate=False)
                vertices+=(np.array([s.start for s in box])-1)*atlas.spacing_um_zyx
                region_meshes.append((identifier,vertices,faces))
        if entries is not None:
            from .surface_entries import atlas_surface_mesh
            surface=atlas_surface_mesh(raw_volume(owner.project.reference),max_dimension=max(owner.project.reference.shape_zyx),
                max_voxels=max(1_000_000,psutil.virtual_memory().available*.4/20))
        return volumes,factors,region_meshes,surface
    def ready(result):
        from vispy.visuals.transforms import MatrixTransform
        from .desktop_views import visual_matrix
        from .desktop_rendering import apply_render_settings
        volumes,factors,regions,surface=result; old=[]; old_meshes=[]
        guard=view.gl_guard; guard.suspended=True
        try:
            for i,(data,spacing) in volumes.items():
                visual=view.visuals[i]; old.append((i,visual.transform,visual.source_spacing_zyx))
                visual.set_data(data); visual.source_spacing_zyx=spacing
                visual.transform=MatrixTransform(visual_matrix(spacing,np.eye(4) if i==0 else matrix))
                record=dict(settings[i],sampling_step=.5); apply_render_settings(visual,record)
            for identifier,vertices,faces in regions:
                mesh=view.label_visuals[identifier]; old_meshes.append((mesh,mesh.mesh_data.get_vertices().copy(),mesh.mesh_data.get_faces().copy()))
                mesh.set_data(vertices=vertices[:,::-1],faces=faces)
            if surface is not None:
                mesh=entries.mesh; old_meshes.append((mesh,mesh.mesh_data.get_vertices().copy(),mesh.mesh_data.get_faces().copy()))
                mesh.set_data(vertices=surface[0][:,::-1],faces=surface[1])
            from .desktop_gl import render_frame
            rendered=render_frame(view.canvas,size)
            write_png(path,array_image(rendered),dict(kind='3D',source='original TIFF data',native_strides=factors,
                camera=camera,rendering=settings,pixels=size,
                surface_spacing_um_zyx=None if surface is None else list(surface[2])))
            owner.statusBar().showMessage(f'Saved figs/{path.name}',5000)
        finally:
            for mesh,vertices,faces in old_meshes: mesh.set_data(vertices=vertices,faces=faces)
            for i,transform,spacing in old:
                visual=view.visuals[i]; visual.set_data(previews[i].data); visual.source_spacing_zyx=spacing
                visual.transform=transform; apply_render_settings(visual,settings[i])
            guard.suspended=False; view.canvas.update()
    owner._start_job(prepare,lambda result:_finish(owner,path,lambda:ready(result)),'Saving native 3D figure…',on_error=lambda error:path.unlink(missing_ok=True))

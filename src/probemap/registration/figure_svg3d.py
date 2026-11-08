"""SVG projection of visible 3D meshes over a scan-volume image."""
import base64
from html import escape
from pathlib import Path
import numpy as np


def export_scene_svg(view,path,owner=None):
    """Keep scan intensities raster; project region/probe meshes as editable paths."""
    from vispy import scene
    from qtpy.QtCore import QBuffer,QIODevice
    from .desktop_figures import array_image
    from .desktop_gl import render_frame
    from .figure_files import temporary_figure_path,publish_figure
    width,height=view.canvas.size
    nodes=[]
    def visit(node,visible=True):
        visible=visible and node.visible
        if not visible: return
        if isinstance(node,(scene.visuals.Mesh,scene.visuals.Line,scene.visuals.Markers)):
            nodes.append(node)
        for child in node.children: visit(child,visible)
    visit(view.view.scene)
    snapshots=[]; captured=[]
    labels={id(node):f'region-{identifier}' for identifier,node in view.label_visuals.items()}
    # Resolve all transforms before hiding visuals. Canvas coordinates match the background.
    for index,node in enumerate(nodes):
        transform=node.get_transform(map_from='visual',map_to='canvas')
        def project(points):
            result=np.asarray(transform.map(points),float)
            if result.shape[1]==4: result=result[:,:3]/result[:,3,None]
            return result
        record=dict(order=node.order,index=index,identifier=labels.get(id(node),f'object-{index}'))
        if isinstance(node,scene.visuals.Mesh):
            mesh=node.mesh_data; vertices=mesh.get_vertices(); faces=mesh.get_faces()
            if vertices is None or faces is None: continue
            projected=project(vertices)
            colors=mesh.get_face_colors()
            if colors is None:
                vertex_colors=mesh.get_vertex_colors()
                colors=vertex_colors[faces].mean(axis=1) if vertex_colors is not None else np.tile(node.color.rgba,(len(faces),1))
            colors=colors.copy(); colors[:,3]*=getattr(node,'opacity',1)
            record.update(kind='mesh',projected=projected.copy(),faces=faces.copy(),colors=colors)
        elif isinstance(node,scene.visuals.Line):
            pos=getattr(node,'_pos',None)
            if pos is None or isinstance(pos,str): continue
            xy=project(pos)[:,:2]; color=np.asarray(getattr(node,'_color',[1,1,1,1]))
            if color.ndim!=1 or color.size!=4: continue
            rgb='#'+''.join(f'{int(np.clip(c,0,1)*255):02x}' for c in color[:3])
            connect=getattr(node,'_connect','strip')
            segments=[xy[i:i+2] for i in range(0,len(xy)-1,2)] if isinstance(connect,str) and connect=='segments' else [xy]
            record.update(kind='line',segments=segments,color=color.copy(),rgb=rgb,width=getattr(node,'_width',1))
        else:
            data=getattr(node,'_data',None)
            if data is None or not data.dtype.names or 'a_position' not in data.dtype.names: continue
            xy=project(data['a_position'])[:,:2]
            record.update(kind='markers',xy=xy.copy(),data=data.copy())
        snapshots.append(record); captured.append(node)
    old=[node.visible for node in captured]
    try:
        for node in captured: node.visible=False
        image=array_image(render_frame(view.canvas,(width,height)))
    finally:
        for node,visible in zip(captured,old): node.visible=visible
        view.canvas.update()
    buffer=QBuffer(); buffer.open(QIODevice.OpenModeFlag.WriteOnly); image.save(buffer,'PNG')
    encoded=base64.b64encode(bytes(buffer.data())).decode('ascii')
    def build(progress):
        groups=[]
        for number,record in enumerate(snapshots):
            elements=[]
            if record['kind']=='mesh':
                projected,faces,colors=(record[k] for k in ('projected','faces','colors'))
                # Back-to-front within each editable object.
                for face_number,face_index in enumerate(np.argsort(projected[faces,2].mean(axis=1))[::-1]):
                    if face_number%10000==0:
                        progress(round(90*(number+face_number/max(1,len(faces)))/max(1,len(snapshots))),'Writing SVG regions and electrodes…')
                    xy=projected[faces[face_index],:2]; rgba=colors[face_index]
                    if not np.isfinite(xy).all() or rgba[3]<=0: continue
                    if xy[:,0].max()<0 or xy[:,0].min()>width or xy[:,1].max()<0 or xy[:,1].min()>height: continue
                    color='#'+''.join(f'{int(np.clip(c,0,1)*255):02x}' for c in rgba[:3])
                    points=' L '.join(f'{x:.3f} {y:.3f}' for x,y in xy)
                    elements.append(f'<path d="M {points} Z" fill="{color}" fill-opacity="{rgba[3]:.4g}"/>')
            elif record['kind']=='line':
                segments,color,rgb=(record[k] for k in ('segments','color','rgb')); line_width=record['width']
                for segment in segments:
                    if not np.isfinite(segment).all(): continue
                    points=' '.join(f'{x:.3f},{y:.3f}' for x,y in segment)
                    elements.append(f'<polyline points="{points}" fill="none" stroke="{rgb}" stroke-opacity="{color[3]:.4g}" stroke-width="{line_width}"/>')
            else:
                xy,data=record['xy'],record['data']
                for point,row in zip(xy,data):
                    if not np.isfinite(point).all(): continue
                    color=row['a_facecolor']; rgb='#'+''.join(f'{int(np.clip(c,0,1)*255):02x}' for c in color[:3])
                    elements.append(f'<circle cx="{point[0]:.3f}" cy="{point[1]:.3f}" r="{row["a_size"]/2:.3f}" fill="{rgb}" fill-opacity="{color[3]:.4g}"/>')
            groups.append((record['order'],record['index'],record['identifier'],elements))
            progress(round(90*(number+1)/max(1,len(snapshots))),'Writing SVG regions and electrodes…')
        parts=[f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
               '<title>ProbeMap 3D view</title>',
               '<desc>Scan volumes are an image. Regions, probes and contacts are editable projected vectors.</desc>',
               f'<defs><clipPath id="viewport"><rect width="{width}" height="{height}"/></clipPath></defs>',
               f'<image id="scan-volumes" width="{width}" height="{height}" xlink:href="data:image/png;base64,{encoded}"/>',
               '<g clip-path="url(#viewport)">']
        for _,_,identifier,elements in sorted(groups):
            parts.append(f'<g id="{escape(identifier)}">'+''.join(elements)+'</g>')
        parts.append('</g></svg>')
        temporary=temporary_figure_path(path)
        try:
            temporary.write_text('\n'.join(parts),encoding='utf-8'); publish_figure(temporary,path)
        finally: temporary.unlink(missing_ok=True)
        progress(100,'SVG saved')
        return path
    if owner is None: return build(lambda *args:None)
    owner._start_job(build,lambda saved:owner.statusBar().showMessage(f'Saved figs/{saved.name}',5000),
                     'Saving vector 3D figure…',on_error=lambda error:Path(path).unlink(missing_ok=True))

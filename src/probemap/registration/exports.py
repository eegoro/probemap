"""Named, selective exports. Export destinations never change the project folder."""
import csv
import json
from pathlib import Path

import numpy as np

from .coarse import bounding_grid, resample_tiff
from .review import registration_report
from .volumes import spacing_zyx


def export_grid(project, source, matrix, spacing, extent):
    spacing = spacing_zyx(spacing)
    if extent=='full':
        return bounding_grid(source.shape_zyx, source.spacing_um_zyx, matrix, spacing)
    if extent!='atlas':
        raise ValueError('Unknown export extent.')
    # Cover the atlas voxel-cell bounds on a grid with the chosen spacing.
    lower = -np.asarray(project.reference.spacing_um_zyx)/2
    upper = (np.asarray(project.reference.shape_zyx)-.5)*project.reference.spacing_um_zyx
    return dict(shape_zyx=np.maximum(1,np.ceil((upper-lower)/spacing-1e-10).astype(int)).tolist(),
                spacing_um_zyx=spacing.tolist(), origin_um_zyx=(lower+spacing/2).tolist())


def export_selection(project, options, progress=None, cancelled=lambda:False):
    path = Path(options['path']).resolve()
    kind, stage, fmt = (options[key] for key in ('kind','stage','format'))
    if kind not in ('brain','electrodes','transforms','report','analysis','reconstruction') or stage not in ('placement','registration'):
        raise ValueError('Unknown export selection.')
    if (stage=='registration' or kind in ('report','analysis','reconstruction')) and not project.fit_is_current:
        raise ValueError('Apply the current registration before exporting this result.')
    source = project.track if kind=='electrodes' else project.moving
    if source is None:
        raise ValueError('Add an electrodes scan first.')
    matrix = project.coarse.copy() if stage=='placement' else project.total.copy()
    sidecar = path.with_suffix('.geometry.json')
    protected = {s.path.resolve() for s in (project.reference,project.moving,project.track) if s is not None}
    protected.add((Path(project.output_dir)/'project.json').resolve())
    if project.labels_dir:
        protected.update(p.resolve() for p in Path(project.labels_dir).iterdir() if p.is_file())
    if path in protected or sidecar in protected:
        raise ValueError('The export must not overwrite a project or source image.')
    if cancelled():
        raise InterruptedError('Export cancelled.')
    path.parent.mkdir(parents=True, exist_ok=True)
    if kind in ('brain','electrodes'):
        if fmt not in ('tiff','ome'):
            raise ValueError('Choose TIFF or OME-TIFF for a volume.')
        grid = export_grid(project, source, matrix, options['spacing_um_zyx'], options['extent'])
        def update(done, total):
            if cancelled():
                raise InterruptedError('Export cancelled.')
            if progress: progress(round(99*done/total),f'Writing {path.name} · {done}/{total}')
        resample_tiff(source.path,source.spacing_um_zyx,matrix,grid,path,overwrite=True,
                      progress=update,order=0 if options.get('interpolation')=='nearest' else 1,ome=fmt=='ome')
        source_scale=np.eye(4); source_scale[:3,:3]=np.diag(source.spacing_um_zyx)
        target_scale=np.eye(4); target_scale[:3,:3]=np.diag(grid['spacing_um_zyx'])
        shift=np.eye(4); shift[:3,3]=-np.asarray(grid['origin_um_zyx'])
        voxel_map=np.linalg.inv(target_scale)@shift@matrix@source_scale
        geometry=dict(**grid, axes='ZYX',unit='um',stage=stage,source_role=kind,
                      source=source.record(),reference=project.reference.record(),
                      source_to_atlas_zyx_um=matrix.tolist(),atlas_to_source_zyx_um=np.linalg.inv(matrix).tolist(),
                      moving_to_output_voxel_zyx=voxel_map.tolist(),
                      output_voxel_to_moving_zyx=np.linalg.inv(voxel_map).tolist(),
                      interpolation=options.get('interpolation','linear'))
        sidecar.write_text(json.dumps(geometry,indent=2)+'\n', encoding='utf-8')
    elif kind=='transforms':
        transforms=dict(placement_zyx_um=project.coarse,total_zyx_um=matrix,
                        inverse_zyx_um=np.linalg.inv(matrix),
                        registration_zyx_um=project.residual if stage=='registration' else np.eye(4))
        atlas_scale=np.eye(4); atlas_scale[:3,:3]=np.diag(project.reference.spacing_um_zyx)
        for name,spec in (('brain',project.moving),('electrodes',project.track)):
            if spec is None: continue
            source_scale=np.eye(4); source_scale[:3,:3]=np.diag(spec.spacing_um_zyx)
            transforms[name+'_to_atlas_zyx_voxel']=np.linalg.inv(atlas_scale)@matrix@source_scale
            transforms['atlas_to_'+name+'_zyx_voxel']=np.linalg.inv(transforms[name+'_to_atlas_zyx_voxel'])
        if fmt=='json':
            path.write_text(json.dumps(dict(stage=stage,units='um',coordinate_order='ZYX',
                direction='source_to_atlas',reference=project.reference.record(),brain=project.moving.record(),
                electrodes=project.track.record() if project.track else None,
                **{k:v.tolist() for k,v in transforms.items()}),indent=2)+'\n', encoding='utf-8')
        elif fmt=='npz':
            with path.open('wb') as stream: np.savez(stream,**transforms)
        elif fmt=='ants':
            from .landmarks import write_ants_initial_transform
            write_ants_initial_transform(matrix,path)
        else: raise ValueError('Unknown transform format.')
    elif kind=='reconstruction':
        import tempfile
        import zipfile
        from .reconstruction import project_reconstruction_report,write_reconstruction_report
        from .surface_entries import build_atlas_surface
        from .track_model import raw_volume,load_label_atlas
        if fmt not in ('json','zip'): raise ValueError('Choose JSON or ZIP for reconstruction reports.')
        if progress: progress(10,'Finding atlas surface')
        surface=build_atlas_surface(raw_volume(project.reference))
        labels=None
        if project.labels_dir:
            labels=load_label_atlas(project.labels_dir,project.reference,Path(tempfile.gettempdir())/'probemap'/'native-labels',progress,spacing_um_zyx=project.label_spacing)
        report=project_reconstruction_report(project,surface,labels)
        if cancelled(): raise InterruptedError('Export cancelled.')
        if fmt=='json': path.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n', encoding='utf-8')
        else:
            with tempfile.TemporaryDirectory(prefix='probemap-reconstruction-') as temporary:
                write_reconstruction_report(report,temporary)
                with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED) as archive:
                    for item in sorted(Path(temporary).iterdir()): archive.write(item,item.name)
    elif kind=='analysis':
        import tempfile
        import zipfile
        from .neuropixels_model import write_reports
        if fmt!='zip': raise ValueError('Choose ZIP for analysis reports.')
        with tempfile.TemporaryDirectory(prefix='probemap-export-') as temporary:
            folder=Path(temporary)
            write_reports(project,folder)
            (folder/'insertions_and_tracks.json').write_text(json.dumps(project.track_analysis,indent=2)+'\n', encoding='utf-8')
            with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED) as archive:
                for item in sorted(folder.iterdir()): archive.write(item,item.name)
    else:
        report=registration_report(project)
        if fmt=='json':
            path.write_text(json.dumps(report,indent=2)+'\n', encoding='utf-8')
        elif fmt=='csv':
            with path.open('w',newline='',encoding='utf-8') as stream:
                writer=csv.DictWriter(stream,fieldnames=['pair','before_um','after_um'])
                writer.writeheader(); writer.writerows(report['pair_errors'])
        else: raise ValueError('Choose JSON or CSV for the registration report.')
    if progress: progress(100,'Export complete')
    return path

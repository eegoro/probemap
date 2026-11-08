"""Registration review in atlas physical coordinates, independent of display sampling."""
import numpy as np

from .landmarks import fit_landmarks, transform_points


def placement_without_scale(project):
    """Remove stretch/shear, retain axes/reflections and the placed scan centre.

    The polar factor is exact for Placement's S @ R @ axis-assignment map.
    It also handles imported affine placements without relying on saved UI knobs.
    """
    matrix = project.coarse.copy()
    u, _, vt = np.linalg.svd(matrix[:3, :3])
    rotation = u @ vt
    center = (np.asarray(project.moving.shape_zyx)-1)*project.moving.spacing_um_zyx/2
    placed_center = transform_points(center[None], matrix)[0]
    matrix[:3, :3] = rotation
    matrix[:3, 3] = placed_center-rotation@center
    return matrix


def _summary(errors):
    return dict(rms=float(np.sqrt(np.mean(errors**2))), median=float(np.median(errors)),
                p95=float(np.percentile(errors, 95)), maximum=float(np.max(errors)))


def registration_report(project):
    """Fit residuals and leave-one-out prediction errors; never label FRE as TRE."""
    complete = [(i+1, p) for i, p in enumerate(project.pairs)
                if p.get('reference') is not None and p.get('moving') is not None]
    report = dict(model=project.fit_result['model'] if project.fit_result else None,
                  pairs=len(complete), current=project.fit_is_current,
                  error_frame='atlas physical', units='um', before_definition='placement without scale',
                  placement_without_scale_zyx_um=placement_without_scale(project).tolist(),
                  placement_zyx_um=project.coarse.tolist(), total_zyx_um=project.total.tolist(),
                  principal_stretches=np.linalg.svd(project.total[:3, :3], compute_uv=False).tolist(),
                  volume_ratio=float(abs(np.linalg.det(project.total[:3, :3]))),
                  source_voxel_um_xyz=list(project.moving.spacing_um_zyx[::-1]),
                  atlas_voxel_um_xyz=list(project.reference.spacing_um_zyx[::-1]),
                  before=None, after=None, leave_one_out_rms=None, leave_one_out_count=0,
                  pair_errors=[], accuracy_note='Fit residuals are not independent target registration accuracy.')
    report.update(atlas_size_mm_xyz=(np.asarray(project.reference.shape_zyx[::-1])*
                  np.asarray(project.reference.spacing_um_zyx[::-1])/1000).tolist(),
                  rms_atlas_voxels=None,rms_reduction_percent=None)
    if not complete or not project.fit_is_current:
        return report
    ids = [i for i, _ in complete]
    ref = np.asarray([p['reference'] for _, p in complete])
    placed = np.asarray([p['moving'] for _, p in complete])
    raw = transform_points(placed, np.linalg.inv(project.coarse))
    before = np.linalg.norm(transform_points(raw, placement_without_scale(project))-ref, axis=1)
    after = np.linalg.norm(transform_points(placed, project.residual)-ref, axis=1)
    voxel_errors=(transform_points(placed,project.residual)-ref)/project.reference.spacing_um_zyx
    report.update(before=_summary(before), after=_summary(after),
                  rms_atlas_voxels=float(np.sqrt(np.mean(np.sum(voxel_errors**2,axis=1)))),
                  largest_pair=ids[int(np.argmax(after))],
                  pair_errors=[dict(pair=i, before_um=float(b), after_um=float(a))
                               for i, b, a in zip(ids, before, after)])
    if report['before']['rms']>1e-12:
        report['rms_reduction_percent']=100*(1-report['after']['rms']/report['before']['rms'])
    minimum = 4 if report['model']=='affine' else 3
    if len(complete)>minimum:
        errors = []
        for i in range(len(complete)):
            keep = np.arange(len(complete)) != i
            try:
                fit = fit_landmarks(ref[keep], placed[keep], model=report['model'])
                predicted = transform_points(placed[i:i+1], fit['moving_to_reference_zyx_um'])[0]
                errors.append(float(np.linalg.norm(predicted-ref[i])))
            except ValueError:
                continue
        report['leave_one_out_count'] = len(errors)
        # An incomplete set of folds could hide difficult/degenerate points.
        if len(errors)==len(complete):
            report['leave_one_out_rms'] = _summary(np.asarray(errors))['rms']
    return report

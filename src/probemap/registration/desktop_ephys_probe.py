"""Permanent physical probe view and contact hover information."""
import numpy as np
from matplotlib.patches import Polygon
from qtpy.QtGui import QColor, QCursor
from qtpy.QtWidgets import QToolTip

from .ephys_anatomy import probe_site_layout, recording_depth_limit
from .neuropixels_model import geometry_for


def probe_geometry(page):
    if page.anatomy: return page.anatomy['geometry']
    rec = page.current() or {}
    insertion = next((i for i in page._items() if i['id'] == rec.get('insertion_id')), {})
    return insertion.get('probe',{}).get('geometry') or geometry_for()


def depth_limits(page):
    return 0., recording_depth_limit(page.current() or {}, probe_geometry(page))


def draw_probe(page):
    ax = page.anatomy_ax; ax.clear(); ax.set_facecolor('#1c222c')
    page._hover_tree = None; page._hover_key = None; page._hover_sites = None
    rec = page.current(); a = page.anatomy
    if not rec:
        ax.axis('off'); page.anatomy_caption.setText('Probe'); page.anatomy_canvas.draw_idle(); return
    geometry = probe_geometry(page); sites = a['sites'] if a else probe_site_layout(geometry)
    low, high = depth_limits(page)
    pitch, width, taper = (geometry[k] for k in ('shank_pitch_um','shank_width_um','taper_length_um'))
    left, right = -140, (geometry['shanks']-1)*pitch+140
    regions = page.region_editor
    if a:
        gamma = regions.atlas_gamma.value()
        image = a['images'][page.source.currentText()]
        ax.imshow(image if gamma == 1 else image**(1/gamma),extent=a['extent'],aspect='auto',origin='upper')
        if page.labels.isChecked() and a['label_ids'] is not None:
            overlay = np.zeros((*a['label_ids'].shape,4),np.float32)
            styles = (regions.session or {}).get('regions',{})
            for region in a['label_regions']:
                identifier = str(region['id']); style = styles.get(identifier,{})
                if not style.get('visible',True): continue
                mask = a['label_ids'] == region['id']
                if not np.any(mask): continue
                overlay[mask,:3] = np.asarray(style.get('color_rgb',region['color_rgb']))/255
                overlay[mask,3] = regions.region_opacity.value()/100
                if getattr(page,'_vector_export',False):
                    from .figure_regions import matplotlib_region
                    matplotlib_region(ax,mask,a['extent'],overlay[mask][0],f'region-{identifier}')
                if style.get('label',False):
                    x,y = a['region_positions'][identifier]
                    ax.text(x,y,style.get('name',region['name']),fontsize=7,color='#f5f8ff',ha='center',va='center',clip_on=True,zorder=7,
                            bbox=dict(facecolor='#202733',edgecolor='none',alpha=.7,pad=2))
            if not getattr(page,'_vector_export',False):
                ax.imshow(overlay,extent=a['extent'],aspect='auto',origin='upper',zorder=2)
    insertion = next((i for i in page._items() if i['id'] == rec.get('insertion_id')), {})
    body_color = insertion.get('color',page.record_color(rec)); active_color = page.record_color(rec)
    lookup = {(c['shank'],c['row'],c['column']):c['channel'] for c in rec['channels']}
    keys = list(zip(sites['shanks'],sites['rows'],sites['columns']))
    recorded = np.array([key in lookup for key in keys])
    choices=page.plot_board.column_choices()
    selected_columns={0,1} if 'mean' in choices else {int(c) for c in choices}
    active=recorded & np.isin(sites['columns'],list(selected_columns))
    if page.probe_body.isChecked():
        for shank in range(geometry['shanks']):
            x = shank*pitch; top = geometry['shank_length_um']
            ax.add_patch(Polygon([(x,0),(x-width/2,taper),(x-width/2,top),(x+width/2,top),(x+width/2,taper)],
                closed=True,facecolor=body_color,edgecolor=body_color,alpha=page.probe_opacity.value()/100,linewidth=.8,zorder=3))
    size = page.contact_size.value()
    shown = np.ones(len(recorded),bool) if page.inactive_contacts.isChecked() else active
    if page.inactive_contacts.isChecked():
        ax.scatter(sites['xs'][~active],sites['heights'][~active],s=size*.4,marker='s',c=body_color,alpha=.22,linewidths=0,zorder=4)
    ax.scatter(sites['xs'][active],sites['heights'][active],s=size,marker='s',c=active_color,edgecolors='#f5f8ff',linewidths=.25,zorder=5)
    if a:
        for shank, depth in a['depths'].items():
            x = shank*pitch; ax.plot([x-65,x+65],[depth]*2,color=body_color,ls='--',lw=1,zorder=6)
    if page.boundary.isChecked():
        x = (page.shank.currentData() or 0)*pitch
        ax.plot([x-65,x+65],[page.boundary_value.value()]*2,color='#7ac7e5',lw=1,zorder=6)
    ax.set_xlim(left,right); ax.set_ylim(low,high)
    ax.set_xticks(np.arange(geometry['shanks'])*pitch, [f'S{s}' for s in range(geometry['shanks'])])
    ax.tick_params(axis='x',top=True,labeltop=True,bottom=False,labelbottom=False,length=0,pad=6)
    ax.set_ylabel('Depth from tip · µm',color='#aebdd0'); ax.tick_params(colors='#aebdd0',labelsize=8)
    for spine in ax.spines.values(): spine.set_color('#455369')
    page.anatomy_caption.setText(page.source.currentText() if a else 'Probe · assign insertion')
    page._hover_sites = (sites,recorded,shown,lookup)
    page.anatomy_canvas.draw_idle()


def hover_contact(page,event):
    if page._busy or event.inaxes is not page.anatomy_ax or page._hover_sites is None:
        page._hover_text = ''; QToolTip.hideText(); return
    from scipy.spatial import cKDTree
    sites,recorded,shown,lookup = page._hover_sites
    ax = page.anatomy_ax
    key = (*ax.get_xlim(),*ax.get_ylim(),*ax.bbox.bounds)
    if key != page._hover_key:
        points = ax.transData.transform(np.column_stack((sites['xs'],sites['heights'])))
        mask = shown & (sites['heights']>=ax.get_ylim()[0]) & (sites['heights']<=ax.get_ylim()[1])
        page._hover_indices = np.flatnonzero(mask)
        page._hover_tree = cKDTree(points[mask]); page._hover_key = key
    distance,index = page._hover_tree.query([event.x,event.y])
    if distance > 7 or not len(page._hover_indices):
        page._hover_text = ''; QToolTip.hideText(); return
    index = page._hover_indices[index]
    shank,row,column = (int(sites[k][index]) for k in ('shanks','rows','columns'))
    channel = lookup.get((shank,row,column))
    state = f'Recording · channel {channel}' if recorded[index] else 'Not recorded'
    text = f'Shank {shank} · electrode {row*2+column}\n{sites["heights"][index]:g} µm from tip\n{state}'
    page._hover_text = text
    QToolTip.showText(QCursor.pos(),text,page.anatomy_canvas)


def probe_draw_key(page):
    """Only anatomy/display changes invalidate the expensive image overlay."""
    import json
    rec=page.current() or {}
    insertion=next((i for i in page._items() if i['id']==rec.get('insertion_id')), {})
    regions=page.region_editor
    return (rec.get('id'),id(page.anatomy),id(rec.get('channels')),page.record_color(rec),
            insertion.get('color'),json.dumps(probe_geometry(page),sort_keys=True,default=str),
            page.source.currentText(),page.labels.isChecked(),tuple(page.plot_board.column_choices()),
            page.probe_body.isChecked(),page.inactive_contacts.isChecked(),page.probe_opacity.value(),
            page.contact_size.value(),page.shank.currentData(),page.boundary.isChecked(),page.boundary_value.value(),
            regions.atlas_gamma.value(),regions.region_opacity.value(),
            json.dumps((regions.session or {}).get('regions',{}),sort_keys=True,default=str))

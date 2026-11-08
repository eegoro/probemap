"""A dedicated legend footer for each panel, also used by vector PDF export."""
import math
import textwrap
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from .ephys_analysis import BANDS
from .desktop_ephys_plots import COLORS,series_style


def column_handle(kind,**style):
    line=Line2D([],[],**style); line._ephys_column=kind
    return line


def attach(figure,axes,handles,labels,title=None,group=None):
    if not handles: return
    legend=figure.legend(handles,labels,loc='lower center',frameon=False,fontsize=7,
        labelcolor='#dce6f0',handlelength=1.7,columnspacing=1,handletextpad=.6,title=title,title_fontsize=7)
    legend.get_title().set_color('#dce6f0'); legend._ephys_group=group; legend._ephys_title=title
    legend._ephys_axes=axes; legend._ephys_handles=handles; legend._ephys_labels=labels; legend._ephys_layout=None


def plot_legend(page,figure,key,axes):
    if not page.plot_board.legends.isChecked(): return
    board=page.plot_board; choices=board.column_choices(); handles=[]; labels=[]
    options=board.plot_options(key)
    if key in ('lfp','lfp_gradient'):
        for number,kind in enumerate(choices):
            visible=[(i,b) for i,b in enumerate(options['bands']) if b['visible']]
            lines=[]
            for band,b in visible:
                actual=next((line for ax in axes for line in ax.lines if getattr(line,'_ephys_column',None)==kind and getattr(line,'_ephys_band',None)==band),None)
                style=series_style(b['color'],kind,board.opacity(kind))
                if actual is not None: style.update(color=actual.get_color(),lw=actual.get_linewidth(),alpha=actual.get_alpha())
                lines.append(column_handle(kind,**style))
            attach(figure,axes,lines,[f'{b["low"]:g}–{b["high"]:g} Hz' for i,b in visible],board.column_label(kind),(number,len(choices)))
        if key=='lfp' and options['fits']:
            handles.append(Line2D([],[],color='#aebdd0',lw=1.4*options['line_width'])); labels.append('Piecewise linear fit')
        if key=='lfp' and options['transitions']:
            handles.append(Line2D([],[],color='#aebdd0',lw=.5)); labels.append('Fitted transition depths')
    if key=='rms':
        for kind in choices:
            color=options['color'] or (page.record_color(page.current()) if page.current() else COLORS[0])
            handles.append(column_handle(kind,**{**series_style(color,kind,board.opacity(kind)),'lw':series_style(color,kind)['lw']*options['line_width']})); labels.append(board.column_label(kind))
    if key.startswith('compare_'):
        for rec,_ in page.comparisons:
            for kind in choices:
                handles.append(column_handle(kind,**{**series_style(options['color'] or page.record_color(rec),kind,board.opacity(kind)),'lw':series_style(page.record_color(rec),kind)['lw']*options['line_width']}))
                labels.append(rec['name']+' · '+board.column_label(kind))
    if key=='region_lfp':
        handles.append(Patch(facecolor=COLORS[0])); labels.append('Mean of 1.5–4, 4–10, 10–30 Hz bands')
    if key=='traces' and not board.notebook_mode.isChecked():
        for number,kind in enumerate(choices):
            attach(figure,axes,[column_handle(kind,color=options['color'] or COLORS[0],lw=.35*options['line_width'],alpha=board.opacity(kind))],[f'LFP · P95 scaled to {10*options["trace_scale"]:g} µm'],board.column_label(kind),(number,len(choices)))
    if key=='low40_plain' or (key=='traces' and board.notebook_mode.isChecked()):
        for number,kind in enumerate(choices):
            attach(figure,axes,[column_handle(kind,color=options['color'] or COLORS[0],alpha=board.opacity(kind))],
                   [f'LFP · traces ×{options["trace_scale"]:g}'],board.column_label(kind),(number,len(choices)))
    if key=='traces_clustered' and not board.notebook_mode.isChecked() and page.trace is not None and len(page.trace)>2 and page.trace[2] is not None:
        from matplotlib import colormaps
        for number,kind in enumerate(choices):
            if kind not in page.trace[2]: continue
            available=sorted(set(page.trace[2][kind].tolist()))
            handles_for_kind=[column_handle(kind,color=options['color'] or colormaps['tab10'](int(i)%10),alpha=board.opacity(kind)) for i in available]
            attach(figure,axes,handles_for_kind,[f'Cluster {int(i)+1}' for i in available],board.column_label(kind),(number,len(choices)))
    heatmaps=dict(spectrum='Power · dB (µV²/Hz)',correlation='Pearson r',activity='RMS · µV',csd='CSD · relative units')
    if key in heatmaps:
        for ax in axes:
            if not ax.collections: continue
            collection=ax.collections[0]; lo,hi=collection.get_clim()
            if lo is None or hi is None: continue
            color_scale(figure,axes,collection,heatmaps[key])
            break
    if key=='correlation_profile':
        for number,kind in enumerate(choices):
            lines=[]; names=[]
            if options['raw']: lines.append(column_handle(kind,color=options['color'] or COLORS[0],lw=.85*options['line_width'],alpha=board.opacity(kind))); names.append('Mean |Pearson r|')
            if options['fits']: lines.append(column_handle(kind,color=options['color'] or COLORS[1],lw=1.2*options['line_width'],alpha=board.opacity(kind))); names.append('Piecewise linear fit')
            attach(figure,axes,lines,names,board.column_label(kind),(number,len(choices)))
    if key=='csd':
        for kind in choices:
            handles.append(column_handle(kind,color=series_style(options['color'],kind)['color'] if options['color'] else 'black',lw=.4*options['line_width'],alpha=board.opacity(kind))); labels.append(board.column_label(kind)+f' · traces ×{options["trace_scale"]:g}')
    if page.notebook is not None and (key in ('low40','low40_plain','csd','correlation_profile') or (board.notebook_mode.isChecked() and key in ('lfp','spectrum','correlation','traces','traces_clustered'))):
        handles.append(Line2D([],[],color='#e0bf55',lw=.7)); labels.append('Mean band boundary · estimate')
    if key in ('traces_clustered','low40') and page.notebook is not None and (key=='low40' or board.notebook_mode.isChecked()):
        from matplotlib import colormaps
        for number,kind in enumerate(choices):
            name=kind+('_low40_clusters' if key=='low40' else '_clusters')
            if name not in page.notebook: continue
            available=sorted(set(page.notebook[name].tolist()))
            lines=[column_handle(kind,color=series_style(options['color'],kind)['color'] if options['color'] else colormaps['tab10'](i),alpha=board.opacity(kind),lw=.4*options['line_width']) for i in available]
            attach(figure,axes,lines,[f'Cluster {i+1}' for i in available],board.column_label(kind),(number,len(choices)))
    attach(figure,axes,handles,labels)


def anatomy_legend(page,figure,ax):
    a=page.anatomy
    if not page.plot_board.legends.isChecked(): return
    if not a or not page.labels.isChecked(): return
    styles=(page.region_editor.session or {}).get('regions',{})
    handles=[]; labels=[]
    for region in a['label_regions']:
        style=styles.get(str(region['id']),{})
        if not style.get('visible',True): continue
        color=style.get('color_rgb',region['color_rgb'])
        handles.append(Patch(facecolor=[x/255 for x in color],alpha=page.region_editor.region_opacity.value()/100))
        labels.append(style.get('name',region['name']))
    attach(figure,[ax],handles,labels)


def color_scale(figure,axes,collection,label):
    from matplotlib.colorbar import ColorbarBase
    from matplotlib.ticker import FuncFormatter
    cax=figure.add_axes([.1,.03,.8,.015]); cax._ephys_footer=True; cax.set_navigate(False)
    cax.set_facecolor('#242b36')
    bar=ColorbarBase(cax,cmap=collection.cmap,norm=collection.norm,orientation='horizontal',alpha=collection.get_alpha())
    low,high=bar.norm.vmin,bar.norm.vmax
    crosses_zero=low<0<high
    ticks=np.unique(np.r_[low,([0.] if crosses_zero else [(low+high)/2]),high])
    # Three well-separated values avoid crowding min/max with nearby locator
    # ticks. A zero very close to an endpoint gets its label above the bar.
    zero_above=crosses_zero and min(-low,high)/(high-low)<.15
    bar.set_ticks(ticks)
    bar.formatter=FuncFormatter(lambda value,pos:('' if zero_above else '0') if value==0 else f'{value:.3g}')
    bar.update_ticks()
    if zero_above:
        cax.text((0-low)/(high-low),1.7,'0',transform=cax.transAxes,ha='center',va='bottom',color='#aebdd0',fontsize=7)
    bar.set_label(label,color='#dce6f0',fontsize=7,labelpad=3)
    cax.tick_params(colors='#aebdd0',labelsize=7,length=3,pad=2); bar.outline.set_edgecolor('#455369')
    if bar.solids is not None: bar.solids.set_rasterized(False)
    figure._ephys_colorbars=[*getattr(figure,'_ephys_colorbars',[]),dict(axes=axes,bar=bar)]


def legend_span(legend,width):
    axes=legend._ephys_axes
    left=min(a.get_position().x0 for a in axes); right=max(a.get_position().x1 for a in axes)
    group=legend._ephys_group
    if group is not None:
        index,count=group; across=max(1,min(count,int((right-left)*width//95)))
        span=(right-left)/across; left+=(index%across)*span; right=left+span
    return left,right


def legend_rows(legend,width):
    if legend._ephys_group is not None: return len(legend.get_texts())
    left,right=legend_span(legend,width); available=max(100,width*(right-left))
    return math.ceil(len(legend.get_texts())/max(1,int(available//130)))


def group_stack(figure,axes,width):
    groups=[l for l in figure.legends if l._ephys_axes==axes and l._ephys_group is not None]
    if not groups: return 0,1,0
    left=min(a.get_position().x0 for a in axes); right=max(a.get_position().x1 for a in axes)
    count=max(l._ephys_group[1] for l in groups)
    across=max(1,min(count,int((right-left)*width//95)))
    block=max(16+len(l.get_texts())*14 for l in groups)
    return math.ceil(count/across)*block,across,block


def footer_height(figure,width):
    height=80
    # Grouped column legends sit above shared annotation keys, never overlap.
    for axes in [l._ephys_axes for l in figure.legends]:
        plain=max([legend_rows(l,width)*25 for l in figure.legends if l._ephys_axes==axes and l._ephys_group is None] or [0])
        grouped=group_stack(figure,axes,width)[0]
        height=max(height,70+plain+grouped)
    for scale in getattr(figure,'_ephys_colorbars',[]):
        plain=max([legend_rows(l,width)*25 for l in figure.legends if l._ephys_axes==scale['axes'] and l._ephys_group is None] or [0])
        grouped=group_stack(figure,scale['axes'],width)[0]
        height=max(height,130+plain+grouped)
    return height


def place_legends(figure,width,height):
    # Each legend is attached to its own plot's horizontal span, not the whole PDF.
    for legend in list(figure.legends):
        axes=legend._ephys_axes
        left,right=legend_span(legend,width)
        available=max(65,(right-left)*width)
        ncol=1 if legend._ephys_group is not None else max(1,int(available//130)); wrap=max(12,int(available/ncol/4.5))
        layout=(ncol,wrap)
        if legend._ephys_layout!=layout:
            labels=['\n'.join(textwrap.wrap(text,width=wrap)) for text in legend._ephys_labels]
            handles=legend._ephys_handles; originals=legend._ephys_labels; group=legend._ephys_group; title=legend._ephys_title
            legend.remove()
            legend=figure.legend(handles,labels,loc='lower center',ncol=ncol,frameon=False,fontsize=7,
                labelcolor='#dce6f0',handlelength=1.7,columnspacing=1,handletextpad=.6,title=title,title_fontsize=7)
            legend.get_title().set_color('#dce6f0'); legend._ephys_group=group; legend._ephys_title=title
            legend._ephys_axes=axes; legend._ephys_handles=handles; legend._ephys_labels=originals; legend._ephys_layout=layout
        base=8
        if legend._ephys_group is not None:
            base+=max([legend_rows(l,width)*25 for l in figure.legends if l._ephys_axes==axes and l._ephys_group is None] or [0])
            stack,across,block=group_stack(figure,axes,width)
            base+=stack-block*(legend._ephys_group[0]//across+1)
        legend.set_bbox_to_anchor(((left+right)/2,base/height),transform=figure.transFigure)
    for scale in getattr(figure,'_ephys_colorbars',[]):
        axes=scale['axes']; left=min(a.get_position().x0 for a in axes); right=max(a.get_position().x1 for a in axes)
        plain=max([legend_rows(l,width)*25 for l in figure.legends if l._ephys_axes==axes and l._ephys_group is None] or [0])
        grouped=group_stack(figure,axes,width)[0]
        scale['bar'].ax.set_position([left,(50+plain+grouped)/height,right-left,10/height])

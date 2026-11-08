"""Independent analysis plots; depth stays in physical micrometres from the tip."""
from dataclasses import dataclass
import numpy as np
from matplotlib.colors import to_rgb

from .ephys_analysis import BANDS
from .desktop_ephys_probe import depth_limits


@dataclass(frozen=True)
class PlotSpec:
    key: str
    title: str
    depth: bool = True
    source: str = 'signal'


PLOTS = [PlotSpec('lfp','LFP power'), PlotSpec('lfp_gradient','Spectral depth gradient'), PlotSpec('rms','AP RMS'),
    PlotSpec('spectrum','LFP spectrum'), PlotSpec('correlation','LFP correlation'),
    PlotSpec('correlation_profile','Correlation profile'),
    PlotSpec('low40','Low-pass traces · clusters'), PlotSpec('low40_plain','Low-pass traces · no clusters'), PlotSpec('csd','Current source density'),
    PlotSpec('traces','LFP traces · no clusters'), PlotSpec('traces_clustered','LFP traces · clusters'), PlotSpec('activity','AP RMS over time'),
    PlotSpec('raster','Unit raster',source='units'), PlotSpec('density','Unit density',source='units'),
    PlotSpec('waveform','Unit waveform · time / amplitude',False,'units'),
    PlotSpec('isi','Unit ISI · interval / count',False,'units'),
    PlotSpec('acg','Unit autocorrelation · lag / count',False,'units'),
    PlotSpec('rate','Unit firing rate · time / Hz',False,'units'),
    PlotSpec('amplitude','Unit amplitudes · time / amplitude',False,'units'),
    PlotSpec('region_lfp','LFP by region',False), PlotSpec('region_rms','AP RMS by region',False),
    PlotSpec('region_units','Units by region',False,'units'),
    PlotSpec('compare_lfp','Compare LFP power',source='compare'),
    PlotSpec('compare_rms','Compare AP RMS',source='compare')]
SPECS = {p.key:p for p in PLOTS}
COLORS = ['#7ac7e5','#e0bf55','#b69be3','#ee9a7c','#84ce9d']
TOP, BOTTOM = 76., 170.


def axis_style(ax):
    ax.set_facecolor('#1c222c'); ax.tick_params(colors='#aebdd0',labelsize=8)
    for spine in ax.spines.values(): spine.set_color('#455369')
    ax.xaxis.label.set_color('#aebdd0'); ax.yaxis.label.set_color('#aebdd0'); ax.title.set_color('#dce6f0')


def column_series(record, indices, heights, values, choices):
    """Mean in linear signal/power units, only at actually sampled depths."""
    cols = np.array([record['channels'][i]['column'] for i in indices])
    values = np.asarray(values)
    for key in choices:
        keep = np.ones(len(indices),bool) if key == 'mean' else cols == int(key)
        if not np.any(keep): continue
        h, v = heights[keep], values[keep]
        levels, inverse = np.unique(h,return_inverse=True)
        averaged = np.stack([v[inverse==i].mean(axis=0) for i in range(len(levels))])
        yield key, levels, averaged


def series_style(color,kind,alpha=1.):
    rgb=np.array(to_rgb(color))
    if kind=='0': rgb=rgb*.65+.35
    if kind=='1': rgb=rgb*.72
    return dict(color=rgb,lw=1.6 if kind=='mean' else .85,ls='-',alpha=alpha)


def line_profile(ax, values, heights, color, kind, pitch=15.):
    h, v = np.asarray(heights,float), np.asarray(values,float)
    breaks = np.flatnonzero(np.diff(h)>pitch*1.5)+1
    h, v = np.insert(h,breaks,np.nan), np.insert(v,breaks,np.nan)
    line=ax.plot(v,h,**series_style(color,kind))[0]; line._ephys_column=kind
    return line


def depth_gradient(values,heights):
    values=np.asarray(values,float); heights=np.asarray(heights,float); result=np.full_like(values,np.nan)
    breaks=np.r_[0,np.flatnonzero(np.diff(heights)>22.5)+1,len(heights)]
    for a,b in zip(breaks[:-1],breaks[1:]):
        if b-a>1: result[a:b]=np.gradient(values[a:b],heights[a:b])
    return result


def draw_profile_fits(page,ax,fits,options):
    for kind in page.plot_board.column_choices():
        if kind not in fits: continue
        result=fits[kind]; heights=result['heights']
        for band,style in enumerate(options['bands']):
            if not style['visible'] or band>=len(result['band_fit']): continue
            if options['fits']:
                line=line_profile(ax,result['band_fit'][band],heights,style['color'],kind); line.set_linewidth(1.4); line._ephys_role='fit'; line._ephys_band=band
            if options['transitions']:
                for y in np.asarray(result['band_breaks'][band])[np.isfinite(result['band_breaks'][band])][1:-1]:
                    if not np.isfinite(y): continue
                    line=ax.axhline(y,**series_style(style['color'],kind)); line.set_linewidth(.5); line._ephys_column=kind; line._ephys_role='transition'


def filtered_units(page):
    return page._visible_units()


def add_axes(figure, bounds, number=1):
    left,bottom,width,height = bounds
    gap = min(.04,width*.09/number)
    each = (width-gap*(number-1))/number
    return [figure.add_axes([left+i*(each+gap),bottom,each,height]) for i in range(number)]


def render_plot(page, figure, key, bounds, export=False):
    """Render cached arrays only. No source-file reads or worker creation here."""
    spec = SPECS[key]; rec = page.current()
    notebook_keys=('lfp','lfp_gradient','spectrum','correlation','traces','traces_clustered')
    if key in ('correlation_profile','low40','low40_plain','csd') or (page.plot_board.notebook_mode.isChecked() and key in notebook_keys):
        from .desktop_ephys_notebook_plots import render_notebook
        return finish_plot(page,render_notebook(page,figure,key,bounds),spec)
    choices = page.plot_board.column_choices()
    idx, heights = page._indices_heights() if rec else (np.array([],int),np.array([]))
    facets = key in ('spectrum','correlation','traces','traces_clustered','activity')
    available = list(column_series(rec,idx,heights,np.zeros(len(idx)),choices)) if rec and facets else []
    axes = add_axes(figure,bounds,max(1,len(available)) if facets else 1)
    if key == 'waveform' and page.sorting:
        for ax in axes: figure.delaxes(ax)
        left,bottom,width,height = bounds
        count = max(1,len(page._waveform_units()))
        slot = height / count
        axes = [figure.add_axes([left,bottom+(count-i-1)*slot+slot*.16,width,slot*.68]) for i in range(count)]
    for ax in axes:
        axis_style(ax); ax.set_title(spec.title.split(' · ')[0],fontsize=10,pad=12)
        if spec.depth:
            ax.set_ylim(page.plot_board.depth_range())
            ax.set_yticks(page.anatomy_ax.get_yticks()); ax.set_ylim(page.plot_board.depth_range())
            ax.set_ylabel('Depth from tip · µm')
    def empty(message):
        axes[0].text(.5,.5,message,ha='center',va='center',color='#9aafc1',fontsize=9,wrap=True,transform=axes[0].transAxes)
        return axes
    if not rec: return empty('Select a recording')
    if spec.source == 'signal':
        if not page.summary: return empty('Prepare signal')
        data = page.summary[1]
        if key in ('lfp','lfp_gradient','rms'):
            ax = axes[0]
            options=page.plot_board.plot_options(key)
            from .ephys_profiles import band_power
            band_values=band_power(data,options['bands']) if key in ('lfp','lfp_gradient') else None
            bands=range(len(options['bands'])) if band_values is not None else [None]
            for band in bands:
                if band is not None and not options['bands'][band]['visible']: continue
                time_mask=(data['ap_times']>=page.time.value())&(data['ap_times']<=page.time.value()+page.window_duration.value()) if page.plot_board.notebook_mode.isChecked() else np.ones(len(data['ap_times']),bool)
                values=band_values[band,idx] if band is not None else np.median(data['ap_rms'][idx][:,time_mask],axis=1) if np.any(time_mask) else np.full(len(idx),np.nan)
                for kind,h,v in column_series(rec,idx,heights,values,choices):
                    if band is not None: v = 10*np.log10(np.maximum(v,1e-20))
                    if key=='lfp_gradient': v=depth_gradient(v,h)
                    if key!='lfp' or options['raw']:
                        line=line_profile(ax,v,h,options['bands'][band]['color'] if band is not None else (options['color'] or page.record_color(rec)),kind); line._ephys_band=band
            if key=='lfp' and page.profile_fits:
                draw_profile_fits(page,ax,page.profile_fits,options)
            ax.set_xlabel('Gradient · dB / µm' if key=='lfp_gradient' else 'Power · dB (µV²/Hz)' if key=='lfp' else 'RMS · µV')
        elif facets:
            values = data['power'][idx] if key=='spectrum' else data['ap_rms'][idx] if key=='activity' else (
                page.trace[1][idx] if key in ('traces','traces_clustered') and page.trace is not None and abs(page.trace[0]-page.time.value())<.01 else np.zeros((len(idx),1)))
            groups = list(column_series(rec,idx,heights,values,choices))
            for ax,(kind,h,v) in zip(axes,groups):
                suffix = page.plot_board.column_label(kind)
                ax.set_title(f'{spec.title}\n{suffix}',fontsize=9,pad=9)
                if key=='spectrum':
                    freq = data['frequencies']; keep = freq>=0
                    levels,power = page._depth_grid(h,v)
                    ax.pcolormesh(freq[keep],levels,10*np.log10(np.maximum(power[:,keep],1e-20)),shading='nearest',cmap='magma',rasterized=True)
                    ax.set_xlabel('Frequency · Hz')
                elif key=='activity':
                    levels,rms = page._depth_grid(h,v)
                    ax.pcolormesh(data['ap_times'],levels,rms,shading='nearest',cmap='magma',rasterized=True)
                    ax.set_xlabel('Time · s')
                elif key=='correlation':
                    cols = np.array([rec['channels'][i]['column'] for i in idx])
                    chosen = idx if kind=='mean' else idx[cols==int(kind)]
                    # Average correlation blocks for duplicated depths in the mean view.
                    raw_h = heights if kind=='mean' else heights[cols==int(kind)]
                    corr = data['correlation'][np.ix_(chosen,chosen)]
                    levels = np.unique(raw_h)
                    weights = (levels[:,None]==raw_h[None,:]).astype(float); weights /= weights.sum(axis=1,keepdims=True)
                    averaged = weights @ corr @ weights.T
                    ax.pcolormesh(levels,levels,averaged,shading='nearest',cmap='RdBu_r',vmin=-1,vmax=1,rasterized=True)
                    ax.set_xlabel('Depth from tip · µm')
                elif page.trace is not None and v.shape[1]>1:
                    fs = page.summary[0]['sampling_rate']; ts = page.trace[0]+np.arange(v.shape[1])/fs
                    scale = max(1e-6,float(np.percentile(np.abs(v),95)))
                    stride = max(1,len(h)//64)
                    from matplotlib import colormaps
                    labels = page.trace[2].get(kind) if key=='traces_clustered' and len(page.trace)>2 and page.trace[2] is not None else None
                    if key=='traces_clustered' and labels is None:
                        ax.text(.5,.5,'Preparing clusters…',ha='center',transform=ax.transAxes); continue
                    for row in range(0,len(h),stride):
                        color=colormaps['tab10'](int(labels[row])%10) if labels is not None else COLORS[0]
                        line=ax.plot(ts,h[row]+v[row]/scale*10*page.plot_board.plot_options(key)['trace_scale'],lw=.35,c=color)[0]; line._ephys_column=kind
                    ax.set_xlabel('Time · s')
                else: ax.set_xlabel('Time · s')
        elif key.startswith('region_'):
            a = page.anatomy
            if not a or not a['contact_regions']: return empty('Assign anatomy with region labels')
            ids = np.array(a['contact_regions']); channel_indices = np.array([i for i,c in enumerate(rec['channels']) if c['shank']==page.shank.currentData()])
            valid = [i for i in np.unique(ids) if i>0]
            names = [a['regions'].get(str(i),str(i)) for i in valid]
            vals = [10*np.log10(max(float(np.mean(data['bands'][:3,channel_indices[ids==i]])),1e-20)) if key=='region_lfp'
                    else float(np.median(data['ap_rms'][channel_indices[ids==i]])) for i in valid]
            axes[0].barh(names,vals,color=COLORS[0]); axes[0].set_xlabel('Power · dB (µV²/Hz)' if key=='region_lfp' else 'RMS · µV')
    elif spec.source == 'units':
        if not page.sorting: return empty('Connect Kilosort in the recordings table')
        info,data = page.sorting; units = filtered_units(page); ax = axes[0]
        view = page.unit_view if page.unit_view is not None and page.unit_view['key']==page._unit_view_key() else None
        if key=='raster':
            if not page._filtered_units(): return empty('No units match this shank and All/Good filter')
            if view:
                positions = page._raster_positions()
                ax._unit_artists = {}
                visible = {u['id'] for _,u in units}
                ids,starts,counts=np.unique(view['raster_ids'],return_index=True,return_counts=True)
                slices={int(uid):slice(int(start),int(start+count)) for uid,start,count in zip(ids,starts,counts)}
                for _,unit in page._filtered_units():
                    uid = unit['id']
                    times = view['raster_times'][slices.get(uid,slice(0,0))]
                    if not len(times): continue
                    points = ax.scatter(times,np.full(len(times),positions[uid]),s=.7,
                                        c=COLORS[int(uid)%len(COLORS)],linewidths=0,
                                        rasterized=False,picker=5,animated=not export)
                    points._unit_ids = np.full(len(times),uid)
                    points._unit_depth = unit['height_um']; points._raster_y = positions[uid]
                    points.set_visible(uid in visible)
                    ax._unit_artists[uid] = points
                span = page._spike_span(); ax.set_xlim(page.time.value(),page.time.value()+span) if span else ax.set_xlim(0,info['duration_s'])
            else: ax.text(.5,.5,'Reading cached spikes…',ha='center',color='#9aafc1',transform=ax.transAxes)
            ax.set_xlabel('Time · s'); ax.set_ylabel('Raster Y · µm')
        elif key=='waveform':
            chosen = page._waveform_units()
            if not chosen: return empty('Select units in the Waveforms table')
            selected_waves = np.asarray([data['waveforms'][i] for i,_ in chosen])
            finite = selected_waves[np.isfinite(selected_waves)]
            limits = None
            if finite.size:
                lo,hi = float(finite.min()),float(finite.max())
                pad = max((hi-lo)*.08,1e-6)
                limits = (lo-pad,hi+pad)
            for ax,(index,unit) in zip(axes,chosen):
                wave = data['waveforms'][index]
                if not np.isfinite(wave).any():
                    ax.text(.5,.5,'No Kilosort template available',ha='center',transform=ax.transAxes)
                ax.plot(np.arange(len(wave))/info['sampling_rate']*1000,wave,
                        c=COLORS[int(unit['id'])%len(COLORS)])
                ax.set_title(f"Unit {unit['id']} · {unit['height_um']:.0f} µm · {unit['label']}",fontsize=9,pad=5)
                ax.set_xlabel('Time · ms'); ax.set_ylabel('Template units')
                if limits is not None: ax.set_ylim(limits)
        elif key=='density':
            ax.hist([u['height_um'] for _,u in units],bins=np.arange(0,depth_limits(page)[1]+150,150),orientation='horizontal',color=COLORS[0])
            ax.set_xlabel('Units / 150 µm')
        elif key=='region_units':
            if not page.anatomy or not page.anatomy['contact_regions']: return empty('Assign anatomy with region labels')
            a=page.anatomy; counts={}
            for _,u in units:
                nearest=int(np.argmin(abs(a['heights']-u['height_um'])))
                name=a['regions'].get(str(a['contact_regions'][nearest]),'Unassigned'); counts[name]=counts.get(name,0)+1
            ax.barh(list(counts),list(counts.values()),color=COLORS[0]); ax.set_xlabel('Units')
        else:
            row=page.units.currentRow()
            if row<0: return empty('Select a unit · use All units if no good labels are available')
            index=page.units.item(row,0).data(256); unit=info['units'][index]
            if key=='isi':
                ax.bar(np.arange(50)*2+1,unit['isi_hist'],width=1.8,color=COLORS[0]); ax.set_xlabel('ISI · ms'); ax.set_ylabel('Spikes')
            elif key=='acg':
                if view: ax.bar(np.arange(-49.5,50),view['acg'],width=.9,color=COLORS[0])
                ax.set_xlabel('Lag · ms'); ax.set_ylabel('Coincidences · sampled')
            elif key=='rate':
                edges=np.linspace(0,info['duration_s'],61); ax.stairs(unit['rate_bins'],edges,color=COLORS[0]); ax.set_xlabel('Time · s'); ax.set_ylabel('Hz')
            elif key=='amplitude':
                if view: ax.scatter(view['times'],view['amplitudes'],s=2,c=COLORS[0],linewidths=0,rasterized=False)
                ax.set_xlabel('Time · s'); ax.set_ylabel('Exported amplitude units')
            ax.set_title(f'Unit {unit["id"]} · {spec.title.split(" · ")[0]}',fontsize=10,pad=12)
    else:
        if len({r['insertion_id'] for r in page.selected()})>1: return empty('Select recordings from the same insertion')
        for record,(_,data) in page.comparisons:
            indices,h=page._indices_heights(record)
            if not len(indices): continue
            values=np.mean(data['bands'][:3,indices],axis=0) if key=='compare_lfp' else np.median(data['ap_rms'][indices],axis=1)
            for kind,levels,vs in column_series(record,indices,h,values,choices):
                if key=='compare_lfp': vs=10*np.log10(np.maximum(vs,1e-20))
                line_profile(axes[0],vs,levels,page.record_color(record),kind)
        axes[0].set_xlabel('Power · dB (µV²/Hz)' if key=='compare_lfp' else 'RMS · µV')
    return finish_plot(page,axes,spec)


def finish_plot(page,axes,spec):
    from matplotlib.collections import QuadMesh
    meshes=[collection for ax in axes for collection in ax.collections if isinstance(collection,QuadMesh)]
    if meshes:
        low=min(c.get_clim()[0] for c in meshes); high=max(c.get_clim()[1] for c in meshes)
        for collection in meshes: collection.set_clim(low,high)
    for ax in axes:
        for line in ax.lines:
            kind=getattr(line,'_ephys_column',None)
            if kind is not None: line.set_alpha(page.plot_board.opacity(kind))
        for collection in ax.collections:
            kind=getattr(collection,'_ephys_column',None)
            if kind is not None: collection.set_alpha(page.plot_board.opacity(kind))
    if page.plot_board.notebook_mode.isChecked() and spec.key in ('activity','raster','rate','amplitude'):
        for ax in axes: ax.set_xlim(page.time.value(),page.time.value()+page.window_duration.value())
    from .desktop_ephys_settings import apply_visual_settings
    apply_visual_settings(page,axes,spec)
    if spec.depth:
        for ax in axes:
            ax.set_yticks(page.anatomy_ax.get_yticks()); ax.set_ylim(page.plot_board.depth_range())
            if page.anatomy and page.anatomy['depth_um'] is not None: ax.axhline(page.anatomy['depth_um'],c='#e0bf55',ls='--',lw=.8)
            if page.boundary.isChecked(): ax.axhline(page.boundary_value.value(),c='#7ac7e5',lw=.8)
    return axes

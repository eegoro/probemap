"""The notebook's seven views, using the prepared two-second source profile."""
import numpy as np
from matplotlib import colormaps
from .desktop_ephys_plots import COLORS,add_axes,axis_style,line_profile,draw_profile_fits,depth_gradient


def render_notebook(page,figure,key,bounds):
    data=page.notebook; choices=page.plot_board.column_choices()
    groups=[kind for kind in choices if data is not None and kind+'_heights' in data]
    facets=key not in ('lfp','lfp_gradient'); axes=add_axes(figure,bounds,max(1,len(groups)) if facets else 1)
    options=page.plot_board.plot_options(key)
    title={'lfp':'LFP power','lfp_gradient':'Spectral depth gradient','spectrum':'LFP spectrum','correlation':'LFP correlation',
        'correlation_profile':'Correlation profile','traces':'LFP traces · no clusters','traces_clustered':'LFP traces · clusters','low40_plain':'Low-pass traces · no clusters','low40':f'Low-pass traces · {page.plot_board.analysis_fields["lowpass"].value():g} Hz','csd':'Current source density'}[key]
    for ax in axes:
        axis_style(ax); ax.set_title(title,fontsize=10,pad=12)
        ax.set_ylabel('Depth from tip · µm'); ax.set_ylim(page.plot_board.depth_range())
    if data is None:
        axes[0].text(.5,.5,'Update analysis to view\nthis time interval',ha='center',va='center',color='#9aafc1',transform=axes[0].transAxes)
        return axes
    for number,kind in enumerate(groups):
        ax=axes[number] if facets else axes[0]; line_start=len(ax.lines)
        def get(name): return data[kind+'_'+name]
        h=get('heights'); boundary=get('boundaries'); finite=boundary[np.isfinite(boundary)]
        if facets: ax.set_title(title+'\n'+page.plot_board.column_label(kind),fontsize=9,pad=9)
        if key in ('lfp','lfp_gradient'):
            for band,style in enumerate(options['bands']):
                if not style['visible'] or (key=='lfp' and band>=len(get('bands'))): continue
                color=style['color']
                if key=='lfp_gradient':
                    from .ephys_profiles import band_power
                    powers=band_power(dict(frequencies=get('frequencies'),power=get('power')),options['bands'])[band]
                else: powers=get('bands')[band]
                values=10*np.log10(np.maximum(powers,np.finfo(float).tiny))
                if key=='lfp_gradient': values=depth_gradient(values,h)
                if key!='lfp' or options['raw']:
                    line=line_profile(ax,values,h,color,kind); line._ephys_band=band
            if key=='lfp':
                edge=get('band_breaks') if kind+'_band_breaks' in data else [np.array([])]*len(get('bands'))
                draw_profile_fits(page,ax,{kind:dict(heights=h,band_fit=get('band_fit'),band_breaks=edge)},options)
            ax.set_xlabel('Gradient · dB / µm' if key=='lfp_gradient' else 'Power · dB (µV²/Hz)')
        elif key=='spectrum':
            freq=get('frequencies'); keep=freq>=0
            levels,power=page._depth_grid(h,get('power'))
            if len(levels)>1:
                ax.pcolormesh(freq[keep],levels,10*np.log10(np.maximum(power[:,keep],np.finfo(float).tiny)),shading='nearest',cmap='magma',rasterized=True)
            if not options['auto_range']: ax.set_xlim(options['xmin'],options['xmax'])
            ax.set_xlabel('Frequency · Hz')
        elif key=='correlation':
            if len(h)>1: ax.pcolormesh(h,h,get('correlation'),shading='nearest',cmap='RdBu_r',vmin=-.6,vmax=1,rasterized=True)
            ax.set_xlabel('Depth from tip · µm')
        elif key=='correlation_profile':
            if options['raw']:
                ax.plot(get('correlation_profile'),h,color=options['color'] or COLORS[0],lw=.85)
                scatter=ax.scatter(get('correlation_profile'),h,color=options['color'] or COLORS[0],s=3,linewidths=0); scatter._ephys_column=kind
            if options['fits']: ax.plot(get('correlation_fit'),h,color=COLORS[1],lw=1.2)
            if options['transitions']:
                for y in get('correlation_breaks')[1:-1]:
                    line=ax.axhline(y,c=COLORS[1],lw=.5); line._ephys_role='transition'
            ax.set_xlabel('Mean |Pearson r|'); ax.set_xlim(0,1)
        else:
            slow=key in ('low40','low40_plain','csd'); wave=get('low40' if slow else 'lfp')
            fs=float(get('low40_fs')) if slow else float(data['fs'])
            time=data['low40_times'] if slow else (data['times']-data['times'][0])*1000
            if key=='csd' and len(h)>1:
                csd=get('csd'); vmax=max(1e-12,float(np.percentile(np.abs(csd),95)))
                ax.pcolormesh(time,h,csd,shading='nearest',cmap='bwr',vmin=-vmax,vmax=vmax,alpha=.6,rasterized=True)
            clustered=key in ('traces_clustered','low40')
            labels=get('low40_clusters' if slow else 'clusters') if clustered else None
            for row,(height,trace) in enumerate(zip(h,wave)):
                ax.plot(time,height+trace*options['trace_scale'],color='black' if key=='csd' else colormaps['tab10'](int(labels[row])%10) if labels is not None else COLORS[0],lw=.4)
            ax.set_xlabel('Time · ms')
        for line in ax.lines[line_start:]: line._ephys_column=kind
        if len(finite): ax.axhline(np.mean(finite),color='#e0bf55',lw=.7)
    return axes

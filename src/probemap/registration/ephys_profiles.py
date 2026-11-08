"""Depth-profile helpers; no GUI and no original-recording reads."""
import numpy as np
from .ephys_analysis import check_cancel
from .ephys_notebook import fit_profile


def band_power(data,bands):
    freq=np.asarray(data['frequencies']); power=np.asarray(data['power'])
    return np.asarray([np.mean(power[:,(freq>=b['low'])&(freq<=b['high'])],axis=1)
        if np.any((freq>=b['low'])&(freq<=b['high'])) else np.full(len(power),np.nan) for b in bands])


def fit_overview(record,indices,heights,data,bands,segments,progress,cancel):
    power=band_power(data,bands)[:,indices]; columns=np.array([record['channels'][i]['column'] for i in indices]); result={}
    for step,kind in enumerate(('0','1','mean')):
        mask=np.ones(len(indices),bool) if kind=='mean' else columns==int(kind)
        if not np.any(mask): continue
        levels=np.unique(heights[mask]); curves=np.asarray([power[:,mask][:,heights[mask]==h].mean(axis=1) for h in levels]).T
        fits=[]; breaks=[]; candidates=[]
        for number,curve in enumerate(curves):
            check_cancel(cancel); progress(100*(step*len(bands)+number)/(3*len(bands)),'Fitting depth profiles')
            values=10*np.log10(np.maximum(curve,1e-20))
            if np.all(np.isfinite(values)):
                edge,fit=fit_profile(levels,values,segments)
            else: edge=np.array([]); fit=np.full(len(levels),np.nan)
            fits.append(fit); breaks.append(edge); candidates.append(edge[-2] if len(edge)>2 else np.nan)
        result[kind]=dict(heights=levels,band_fit=np.asarray(fits),band_breaks=breaks,boundaries=np.asarray(candidates))
    return result

"""Bounded source reads reproducing the supplied notebook's two-second workflow.

FFT resampling is the CPU equivalent of neuropix's real-input torch FFT routine.
No notebook cells or user-supplied Python files are executed.
"""
from pathlib import Path
import hashlib
import json
import tempfile
import time
from zipfile import BadZipFile
import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter,gaussian_filter1d
from .ephys_analysis import BANDS,check_cancel

from collections import OrderedDict
from threading import RLock

# Keep recent small intervals, not entire recordings. Time navigation must not
# repeat filtering/fitting when the user returns to the same parameters.
_WINDOW_LIMIT = 64 * 1024**2
_WINDOW_MEMORY = OrderedDict()
_WINDOW_LOCK = RLock()


def _remember_window(key, values):
    size=sum(v.nbytes for v in values.values() if isinstance(v,np.ndarray))
    if size > _WINDOW_LIMIT: return values
    with _WINDOW_LOCK:
        _WINDOW_MEMORY.pop(key,None)
        while _WINDOW_MEMORY and sum(item[0] for item in _WINDOW_MEMORY.values())+size > _WINDOW_LIMIT:
            _WINDOW_MEMORY.popitem(last=False)
        # Cap metadata/zero-length results as well as array bytes.
        while len(_WINDOW_MEMORY) >= 8: _WINDOW_MEMORY.popitem(last=False)
        for value in values.values():
            if isinstance(value,np.ndarray): value.flags.writeable=False
        _WINDOW_MEMORY[key]=(size,dict(values))
    return values


def _recalled_window(key):
    with _WINDOW_LOCK:
        item=_WINDOW_MEMORY.get(key)
        if item is None: return None
        _WINDOW_MEMORY.move_to_end(key)
        return dict(item[1])


VERSION=2
FIT_OPTIONS=dict(seed=7,maxiter=200,popsize=8,tol=1e-4)


def c_factor(data,sigma=3.,common_reference=False):
    centered=np.asarray(data,float)-np.median(data,axis=1,keepdims=True)
    if common_reference: centered-=np.median(centered,axis=0,keepdims=True)
    expected=gaussian_filter1d(centered,sigma,axis=0,mode='nearest') if sigma>0 else centered
    ratios=np.divide(expected,centered,out=np.ones_like(centered),where=centered!=0)
    factor=np.median(ratios,axis=1); factor[~np.isfinite(factor)]=1
    return np.asarray(centered*factor[:,None],np.float32),factor


def metrics(data,fs,bands=BANDS,nperseg=256,nfft=4096):
    # Notebook intentionally leaves nperseg at SciPy's default 256, not nfft.
    freq,power=signal.welch(data,fs=int(fs),nfft=nfft,nperseg=min(nperseg,data.shape[1],nfft),axis=1)
    for lo,hi in bands:
        if not np.any((freq>=lo)&(freq<=hi)): raise ValueError(f'No frequencies in {lo:g}–{hi:g} Hz. Widen the band or increase FFT length.')
    bands=np.array([np.mean(power[:,(freq>=lo)&(freq<=hi)],axis=1) for lo,hi in bands])
    with np.errstate(invalid='ignore',divide='ignore'): corr=np.corrcoef(data)
    return freq,power,bands,np.nan_to_num(np.atleast_2d(corr),nan=0,posinf=0,neginf=0)


def clusters(corr):
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.neighbors import kneighbors_graph
    n=len(corr)
    if n<2: return np.zeros(n,int)
    connection=kneighbors_graph(np.arange(n)[:,None],n_neighbors=min(6,n-1),mode='connectivity')
    return AgglomerativeClustering(n_clusters=min(10,n),connectivity=connection,linkage='ward').fit_predict(corr)


def fit_profile(heights,values,segments=6):
    import pwlf
    heights=np.asarray(heights); values=np.asarray(values)
    if len(heights)<8 or np.ptp(values)<1e-10:
        return np.array([]),np.full_like(values,np.nan,dtype=float)
    model=pwlf.PiecewiseLinFit(heights,values)
    # Six segments as in the notebook; reduce only for short/sparse test maps.
    breaks=model.fit(min(segments,len(heights)//2-1),**FIT_OPTIONS)
    return breaks,model.predict(heights)


def source_window(record,start,duration,indices,folder,progress,cancel):
    """Reconstruct only notebook batches overlapping the requested output interval.

    Preserves its 1,048,576 sample batches, 1024 overlap, rounded FFT lengths,
    integer crop indices and effective output sampling rate/time linspace.
    """
    fs=record['sampling_rate']; total=record['n_samples']; batch=1024**2
    plan=[]; output=0
    for begin in range(0,total,batch):
        end=min(begin+batch,total); left=max(0,begin-1024); right=min(total,end+1024)
        size=right-left; reduced=round(size/fs*1000)
        a=int((begin-left)*reduced/size); b=int((end-left)*reduced/size)
        plan.append((left,right,reduced,a,b,output)); output+=b-a
    new_fs=output/(total/fs); first=int(start*new_fs); last=min(output,first+int(duration*new_fs))
    if last-first<256: raise ValueError('Choose a window with at least 256 LFP samples.')
    needed=[p for p in plan if p[5]<last and p[5]+p[4]-p[3]>first]
    n=len(indices); result=np.empty((n,last-first),np.float64)
    file_cols=[record['channels'][i]['file_index'] for i in indices]
    count=record['n_saved_channels']; to_read=sum((p[1]-p[0])*count*2 for p in needed); read_bytes=0
    notch=signal.iirnotch(50/(fs/2),30); low=signal.butter(2,300/(fs/2),btype='low')
    seconds_read=0.
    progress(0,f'Original-source filters · reading {to_read/1024**2:.0f} MiB from source')
    # Stage outside an iCloud/network project folder: these large disposable
    # blocks must not trigger cloud uploads while reading the server.
    from ..platform_io import close_memmap
    with tempfile.TemporaryDirectory(prefix='probemap-notebook-') as temporary:
        for batch_number,(left,right,reduced,a,b,offset) in enumerate(needed):
            check_cancel(cancel)
            local=np.memmap(Path(temporary)/'batch.bin',mode='w+',dtype='<i2',shape=(right-left,n))
            try:
                with open(record['bin_path'],'rb',buffering=0) as stream:
                    stream.seek(left*count*2)
                    block=max(1,4*1024**2//(count*2))
                    for pos in range(left,right,block):
                        check_cancel(cancel); length=min(block,right-pos)
                        raw=np.empty((length,count),dtype='<i2'); dest=memoryview(raw).cast('B'); done=0; tick=time.perf_counter()
                        while done<len(dest):
                            check_cancel(cancel); amount=stream.readinto(dest[done:])
                            if not amount: raise ValueError('Recording binary is truncated.')
                            done+=amount; read_bytes+=amount
                        seconds_read+=time.perf_counter()-tick
                        local[pos-left:pos-left+length]=raw[:,file_cols]
                        progress(70/len(needed)*(batch_number+.78*(pos-left+length)/(right-left)),f'Reading original source · {read_bytes/1024**2:.0f}/{to_read/1024**2:.0f} MiB · {read_bytes/max(seconds_read,1e-9)/1024**2:.1f} MiB/s')
                take0=max(first,offset); take1=min(last,offset+b-a)
                for ch in range(0,n,8):
                    check_cancel(cancel)
                    raw=np.asarray(local[:,ch:ch+8].T,dtype=float)*record['uv_per_count']
                    filtered=signal.filtfilt(*low,signal.filtfilt(*notch,raw,axis=1),axis=1)
                    reduced_data=signal.resample(filtered,reduced,axis=1)
                    result[ch:ch+8,take0-first:take1-first]=reduced_data[:,a+take0-offset:a+take1-offset]
                    progress(70/len(needed)*(batch_number+.78+.22*min(n,ch+8)/max(1,n)),'Filtering source window')
            finally:
                close_memmap(local)
            del local
    times=np.arange(first,last)*(total-1)/fs/max(1,output-1)
    return result,new_fs,times


def low40_window(record,start,duration,indices,progress,cancel,cutoff=40.):
    fs=record['sampling_rate']; first=max(0,int(start*fs)); last=min(record['n_samples'],int((start+duration)*fs))
    count=record['n_saved_channels']; length=last-first
    with open(record['bin_path'],'rb',buffering=0) as stream:
        stream.seek(first*count*2)
        raw=np.empty((length,count),dtype='<i2'); dest=memoryview(raw).cast('B'); done=0
        while done<len(dest):
            check_cancel(cancel); amount=stream.readinto(dest[done:done+4*1024**2])
            if not amount: raise ValueError('Recording binary is truncated.')
            done+=amount; progress(71+4*done/len(dest),'Reading low-pass 40 Hz window')
    data=np.asarray(raw[:,[record['channels'][i]['file_index'] for i in indices]].T,float)*record['uv_per_count']
    notch=signal.iirnotch(50/(fs/2),30)
    data=signal.resample(signal.filtfilt(*notch,data,axis=1),int(length/fs*1000),axis=1)
    # Notebook uses the effective sampling rate returned by apply_preprocess.
    effective=data.shape[1]/(length/fs)
    return signal.filtfilt(*signal.butter(2,cutoff/(effective/2),btype='low'),data,axis=1),effective


def cached_window(record,start,duration,indices,fs,progress,cancel,cutoff=40.):
    """Read a few seconds from the local LFP, never reopen the source binary."""
    check_cancel(cancel); progress(5,'Reading prepared LFP window')
    path=Path(record['cache_path'])/'lfp.npy'
    lfp=np.load(path,mmap_mode='r',allow_pickle=False)
    first=int(start*fs); last=min(lfp.shape[1],first+int(duration*fs))
    if last-first<256: raise ValueError('Choose a window with at least 256 LFP samples.')
    left=max(0,first-round(fs)); right=min(lfp.shape[1],last+round(fs))
    # Slice time before copying channels; never materialize the whole recording.
    values=np.asarray(lfp[np.asarray(indices),left:right],dtype=float)
    del lfp
    check_cancel(cancel); progress(35,'Filtering the prepared window · 40 Hz')
    slow=signal.filtfilt(*signal.butter(2,cutoff/(fs/2),btype='low'),values,axis=1)
    core=slice(first-left,last-left)
    return values[:,core],fs,np.arange(first,last)/fs,slow[:,core],fs


def _prepare_window(record,shank,indices,heights,start,source_key,progress=lambda *args:None,cancel=lambda:False,allow_source=True,mode='original',lfp_fs=None,options=None):
    options=options or {}; duration=options.get('duration',2); bands=options.get('bands',BANDS)
    started=time.perf_counter()
    folder=Path(record['cache_path']); target=folder/('cached_window.npz' if mode=='cached' else 'notebook_window.npz')
    config=dict(version=VERSION,source=source_key,shank=int(shank),indices=list(map(int,indices)),heights=list(map(float,heights)),start=float(start),duration=duration,fit=FIT_OPTIONS)
    if options: config['analysis']=options
    if mode=='cached':
        if lfp_fs is None:
            lfp_fs=float(json.loads((folder/'manifest.json').read_text(encoding='utf-8'))['sampling_rate'])
        stat=(folder/'lfp.npy').stat()
        config.update(mode='cached-lfp-v1',lfp_fs=float(lfp_fs),lfp_signature=[stat.st_size,stat.st_mtime_ns])
    elif mode!='original': raise ValueError('Unknown profile signal source.')
    key=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    recalled=_recalled_window(key)
    if recalled is not None:
        progress(100,'Using recent interval'); return recalled
    if target.exists():
        try:
            with np.load(target,allow_pickle=False) as saved:
                if str(saved['key'])==key and all(k in saved for k in ('times','fs','start','indices')):
                    progress(100,'Using cached window profile'); return _remember_window(key,{k:saved[k] for k in saved.files})
        except (OSError,ValueError,KeyError,EOFError,BadZipFile): pass
    if not allow_source: return None
    indices=np.asarray(indices); heights=np.asarray(heights)
    read_started=time.perf_counter()
    if mode=='cached':
        lfp,fs,times,low40,lowfs=cached_window(record,start,duration,indices,lfp_fs,progress,cancel,cutoff=options.get('lowpass',40.))
    else:
        lfp,fs,times=source_window(record,start,duration,indices,folder,progress,cancel)
        low40,lowfs=low40_window(record,float(times[0]),duration,indices,progress,cancel,cutoff=options.get('lowpass',40.))
    signal_seconds=time.perf_counter()-read_started
    rawfs=record['sampling_rate']; low_start=float(times[0])
    low_count=min(record['n_samples'],int((low_start+duration)*rawfs))-max(0,int(low_start*rawfs))
    out=dict(key=np.array(key),start=np.array(start),times=times,fs=np.array(fs),indices=indices,
        low40_times=((times-times[0])*1000 if mode=='cached' else np.linspace(0,(low_count-1)/rawfs,low40.shape[1])*1000),
        source_mode=np.array(mode),band_edges=np.asarray(bands),duration=np.array(duration))
    columns=np.array([record['channels'][i]['column'] for i in indices])
    for number,kind in enumerate(('0','1','mean')):
        check_cancel(cancel); mask=np.ones(len(indices),bool) if kind=='mean' else columns==int(kind)
        if not mask.any(): continue
        h=heights[mask]; levels,inverse=np.unique(h,return_inverse=True)
        def group(data):
            v=data[mask]; return np.stack([v[inverse==i].mean(axis=0) for i in range(len(levels))])
        # Correct each physical column independently; Mean is corrected mean voltage.
        def correct(data):
            if options.get('c_factor',True): return c_factor(data,options.get('sigma',3.),options.get('reference',False))
            centered=data-np.median(data,axis=1,keepdims=True)
            if options.get('reference',False): centered-=np.median(centered,axis=0,keepdims=True)
            return centered.astype(np.float32),np.ones(len(centered))
        corrected,factor=correct(group(lfp)); slow,_=correct(group(low40))
        freq,power,band_power,corr=metrics(corrected,fs,bands=bands,nperseg=options.get('nperseg',256),nfft=options.get('nfft',4096))
        values=dict(heights=levels,lfp=corrected,c_factor=factor,frequencies=freq,power=power,bands=band_power,correlation=corr,
            clusters=clusters(corr),low40=slow,low40_fs=np.array(lowfs),low40_clusters=clusters(metrics(slow,lowfs)[3]),
            csd=-np.gradient(np.gradient(gaussian_filter(slow,[5,5]),axis=0),axis=0) if len(slow)>1 else np.zeros_like(slow))
        progress(76+8*number,'Fitting depth profiles')
        candidates=[]; fits=[]; breakpoints=[]
        for band in band_power:
            check_cancel(cancel)
            curve=10*np.log10(np.maximum(band,np.finfo(float).tiny))
            breaks,fit=fit_profile(levels,curve) if options.get('segments',6)==6 else fit_profile(levels,curve,options['segments'])
            candidates.append(breaks[-2] if len(breaks)>2 else np.nan); fits.append(fit)
            breakpoints.append(np.pad(breaks,(0,max(0,options.get('segments',6)+1-len(breaks))),constant_values=np.nan))
        profile=np.mean(np.abs(corr),axis=0)
        breaks,fit=fit_profile(levels,profile) if options.get('segments',6)==6 else fit_profile(levels,profile,options['segments'])
        values.update(band_fit=fits,band_breaks=breakpoints,boundaries=candidates,correlation_profile=profile,correlation_breaks=breaks,correlation_fit=fit)
        out.update({kind+'_'+k:np.asarray(v) for k,v in values.items()})
        progress(76+8*(number+1),'Fitting depth profiles')
    out['timings_s']=np.array(json.dumps(dict(signal=signal_seconds,metrics_and_fits=time.perf_counter()-read_started-signal_seconds,total=time.perf_counter()-started)))
    temporary=target.with_name('.'+target.stem+'.partial.npz')
    try:
        check_cancel(cancel); np.savez_compressed(temporary,**out); check_cancel(cancel); temporary.replace(target)
    finally: temporary.unlink(missing_ok=True)
    progress(100,'Window profile ready'); return _remember_window(key,out)


def prepare_window(record,*args,**kwargs):
    # Share the signal writer lock: two app windows must never replace each
    # other's derived profile or remove a source staging file mid-analysis.
    import shutil
    from ..platform_io import exclusive_file_lock
    folder=Path(record['cache_path']); root=folder.parent if folder.name=='signal' else folder
    if kwargs.get('allow_source',True) is False: return _prepare_window(record,*args,**kwargs)
    with exclusive_file_lock(root/'.prepare.lock'):
        for stale in folder.glob('.notebook-read-*'):
            if stale.is_dir() and not stale.is_symlink(): shutil.rmtree(stale)
        return _prepare_window(record,*args,**kwargs)

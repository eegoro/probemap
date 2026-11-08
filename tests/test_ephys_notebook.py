"""Numerical comparisons with pure functions from the user's actual notebook."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
from scipy import signal
from probemap.registration.ephys_notebook import c_factor,metrics,clusters,source_window,low40_window,fit_profile,FIT_OPTIONS,prepare_window

spec=importlib.util.spec_from_file_location('reference',Path(__file__).parent/'fixtures/notebook_profiling_reference.py')
reference=importlib.util.module_from_spec(spec); spec.loader.exec_module(reference)


def test_actual_notebook_c_factor_welch_bands_correlation_and_clusters():
    rng=np.random.default_rng(213); fs=999.98; time=np.arange(2000)/fs
    data=rng.normal(size=(32,2000))+np.sin(time[None,:]*np.linspace(3,80,32)[:,None]*2*np.pi)*4+np.arange(32)[:,None]
    expected,factor=reference.apply_c_factor(data); actual,actual_factor=c_factor(data)
    np.testing.assert_array_equal(actual,expected); np.testing.assert_array_equal(actual_factor,factor)
    freq,power,bands,corr=metrics(actual,fs)
    expected_freq,expected_power,expected_bands=reference.compute_power_metrics(expected,fs)
    np.testing.assert_array_equal(freq,expected_freq)
    np.testing.assert_allclose(10*np.log10(power),expected_power,atol=1e-5)
    np.testing.assert_allclose(10*np.log10(bands),expected_bands,atol=1e-5)
    np.testing.assert_array_equal(corr,reference.compute_correlation(expected))
    np.testing.assert_array_equal(clusters(corr),reference.cluster_lfp_by_correlation(corr))
    flat,_=c_factor(np.zeros((3,2000))); assert np.isfinite(flat).all()


def source(tmp_path,seconds=36):
    fs=30000.282338; count=4; n=int(fs*seconds)
    t=np.arange(n)/fs
    raw=np.stack([np.sin(2*np.pi*f*t)*100 for f in (10,30,70,50)],axis=1).astype('<i2')
    path=tmp_path/'test.bin'; raw.tofile(path)
    rec=dict(bin_path=str(path),sampling_rate=fs,n_samples=n,n_saved_channels=count,uv_per_count=3.,
             channels=[dict(file_index=i,column=i%2) for i in range(count)])
    return rec,raw


def test_source_window_matches_full_notebook_batches_across_boundary(tmp_path):
    record,raw=source(tmp_path); fs=record['sampling_rate']; selected=[0,2]
    # Independent transcription of neuropix.pipeline._process_by_batch and its FFT resampler.
    full=raw[:,selected].T*record['uv_per_count']; pieces=[]
    for begin in range(0,len(raw),1024**2):
        end=min(len(raw),begin+1024**2); left=max(0,begin-1024); right=min(len(raw),end+1024)
        batch=full[:,left:right].copy()
        batch=signal.filtfilt(*signal.iirnotch(50/(fs/2),30),batch,axis=1)
        batch=signal.filtfilt(*signal.butter(2,300/(fs/2),btype='low'),batch,axis=1)
        # Real FFT equivalent to the notebook's torch implementation, including Nyquist scaling.
        n=round((right-left)/fs*1000); spectrum=np.fft.rfft(batch,axis=1)
        truncated=spectrum[:,:n//2+1].copy()
        if n%2==0: truncated[:,-1]*=2
        reduced=np.fft.irfft(truncated,n=n,axis=1)*(n/(right-left))
        a=int((begin-left)*n/(right-left)); b=int((end-left)*n/(right-left))
        pieces.append(reduced[:,a:b])
    reference_lfp=np.concatenate(pieces,axis=1); effective=reference_lfp.shape[1]/(len(raw)/fs)
    actual,newfs,times=source_window(record,34,2,selected,tmp_path,lambda *args:None,lambda:False)
    first=int(34*effective); last=min(reference_lfp.shape[1],first+int(2*effective))
    np.testing.assert_allclose(actual,reference_lfp[:,first:last],atol=1e-10)
    assert newfs==effective
    np.testing.assert_allclose(times,np.linspace(0,(len(raw)-1)/fs,reference_lfp.shape[1])[first:last],atol=1e-12)
    assert not list(tmp_path.glob('.notebook-read-*'))


def test_low40_source_pipeline_and_pwlf_model(tmp_path):
    record,raw=source(tmp_path,5); fs=record['sampling_rate']; start=.123456; first=int(start*fs); last=int((start+2)*fs)
    selected=[0,2]; data=raw[first:last,selected].T*record['uv_per_count']
    data=signal.filtfilt(*signal.iirnotch(50/(fs/2),30),data,axis=1)
    data=signal.resample(data,int((last-first)/fs*1000),axis=1); effective=data.shape[1]/((last-first)/fs)
    expected=signal.filtfilt(*signal.butter(2,40/(effective/2),btype='low'),data,axis=1)
    actual,actual_fs=low40_window(record,start,2,selected,lambda *args:None,lambda:False)
    np.testing.assert_array_equal(actual,expected); assert actual_fs==effective
    import pwlf
    h=np.linspace(217,3000,24); values=np.sin(h/500)+h/1000
    breaks,fitted=fit_profile(h,values)
    model=pwlf.PiecewiseLinFit(h,values); expected=model.fit(6,**FIT_OPTIONS)
    np.testing.assert_array_equal(breaks,expected); np.testing.assert_array_equal(fitted,model.predict(h))


def test_window_cache_reuses_without_source_and_preserves_cache_on_cancel(tmp_path,monkeypatch):
    record,raw=source(tmp_path,5); record['cache_path']=str(tmp_path)
    # Keep this test focused on cache and CSD; optimizer is independently verified above.
    import probemap.registration.ephys_notebook as module
    monkeypatch.setattr(module,'fit_profile',lambda h,v:(np.array([]),np.full_like(v,np.nan)))
    result=prepare_window(record,0,np.arange(4),np.array([217,217,232,232]),0,'test-source')
    assert all(k in result for k in ('0_power','1_low40','mean_csd','mean_clusters'))
    stamp=(tmp_path/'notebook_window.npz').stat().st_mtime_ns
    Path(record['bin_path']).unlink()
    cached=prepare_window(record,0,np.arange(4),np.array([217,217,232,232]),0,'test-source')
    np.testing.assert_array_equal(cached['0_power'],result['0_power'])
    assert (tmp_path/'notebook_window.npz').stat().st_mtime_ns==stamp
    assert prepare_window(record,0,np.arange(4),np.array([217,217,232,232]),1,'test-source',allow_source=False) is None

    with pytest.raises(InterruptedError):
        prepare_window(record,0,np.arange(4),np.array([217,217,232,232]),1,'test-source',cancel=lambda:True)
    assert (tmp_path/'notebook_window.npz').stat().st_mtime_ns==stamp
    assert not list(tmp_path.glob('.notebook-read-*'))


def test_fast_profile_reads_only_prepared_lfp_and_preserves_original_result(tmp_path,monkeypatch):
    import probemap.registration.ephys_notebook as module
    rng=np.random.default_rng(41); fs=999.98; saved=rng.normal(size=(16,6000)).astype(np.float32)
    np.save(tmp_path/'lfp.npy',saved)
    (tmp_path/'manifest.json').write_text('{"sampling_rate":999.98}')
    original=tmp_path/'notebook_window.npz'; original.write_bytes(b'previous exact notebook result')
    rec=dict(cache_path=str(tmp_path),bin_path='/unavailable/server.bin',n_samples=180000,
        sampling_rate=30000.,channels=[dict(column=i%2) for i in range(16)])
    def unavailable(*args,**kwargs): raise AssertionError('Fast mode must not read the source binary')
    monkeypatch.setattr(module,'source_window',unavailable); monkeypatch.setattr(module,'low40_window',unavailable)
    monkeypatch.setattr(module,'fit_profile',lambda h,v:(np.array([]),np.full_like(v,np.nan)))
    indices=np.arange(16); heights=217+np.arange(16)//2*15
    start=2.
    lfp,actual_fs,times,slow,slow_fs=module.cached_window(rec,start,2,indices,fs,lambda *args:None,lambda:False)
    first=int(start*fs); last=first+int(2*fs); left=first-round(fs); right=last+round(fs)
    np.testing.assert_array_equal(lfp,saved[:,first:last]); assert actual_fs==slow_fs==fs
    expected=signal.filtfilt(*signal.butter(2,40/(fs/2),btype='low'),saved[:,left:right].astype(float),axis=1)
    np.testing.assert_array_equal(slow,expected[:,first-left:last-left])
    result=module.prepare_window(rec,0,indices,heights,start,'source',mode='cached',lfp_fs=fs)
    assert str(result['source_mode'])=='cached' and 'timings_s' in result
    assert original.read_bytes()==b'previous exact notebook result'
    stamp=(tmp_path/'cached_window.npz').stat().st_mtime_ns
    monkeypatch.setattr(module,'cached_window',unavailable)
    reopened=module.prepare_window(rec,0,indices,heights,start,'source',mode='cached',lfp_fs=fs)
    np.testing.assert_array_equal(reopened['0_power'],result['0_power'])
    assert (tmp_path/'cached_window.npz').stat().st_mtime_ns==stamp
    assert module.prepare_window(rec,0,indices,heights,start+1,'source',mode='cached',lfp_fs=fs,allow_source=False) is None
    # Preparing a new base signal must invalidate the derived fast profile.
    import os
    base=tmp_path/'lfp.npy'; stat=base.stat(); os.utime(base,ns=(stat.st_atime_ns,stat.st_mtime_ns+1_000_000_000))
    assert module.prepare_window(rec,0,indices,heights,start,'source',mode='cached',lfp_fs=fs,allow_source=False) is None


def test_custom_interval_fft_bands_and_reference_are_applied_to_cached_signal(tmp_path,monkeypatch):
    import probemap.registration.ephys_notebook as module
    rng=np.random.default_rng(51); fs=1000.; saved=rng.normal(size=(16,6000)).astype(np.float32)
    np.save(tmp_path/'lfp.npy',saved)
    rec=dict(cache_path=str(tmp_path),bin_path='/unavailable/server.bin',n_samples=180000,sampling_rate=30000.,channels=[dict(column=i%2) for i in range(16)])
    monkeypatch.setattr(module,'fit_profile',lambda h,v,segments=6:(np.linspace(min(h),max(h),segments+1),np.asarray(v)))
    options=dict(duration=1,bands=[[2,8],[20,40]],nperseg=512,nfft=2048,c_factor=False,reference=True,sigma=3,segments=4,lowpass=30)
    result=module.prepare_window(rec,0,np.arange(16),217+np.arange(16)//2*15,2.,'source',mode='cached',lfp_fs=fs,options=options)
    raw=saved[::2,2000:3000].astype(float); expected=raw-np.median(raw,axis=1,keepdims=True); expected-=np.median(expected,axis=0,keepdims=True)
    np.testing.assert_array_equal(result['0_lfp'],expected.astype(np.float32))
    freq,power=signal.welch(expected.astype(np.float32),fs=1000,nperseg=512,nfft=2048,axis=1)
    np.testing.assert_allclose(result['0_power'],power); assert len(result['0_frequencies'])==1025
    assert result['0_bands'].shape==(2,8) and result['0_band_breaks'].shape==(2,5)
    np.testing.assert_array_equal(result['band_edges'],options['bands'])
    changed=dict(options,nfft=4096)
    assert module.prepare_window(rec,0,np.arange(16),217+np.arange(16)//2*15,2.,'source',mode='cached',lfp_fs=fs,options=changed,allow_source=False) is None


def test_depth_gradient_preserves_unsampled_gaps():
    from probemap.registration.desktop_ephys_plots import depth_gradient
    np.testing.assert_allclose(depth_gradient(np.array([0,30,60,120,150]),np.array([217,232,247,400,415])),2)
    assert np.isnan(depth_gradient([1],[217])[0])

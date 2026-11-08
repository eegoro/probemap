"""Pure reference functions extracted from electrophysiological_profiling.ipynb.
No notebook loading, file reads, plotting or hardcoded recording paths.
Notebook SHA256: d6eb004904abe06a73d7fd37c10a3f1c09ec1357bd5ae055479188a5c5381708
"""
import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
APPLY_COMMON_MEDIAN_REFERENCE=False
C_FACTOR_SIGMA_CHANNELS=3
C_FACTOR_CLIP=None
N_LFP_TRACE_CLUSTERS=10
NFFT=4096
FREQUENCY_BANDS=[(1.5,4),(4,10),(10,30),(30,80),(80,200)]


def apply_c_factor(data):
    centered = data - np.median(data, axis=1, keepdims=True)
    if APPLY_COMMON_MEDIAN_REFERENCE:
        centered = centered - np.median(centered, axis=0, keepdims=True)
    expected = gaussian_filter1d(centered, sigma=C_FACTOR_SIGMA_CHANNELS, axis=0, mode='nearest')
    c_samples = np.divide(expected, centered, out=np.ones_like(centered), where=centered != 0)
    c_factor = np.median(c_samples, axis=1)
    c_factor = np.where(np.isfinite(c_factor), c_factor, 1.0)
    if C_FACTOR_CLIP is not None:
        c_factor = np.clip(c_factor, C_FACTOR_CLIP[0], C_FACTOR_CLIP[1])
    corrected = centered * c_factor[:, None]
    return (corrected.astype(np.float32, copy=False), c_factor)


def cluster_lfp_by_correlation(corr_matrix, n_clusters=N_LFP_TRACE_CLUSTERS, max_gap=3):
    indices = np.arange(corr_matrix.shape[0]).reshape(-1, 1)
    connectivity = kneighbors_graph(indices, n_neighbors=2 * max_gap, mode='connectivity')
    clustering = AgglomerativeClustering(n_clusters=n_clusters, connectivity=connectivity, linkage='ward')
    return clustering.fit_predict(corr_matrix)


def compute_power_metrics(data, fs):
    freqs, power = signal.welch(data, fs=int(fs), nfft=NFFT, axis=1)
    power_db = 10.0 * np.log10(np.maximum(power, np.finfo(float).tiny))
    band_power = []
    for low, high in FREQUENCY_BANDS:
        idx = (freqs >= low) & (freqs <= high)
        if not np.any(idx):
            raise ValueError(f'No Welch frequencies inside band {low}-{high} Hz')
        band_power.append(10.0 * np.log10(np.maximum(np.mean(power[:, idx], axis=1), np.finfo(float).tiny)))
    return (freqs, power_db, band_power)


def compute_correlation(data):
    corr = np.corrcoef(data)
    return np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

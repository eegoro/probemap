# Ephys calculation methods

This note records the comparison with `electrophysiological_profiling.ipynb`
and its `neuropix` helpers, reviewed on 2 November 2026. ProbeMap does not run
the notebook. Reference functions used by tests are in
[`tests/fixtures/notebook_profiling_reference.py`](../../tests/fixtures/notebook_profiling_reference.py).

## Analysis modes

| Mode | Input and processing |
| --- | --- |
| Whole-record overview | Prepared LFP: 50 Hz notch, fourth-order 300 Hz low-pass, polyphase resampling to about 1 kHz. Welch uses `nperseg=nfft=4096` in non-overlapping 20 s windows. No C-factor. |
| Selected interval · Prepared LFP | Read the interval plus 1 s of filter context on each side from `lfp.npy`. Apply interval correction and metrics to the prepared signal. |
| Selected interval · Original recording | Read raw batches and apply the notebook's filtering and FFT resampling before calculating interval metrics. |

The prepared-signal modes use different preprocessing from the notebook.
AP RMS is calculated separately with a 300 Hz high-pass filter in 2 s bins.

In the UI, use **Selected interval** and choose the source under
**Plots → Signal processing**. The source names are **Prepared LFP · fast** and
**Original recording · reference filters**. Use **Update analysis** when needed.
Window duration and processing settings can be changed.

## Compare with the notebook

Set the start to **99 s** and duration to **2 s**. This was the notebook's final
executed interval; an earlier cell used 100 s. Select the same shank and one
electrode column, such as X = 27 µm. Use the original-recording source for a
comparison of the source filters.

A 2 s result can require 35–70 s of raw input because FFT resampling uses larger
batches. In the checked recording this meant about 0.8–1.6 GB of source data.

## Original-recording defaults

1. Read overlapping batches of 1,048,576 input samples with 1024-sample overlaps.
   Keep the notebook's rounded output lengths, crop indices, and effective sample rate.
2. Apply a 50 Hz notch (`Q=30`, `filtfilt`), a second-order Butterworth 300 Hz
   low-pass (`filtfilt`), and real FFT resampling to about 1 kHz.
3. Subtract each channel's temporal median. Estimate the spatial signal with
   `gaussian_filter1d`, sigma 3 along channels, `mode='nearest'`.
4. Calculate C-factor as `median(expected / centered)` for each channel.
   Zero denominators contribute 1; non-finite factors become 1. Apply no clipping
   or common median reference by default. Store corrected samples as float32.
5. Calculate Welch PSD with `fs=int(effective_fs)`, `nperseg=256`, `nfft=4096`.
   Average linear power within inclusive bands 1.5–4, 4–10, 10–30, 30–80, and
   80–200 Hz, then convert to dB.
6. Calculate `np.corrcoef` of the corrected signal. The correlation profile is
   `mean(abs(correlation), axis=0)`.
7. Fit six continuous linear segments with `pwlf`. The boundary candidate for
   each band is `breaks[-2]`; average candidates across selected bands/series.
8. Cluster correlation rows with Ward clustering, ten clusters, and a
   six-neighbour channel-index graph.

SciPy's CPU FFT uses the same resampling convention as the notebook's torch/CUDA
code. Floating-point results can differ.

For low-pass traces and CSD, read the raw interval from the first LFP timestamp.
Apply notch, FFT resampling, a second-order 40 Hz low-pass, and C-factor.
CSD is the negative second spatial difference after Gaussian smoothing with
`sigma=[5,5]`. It is in relative units, not calibrated current density.
The default CSD color range is ± the 95th percentile of absolute values.

## Differences to account for

- `pwlf` uses seed 7, `maxiter=200`, `popsize=8`, and `tol=1e-4`.
  The notebook uses unseeded library defaults, so breakpoints may differ.
- Short maps use fewer segments or clusters. Fewer than eight sites or a constant
  profile receive no boundary fit.
- ProbeMap depth starts at the physical tip. The notebook's Y coordinates start
  at the first site. Add the first-site offset when comparing depths (normally
  217 µm in this example).
- **Mean** averages the two column voltages at matching depths before interval
  correction. It does not average dB curves. Whole-record power uses linear means.
- Missing electrode rows remain gaps. Cluster IDs and fitted transitions do not
  identify anatomical regions.
- Short inputs are processed directly; the notebook helper's broken short-record
  call is not used.

## Stored results

| File under the prepared recording | Contents |
| --- | --- |
| `signal/notebook_window.npz` | Latest original-source interval result |
| `signal/cached_window.npz` | Latest prepared-LFP interval result |

Results include metrics, traces, fits, clusters, and CSD. They are reused only
when the source, interval, channels, geometry, and analysis settings match.
The prepared-LFP cache also checks `lfp.npy` size, modification time, and sample rate.
Each mode replaces its previous interval file and leaves `lfp.npy` unchanged.

Source data is read in 4 MiB requests and processed in groups of eight channels.
Temporary source batches use the system temporary folder and are removed after
completion or cancellation. A recording lock prevents simultaneous cache writes.

## Checks

Tests compare C-factor, PSD, band power, correlation, and clustering with the
notebook reference functions. They also check FFT batching across a batch boundary,
low-pass filtering, seeded fits, cache reuse, and PDF export.

A local check on PV233 used 384 contacts, both columns and Mean, starting at 99 s.
Prepared-LFP calculation took 4.877 s, including cache writing; reopening took
0.065 s. The raw binary was unavailable during this check, so these timings do
not measure original-recording processing.

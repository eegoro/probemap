# Ephys performance notes

These measurements used the prepared PV233 `5_depthTest_before_g0` recording
on macOS with offscreen Qt. The plot check used a probe without an atlas image.
They measure plot preparation and cache loading, not screen frame rate or network
transfer speed. Filesystem caches may have been warm.

## Measurements

| Operation | Before changes | After changes |
| --- | ---: | ---: |
| Prepare artists after one RMS setting change | 45.6 ms | 9.4 ms |
| Load Kilosort arrays | 27–34 ms | 4.3 ms |
| Load the same Kilosort arrays again | — | 2.1 ms |
| Kilosort array data read | about 31 MiB | 0.032 MiB |

Timings exclude the 60 ms settings delay and Qt painting. Data came from the
local prepared cache, not the AP binary. Other recordings may take longer.

## Implementation

- Wait 60 ms after display setting changes, then redraw affected panels.
- Move existing canvases when plots are reordered.
- Keep raster artists when unit visibility changes. Read spikes again when
  the recording, sorting, filter, or interval changes.
- Load amplitude data and calculate autocorrelation only when their plots are open.
- Scan for prepared recordings in a worker. Keep edits to names, insertion links,
  unit settings, and sorting selections when merging scan results.
- Read only requested fields from `units.npz`. The reader still supports the
  older `raster` and `raster_good` arrays.
- Keep up to four prepared array sets in memory. Reload them if the source
  manifest or archive changes. Shared arrays are read-only.
- Cache up to eight interval results, with a total limit of 64 MiB.
  Include source, time, channel, geometry, and analysis settings in cache keys.

These changes did not alter filtering, spectral calculations, unit depths,
or waveform calibration. Processing runs on the CPU.

## Tests

From an environment with the test dependencies installed, run:

```sh
python -m pytest tests/test_ephys_performance.py tests/test_ephys_unit_controls.py tests/test_ephys_notebook.py tests/test_ephys_portable.py tests/test_ephys_cache.py tests/test_ephys.py -q
```

Tests cover cache reuse and invalidation, memory limits, plot updates, background
jobs, sorting changes, saved settings, depth alignment, and calculation results.
GUI tests need Qt; the full interface needs a display and working OpenGL.

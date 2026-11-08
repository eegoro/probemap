# Electrophysiology

Use this page to compare recordings with anatomy along the probe.
Depth is measured in µm from the physical tip.

## Open a recording

1. In **Recordings**, click **+** and select a recording folder.
2. Link the recording to an insertion and choose a shank.
3. Click **Prepare signal** if the recording has no prepared data.
4. Right-click the recording and choose **Connect Kilosort…** to add sorting results.

Prepared recordings can be reopened without the original binary. Preparing new
signal data or filtering the original recording requires that file.

## Sidebar

| Tab | Use it to |
| --- | --- |
| Recordings | Add recordings, link insertions, and choose Kilosort results |
| Plots | Add and arrange plots; change display and processing settings |
| Anatomy | Choose the image, show labels, and edit region display |
| Marks | Add depth levels and mark the brain boundary |
| Units | Choose units for the raster and waveforms |

## Arrange plots

Click **Add plot…** to open a view. Drag entries in the plot list to reorder them.
Uncheck an entry to hide it; its settings are kept.
Double-click a plot to open its settings. **Reset plot** resets only that plot.
Drag the dividers to change plot widths.

Choose electrode columns and their opacity in **Plots**. **Mean** averages both
columns and highlights contacts from both. Use **Legends** to show or hide legends
and color scales. These settings also apply to PDF export.

The anatomy stays on the left. Plots that show depth use the same depth scale.
Waveforms, ISI, and other plots with different units use their own axes.

## Choose a time interval

Use the bottom timeline, start time, and **Window** (0.5–60 s).
Moving the timeline turns on **Selected interval**. Release the slider to update
plots for that interval.

In **Plots → Signal processing**, choose:

- **Prepared LFP · fast**: use the local LFP cache.
- **Original recording · reference filters**: filter the raw recording again.
  This can read hundreds of MB, even for a short interval. Click **Update analysis**
  when a new calculation is needed.

AP RMS uses prepared 2 s bins. A short interval may contain no bin.
Recording comparisons use whole-record summaries. Region unit counts describe
the selected population, not the number of spikes in the time window.

## Change settings

Each plot has its own axis ranges, scales, colors, and line or heatmap settings.
LFP plots also have editable frequency bands, profile fits, and transition lines.
Display smoothing stops at gaps between recorded sites.

**Signal processing** contains C-factor correction, spatial smoothing, common
median reference, Welch window, FFT length, linear segment count, and the trace
low-pass cutoff. Changing these recalculates the interval, not the full recording.
Changing display colors or line widths does not recalculate signal data.

Band power is averaged in linear units, then converted to dB. CSD is shown in
relative units. Fitted transitions and cluster IDs are not anatomical labels;
check them against the scan and atlas.

## Save

Save the project to keep the plot order and settings. Use the save icon to export
PDF. See [figure export](figure-export.md).

For calculation details, see [methods](development/ephys_notebook_comparison.md).

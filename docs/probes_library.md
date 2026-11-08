# Probe library and designer

Open **Probes Library → Probes Library…** from the top menu.
Choose a template, click **Duplicate** to edit a copy, or click **Create probe…**.
Built-in templates cannot be edited directly. Check template dimensions against
your probe before use.

## 1. Set the body

In **Body**, choose a wire, flat shape, cylinder, shank/grid, custom cross-section,
or contacts only. Set the dimensions in µm.

A new probe starts as a wire with a contact at its end. Changing the body keeps
existing contacts; remove any unwanted ones in **Contacts**.

For a custom outline, click **Draw contour** and click its vertices. Click the
first point or press Enter to close it; Esc cancels. Use the vertex table for
exact coordinates. Outlines must not cross themselves.

Flat outlines are drawn in X/Y and given a thickness in Z. Custom cross-sections
are drawn in X/Z and extended along Y.

## 2. Add contacts

In **Add contacts**, choose a layout: individual contacts, rows/grids, contacts
along an edge, terminal contacts, rings, or sectors. Available layouts depend
on the body. Set the contact dimensions separately from the body dimensions.

Choose the recording surface: front, back, edge, prism face, or cylinder surface.
For rings and sectors, set their angular position and extent.
New contacts are added to the existing list. Use Undo to reverse an edit.

In **Contacts**, edit IDs, X/Y/Z positions, sizes, and shapes. Select rows to
delete or duplicate contacts. With the canvas focused, arrow keys move selected
contacts by the snap step; Shift moves them ten times farther.

## 3. Make an array

In **Array**, set the number of copies and their spacing. Applying the array
replaces previous copies of element 0. Undo restores them.
All copies move together. Use separate insertions for probes that need independent
placement. Flexible probe shapes are not supported.

## 4. Save and use

Click **Save probe**, then select the model on the **Probes** page.
Each project stores a copy of its assigned geometry. Later library edits do not
change existing reconstructions.

Custom geometry can be used for placement. Raw recording import currently supports
four-shank Neuropixels 2.0; a new geometry does not add another recording format.
Region overlap over the full area of large contacts is not calculated.

## Import and export

- **JSON**: the complete probe geometry, including bodies and contact surfaces.
- **CSV**: contact centres only. Import replaces the contact list and can be undone.
  Columns: `id,shank,electrode,x_um,y_um,z_um`. Only X/Y are required.

Coordinates are in µm. Y runs from the distal reference toward the base.
X is relative to the element centre; Z is relative to its reference plane.

## Move around the view

Use **− / +**, the mouse wheel, or trackpad pinch to zoom. Use two-finger scroll,
Space + drag, middle-drag, or the Pan tool to pan. Drag in 3D to rotate the view.
Use front, side, or top views for editing. **Fit body**, **Fit contacts**, and
**Selected** centre the view on those objects.

## Template files

| System | Personal templates |
| --- | --- |
| macOS | `~/Library/Application Support/ProbeMap/probes_library/` |
| Windows | `%APPDATA%/ProbeMap/probes_library/` |
| Linux | `$XDG_DATA_HOME/probemap/probes_library/`, or `~/.local/share/probemap/probes_library/` |

Set `PROBEMAP_PROBES_LIBRARY` to use another folder.

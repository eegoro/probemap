"""Step 9 controls and views, sharing the insertion workspace's single GL canvas."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (QCheckBox,QComboBox,QDoubleSpinBox,QFileDialog,QFormLayout,
    QGroupBox,QHBoxLayout,QHeaderView,QInputDialog,QLabel,QTabWidget,QTableWidget,QTableWidgetItem,QVBoxLayout,QWidget)

from .desktop_tracks import action,note
from .landmarks import transform_points
from .neuropixels_model import (add_recording,channel_rows,find_surface_depths,fit_probe,geometry_for,
    pose_current,probe_coordinates,probe_plane,probe_settings,shank_mesh,write_reports,suggest_shank_assignments)
from .track_model import plane_geometry,plane_covering_volumes


class NeuropixelsWorkspaceMixin:
    def _probe_controls(self):
        page=QWidget(); layout=QVBoxLayout(page)
        layout.addWidget(note('Track start = deep tip; the second click points toward the brain surface. Shanks follow that direction. Assign shank IDs, fit, then verify placement. Dimensions are in source scan µm.'))
        self.probe_tabs=QTabWidget()
        fit_page,record_page,geometry_page=QWidget(),QWidget(),QWidget()
        fit_layout,record_layout,geometry_layout=QVBoxLayout(fit_page),QVBoxLayout(record_page),QVBoxLayout(geometry_page)
        for title,child in [('Fit',fit_page),('Recordings',record_page),('Geometry',geometry_page)]:
            self.probe_tabs.addTab(child,title)
        layout.addWidget(self.probe_tabs)
        self.probe_geometry_box=QGroupBox('Probe geometry')
        form=QFormLayout(self.probe_geometry_box)
        self.probe_model=QComboBox(); self.probe_model.addItems(['NP2013','NP2014','Legacy NP2'])
        form.addRow('Hardware model',self.probe_model)
        self.probe_dimensions={}
        fields=[('shank_pitch_um','Shank pitch',250),('shank_length_um','Tip to base',10000),
                ('shank_width_um','Shank width',70),('shank_thickness_um','Thickness',24),
                ('taper_length_um','Taper length',206),('first_site_center_from_tip_um','Tip to first site centre',217)]
        for key,title,value in fields:
            box=QDoubleSpinBox(); box.setRange(1,20000); box.setDecimals(1); box.setValue(value); box.setSuffix(' µm')
            self.probe_dimensions[key]=box; form.addRow(title,box)
            box.valueChanged.connect(self._probe_geometry_changed)
        form.addRow(note('Sites: 12 × 12 µm; rows: 15 µm; columns: 32 µm. First-site offset includes an 11 µm contact-centre offset; verify your probe version.'))
        self.probe_model.currentTextChanged.connect(self._probe_model_changed)
        geometry_layout.addWidget(self.probe_geometry_box); geometry_layout.addStretch()
        self.probe_assignment=QTableWidget(0,2); self.probe_assignment.setHorizontalHeaderLabels(['Traced track','Shank ID'])
        self.probe_assignment.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.probe_assignment.setMaximumHeight(160)
        fit_layout.addWidget(self.probe_assignment)
        order_row=QHBoxLayout()
        self.probe_suggest_button=action('Suggest shank order',self.suggest_probe_order)
        order_row.addWidget(self.probe_suggest_button)
        order_row.addWidget(action('Reverse order',self.reverse_probe_order))
        fit_layout.addLayout(order_row)
        self.probe_order_info=note('Shank IDs link the stain to channel maps. A spatial suggestion cannot identify which side is shank 0.')
        fit_layout.addWidget(self.probe_order_info)
        pose_box=QGroupBox('Probe placement'); pose_form=QFormLayout(pose_box)
        self.probe_tip_mode=QComboBox()
        for label,key in [('Marked deep end (track start)','track_start'),
                          ('Second marked point (override)','direction_point'),('4 mm end (override)','track_end')]:
            self.probe_tip_mode.addItem(label,key)
        self.probe_tip_mode.setToolTip('The first click on step 6 marks the deep end of the electrode trace. The second points toward the surface. Other anchors are manual overrides, not detected tips.')
        pose_form.addRow('Tip anchor',self.probe_tip_mode)
        self.probe_tip_mode.currentIndexChanged.connect(self._probe_pose_edited)
        self.probe_reverse=QCheckBox('Reverse direction (override)')
        self.probe_reverse.toggled.connect(self._probe_pose_edited); pose_form.addRow(self.probe_reverse)
        self.probe_pose_controls={}
        for key,title,unit in [('tip_shift_um','Tip shift (+ deeper)',' µm'),('lateral_shift_um','Lateral shift',' µm'),
                               ('normal_shift_um','Normal shift',' µm'),('roll_deg','Probe roll','°')]:
            box=QDoubleSpinBox(); box.setRange(-20000,20000); box.setDecimals(1); box.setSuffix(unit)
            if key=='roll_deg': box.setRange(-180,180)
            self.probe_pose_controls[key]=box; pose_form.addRow(title,box)
            box.valueChanged.connect(self._probe_pose_edited)
        self.probe_fit_button=action('Fit / update probe',self.fit_current_probe); self.probe_fit_button.setObjectName('primary')
        self.probe_fit_button.setToolTip('Four unassigned traces receive a provisional spatial order. Verify the shank-0 side before confirming channel locations.')
        fit_layout.addWidget(self.probe_fit_button)
        self.probe_fit_info=note('No probe fitted. Shank IDs cannot be inferred reliably from an unlabeled stain.')
        fit_layout.addWidget(self.probe_fit_info)
        self.probe_confirm=QCheckBox('Tip placement and shank order verified')
        self.probe_confirm.toggled.connect(self._probe_confirmed)
        pose_form.addRow(self.probe_confirm); fit_layout.addWidget(pose_box)
        surface=QGroupBox('Brain entry surface · zero depth'); surface_form=QFormLayout(surface)
        surface_form.addRow(note('Automatic depths use the brain mask from step 6. Check each entry. Optional override: tip-to-surface distance, positive in source scan µm.'))
        self.probe_surface_boxes=[]
        for shank in range(4):
            box=QDoubleSpinBox(); box.setRange(-1,20000); box.setValue(-1); box.setDecimals(1)
            box.setSpecialValueText('Automatic'); box.setSuffix(' µm')
            surface_form.addRow(f'Shank {shank}',box); self.probe_surface_boxes.append(box)
            box.valueChanged.connect(self._probe_surface_edited)
        surface_form.addRow(action('Recalculate entry surfaces',self._probe_recompute_surface))
        fit_layout.addWidget(surface)
        recordings=QGroupBox('Recordings / channel maps'); rec_layout=QVBoxLayout(recordings)
        row=QHBoxLayout(); row.addWidget(action('Add maps…',self.add_probe_recordings)); row.addWidget(action('Remove',self.remove_probe_recording))
        rec_layout.addLayout(row)
        self.recording_table=QTableWidget(0,4); self.recording_table.setHorizontalHeaderLabels(['Recording','Show','Opacity','Depth shift µm'])
        self.recording_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.recording_table.horizontalHeader().setSectionResizeMode(0,QHeaderView.ResizeMode.Stretch)
        self.recording_table.setMaximumHeight(190); self.recording_table.itemChanged.connect(self._probe_recording_name)
        rec_layout.addWidget(self.recording_table)
        rec_layout.addWidget(note('TXT / IMRO / META. Maps are embedded in the project. Depth shift is optional for recordings made at different depths along the same trajectory.'))
        self.probe_recording_filter=QComboBox(); self.probe_recording_filter.currentIndexChanged.connect(self._probe_filter_changed)
        rec_layout.addWidget(self.probe_recording_filter); record_layout.addWidget(recordings); record_layout.addStretch()
        view_box=QGroupBox('Probe display'); display_form=QFormLayout(view_box)
        self.probe_show_shanks=QCheckBox('Show shanks'); self.probe_show_shanks.setChecked(True)
        self.probe_show_sites=QCheckBox('Show recording sites'); self.probe_show_sites.setChecked(True)
        for box in (self.probe_show_shanks,self.probe_show_sites):
            display_form.addRow(box); box.toggled.connect(self._probe_display_changed)
        self.probe_alpha=QDoubleSpinBox(); self.probe_alpha.setRange(.01,1); self.probe_alpha.setSingleStep(.05); self.probe_alpha.setValue(.18)
        display_form.addRow('Shank opacity',self.probe_alpha); self.probe_alpha.valueChanged.connect(self._probe_display_changed)
        self.probe_plane_controls={}
        self.probe_plane_auto=QCheckBox('Show complete scan slices'); self.probe_plane_auto.setChecked(True)
        display_form.addRow(self.probe_plane_auto)
        self.probe_plane_auto.toggled.connect(self._probe_display_changed)
        for key,title,lo,hi,value in [('plane_offset_um','Plane normal offset',-10000,10000,0),
                                    ('plane_width_um','Plane width',100,20000,2200),('plane_height_um','Plane height',100,20000,6000)]:
            box=QDoubleSpinBox(); box.setRange(lo,hi); box.setValue(value); box.setSuffix(' µm')
            display_form.addRow(title,box); self.probe_plane_controls[key]=box
            box.valueChanged.connect(self._probe_display_changed)
        display_form.addRow(action('Fit complete slices',self._probe_view_plane))
        self.probe_display_box=view_box
        self.display_content.layout().addWidget(view_box)
        view_box.hide()
        self.probe_channel_info=note('Load channel maps to inspect recording sites and region assignments.')
        fit_layout.addWidget(self.probe_channel_info); fit_layout.addStretch()
        for combo in (self.probe_model,self.probe_tip_mode,self.probe_recording_filter):
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(10)
            combo.setMinimumWidth(0)
        for form in (form,pose_form,surface_form,display_form):
            form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
            form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        return page

    def _refresh_probe_controls(self):
        item=self.current_insertion()
        if item is None: return
        s=probe_settings(item)
        migration_error=None
        if s.get('pose') and s['pose'].get('direction_convention')!='deep_tip_to_surface':
            try:
                fit_probe(item,self.project.total)
                if self.brain_mask is not None:
                    find_surface_depths(item,self.brain_mask,self.brain_preview.spacing_um_zyx)
                self._probe_fit_view_pending=True
                for canvas in self.slices: canvas.fit_image()
            except (ValueError,np.linalg.LinAlgError) as error:
                # The earlier pose is retained in previous_placement. Do not
                # render its reversed geometry as if it were the new fit.
                s.pop('pose',None)
                migration_error=str(error)
        self._updating=True
        self.probe_model.setCurrentText(s['geometry']['model'])
        for key,box in self.probe_dimensions.items(): box.setValue(s['geometry'][key])
        self.probe_tip_mode.setCurrentIndex(self.probe_tip_mode.findData(s['tip_mode']))
        self.probe_reverse.setChecked(s['flip_direction'])
        for key,box in self.probe_pose_controls.items(): box.setValue(s[key])
        self.probe_confirm.setChecked(s['confirmed'])
        for table in (self.probe_assignment,self.recording_table):
            for row in range(table.rowCount()):
                for column in range(table.columnCount()):
                    old=table.cellWidget(row,column)
                    if old is not None:
                        old.hide()
                        table.removeCellWidget(row,column)
        self.probe_assignment.setRowCount(len(item['tracks']))
        for row,t in enumerate(item['tracks']):
            cell=QTableWidgetItem(t['name']); cell.setFlags(cell.flags()&~Qt.ItemFlag.ItemIsEditable)
            self.probe_assignment.setItem(row,0,cell)
            combo=QComboBox(); combo.addItem('Ignore',None)
            for i in range(4): combo.addItem(str(i),i)
            combo.setCurrentIndex(combo.findData(s['assignments'].get(t['id'])))
            combo.currentIndexChanged.connect(lambda _,track=t,box=combo:self._probe_assignment_edited(track,box))
            self.probe_assignment.setCellWidget(row,1,combo)
        for i,box in enumerate(self.probe_surface_boxes):
            value=s['surface_depths_um'].get(str(i)); box.setValue(-1 if value is None else value)
        self.recording_table.setRowCount(len(item.get('recordings',[])))
        self.probe_recording_filter.clear(); self.probe_recording_filter.addItem('All visible recordings','all')
        for row,rec in enumerate(item.get('recordings',[])):
            name=QTableWidgetItem(rec['name']); name.setData(Qt.ItemDataRole.UserRole,rec['id']); self.recording_table.setItem(row,0,name)
            show=QCheckBox(); show.setChecked(rec.get('visible',True)); self.recording_table.setCellWidget(row,1,show)
            alpha=QDoubleSpinBox(); alpha.setRange(.05,1); alpha.setSingleStep(.05); alpha.setValue(rec.get('opacity',1)); self.recording_table.setCellWidget(row,2,alpha)
            depth=QDoubleSpinBox(); depth.setRange(-20000,20000); depth.setValue(rec.get('depth_offset_um',0)); self.recording_table.setCellWidget(row,3,depth)
            show.toggled.connect(lambda val,r=rec:self._probe_rec_edited(r,'visible',val))
            alpha.valueChanged.connect(lambda val,r=rec:self._probe_rec_edited(r,'opacity',val))
            depth.valueChanged.connect(lambda val,r=rec:self._probe_rec_edited(r,'depth_offset_um',val))
            self.probe_recording_filter.addItem(rec['name'],rec['id'])
        self.probe_recording_filter.setCurrentIndex(max(0,self.probe_recording_filter.findData(s['display'].get('slice_recording','all'))))
        self.probe_show_shanks.setChecked(s['display']['shanks']); self.probe_show_sites.setChecked(s['display']['sites'])
        self.probe_alpha.setValue(s['display']['shank_opacity'])
        self.probe_plane_auto.setChecked(s['display']['plane_auto_fit'])
        for key,box in self.probe_plane_controls.items(): box.setValue(s['display'].get(key,box.value()))
        for key in ('plane_width_um','plane_height_um'):
            self.probe_plane_controls[key].setEnabled(not self.probe_plane_auto.isChecked())
        p=s.get('pose')
        assigned=sum(s['assignments'].get(t['id']) is not None for t in item['tracks'])
        self.probe_order_info.setText(f"{assigned} / {len(item['tracks'])} traces assigned. "+s.get('assignment_method','Set known shank IDs, or suggest an order for four traces. Verify which side is shank 0.'))
        self.probe_fit_info.setStyleSheet('')
        if p:
            status='Current fit' if pose_current(item,self.project.total) else 'Tracks or geometry changed: refit required'
            depths=', '.join('unavailable' if v is None else f'{v:.0f}' for v in p.get('surface_depths_um',[None]*4))
            self.probe_fit_info.setText(f"{status}\nFit RMS {p['fit_rms_um']:.1f} µm · max {p['fit_max_error_um']:.1f} µm\nTip-to-surface depths (shanks 0–3): {depths} µm")
            check=p.get('spacing_check')
            if check:
                self.probe_fit_info.setText(self.probe_fit_info.text()+f"\nTraced pitch: {check['observed_pitch_um']:.1f} µm; hardware: {check['nominal_pitch_um']:.1f} µm.")
                if abs(check['observed_to_nominal_ratio']-1)>.15:
                    self.probe_fit_info.setStyleSheet('color:#ffcd88;')
                    self.probe_fit_info.setText(self.probe_fit_info.text()+'\nSpacing mismatch: verify tissue preparation, calibration and trace identity. Fitting does not rescale the scan.')
            if s['tip_mode']=='direction_point' and p['fit_rms_um']>250:
                self.probe_fit_info.setText(self.probe_fit_info.text()+'\nDirection points are being used as physical tips. If the second clicks only defined directions, choose another tip anchor and adjust tip depth.')
        else: self.probe_fit_info.setText('Assign known shank IDs, then fit. Tip placement starts as an estimate.')
        if 'previous_placement' in s:
            self.probe_order_info.setText(self.probe_order_info.text()+' Previous placement preserved. Track start now marks the deep tip; verify placement again.')
        self.probe_assignment.doItemsLayout()
        self.recording_table.doItemsLayout()
        self._updating=False
        if migration_error: self._probe_fit_error(migration_error)

    def _probe_assignment_edited(self,track,combo):
        if self._updating: return
        self._remember(); s=probe_settings(self.current_insertion()); s['assignments'][track['id']]=combo.currentData()
        s['confirmed']=False; self._refresh_probe_controls(); self.request_render()

    def suggest_probe_order(self,checked=False):
        if self.current_insertion() is None: return
        try:
            self._remember()
            suggest_shank_assignments(self.current_insertion(),self.project.total)
            self._refresh_probe_controls(); self.request_render(); self.owner.schedule_save()
        except ValueError as error: self._probe_fit_error(str(error))

    def reverse_probe_order(self):
        item=self.current_insertion()
        if item is None: return
        s=probe_settings(item)
        if not any(value is not None for value in s['assignments'].values()):
            self._probe_fit_error('Suggest or assign shank IDs before reversing the order.'); return
        self._remember()
        s['assignments']={key:None if value is None else 3-value for key,value in s['assignments'].items()}
        s['confirmed']=False
        s['assignment_method']='Order reversed manually; verify shank 0 against the probe orientation.'
        self._refresh_probe_controls(); self.request_render(); self.owner.schedule_save()

    def _probe_fit_error(self,message):
        self.probe_fit_info.setStyleSheet('color:#ffb5a8;')
        self.probe_fit_info.setText('Probe was not fitted:\n'+message)
        self.sidebar_toggle.setChecked(True)
        self.sidebar_tabs.setCurrentIndex(0); self.probe_tabs.setCurrentIndex(0)
        self.sidebar_tabs.widget(0).ensureWidgetVisible(self.probe_fit_info)
        self.owner.statusBar().showMessage(message)

    def _probe_model_changed(self,model):
        if self._updating or self.current_insertion() is None: return
        self._remember(); s=probe_settings(self.current_insertion()); s['geometry']=geometry_for(model); s['confirmed']=False
        self._refresh_probe_controls(); self.request_render()

    def _probe_geometry_changed(self):
        if self._updating or self.current_insertion() is None: return
        self._remember(); s=probe_settings(self.current_insertion())
        for key,box in self.probe_dimensions.items(): s['geometry'][key]=box.value()
        s['confirmed']=False; self.request_render()

    def _probe_pose_edited(self):
        if self._updating or self.current_insertion() is None: return
        self._remember(); s=probe_settings(self.current_insertion())
        for key,box in self.probe_pose_controls.items(): s[key]=box.value()
        s['tip_mode']=self.probe_tip_mode.currentData()
        s['flip_direction']=self.probe_reverse.isChecked(); s['confirmed']=False
        if s.get('pose'): self.fit_current_probe(remember=False)

    def _probe_confirmed(self,checked):
        if self._updating or self.current_insertion() is None: return
        if checked and not pose_current(self.current_insertion(),self.project.total):
            self.owner.statusBar().showMessage('Fit the current tracks and geometry before confirming.'); self.probe_confirm.setChecked(False); return
        self._remember(); probe_settings(self.current_insertion())['confirmed']=checked
        self.owner.schedule_save(); self.request_render()

    def fit_current_probe(self,checked=False,remember=True):
        item=self.current_insertion()
        if item is None: return
        try:
            if remember: self._remember()
            from .insertion_model import sync_tracks
            sync_tracks(self.session,self.project)
            settings=probe_settings(item)
            if len(item['tracks'])==4 and not any(settings['assignments'].get(t['id']) is not None for t in item['tracks']):
                suggest_shank_assignments(item,self.project.total)
            fit_probe(item,self.project.total)
            self._probe_fit_view_pending=True
            for canvas in self.slices: canvas.fit_image()
            self._probe_recompute_surface(remember=False)
            self._refresh_probe_controls(); self.request_render(); self.owner.schedule_save()
        except (ValueError,np.linalg.LinAlgError) as error: self._probe_fit_error(str(error))

    def _probe_surface_edited(self):
        if self._updating or self.current_insertion() is None: return
        self._remember(); s=probe_settings(self.current_insertion())
        s['surface_depths_um']={str(i):None if box.value()<0 else box.value() for i,box in enumerate(self.probe_surface_boxes)}
        s['confirmed']=False; self._probe_recompute_surface(remember=False)

    def _probe_recompute_surface(self,checked=False,remember=True):
        item=self.current_insertion()
        if item is None or not probe_settings(item).get('pose') or self.brain_mask is None: return
        if remember: self._remember()
        find_surface_depths(item,self.brain_mask,self.brain_preview.spacing_um_zyx)
        probe_settings(item)['pose']['brain_mask_threshold']=self.session.get('brain_mask',{}).get('threshold')
        self._refresh_probe_controls(); self.request_render(); self.owner.schedule_save()

    def add_probe_recordings(self,checked=False,paths=None):
        if self.current_insertion() is None: return
        if paths is None:
            paths,_=QFileDialog.getOpenFileNames(self,'Add recording channel maps',str(Path.home()),'SpikeGLX channel maps (*.txt *.imro *.meta);;All files (*)')
        if not paths: return
        self._remember(); errors=[]
        for path in paths:
            try: add_recording(self.current_insertion(),path)
            except (OSError,ValueError) as error: errors.append(f'{Path(path).name}: {error}')
        self._refresh_probe_controls(); self.request_render(); self.owner.schedule_save()
        if errors: self.owner._error('\n'.join(errors))

    def remove_probe_recording(self):
        row=self.recording_table.currentRow(); item=self.current_insertion()
        if item is None or row<0: return
        self._remember(); del item['recordings'][row]; self._refresh_probe_controls(); self.request_render()

    def _probe_rec_edited(self,record,key,value):
        if self._updating: return
        self._remember(); record[key]=value; self.request_render(); self.owner.schedule_save()

    def _probe_recording_name(self,cell):
        if self._updating or cell.column()!=0 or self.current_insertion() is None: return
        text=cell.text().strip()
        if not text: self._refresh_probe_controls(); return
        rec=next(r for r in self.current_insertion()['recordings'] if r['id']==cell.data(Qt.ItemDataRole.UserRole))
        self._remember(); rec['name']=text; self._refresh_probe_controls(); self.request_render()

    def _probe_filter_changed(self):
        if self._updating or self.current_insertion() is None: return
        probe_settings(self.current_insertion())['display']['slice_recording']=self.probe_recording_filter.currentData()
        self.request_render(); self.owner.schedule_save()

    def _probe_display_changed(self):
        if self._updating or self.current_insertion() is None: return
        d=probe_settings(self.current_insertion())['display']
        d.update(shanks=self.probe_show_shanks.isChecked(),sites=self.probe_show_sites.isChecked(),shank_opacity=self.probe_alpha.value())
        d['plane_auto_fit']=self.probe_plane_auto.isChecked()
        for key,box in self.probe_plane_controls.items(): d[key]=box.value()
        for key in ('plane_width_um','plane_height_um'):
            self.probe_plane_controls[key].setEnabled(not d['plane_auto_fit'])
        self.request_render(); self.owner.schedule_save()

    def _probe_view_plane(self):
        if self.current_insertion() is None: return
        self.probe_plane_auto.setChecked(True)
        self.request_render()
        for canvas in self.slices: canvas.fit_image()

    def _probe_geometry(self):
        item=self.current_insertion()
        if item and probe_settings(item).get('pose'):
            plane=probe_plane(item,self.project.total)
            if probe_settings(item)['display']['plane_auto_fit']:
                plane=plane_covering_volumes(plane,self._volume_corner_sets)
            return plane_geometry(plane)
        return plane_geometry(self.session['plane'])

    def _hover(self,world):
        super()._hover(world)
        if self._mode!=4 or not self.track_slice.probe_sites: return
        from .track_model import plane_uv
        geometry=self.track_slice.geometry_record
        if geometry is None: return
        sites=self.track_slice.probe_sites
        uv,dist=plane_uv(geometry,[site['atlas_zyx_um'] for site in sites])
        pointer,_=plane_uv(geometry,[world])
        distances=np.linalg.norm(uv-pointer[0],axis=1)
        distances[np.abs(dist)>max(50,geometry['pixel_size_um'])]=np.inf
        nearest=int(np.argmin(distances))
        scale=self.track_slice.image.width()/max(1,self.track_slice.image_rect().width())
        if distances[nearest]<8*scale:
            site=sites[nearest]; depth=site['depth']
            self.hover_info.setText(site['label']+' · surface depth '+('unavailable' if depth is None else f'{depth:+.1f} µm (source scan)'))

    def _hide_probe_visuals(self):
        self.volume_view.annotation_bounds_zyx=[]
        self.volume_view.hide_probe_face()
        self._hide_placement_recordings()
        for mesh,sites in getattr(self,'_probe_visuals',{}).values(): mesh.visible=sites.visible=False
        for canvas in self.slices:
            canvas.probe_sites=[]; canvas.probe_polygons=[]

    def _dispose_probe_visuals(self):
        self._hide_probe_visuals()
        for mesh,sites in getattr(self,'_probe_visuals',{}).values(): mesh.parent=sites.parent=None
        self._probe_visuals={}

    def _draw_probes(self):
        from vispy import scene
        self._hide_probe_visuals()
        if self.volume_view.view is None: return
        self._probe_visuals=getattr(self,'_probe_visuals',{})
        rows_for_slice=[]; polygons=[]; bounds=[]; n_channels=0
        for item in self.session['insertions']:
            if not self.show_all.isChecked() and item['id']!=self.insertion_id: continue
            s=item.get('probe',{}); p=s.get('pose')
            if p is None: continue
            if item['id'] not in self._probe_visuals:
                mesh=scene.visuals.Mesh(parent=self.volume_view.view.scene,shading=None)
                sites=scene.visuals.Mesh(parent=self.volume_view.view.scene,shading=None)
                mesh.set_gl_state('translucent',depth_test=True,cull_face=False)
                sites.set_gl_state('translucent',depth_test=False,cull_face=False)
                mesh.order=25; sites.order=26; self._probe_visuals[item['id']]=(mesh,sites)
            mesh,sites=self._probe_visuals[item['id']]
            rgb=np.array(QColor(item['color']).getRgbF()[:3])
            vertices,faces=shank_mesh(item,self.project.total)
            mesh.set_data(vertices=vertices,faces=faces,color=(*rgb,s['display']['shank_opacity']))
            mesh.visible=s['display']['shanks']
            if mesh.visible: bounds.extend(vertices[:,::-1])
            records={r['id']:r for r in item.get('recordings',[])}
            rows=channel_rows(item,self.project.total,self.labels,self.session['regions'])
            vertices2=[]; faces2=[]; colors=[]
            for r in rows:
                rec=records[r['recording_id']]
                if not rec.get('visible',True): continue
                selected=s['display'].get('slice_recording','all')
                if selected!='all' and selected!=rec['id']: continue
                alpha=rec.get('opacity',1)*s['display'].get('site_opacity',1)
                # Blend brightness as well as opacity so repeated recordings stay legible on a slice.
                rec_rgb=np.array(QColor(rec.get('color') or item['color']).getRgbF()[:3])
                rgba=(*(.35+.65*alpha)*rec_rgb,alpha)
                pos=[r[f'atlas_{axis}_um'] for axis in 'zyx']
                if s['display']['sites']:
                    rows_for_slice.append(dict(atlas_zyx_um=pos,rgba=rgba,label=f"{r['recording']} · ch {r['channel']} · {r['region']}",
                                               depth=r['depth_from_surface_um']))
                    n_channels+=1
                    source=np.array([r[f'source_{axis}_um'] for axis in 'zyx'])
                    half=s['geometry']['site_size_um']/2
                    quad=np.array([source+x*half*np.array(p['lateral_source_zyx'])+y*half*np.array(p['down_source_zyx'])
                                   for x,y in ((-1,-1),(1,-1),(1,1),(-1,1))])
                    at=transform_points(quad,self.project.total)[:,::-1]
                    start=len(vertices2); vertices2.extend(at); faces2.extend([[start,start+1,start+2],[start,start+2,start+3]])
                    colors.extend([rgba]*4)
            if vertices2:
                sites.set_data(vertices=np.asarray(vertices2,np.float32),faces=np.asarray(faces2,np.uint32),vertex_colors=np.asarray(colors,np.float32)); sites.visible=True
                bounds.extend(np.asarray(vertices2)[:,::-1])
            else: sites.visible=False
            if s['display']['shanks'] and item['id']==self.insertion_id:
                for shank in range(4):
                    # Front-face contour projected on each probe-plane slice.
                    g=s['geometry']; poly=[]
                    for x,h in ((0,0),(-g['shank_width_um']/2,g['taper_length_um']),(-g['shank_width_um']/2,g['shank_length_um']),
                                (g['shank_width_um']/2,g['shank_length_um']),(g['shank_width_um']/2,g['taper_length_um'])):
                        poly.append(probe_coordinates(s,shank,h)+x*np.array(p['lateral_source_zyx']))
                    polygons.append((transform_points(poly,self.project.total).tolist(),(*rgb,s['display']['shank_opacity'])))
            if item['id']==self.insertion_id and (mesh.visible or sites.visible):
                self.volume_view.show_probe_face(item,self.project.total)
        for canvas in self.slices:
            canvas.probe_sites=rows_for_slice; canvas.probe_polygons=polygons; canvas.update()
        self.probe_channel_info.setText(f'{n_channels} visible recording sites. Site centres and 12 µm contact geometry are transformed together with the brain. Surface depths: negative inside brain; unavailable if no surface is defined.')
        self.volume_view.annotation_bounds_zyx=bounds
        if getattr(self,'_probe_fit_view_pending',False):
            self._probe_fit_view_pending=False
            self.volume_view.reset_view()
        self.volume_view.canvas.update()

    def export_probe_reports(self):
        self.capture_state()
        self.owner.save_project()
        path=write_reports(self.project,self.project.output_dir,self.labels)
        self.last_export=path
        self.owner.statusBar().showMessage(f'Channel table and reconstruction report saved: {path}')

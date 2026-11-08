"""Two-view native plane and Neuropixels placement workspace."""
import copy

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (QButtonGroup,QFormLayout,QGridLayout,QGroupBox,QHBoxLayout,
    QLabel,QPushButton,QToolButton,QVBoxLayout,QWidget)

from .desktop_widgets import view_button
from .desktop_placement import EditedSpin as QDoubleSpinBox, DragSlider as QSlider, SelectedCombo as QComboBox
from .desktop_slice_tools import tool_icon
from .landmarks import transform_points
from .neuropixels_model import contact_mesh,geometry_for,probe_settings,probe_coordinates,shank_mesh
from .probe_placement import (source_plane,save_source_plane,pose_on_plane,place_probe,probe_contours,
    contact_side,flip_contact_face,contact_preview)
from .desktop_probe_table import ProbePlacementTableMixin
from .desktop_recordings import RecordingTableMixin
from .desktop_surface_entries import SurfaceEntriesMixin
from .track_model import (plane_geometry,plane_world,plane_covering_volumes,
    sample_world,transform_plane,transform_plane_geometry)
from .coarse import transformed_corners


class ProbePlacementMixin(ProbePlacementTableMixin,RecordingTableMixin,SurfaceEntriesMixin):
    def _build_placement_workspace(self,display_layout):
        self._placement_ghost=None
        self._placement_move_base=None
        self._placement_rotate_base=None
        self._placement_angle=180.
        self._placement_scene='atlas'
        self._placement_cameras={}
        self.placement_toolbar=QWidget()
        row=QHBoxLayout(self.placement_toolbar); row.setContentsMargins(0,0,0,0)
        self.placement_tools={}
        self.placement_tool_group=QButtonGroup(self.placement_toolbar)
        self.placement_tool_group.setExclusive(True)
        for name,title in (('draw','Place probe · Click at the deep tips; drag toward the probe base'),
                           ('select','Select / move probe'),('measure','Measure distance')):
            button=QToolButton(); button.setIcon(tool_icon(name)); button.setCheckable(True)
            button.setFixedSize(32,32); button.setToolTip(title); button.setAccessibleName(title)
            button.clicked.connect(lambda checked=False,n=name:self._placement_tool(n))
            self.placement_tool_group.addButton(button)
            row.addWidget(button); self.placement_tools[name]=button
        self.placement_toolbar.setStyleSheet('''QToolButton {background:transparent;border:1px solid transparent;border-radius:5px;}
            QToolButton:hover {background:rgba(122,199,229,28);}
            QToolButton:checked {background:rgba(51,141,177,110);border:1px solid #7ac7e5;}''')
        self.workspace_header.addWidget(self.placement_toolbar)
        self.placement_toolbar.hide()
        panel=QWidget(); layout=QVBoxLayout(panel); layout.setContentsMargins(0,0,0,0)
        self.placement_insertion=QComboBox()
        self.placement_insertion.currentIndexChanged.connect(self._placement_insertion_changed)
        self.placement_insertion.setParent(panel); self.placement_insertion.hide()
        self._build_placement_table(layout)
        self._build_recording_library(layout)
        self.placement_target=QComboBox(); self.placement_target.addItems(['Plane','Probe'])
        self.placement_target.currentIndexChanged.connect(self._placement_target_changed)
        self.placement_target.setParent(panel); self.placement_target.hide()
        self.placement_plane_group=QGroupBox('Plane')
        self.placement_plane_group.setCheckable(True); self.placement_plane_group.setChecked(False)
        plane_layout=QVBoxLayout(self.placement_plane_group)
        self.placement_plane_box=QWidget(); form=QVBoxLayout(self.placement_plane_box)
        plane_layout.addWidget(self.placement_plane_box)
        self.placement_plane_group.toggled.connect(self.placement_plane_box.setVisible)
        form.setContentsMargins(0,0,0,0)
        head=QHBoxLayout(); head.addStretch()
        self.placement_plane_reset=QPushButton('Reset plane')
        self.placement_plane_reset.setIcon(tool_icon('reset'))
        self.placement_plane_reset.setToolTip('Restore the plane calculated from insertion points')
        self.placement_plane_reset.clicked.connect(self._placement_refit)
        head.addWidget(self.placement_plane_reset)
        form.addLayout(head)
        self.placement_plane_fields={}; self.placement_plane_sliders={}
        for name,title,limits,step,suffix in (
            ('center','Position',(-1e6,1e6),25,' µm'),('rotation','Rotation',(-180,180),1,'°')):
            group=QGroupBox(title); grid=QGridLayout(group)
            for i,axis in enumerate('XYZ'):
                spin=QDoubleSpinBox(); spin.setRange(*limits); spin.setDecimals(1)
                spin.setSingleStep(step); spin.setSuffix(suffix); spin.setKeyboardTracking(False)
                spin.valueChanged.connect(self._placement_plane_edited)
                slider=QSlider(Qt.Orientation.Horizontal)
                slider.setRange(-100000,100000) if name=='center' else slider.setRange(-1800,1800)
                factor=1 if name=='center' else 10
                slider.valueChanged.connect(lambda value,s=spin,f=factor:s.setValue(value/f) if not self._updating else None)
                grid.addWidget(QLabel(axis),i,0); grid.addWidget(slider,i,1); grid.addWidget(spin,i,2)
                self.placement_plane_fields[name,i]=spin
                self.placement_plane_sliders[name,i]=slider
            form.addWidget(group)
        normal=QFormLayout(); normal_row=QHBoxLayout()
        self.placement_normal=QDoubleSpinBox(); self.placement_normal.setRange(-50000,50000)
        self.placement_normal.setDecimals(1); self.placement_normal.setSuffix(' µm')
        self.placement_normal.setSingleStep(25); self.placement_normal.setKeyboardTracking(False)
        self.placement_normal_slider=QSlider(Qt.Orientation.Horizontal); self.placement_normal_slider.setRange(-2000,2000)
        self.placement_normal_slider.valueChanged.connect(self.placement_normal.setValue)
        self.placement_normal.valueChanged.connect(self._placement_normal_edited)
        normal_row.addWidget(self.placement_normal_slider,1); normal_row.addWidget(self.placement_normal)
        normal.addRow('Offset',normal_row); form.addLayout(normal)
        layout.addWidget(self.placement_plane_group)
        self.placement_probe_box=QWidget(); probe_layout=QVBoxLayout(self.placement_probe_box)
        probe_layout.setContentsMargins(0,0,0,0)
        head=QHBoxLayout(); head.addWidget(QLabel('Probe'),1)
        head.addWidget(view_button('rotate','Reverse probe direction',self._placement_flip))
        self.placement_face_button=view_button('face','Flip recording face and shank order',self._placement_flip_face)
        head.addWidget(self.placement_face_button)
        head.addWidget(view_button('clear','Remove probe',self._placement_delete))
        probe_layout.addLayout(head)
        form=QFormLayout(); self.placement_model=QComboBox()
        self._refresh_probe_models()
        self.placement_model.currentTextChanged.connect(self._placement_model_changed)
        self.placement_model.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.placement_model.setMinimumContentsLength(10); self.placement_model.setMinimumWidth(0)
        form.addRow('Model',self.placement_model)
        form.addRow(view_button('settings','Open Probes Library',self.owner.open_probes_library))
        self.placement_probe_fields={}
        for key,title,suffix,limits in (('u','U',' µm',(-1e6,1e6)),('v','V',' µm',(-1e6,1e6)),
                                        ('angle','Rotation','°',(-360,360))):
            spin=QDoubleSpinBox(); spin.setRange(*limits); spin.setSuffix(suffix); spin.setDecimals(1)
            spin.setSingleStep(1 if key=='angle' else 25); spin.setKeyboardTracking(False)
            spin.valueChanged.connect(self._placement_probe_edited)
            self.placement_probe_fields[key]=spin
            if key=='angle':
                angle_row=QHBoxLayout()
                self.placement_angle_slider=QSlider(Qt.Orientation.Horizontal); self.placement_angle_slider.setRange(-1800,1800)
                self.placement_angle_slider.valueChanged.connect(lambda value,s=spin:s.setValue(value/10) if not self._updating else None)
                angle_row.addWidget(self.placement_angle_slider,1); angle_row.addWidget(spin)
                form.addRow('Rotation',angle_row)
            else: spin.setParent(self.placement_probe_box); spin.hide()
        probe_layout.addLayout(form)
        layout.addWidget(self.placement_probe_box)
        layout.addStretch()
        self.controls.removeWidget(self.plane_tabs); self.plane_tabs.setParent(self.control_content); self.plane_tabs.hide()
        self.controls.insertWidget(1,panel)
        self.placement_panel=panel
        for check in self.volume_view.layers.checks:
            check.toggled.connect(self._placement_layer_changed)
        self.volume_view.layers.labels_check.toggled.connect(self._placement_layer_changed)
        canvas=self.track_slice
        canvas.probePreview.connect(self._placement_preview)
        canvas.probePlaced.connect(self._placement_commit)
        canvas.probeMoved.connect(self._placement_move)
        canvas.probeRotated.connect(self._placement_rotate)
        canvas.probeSelected.connect(self._placement_select)
        canvas.probeDeleted.connect(self._placement_delete)
        canvas.probePreviewLeft.connect(self._placement_clear_ghost)
        canvas.probeCancelled.connect(self._placement_cancel)
        self.placement_plane_box.hide()
        self._build_surface_entries_controls(display_layout)

    def _configure_placement_page(self):
        active=self._mode==1
        self.placement_toolbar.setVisible(active)
        self.track_slice.placement_enabled=active
        if active:
            self._placement_ghost=self._placement_move_base=None
            self._placement_refresh_controls()
            self._placement_row_active=True
            self._placement_tool('select')
            # Start with the registered electrode channel visible in 3D.
            self.volume_view.layers.checks[1].blockSignals(True)
            self.volume_view.layers.checks[1].setChecked(True)
            self.volume_view.layers.checks[1].blockSignals(False)
            if self.opacity[1].value()==0: self.opacity[1].setValue(100)
        else:
            self.track_slice.placement_selected=False
            for i in (0,): self.volume_view.layers.checks[i].setEnabled(True)
            self.volume_view.layers.labels_check.setEnabled(True)
        self.slice_display.channel.setVisible(not active)
        if active: self.slice_display.channel.setCurrentIndex(0)
        self._sync_surface_entries_mode()

    def _placement_target_changed(self,*args):
        plane=self.placement_target.currentIndex()==0
        self.placement_plane_group.setChecked(plane); self.placement_probe_box.show()

    def _placement_refresh_controls(self):
        if self.project is None or self.session is None: return
        self._updating=True
        self.placement_insertion.clear()
        for item in self.session['insertions']: self.placement_insertion.addItem(item['name'],item['id'])
        self.placement_insertion.setCurrentIndex(self.placement_insertion.findData(self.insertion_id))
        item=self.current_insertion()
        if item:
            native=source_plane(item,self.session['plane'],self.project.total)
            display=transform_plane(native,self._tracing_matrix())
            for name,values in (('center',display['center_um_zyx'][::-1]),('rotation',display['rotation_xyz_deg'])):
                for axis,value in enumerate(values):
                    self.placement_plane_fields[name,axis].setValue(value)
                    self.placement_plane_sliders[name,axis].setValue(round(value*(1 if name=='center' else 10)))
            self.placement_normal.setValue(0); self.placement_normal_slider.setValue(0)
            self._placement_normal_base=copy.deepcopy(native)
            s=probe_settings(item)
            if self.placement_model.findText(s['geometry']['model'])<0: self.placement_model.addItem(s['geometry']['model'])
            self.placement_model.setCurrentText(s['geometry']['model'])
            self.placement_face_button.setToolTip('Rotate 180° about the element axis' if s['geometry'].get('geometry_version',1)>=2 else 'Flip recording face and shank order')
            p=s.get('pose'); g=plane_geometry(native)
            if p:
                center=probe_coordinates(s,(s['geometry']['shanks']-1)/2,0)-np.asarray(g['center_um_zyx'])
                self._placement_angle=float(np.rad2deg(np.arctan2(-np.dot(p['down_source_zyx'],g['u_zyx']),-np.dot(p['down_source_zyx'],g['v_zyx']))))
                for key,value in (('u',center@g['u_zyx']),('v',center@g['v_zyx'])):
                    self.placement_probe_fields[key].setValue(0 if abs(value)<.05 else value)
            self.placement_probe_fields['angle'].setValue(self._placement_angle)
            self.placement_angle_slider.setValue(round(self._placement_angle*10))
            for key in ('u','v'): self.placement_probe_fields[key].setEnabled(p is not None)
        self.placement_plane_group.setEnabled(item is not None)
        self.placement_probe_box.setEnabled(item is not None)
        self._refresh_placement_table()
        self._refresh_recording_library()
        self._updating=False

    def _placement_insertion_changed(self,*args):
        if self._updating or self._mode!=1 or self.session is None: return
        self.insertion_id=self.placement_insertion.currentData()
        self._placement_row_active=True
        self._placement_ghost=self._placement_move_base=None
        self._restore_insertion_plane(); self._placement_refresh_controls()
        self.track_slice.fit_image(); self.request_render()

    def _placement_plane_edited(self,*args):
        if self._updating or self._mode!=1 or self.current_insertion() is None: return
        self._remember_plane()
        plane=copy.deepcopy(source_plane(self.current_insertion(),self.session['plane'],self.project.total))
        display=transform_plane(plane,self._tracing_matrix())
        display['center_um_zyx']=[self.placement_plane_fields['center',i].value() for i in (2,1,0)]
        display['rotation_xyz_deg']=[self.placement_plane_fields['rotation',i].value() for i in range(3)]
        self._placement_set_plane(transform_plane(display,np.linalg.inv(self._tracing_matrix())))

    def _placement_set_plane(self,plane,refresh=True):
        item=self.current_insertion(); old=source_plane(item,self.session['plane'],self.project.total)
        # A placed probe belongs to this plane: carry its tip and orientation
        # through the rigid change of plane, without changing hardware spacing.
        p=probe_settings(item).get('pose')
        if p:
            old_g,new_g=plane_geometry(old),plane_geometry(plane)
            old_basis=np.column_stack([old_g[k] for k in ('u_zyx','v_zyx','normal_zyx')])
            new_basis=np.column_stack([new_g[k] for k in ('u_zyx','v_zyx','normal_zyx')])
            rotation=new_basis@old_basis.T
            p=copy.deepcopy(p)
            p['tip0_source_zyx_um']=(np.asarray(new_g['center_um_zyx'])+rotation@(np.asarray(p['tip0_source_zyx_um'])-old_g['center_um_zyx'])).tolist()
            for k in ('down_source_zyx','lateral_source_zyx','normal_source_zyx'): p[k]=(rotation@p[k]).tolist()
        save_source_plane(item,plane,self.project.total)
        self.session['plane']=copy.deepcopy(item['plane'])
        if p: place_probe(item,p,self.project,self.session)
        self._placement_ghost=None
        if refresh: self._placement_refresh_controls()
        self.request_render(); self.owner.schedule_save()

    def _placement_normal_edited(self,value):
        if self._updating or self._mode!=1 or self.current_insertion() is None: return
        self._remember_plane()
        plane=copy.deepcopy(self._placement_normal_base)
        plane['center_um_zyx']=(np.asarray(plane['center_um_zyx'])+value*np.asarray(plane_geometry(plane)['normal_zyx'])).tolist()
        self.placement_normal_slider.blockSignals(True); self.placement_normal_slider.setValue(round(value)); self.placement_normal_slider.blockSignals(False)
        self._placement_set_plane(plane,refresh=False)
        display=transform_plane(plane,self._tracing_matrix())
        for axis,value in enumerate(display['center_um_zyx'][::-1]):
            spin=self.placement_plane_fields['center',axis]; spin.blockSignals(True); spin.setValue(value); spin.blockSignals(False)
            slider=self.placement_plane_sliders['center',axis]
            slider.blockSignals(True); slider.setValue(round(value)); slider.blockSignals(False)

    def _placement_refit(self):
        item=self.current_insertion()
        if self._mode!=1 or item is None: return
        from .insertion_model import fit_insertion_plane
        try:
            # Fit a copy first: invalid/collinear points must leave the saved
            # plane, probe and undo history intact. The same native-space SVD
            # recovers the original fit, including after saving and reopening.
            fitted=copy.deepcopy(item)
            fit_insertion_plane(fitted,self.session['plane'],self.project.total)
        except ValueError as error:
            self.owner._error(str(error)); return
        self._remember()
        self._placement_set_plane(fitted['source_plane'],refresh=False)
        item['plane_fit']=fitted['plane_fit']
        self._placement_refresh_controls()
        self._update_plane_status()

    def _placement_tool(self,tool):
        self._placement_clear_ghost()
        self.track_slice.placement_tool=tool
        self.track_slice.drawing=False
        self.track_slice.set_interaction_tool('measure' if tool=='measure' else 'annotate')
        self.track_slice.point_mode='view'
        for name,button in self.placement_tools.items(): button.setChecked(name==tool)
        if tool=='draw':
            if not self._placement_row_active and self.session and self.session['insertions']:
                first=self.session['insertions'][0]['id']
                self.placement_insertion.setCurrentIndex(self.placement_insertion.findData(first))
            self._placement_select_row(self.insertion_id)
            if self.current_insertion(): self.current_insertion()['probe_visible']=True
            self._refresh_placement_table()
            self.placement_target.setCurrentIndex(1)
        self.track_slice.setToolTip('Click at the deep tips · Drag toward the base · Shift+drag: pan · Pinch or wheel: zoom' if tool=='draw' else
            'Drag probe: move · Drag circular handle: rotate · Drag background: pan · Pinch or wheel: zoom · Delete: remove probe')

    def _placement_pose_at(self,start,end):
        item=self.current_insertion()
        if item is None or not self._cache_current(): return None
        native=source_plane(item,self.session['plane'],self.project.total)
        start=transform_points([start],np.linalg.inv(self._tracing_matrix()))[0]
        angle=self.placement_probe_fields['angle'].value()
        if end is not None:
            end=transform_points([end],np.linalg.inv(self._tracing_matrix()))[0]
            vector=end-start; g=plane_geometry(native)
            if np.linalg.norm(vector)>1: angle=float(np.rad2deg(np.arctan2(vector@g['u_zyx'],vector@g['v_zyx'])))
        return pose_on_plane(item,native,start,angle)

    def _placement_preview(self,start,end):
        if self._mode!=1 or self.current_insertion() is None: return
        self._placement_ghost=self._placement_pose_at(start,end)
        self._draw_placement_annotations()

    def _placement_clear_ghost(self):
        self._placement_ghost=None
        if self._mode==1 and self.track_native is not None: self._draw_placement_annotations()

    def _placement_commit(self,start,end):
        pose=self._placement_pose_at(start,end)
        if pose is None: return
        self._remember(); place_probe(self.current_insertion(),pose,self.project,self.session)
        self._placement_ghost=None; self._placement_move_base=None
        self._placement_select(True); self._placement_tool('select')
        self._placement_refresh_controls(); self.draw_cached(); self.owner.schedule_save()

    def _placement_select(self,selected):
        self.track_slice.placement_selected=selected
        self._placement_move_base=None
        self._placement_rotate_base=None
        if selected: self._placement_select_row(self.insertion_id)
        elif hasattr(self,'placement_table'):
            self._placement_row_active=False
            self.placement_table.blockSignals(True); self.placement_table.clearSelection(); self.placement_table.setCurrentItem(None); self.placement_table.blockSignals(False)
        if selected: self.placement_target.setCurrentIndex(1)
        self.track_slice.update()

    def _placement_cancel(self):
        item=self.current_insertion()
        if item and self._placement_move_base is not None:
            place_probe(item,self._placement_move_base,self.project,self.session)
        if item and self._placement_rotate_base is not None:
            place_probe(item,self._placement_rotate_base,self.project,self.session)
        self._placement_move_base=None
        self._placement_rotate_base=None
        self._placement_tool('select'); self._placement_select(False)
        self._placement_refresh_controls(); self._draw_placement_annotations()

    def _placement_move(self,delta,finished):
        item=self.current_insertion()
        if item is None or not probe_settings(item).get('pose') or not self._cache_current(): return
        if self._placement_move_base is None:
            self._remember(); self._placement_move_base=copy.deepcopy(item['probe']['pose'])
        pose=copy.deepcopy(self._placement_move_base)
        pose['tip0_source_zyx_um']=(np.asarray(pose['tip0_source_zyx_um'])+np.linalg.inv(self._tracing_matrix())[:3,:3]@delta).tolist()
        place_probe(item,pose,self.project,self.session)
        self._draw_placement_annotations()
        if finished:
            self._placement_move_base=None; self._placement_refresh_controls(); self.owner.schedule_save()

    def _placement_rotate(self,start,end,finished):
        item=self.current_insertion()
        if item is None or not probe_settings(item).get('pose') or not self._cache_current(): return
        if self._placement_rotate_base is None:
            self._remember(); self._placement_rotate_base=copy.deepcopy(item['probe']['pose'])
        p=self._placement_rotate_base
        native=source_plane(item,self.session['plane'],self.project.total); g=plane_geometry(native)
        center=np.asarray(p['tip0_source_zyx_um'])+((item['probe']['geometry']['shanks']-1)/2)*item['probe']['geometry']['shank_pitch_um']*np.asarray(p['lateral_source_zyx'])
        positions=transform_points([start,end],np.linalg.inv(self._tracing_matrix()))-center
        angles=np.arctan2(positions@np.asarray(g['u_zyx']),positions@np.asarray(g['v_zyx']))
        before=np.arctan2(-np.dot(p['down_source_zyx'],g['u_zyx']),-np.dot(p['down_source_zyx'],g['v_zyx']))
        angle=float((np.rad2deg(before+angles[1]-angles[0])+180)%360-180)
        place_probe(item,pose_on_plane(item,native,center,angle),self.project,self.session)
        self._draw_placement_annotations()
        self._updating=True
        self.placement_probe_fields['angle'].setValue(angle); self.placement_angle_slider.setValue(round(angle*10))
        self._updating=False
        self._refresh_placement_table()
        if finished:
            self._placement_rotate_base=None; self._placement_refresh_controls(); self.owner.schedule_save()

    def _placement_probe_edited(self,*args):
        if self._updating or self._mode!=1 or self.current_insertion() is None: return
        self._placement_angle=self.placement_probe_fields['angle'].value()
        self.placement_angle_slider.blockSignals(True); self.placement_angle_slider.setValue(round(self._placement_angle*10)); self.placement_angle_slider.blockSignals(False)
        item=self.current_insertion()
        if not probe_settings(item).get('pose'): return
        self._remember_plane(); native=source_plane(item,self.session['plane'],self.project.total); g=plane_geometry(native)
        center=np.asarray(g['center_um_zyx'])+self.placement_probe_fields['u'].value()*np.asarray(g['u_zyx'])+self.placement_probe_fields['v'].value()*np.asarray(g['v_zyx'])
        place_probe(item,pose_on_plane(item,native,center,self._placement_angle),self.project,self.session)
        self._draw_placement_annotations(); self._refresh_placement_table(); self.owner.schedule_save()

    def _placement_flip(self):
        box=self.placement_probe_fields['angle']; box.setValue((box.value()+360)%360-180)

    def _placement_flip_face(self):
        item=self.current_insertion()
        if self._mode!=1 or item is None: return
        self._remember()
        native=source_plane(item,self.session['plane'],self.project.total)
        pose=flip_contact_face(item,native)
        if pose: place_probe(item,pose,self.project,self.session)
        self._placement_ghost=None
        self._placement_refresh_controls(); self._draw_placement_annotations(); self.owner.schedule_save()

    def _refresh_probe_models(self):
        from ..probe_library import ProbeLibrary
        combo=self.placement_model; current=combo.currentText()
        combo.blockSignals(True); combo.clear()
        combo.addItems([e['geometry']['model'] for e in ProbeLibrary().entries()])
        if current and combo.findText(current)<0: combo.addItem(current)
        if current: combo.setCurrentText(current)
        combo.blockSignals(False)

    def _placement_model_changed(self,model):
        if self._updating or self._mode!=1 or self.current_insertion() is None: return
        item=self.current_insertion(); s=probe_settings(item)
        if model==s['geometry']['model']: return
        try: geometry=geometry_for(model)
        except ValueError as error: self.owner._error(str(error)); return
        # A recording's hardware identity must not be silently reinterpreted.
        if item.get('recordings'):
            self.owner._error('Unassign the recording maps before changing the probe model. The maps describe the current hardware.')
            self._placement_refresh_controls(); return
        self._remember(); p=copy.deepcopy(s.get('pose'))
        center=probe_coordinates(s,(s['geometry']['shanks']-1)/2,0.) if p else None
        s['geometry']=geometry
        if p:
            p['tip0_source_zyx_um']=(center-((geometry['shanks']-1)/2)*geometry['shank_pitch_um']*np.asarray(p['lateral_source_zyx'])).tolist()
            place_probe(item,p,self.project,self.session)
        self._placement_refresh_controls(); self._draw_placement_annotations(); self.owner.schedule_save()

    def _placement_delete(self):
        item=self.current_insertion()
        if self._mode!=1 or item is None or not probe_settings(item).get('pose'): return
        self._remember(); item['probe'].pop('pose',None); item['probe']['assignments']={}
        item['tracks']=[t for t in item['tracks'] if not t.get('probe_placement_generated')]
        from .insertion_model import sync_tracks
        sync_tracks(self.session,self.project)
        self._placement_select(False); self._placement_refresh_controls(); self.draw_cached(); self.owner.schedule_save()

    def _placement_restore_camera(self):
        if self._mode!=1: return
        camera=self._placement_cameras.get(self._placement_scene)
        if camera: self.volume_view.restore_camera(camera)
        else: self.volume_view.reset_view()

    def _placement_layer_changed(self,*args):
        if self._mode!=1 or self._updating or self.track_native is None: return
        layers=self.volume_view.layers
        if args and args[0] is True and self.sender() is layers.checks[0] and self.opacity[0].value()==0:
            self.opacity[0].setValue(65)
        self._update_placement_volume(); self._draw_placement_annotations(); self._placement_show_plane()

    def _update_placement_volume(self):
        # All 3D scan layers share the atlas frame, even with Atlas hidden.
        self.volume_view.update_alignment(self.project.total,*[s.value()/100 for s in self.opacity])
        self.volume_view.set_label_styles(self.session['regions'])

    def _placement_matrix(self):
        """The complete source-to-atlas map is for 3D display only."""
        return self.project.total

    def _placement_geometry(self):
        item=self.current_insertion()
        plane=source_plane(item,self.session['plane'],self.project.total) if item else transform_plane(self.session['plane'],np.linalg.inv(self.project.total))
        corners=[transformed_corners(self.project.track.shape_zyx,self.project.track.spacing_um_zyx,np.eye(4))]
        return plane_geometry(plane_covering_volumes(plane,corners))

    def _placement_show_plane(self):
        if self.volume_view.plane_mesh is None: return
        geometry=transform_plane_geometry(self._placement_geometry(),self._placement_matrix())
        self.volume_view.set_plane(geometry)
        self.volume_view.plane_mesh.visible=self.volume_view.plane_outline.visible=not self._surface_entries_active()

    def _draw_placement_annotations(self):
        if self.session is None or self.project is None or self.track_native is None: return
        from .insertion_model import sync_measurements
        sync_measurements(self.session)
        display=self._tracing_matrix(); inverse=np.linalg.inv(self._tracing_to_atlas())
        canvas=self.track_slice
        canvas.insertion_tracks=[]; canvas.track_lines=[]; canvas.show_tracks=False; canvas.track_hits=False
        canvas.insertion_points=[]; canvas.show_points=False; canvas.probe_sites=[]
        canvas.measurements=[dict(m,points_atlas_zyx_um=transform_points(m['points_atlas_zyx_um'],inverse).tolist()) for m in self.session.get('measurements',[])]
        canvas.measurement_inverse=np.linalg.inv(display)[:3,:3]; canvas.measurement_unit=self.measurement_unit.currentText()
        canvas.probe_polygons=[]; canvas.placement_contacts=[]
        canvas.placement_pivot=canvas.placement_base=None
        self.volume_view.set_tracks([])
        points=[dict(p,color=item['color']) for item in self.session['insertions']
                if item.get('points_visible',item.get('visible',True)) for p in item['points']]
        self.volume_view.set_points(points,(),not self._surface_entries_active())
        self._hide_probe_visuals()
        self._placement_visuals=getattr(self,'_placement_visuals',{})
        self._placement_contact_visuals=getattr(self,'_placement_contact_visuals',{})
        for mesh in [*self._placement_visuals.values(),*self._placement_contact_visuals.values()]: mesh.visible=False
        for item in self.session['insertions']:
            if not item.get('probe_visible',True): continue
            drawn=item
            ghost=item['id']==self.insertion_id and self._placement_ghost is not None
            if ghost:
                drawn=copy.deepcopy(item); probe_settings(drawn)['pose']=self._placement_ghost
            if not drawn.get('probe',{}).get('pose'): continue
            rgb=QColor(item['color']).getRgbF()[:3]
            if item['id']==self.insertion_id:
                canvas.probe_polygons=[(poly.tolist(),(*rgb,.24 if ghost else .4)) for poly in probe_contours(drawn,display)]
                canvas.placement_contacts=contact_preview(drawn,display).tolist()
                native=source_plane(item,self.session['plane'],self.project.total)
                canvas.placement_contact_front=contact_side(drawn,native)>0
                center=probe_coordinates(drawn['probe'],(drawn['probe']['geometry']['shanks']-1)/2,0.)
                canvas.placement_pivot,canvas.placement_base=transform_points([center,center-np.asarray(drawn['probe']['pose']['down_source_zyx'])*500],display)
            if self.volume_view.view is not None and not self._surface_entries_active():
                from vispy import scene
                if item['id'] not in self._placement_visuals:
                    mesh=scene.visuals.Mesh(parent=self.volume_view.view.scene,shading=None)
                    mesh.set_gl_state('translucent',depth_test=True,cull_face=False); mesh.order=24
                    self._placement_visuals[item['id']]=mesh
                vertices,faces=shank_mesh(drawn,self._placement_matrix())
                mesh=self._placement_visuals[item['id']]
                mesh.set_data(vertices=vertices,faces=faces,color=(*rgb,.28)); mesh.visible=True
                if item['id'] not in self._placement_contact_visuals:
                    contacts=scene.visuals.Mesh(parent=self.volume_view.view.scene,shading=None)
                    contacts.set_gl_state('translucent',depth_test=True,cull_face=False); contacts.order=25
                    self._placement_contact_visuals[item['id']]=contacts
                sites=contact_preview(drawn,np.eye(4))
                vertices,faces=contact_mesh(drawn,sites,self._placement_matrix())
                contacts=self._placement_contact_visuals[item['id']]
                contacts.set_data(vertices=vertices,faces=faces,color=(.82,.96,1,.85)); contacts.visible=True
                if item['id']==self.insertion_id:
                    self.volume_view.show_probe_face(drawn,self._placement_matrix())
            self._draw_placement_recordings(drawn,display,
                draw_3d=self.volume_view.view is not None and not self._surface_entries_active())
        if self._surface_entries_active(): self._draw_surface_entries()
        canvas.tools.hide(); canvas.update()
        if self.volume_view.canvas is not None: self.volume_view.canvas.update()

    def _render_placement(self):
        if self._running:
            self._pending=True; return
        from .desktop_app import Job
        from qtpy.QtCore import QThreadPool
        token=self._token; geometry=self._placement_geometry(); volume=self.track_native
        display=self._tracing_matrix().copy()
        def render(progress):
            pixels=sample_world(volume,np.eye(4),plane_world(geometry)).reshape(geometry['shape_hw'])
            return {'token':token,'mode':1,'images':[(pixels,None,transform_plane_geometry(geometry,display),None)]}
        job=Job(render); self._jobs.add(job); self._running=True; self._pending=False
        def finish(result=None,error=None):
            self._jobs.discard(job); self._running=False
            if self._closing: return
            if self._mode==1 and self._token==token:
                if error: self.owner.statusBar().showMessage(error)
                else: self._cache=result; self.draw_cached()
            if self._pending or self._token!=token: self.render_timer.start(0)
        job.signals.done.connect(lambda result:finish(result=result))
        job.signals.failed.connect(lambda error:finish(error=error))
        QThreadPool.globalInstance().start(job)

    def _draw_placement(self,annotations_only=False):
        self._draw_placement_annotations()
        if annotations_only or not self._cache or self._cache['mode']!=1: return
        from .desktop_slice_display import slice_color
        pixels,_,geometry,_=self._cache['images'][0]
        settings=self.slice_display.records[0]
        self.slice_display.observe(0,pixels)
        self.track_slice.smooth_interpolation=settings['interpolation']=='linear'
        self.track_slice.display_interpolation=settings['interpolation']
        self.track_slice.set_plane_image(slice_color(pixels,settings),geometry)
        from .desktop_figures import slice_sampler
        self.track_slice.figure_sampler=slice_sampler([self.track_native],[self._tracing_matrix()],[settings],visible_pixels=[pixels])
        self.slice_titles[0].setText('Electrodes')

    def apply_placement(self):
        if self._mode!=1: return self.apply_tracing()
        from .neuropixels_model import find_surface_depths,pose_current
        for item in self.session['insertions']:
            if pose_current(item,self.project.total) and self.brain_mask is not None:
                find_surface_depths(item,self.brain_mask,self.brain_preview.spacing_um_zyx)
        self.validate_starts(); self.capture_state(); self.owner.schedule_save()

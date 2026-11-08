"""One ordered row of independently resizable, pixel-aligned analysis plots."""
from pathlib import Path
import re
import copy
import numpy as np
from qtpy.QtCore import Qt,QTimer,QEvent,QObject
from qtpy.QtWidgets import (QWidget,QVBoxLayout,QHBoxLayout,QLabel,QListWidget,QListWidgetItem,
    QAbstractItemView,QCheckBox,QSplitter,QScrollArea,QFileDialog,QSlider,QPushButton,QComboBox,QMenu,QFormLayout,QSpinBox)
from .desktop_ephys_settings import PlotInspector,Fold,defaults,number
from .ephys_analysis import BANDS
from .desktop_ephys_plots import PLOTS,SPECS,COLORS,TOP,BOTTOM,render_plot
from .desktop_ephys_probe import draw_probe,depth_limits,probe_geometry,probe_draw_key
from .desktop_ephys_legends import plot_legend,anatomy_legend,place_legends,footer_height


class PlotBoard(QObject):
    def __init__(self,page,side,host):
        super().__init__(page); self.page=page; self.panels={}; self._restoring=False; self._ylim=None; self._record_id=None
        self.options={}; self.inspectors={}; self.active_updating=False
        self._pending_panels=set(); self._pending_analysis=False
        self.settings_timer=QTimer(self); self.settings_timer.setSingleShot(True)
        self.settings_timer.timeout.connect(self.flush_panels)
        self.add_plot=QPushButton('Add plot…'); self.add_plot.clicked.connect(self.plot_menu); side.addWidget(self.add_plot)
        self.list=QListWidget(); self.list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove); self.list.setFixedHeight(0); self.list.hide()
        for spec in PLOTS:
            item=QListWidgetItem(spec.title); item.setData(Qt.ItemDataRole.UserRole,spec.key)
            item.setFlags(item.flags()|Qt.ItemFlag.ItemIsUserCheckable|Qt.ItemFlag.ItemIsDragEnabled|Qt.ItemFlag.ItemIsDropEnabled)
            item.setCheckState(Qt.CheckState.Checked if spec.key in ('lfp','rms') else Qt.CheckState.Unchecked)
            item.setToolTip('Shared depth axis' if spec.depth else 'Natural units on Y; depth levels do not apply')
            self.list.addItem(item)
        self.list.itemChanged.connect(self.changed); self.list.model().rowsMoved.connect(self.changed); side.addWidget(self.list)
        self.legends=QCheckBox('Legends'); self.legends.setChecked(True)
        self.legends.toggled.connect(self.changed); side.addWidget(self.legends)
        self.active_list=QListWidget(); self.active_list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.active_list.itemChanged.connect(self.active_changed); self.active_list.model().rowsMoved.connect(self.active_reordered); side.addWidget(self.active_list)
        self.notebook_mode=QCheckBox('Selected interval')
        self.notebook_mode.setToolTip('Analyze the selected time interval using the signal-processing settings. Prepared signal uses the local cache; original-source filtering may require large server reads.')
        self.notebook_mode.toggled.connect(self.interval_changed); page.window_controls.addWidget(self.notebook_mode)
        self.profile_source=QComboBox(); self.profile_source.addItem('Prepared LFP · fast','cached'); self.profile_source.addItem('Original recording · reference filters','original')
        self.profile_source.setToolTip('Fast: uses the prepared LFP without a server read. Original: repeats filtering and FFT from the raw source; reads up to 35–70 s of raw data.')
        self.profile_source.currentIndexChanged.connect(self.profile_source_changed)
        self.prepare_window=QPushButton('Update analysis'); self.prepare_window.clicked.connect(lambda:page._load_notebook())
        page.window_controls.addWidget(self.prepare_window)
        side.addWidget(QLabel('Electrode columns'))
        row=QHBoxLayout(); self.columns={}; self.column_opacity={}; self.opacity_labels={}
        for key,label,checked in [('0','Column 1',True),('1','Column 2',True),('mean','Mean',False)]:
            group=QVBoxLayout(); check=QCheckBox(label); check.setChecked(checked)
            if key=='mean': check.setToolTip('Average both electrode columns; both are highlighted on the probe.')
            check.toggled.connect(self.changed); self.columns[key]=check; group.addWidget(check)
            opacity_row=QHBoxLayout(); slider=QSlider(Qt.Orientation.Horizontal); slider.setRange(0,100); slider.setValue(100)
            slider.setMinimumWidth(35); label=QLabel('100%'); label.setMinimumWidth(34)
            slider.valueChanged.connect(self.opacity_changed); self.column_opacity[key]=slider; self.opacity_labels[key]=label
            opacity_row.addWidget(slider,1); opacity_row.addWidget(label); group.addLayout(opacity_row); row.addLayout(group,1)
        side.addLayout(row)
        processing=Fold('Signal processing'); self.processing=processing; side.addWidget(processing)
        form=QFormLayout(); processing.content.addLayout(form); form.addRow('Signal source',self.profile_source)
        self.analysis_fields={}
        for key,label,field in [('c_factor','Spatial correction',QCheckBox('C-factor')),('sigma','Smoothing · sites',number(3,0,20)),
            ('reference','Reference',QCheckBox('Common median')),('segments','Linear segments',QSpinBox()),
            ('lowpass','Low-pass traces · Hz',number(40,1,450))]:
            if key=='segments': field.setRange(2,12); field.setValue(6)
            if key=='c_factor': field.setChecked(True)
            form.addRow(label,field); self.analysis_fields[key]=field
        for key,label,values,initial in [('nperseg','Welch window · samples',[128,256,512,1024,2048,4096],256),('nfft','FFT length',[512,1024,2048,4096,8192],4096)]:
            field=QComboBox()
            for value in values: field.addItem(str(value),value)
            field.setCurrentText(str(initial)); form.addRow(label,field); self.analysis_fields[key]=field
        for field in self.analysis_fields.values():
            signal=field.toggled if isinstance(field,QCheckBox) else field.currentIndexChanged if isinstance(field,QComboBox) else field.valueChanged
            signal.connect(self.analysis_setting_changed)
        self.inspector_layout=QVBoxLayout(); side.addLayout(self.inspector_layout)
        self.scroll=QScrollArea(); self.scroll.setWidgetResizable(True); self.scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self.splitter=QSplitter(Qt.Orientation.Horizontal); self.splitter.setChildrenCollapsible(False); self.splitter.setHandleWidth(7)
        self.scroll.setWidget(self.splitter); host.addWidget(self.scroll)
        self.splitter.splitterMoved.connect(self.align_later)
        self.timer=QTimer(self); self.timer.setSingleShot(True); self.timer.timeout.connect(self.align)
        page.visual_splitter.splitterMoved.connect(self.align_later)
        self.scroll.viewport().installEventFilter(self)
        page.anatomy_canvas.installEventFilter(self)
        self.connect_canvas(page.anatomy_canvas)
        self.widths={}
        self.rebuild()

    def plot_options(self,key): return self.options.setdefault(key,defaults(key))

    def analysis_options(self):
        values={k:(f.isChecked() if isinstance(f,QCheckBox) else f.currentData() if isinstance(f,QComboBox) else f.value()) for k,f in self.analysis_fields.items()}
        values['duration']=self.page.window_duration.value()
        values['bands']=[[b['low'],b['high']] for b in self.plot_options('lfp')['bands']]
        return values

    def analysis_setting_changed(self,*args):
        if self._restoring or self.page._updating: return
        nfft=self.analysis_fields['nfft']; segment=self.analysis_fields['nperseg']
        if nfft.currentData()<segment.currentData():
            self._restoring=True
            if self.sender() is nfft: segment.setCurrentText(str(nfft.currentData()))
            else: nfft.setCurrentText(str(segment.currentData()))
            self._restoring=False
        self.notebook_mode.setChecked(True); self.page._analysis_changed()

    def queue_panel(self,key):
        self._pending_panels.add(key); self.settings_timer.start(60)

    def flush_panels(self):
        if self.page._closing: self._pending_panels.clear(); return
        keys=self._pending_panels; self._pending_panels=set()
        analysis=self._pending_analysis; self._pending_analysis=False
        if analysis:
            self.page.draw()
            if self.needs_notebook() and self.profile_source.currentData()=='cached': self.page.window_timer.start(350)
        else:
            for key in keys:
                if key=='__anatomy__': self.render_anatomy()
                else: self.render_panel(key)
            if keys:
                self.align(); self.page.levels.paint(draw=False)

    def render_anatomy(self,connect=True):
        page=self.page
        probe_key=(probe_draw_key(page),self.legends.isChecked())
        if getattr(self,'_probe_draw_key',None)!=probe_key:
            retained=[]
            for ax,cid in page._depth_callbacks:
                if ax is page.anatomy_ax: ax.callbacks.disconnect(cid)
                else: retained.append((ax,cid))
            page._depth_callbacks=retained
            draw_probe(page); self._probe_draw_key=probe_key
            for legend in list(page.anatomy_figure.legends): legend.remove()
            anatomy_legend(page,page.anatomy_figure,page.anatomy_ax)
            if connect:
                page._depth_callbacks.append((page.anatomy_ax,page.anatomy_ax.callbacks.connect('ylim_changed',page._sync_depth)))
        page.anatomy_ax.set_ylim(self.depth_range(),emit=False)
        page.anatomy_ax.set_title(page.anatomy_caption.text(),fontsize=10,pad=30,color='#dce6f0')

    def render_panel(self,key):
        if key not in self.keys() or key not in self.panels: return
        page=self.page; panel=self.panels[key]
        old_axes=panel['axes']
        remaining=[]
        for ax,cid in page._depth_callbacks:
            if ax in old_axes: ax.callbacks.disconnect(cid)
            else: remaining.append((ax,cid))
        page._depth_callbacks=remaining
        page._depth_axes=[ax for ax in page._depth_axes if ax not in old_axes]
        page.levels.preview=None; page.levels.backgrounds.clear()
        figure=panel['figure']; figure.clear(); figure._ephys_colorbars=[]
        panel['axes']=render_plot(page,figure,key,(.2,.1,.75,.8))
        plot_legend(page,figure,key,panel['axes'])
        panel['canvas'].setMinimumWidth(210 if key=='waveform' else 210*len(panel['axes']))
        if SPECS[key].depth:
            page._depth_axes.extend(panel['axes'])
            for ax in panel['axes']:
                page._depth_callbacks.append((ax,ax.callbacks.connect('ylim_changed',page._sync_depth)))
        panel['canvas'].draw_idle()
        if key=='lfp' and page.summary and not self.notebook_mode.isChecked() and not page._busy:
            options=self.plot_options('lfp')
            if (options['fits'] or options['transitions']) and page._overview_attempted!=page._overview_key():
                page.profile_timer.start(120)

    def plot_settings_changed(self,key,analysis=False):
        if self._restoring: return
        if analysis:
            self.page.notebook=None; self.page._notebook_attempted=None
            self.page.profile_fits=None; self.page._overview_attempted=None
            self._pending_analysis=True
        self.queue_panel(key); self.page.owner.schedule_save()

    def plot_menu(self):
        menu=QMenu(self.page)
        categories={'Depth profiles':{'lfp','lfp_gradient','rms','correlation_profile','density'},'Maps and traces':{'spectrum','correlation','traces','traces_clustered','low40','low40_plain','csd','activity'},
            'Units':{'raster','waveform','isi','acg','rate','amplitude'},'Regions':{'region_lfp','region_rms','region_units'},'Compare recordings':{'compare_lfp','compare_rms'}}
        for title,keys in categories.items():
            sub=menu.addMenu(title)
            for spec in PLOTS:
                if spec.key not in keys: continue
                action=sub.addAction(spec.title); action.setCheckable(True); action.setChecked(spec.key in self.keys())
                action.toggled.connect(lambda on,key=spec.key:self.toggle_plot(key,on))
        menu.exec(self.add_plot.mapToGlobal(self.add_plot.rect().bottomLeft()))

    def toggle_plot(self,key,on):
        for i in range(self.list.count()):
            item=self.list.item(i)
            if item.data(Qt.ItemDataRole.UserRole)==key:
                item.setCheckState(Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)
                if on and key in self.inspectors:
                    self.inspectors[key].toggle.setChecked(True); self.page.settings_tabs.setCurrentIndex(1)
                break

    def active_changed(self,item):
        if not self.active_updating: self.toggle_plot(item.data(Qt.ItemDataRole.UserRole),item.checkState()==Qt.CheckState.Checked)

    def active_reordered(self,*args):
        if self.active_updating: return
        order=[self.active_list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.active_list.count())]
        self._restoring=True
        items=[]
        while self.list.count(): items.append(self.list.takeItem(0))
        positions={key:i for i,key in enumerate(order)}
        items.sort(key=lambda item:positions.get(item.data(Qt.ItemDataRole.UserRole),len(order)))
        for item in items: self.list.addItem(item)
        self._restoring=False
        self.rebuild(); self.page.owner.schedule_save()

    def inspector_opened(self,key,on):
        if not on: return
        for other,inspector in self.inspectors.items():
            if other!=key: inspector.toggle.setChecked(False)

    def refresh_inspectors(self):
        keys=self.keys(); self.active_updating=True; self.active_list.clear()
        for key in keys:
            item=QListWidgetItem(SPECS[key].title); item.setToolTip('Drag to reorder · uncheck to close (settings are kept)'); item.setData(Qt.ItemDataRole.UserRole,key); item.setCheckState(Qt.CheckState.Checked); self.active_list.addItem(item)
            if key not in self.inspectors:
                inspector=PlotInspector(self,key); inspector.toggle.setChecked(not self.inspectors); self.inspectors[key]=inspector; self.inspector_layout.addWidget(inspector)
                inspector.toggle.toggled.connect(lambda on,selected=key:self.inspector_opened(selected,on))
            self.inspector_layout.removeWidget(self.inspectors[key]); self.inspector_layout.addWidget(self.inspectors[key]); self.inspectors[key].show()
        self.active_list.setFixedHeight(max(34,min(150,len(keys)*28+8))); self.active_updating=False
        for key,inspector in self.inspectors.items():
            if key not in keys: inspector.hide()

    def keys(self):
        return [self.list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.list.count()) if self.list.item(i).checkState()==Qt.CheckState.Checked]

    def column_label(self,key):
        if key=='mean': return 'Mean'
        g=probe_geometry(self.page)
        x=g['left_site_center_um']+int(key)*g['column_pitch_um']
        return f'X = {x:g} µm'

    def opacity(self,kind): return self.column_opacity[kind].value()/100

    def opacity_changed(self,*args):
        if self._restoring or self.page._updating: return
        if self.page.levels.backgrounds: self.page.levels.clear_preview()
        for kind,slider in self.column_opacity.items(): self.opacity_labels[kind].setText(f'{slider.value()}%')
        # Update artists in place: slider drags never rebuild plots or touch data.
        for key in self.keys():
            panel=self.panels.get(key)
            if panel is None: continue
            for ax in panel['axes']:
                for artist in [*ax.lines,*ax.collections]:
                    kind=getattr(artist,'_ephys_column',None)
                    if kind is not None: artist.set_alpha(self.opacity(kind))
            for legend in panel['figure'].legends:
                for original,handle in zip(legend._ephys_handles,legend.legend_handles):
                    kind=getattr(original,'_ephys_column',None)
                    if kind is not None:
                        original.set_alpha(self.opacity(kind)); handle.set_alpha(self.opacity(kind))
            panel['canvas'].draw_idle()
        self.page.owner.schedule_save()

    def column_choices(self): return [key for key,check in self.columns.items() if check.isChecked()]
    def has_units(self): return any(SPECS[key].source=='units' for key in self.keys())
    def needs_detail(self): return any(key in self.keys() for key in ('raster','amplitude','acg'))
    def needs_notebook(self): return self.notebook_mode.isChecked() or any(k in self.keys() for k in ('correlation_profile','low40','low40_plain','csd'))
    def needs_traces(self): return any(k in self.keys() for k in ('traces','traces_clustered')) and not self.notebook_mode.isChecked()
    def depth_range(self): return self._ylim or depth_limits(self.page)

    def eventFilter(self,watched,event):
        if event.type() in (QEvent.Type.Resize,QEvent.Type.Show): self.align_later()
        return False

    def align_later(self,*args): self.timer.start(0)

    def focus_inspector(self,event):
        if not getattr(event,'dblclick',False) or event.canvas is self.page.anatomy_canvas: return
        key=self.key_for(event.canvas)
        if key in self.inspectors:
            self.page.settings_tabs.setCurrentIndex(1); self.inspectors[key].toggle.setChecked(True)
            self.page.sidebar.ensureWidgetVisible(self.inspectors[key])

    def connect_canvas(self,canvas):
        canvas.mpl_connect('button_press_event',self.focus_inspector)
        canvas.mpl_connect('scroll_event',self.page._plot_scroll)
        canvas.mpl_connect('draw_event',self.page._raster_draw_event)
        canvas.mpl_connect('motion_notify_event',self.page._hover_unit)
        canvas.mpl_connect('pick_event',self.page._pick_unit)
        canvas.mpl_connect('button_press_event',self.page.levels.press)
        canvas.mpl_connect('motion_notify_event',self.page.levels.motion)
        canvas.mpl_connect('button_release_event',self.page.levels.release)
        canvas.mpl_connect('figure_leave_event',self.page.levels.leave)

    def rebuild(self,remember=True):
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg,NavigationToolbar2QT
        sizes={key:panel['canvas'].width() for key,panel in self.panels.items() if not panel['canvas'].isHidden()}
        if remember: self.widths.update(sizes)
        keys=self.keys()
        for key in keys:
            if key not in self.panels:
                figure=Figure(facecolor='#242b36'); canvas=FigureCanvasQTAgg(figure); canvas.setMinimumSize(210,220)
                toolbar=NavigationToolbar2QT(canvas,self.page,coordinates=False); toolbar.hide()
                self.panels[key]=dict(figure=figure,canvas=canvas,axes=[],toolbar=toolbar)
                canvas.installEventFilter(self); self.connect_canvas(canvas)
            self.splitter.insertWidget(keys.index(key),self.panels[key]['canvas']); self.panels[key]['canvas'].show()
        for key,panel in self.panels.items():
            if key not in keys: panel['canvas'].hide()
        self.splitter.setSizes([self.widths.get(next((k for k,v in self.panels.items() if v['canvas'] is self.splitter.widget(i)),''),330) if not self.splitter.widget(i).isHidden() else 0 for i in range(self.splitter.count())])
        if keys:
            first=self.panels[keys[0]]; self.page.figure=first['figure']; self.page._canvas=first['canvas']; self.page._toolbar=first['toolbar']
        else: self.page.figure=self.page.anatomy_figure; self.page._canvas=self.page.anatomy_canvas
        self.refresh_inspectors(); self.align_later()

    def changed(self,*args):
        if self._restoring or self.page._updating: return
        self.rebuild(); self.page.draw(); self.page.owner.schedule_save()
        if self.has_units() and self.page.sorting is None and self.page.current() and self.page.current().get('selected_sorting') and not self.page._busy:
            self.page._load_selection()

    def interval_changed(self,*args):
        if self._restoring or self.page._updating: return
        self.changed()
        if self.notebook_mode.isChecked(): self.page._analysis_changed()

    def profile_source_changed(self,*args):
        if self._restoring or self.page._updating: return
        self.page.notebook=None; self.page._notebook_attempted=None; self.changed()

    def set_checked(self,keys):
        self._restoring=True
        for i in range(self.list.count()):
            item=self.list.item(i); item.setCheckState(Qt.CheckState.Checked if item.data(Qt.ItemDataRole.UserRole) in keys else Qt.CheckState.Unchecked)
        self._restoring=False; self.changed()

    def state(self):
        return dict(plot_settings=copy.deepcopy(self.options),analysis={k:v for k,v in self.analysis_options().items() if k not in ('bands','duration')},duration=self.page.window_duration.value(),
            expanded={k:v.toggle.isChecked() for k,v in self.inspectors.items()},order=[self.list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.list.count())],checked=self.keys(),
            columns=self.column_choices(),legends=self.legends.isChecked(),notebook=self.notebook_mode.isChecked(),profile_source=self.profile_source.currentData(),start=self.page.time.value(),column_opacities={k:s.value() for k,s in self.column_opacity.items()},widths={k:p['canvas'].width() for k,p in self.panels.items() if k in self.keys()})

    def restore(self,state):
        self._restoring=True
        self.options.clear()
        for key,options in state.get('plot_settings',{}).items():
            if key in SPECS: self.options[key]={**defaults(key),**copy.deepcopy(options)}
        for key,inspector in self.inspectors.items():
            inspector.options=self.plot_options(key); inspector.load(); inspector.toggle.setChecked(state.get('expanded',{}).get(key,False))
        self.page.window_duration.setMaximum(60); self.page.window_duration.setValue(state.get('duration',2))
        for key,field in self.analysis_fields.items():
            value=state.get('analysis',{}).get(key,dict(c_factor=True,sigma=3,reference=False,segments=6,lowpass=40,nperseg=256,nfft=4096)[key])
            if isinstance(field,QCheckBox): field.setChecked(value)
            elif isinstance(field,QComboBox): field.setCurrentText(str(value))
            else: field.setValue(value)
        items={}
        while self.list.count():
            item=self.list.takeItem(0); items[item.data(Qt.ItemDataRole.UserRole)]=item
        order=state.get('order',list(SPECS)); order=list(dict.fromkeys([k for k in order if k in items]+list(items)))
        for key in order:
            item=items[key]; item.setCheckState(Qt.CheckState.Checked if key in state.get('checked',['lfp','rms']) else Qt.CheckState.Unchecked); self.list.addItem(item)
        for key,check in self.columns.items(): check.setChecked(key in state.get('columns',['0','1']))
        self.legends.setChecked(state.get('legends',True))
        self.notebook_mode.setChecked(state.get('notebook',False))
        self.profile_source.setCurrentIndex(max(0,self.profile_source.findData(state.get('profile_source','cached'))))
        self.page.time.setMaximum(1e9); self.page.time.setValue(state.get('start',0))
        for key,slider in self.column_opacity.items(): slider.setValue(state.get('column_opacities',{}).get(key,state.get('line_opacity',100)))
        self.widths=state.get('widths',{}).copy(); self._restoring=False; self.rebuild(remember=False)

    def canvases(self): return [self.page.anatomy_canvas,*[self.panels[k]['canvas'] for k in self.keys()]]

    def render(self):
        page=self.page; page.levels.preview=None; page.levels.backgrounds.clear()
        for key,check in self.columns.items(): check.setText(self.column_label(key))
        for key,slider in self.column_opacity.items():
            self.opacity_labels[key].setText(f'{slider.value()}%'); slider.setEnabled(self.columns[key].isChecked())
            slider.setToolTip('Line opacity · '+self.column_label(key))
        self.prepare_window.setVisible(self.needs_notebook())
        self.prepare_window.setEnabled(page.summary is not None)
        rec=page.current(); recid=(rec or {}).get('id')
        if recid!=self._record_id: self._ylim=None; self._record_id=recid
        for ax,cid in getattr(page,'_depth_callbacks',[]): ax.callbacks.disconnect(cid)
        page._depth_callbacks=[]; page._depth_axes=[]
        self.render_anatomy(connect=False)
        for key in self.keys():
            panel=self.panels[key]; figure=panel['figure']; figure.clear(); figure._ephys_colorbars=[]
            panel['axes']=render_plot(page,figure,key,(.2,.1,.75,.8))
            plot_legend(page,figure,key,panel['axes'])
            panel['canvas'].setMinimumWidth(210 if key == 'waveform' else 210*len(panel['axes']))
            if SPECS[key].depth: page._depth_axes.extend(panel['axes'])
        for ax in [page.anatomy_ax,*page._depth_axes]:
            page._depth_callbacks.append((ax,ax.callbacks.connect('ylim_changed',page._sync_depth)))
        self.align(); page.levels.refresh(); page.levels.paint(draw=False)
        for canvas in self.canvases(): canvas.draw_idle()
        if rec and page.summary and 'lfp' in self.keys() and not self.notebook_mode.isChecked() and not page._busy:
            options=self.plot_options('lfp')
            if (options['fits'] or options['transitions']) and page._overview_attempted!=page._overview_key(): page.profile_timer.start(120)
        if rec and self.needs_notebook() and page.summary and page.notebook is None and not page._busy:
            attempt=(rec['id'],page.shank.currentData(),page.time.value(),self.profile_source.currentData())
            if page._notebook_attempted!=attempt:
                page._notebook_attempted=attempt
                QTimer.singleShot(0,lambda:page._load_notebook(cache_only=True))
        if rec and self.needs_traces() and page.summary and not page._trace_ready() and not page._busy: page.trace_timer.start(50)
        if rec and self.needs_detail() and page.sorting and (page.unit_view is None or page.unit_view['key']!=page._unit_view_key()) and not page._busy: page.unit_timer.start(80)

    def align(self):
        if self.page._closing: return
        if self.page.levels.backgrounds:
            self.page.levels.preview=None; self.page.levels.backgrounds.clear(); self.page.levels.paint(draw=False)
        canvases=self.canvases()
        if not canvases: return
        # Global logical pixels account for scrollbars and differing canvas origins.
        origin=lambda c:c.mapToGlobal(c.rect().topLeft()).y()
        footer=max(BOTTOM,*[footer_height(c.figure,c.width()) for c in canvases]) if self.legends.isChecked() else 66.
        top=max(origin(c)+TOP for c in canvases); bottom=min(origin(c)+c.height()-footer for c in canvases)
        if bottom<=top: return
        for canvas in canvases:
            figure=canvas.figure; axes=([self.page.anatomy_ax] if canvas is self.page.anatomy_canvas else self.panels[self.key_for(canvas)]['axes']); width=max(1,canvas.width()); height=max(1,canvas.height()); y=(origin(canvas)+height-bottom)/height
            h=(bottom-top)/height
            if canvas is self.page.anatomy_canvas:
                if axes: axes[0].set_position([min(.4,65/width),y,max(.1,1-77/width),h])
            else:
                count=len(axes)
                if not count: continue
                if self.key_for(canvas) == 'waveform':
                    slot=h/count
                    for i,ax in enumerate(axes):
                        ax.set_position([64/width,y+(count-i-1)*slot+slot*.16,max(.03,1-80/width),slot*.68])
                    place_legends(figure,width,height); canvas.draw_idle(); continue
                slot=width/count
                for i,ax in enumerate(axes):
                    left=100 if any(isinstance(label.get_text(),str) and len(label.get_text())>12 for label in ax.get_yticklabels()) and not SPECS[self.key_for(canvas)].depth else 64
                    ax.set_position([(i*slot+left)/width,y,max(.03,(slot-left-16)/width),h])
            place_legends(figure,width,height)
            canvas.draw_idle()
        # Tick density can change on resize; keep the depth axes identical too.
        page=self.page; previous=getattr(page,'_syncing_depth',False); page._syncing_depth=True
        try:
            limits=self.depth_range(); ticks=page.anatomy_ax.get_yticks()
            for ax in page._depth_axes:
                if not np.array_equal(ax.get_yticks(),ticks): ax.set_yticks(ticks)
                if ax.get_ylim()!=limits: ax.set_ylim(limits,emit=False)
        finally: page._syncing_depth=previous

    def key_for(self,canvas): return next(k for k,p in self.panels.items() if p['canvas'] is canvas)

    def export_pdf(self,path=None):
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib import rc_context
        page=self.page; rec=page.current(); interactive=path is None
        if not rec: return
        if path is None:
            name=re.sub(r'[^\w.-]+','_',rec['name']).lstrip('.') or 'recording'
            default=Path(page.project.output_dir)/'ephys'/'figures'/f'{name}.pdf'
            path,_=QFileDialog.getSaveFileName(page,'Save figure as PDF',str(default),'PDF (*.pdf)')
            if not path: return
        from .figure_files import visible_figure_path, temporary_figure_path, publish_figure
        path=visible_figure_path(path,'.pdf'); path.parent.mkdir(parents=True,exist_ok=True)
        widths=[max(230,page.anatomy_canvas.width()),*[max(210,self.panels[k]['canvas'].width()) for k in self.keys()]]
        height=max(400,page.anatomy_canvas.height()); total=sum(widths)
        figure=Figure(figsize=(total/100,height/100),dpi=100,facecolor='#242b36'); canvas=FigureCanvasAgg(figure)
        class Proxy:
            def __getattr__(self,name): return getattr(page,name)
        proxy=Proxy(); proxy._vector_export=True; proxy.anatomy_figure=figure; proxy.anatomy_canvas=canvas; proxy.anatomy_ax=figure.add_axes([.01,.1,.2,.8])
        draw_probe(proxy); proxy.anatomy_ax.set_ylim(self.depth_range()); proxy.anatomy_ax.set_yticks(page.anatomy_ax.get_yticks())
        proxy.anatomy_ax.set_title(page.anatomy_caption.text(),fontsize=10,pad=30,color='#dce6f0')
        base_footer=BOTTOM if self.legends.isChecked() else 66.
        bottom=base_footer/height; h=1-(TOP+base_footer)/height
        proxy.anatomy_ax.set_position([65/total,bottom,(widths[0]-77)/total,h])
        anatomy_legend(page,figure,proxy.anatomy_ax)
        depth_axes=[proxy.anatomy_ax]; offset=widths[0]
        for key,width in zip(self.keys(),widths[1:]):
            axes=render_plot(page,figure,key,((offset+64)/total,bottom,(width-80)/total,h),export=True)
            slot=width/len(axes)
            for i,ax in enumerate(axes):
                if key == 'waveform':
                    vslot=h/len(axes)
                    ax.set_position([(offset+64)/total,bottom+(len(axes)-i-1)*vslot+vslot*.16,(width-80)/total,vslot*.68])
                else: ax.set_position([(offset+i*slot+64)/total,bottom,max(.01,(slot-80)/total),h])
            plot_legend(page,figure,key,axes)
            if SPECS[key].depth: depth_axes.extend(axes)
            offset+=width
        footer=max(BOTTOM,footer_height(figure,total)) if self.legends.isChecked() else 66.
        for ax in figure.axes:
            if getattr(ax,'_ephys_footer',False): continue
            pos=ax.get_position(); available=1-(TOP+footer)/height
            ax.set_position([pos.x0,footer/height+(pos.y0-bottom)/h*available,pos.width,pos.height/h*available])
        place_legends(figure,total,height)
        page.levels.decorate(depth_axes)
        # TrueType fonts and native paths/collections remain editable in Illustrator.
        def write(progress,cancel):
            from .ephys_analysis import check_cancel
            temporary=temporary_figure_path(path)
            try:
                check_cancel(cancel); progress(0,'Writing vector PDF')
                with rc_context({'pdf.fonttype':42,'pdf.compression':6}): figure.savefig(temporary,format='pdf',facecolor=figure.get_facecolor())
                check_cancel(cancel); publish_figure(temporary,path); progress(100,'PDF saved')
                return path
            finally: temporary.unlink(missing_ok=True)
        def ready(saved): page.message.setText(f'Saved figure: {saved}')
        if interactive:
            page._task(write,ready,'Saving PDF')
            return
        ready(write(lambda *args:None,lambda:False))
        return figure,path

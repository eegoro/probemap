"""Contextual plot inspectors. Hidden plots retain their independent settings."""
import copy
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (QWidget,QVBoxLayout,QHBoxLayout,QFormLayout,QToolButton,QLabel,
    QPushButton,QComboBox,QDoubleSpinBox,QSpinBox,QCheckBox,QColorDialog,QTableWidget,QTableWidgetItem)
from qtpy.QtGui import QColor
from .ephys_analysis import BANDS
from .desktop_ephys_plots import COLORS,SPECS


def defaults(key):
    return dict(scale='linear',auto_range=True,xmin=0.,xmax=100.,auto_color=True,cmin=-1.,cmax=1.,
        cmap='RdBu_r' if key=='correlation' else 'bwr' if key=='csd' else 'magma',
        line_width=1.,marker_size=.7 if key=='raster' else 2.,trace_scale=.25 if key in ('low40','low40_plain','csd') else 1.,smoothing=0.,
        fits=True,transitions=False,raw=True,color=None,
        bands=[dict(low=lo,high=hi,color=c,visible=True) for (lo,hi),c in zip(BANDS,COLORS)])


def number(value,low,high,decimals=2):
    field=QDoubleSpinBox(); field.setRange(low,high); field.setDecimals(decimals); field.setValue(value); field.setKeyboardTracking(False)
    return field


class Fold(QWidget):
    def __init__(self,title,expanded=False):
        super().__init__(); layout=QVBoxLayout(self); layout.setContentsMargins(0,0,0,0); layout.setSpacing(6)
        self.toggle=QToolButton(); self.toggle.setText(title); self.toggle.setCheckable(True); self.toggle.setChecked(expanded)
        self.toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon); self.toggle.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self.toggle.setStyleSheet('QToolButton{text-align:left;border:0;padding:7px 0;font-weight:600;}')
        layout.addWidget(self.toggle); self.body=QWidget(); self.content=QVBoxLayout(self.body); self.content.setContentsMargins(4,0,4,8); layout.addWidget(self.body); self.body.setVisible(expanded)
        self.toggle.toggled.connect(lambda on:(self.body.setVisible(on),self.toggle.setArrowType(Qt.ArrowType.DownArrow if on else Qt.ArrowType.RightArrow)))


class PlotInspector(Fold):
    def __init__(self,board,key):
        super().__init__(SPECS[key].title); self.board=board; self.key=key; self.loading=False
        self.options=board.options.setdefault(key,defaults(key)); form=QFormLayout(); self.content.addLayout(form)
        self.fields={}
        scale=QComboBox(); scale.addItems(['linear','log','symlog']); self.add(form,'X scale','scale',scale)
        auto=QCheckBox('Automatic'); self.add(form,'X range','auto_range',auto)
        self.range_widget=QWidget(); row=QHBoxLayout(self.range_widget); row.setContentsMargins(0,0,0,0); minimum=number(0,-1e9,1e9); maximum=number(100,-1e9,1e9); row.addWidget(minimum); row.addWidget(maximum); form.addRow('From / to',self.range_widget); self.range_label=form.labelForField(self.range_widget)
        self.bind('xmin',minimum); self.bind('xmax',maximum)
        if key in ('spectrum','correlation','activity','csd'):
            cmap=QComboBox(); cmap.addItems(['magma','viridis','cividis','RdBu_r','bwr','gray']); self.add(form,'Color map','cmap',cmap)
            self.add(form,'Color range','auto_color',QCheckBox('Automatic'))
            self.color_range_widget=QWidget(); row=QHBoxLayout(self.color_range_widget); row.setContentsMargins(0,0,0,0); lo=number(-1,-1e12,1e12,3); hi=number(1,-1e12,1e12,3); row.addWidget(lo); row.addWidget(hi); form.addRow('Min / max',self.color_range_widget); self.color_range_label=form.labelForField(self.color_range_widget)
            self.bind('cmin',lo); self.bind('cmax',hi)
        appearance=Fold('Appearance'); appearance_form=QFormLayout(); appearance.content.addLayout(appearance_form)
        if key not in ('spectrum','correlation','activity','density','isi','acg','amplitude','region_lfp','region_rms','region_units'):
            self.add(appearance_form,'Line width','line_width',number(1,.2,5))
        if key not in ('lfp','lfp_gradient','spectrum','correlation','activity'):
            color=QPushButton('Choose…'); appearance_form.addRow('Line color',color); color.clicked.connect(self.choose_color); self.color_button=color
        if key in ('raster','amplitude'): self.add(appearance_form,'Point size','marker_size',number(self.options['marker_size'],.1,20,1))
        if key in ('lfp','rms','correlation_profile','compare_lfp','compare_rms'):
            self.add(appearance_form,'Display smoothing · µm','smoothing',number(0,0,500,0))
        if key in ('traces','traces_clustered','low40','low40_plain','csd'):
            self.add(appearance_form,'Trace amplitude','trace_scale',number(self.options['trace_scale'],.001,1000,3))
        if key in ('lfp','correlation_profile'):
            self.add(form,'Profiles','raw',QCheckBox('Measured profile'))
            self.add(form,'','fits',QCheckBox('Piecewise linear fit'))
            self.add(form,'','transitions',QCheckBox('Transition depths'))
        if appearance_form.rowCount(): self.content.addWidget(appearance)
        else: appearance.hide()
        if key in ('lfp','lfp_gradient'):
            bands=Fold('Frequency bands'); self.content.addWidget(bands)
            table=QTableWidget(0,4); self.bands=table; table.setHorizontalHeaderLabels(['Show','From · Hz','To · Hz','Color']); table.verticalHeader().hide()
            table.setFixedHeight(178); table.setColumnWidth(0,42); table.setColumnWidth(1,76); table.setColumnWidth(2,76); table.setColumnWidth(3,60)
            bands.content.addWidget(table); table.itemChanged.connect(self.band_changed); table.cellDoubleClicked.connect(self.band_color)
            row=QHBoxLayout(); add=QPushButton('Add band'); remove=QPushButton('Remove'); row.addWidget(add); row.addWidget(remove); bands.content.addLayout(row)
            add.clicked.connect(self.add_band); remove.clicked.connect(self.remove_band)
        reset=QPushButton('Reset plot'); reset.clicked.connect(self.reset); self.content.addWidget(reset)
        self.load()

    def add(self,form,label,key,field): form.addRow(label,field); self.bind(key,field)
    def bind(self,key,field):
        self.fields[key]=field
        signal=field.toggled if isinstance(field,QCheckBox) else field.currentTextChanged if isinstance(field,QComboBox) else field.valueChanged
        signal.connect(lambda value:self.changed(key,value))
    def load(self):
        self.loading=True
        for key,field in self.fields.items():
            value=self.options[key]
            if isinstance(field,QCheckBox): field.setChecked(value)
            elif isinstance(field,QComboBox): field.setCurrentText(value)
            else: field.setValue(value)
        if hasattr(self,'bands'):
            table=self.bands; table.setRowCount(len(self.options['bands']))
            for row,band in enumerate(self.options['bands']):
                for col,text in enumerate(('',str(band['low']),str(band['high']),band['color'])):
                    cell=QTableWidgetItem(text)
                    if col==0: cell.setCheckState(Qt.CheckState.Checked if band['visible'] else Qt.CheckState.Unchecked)
                    table.setItem(row,col,cell)
        self.loading=False; self.enable_fields()
    def enable_fields(self):
        self.range_widget.setVisible(not self.options['auto_range']); self.range_label.setVisible(not self.options['auto_range'])
        if hasattr(self,'color_range_widget'):
            self.color_range_widget.setVisible(not self.options['auto_color']); self.color_range_label.setVisible(not self.options['auto_color'])
        for key in ('xmin','xmax'): self.fields[key].setEnabled(not self.options['auto_range'])
        for key in ('cmin','cmax'):
            if key in self.fields: self.fields[key].setEnabled(not self.options['auto_color'])
    def changed(self,key,value):
        if self.loading: return
        self.options[key]=value; self.enable_fields(); self.board.plot_settings_changed(self.key)
    def choose_color(self):
        chosen=QColorDialog.getColor(QColor(self.options['color'] or COLORS[0]),self,'Line color')
        if chosen.isValid(): self.changed('color',chosen.name())
    def band_changed(self,cell):
        if self.loading: return
        row,col=cell.row(),cell.column(); band=self.options['bands'][row]
        if col==0: band['visible']=cell.checkState()==Qt.CheckState.Checked
        elif col in (1,2):
            try: value=float(cell.text())
            except ValueError: self.load(); return
            if not 0<=value<=500: self.load(); return
            candidate=dict(band); candidate['low' if col==1 else 'high']=value
            if candidate['low']>=candidate['high']: self.load(); return
            band.update(candidate)
        elif col==3:
            if not QColor(cell.text()).isValid(): self.load(); return
            band['color']=cell.text()
        self.board.plot_settings_changed(self.key,analysis=col in (1,2))
    def band_color(self,row,col):
        if col!=3: return
        chosen=QColorDialog.getColor(QColor(self.options['bands'][row]['color']),self,'Band color')
        if chosen.isValid(): self.bands.item(row,3).setText(chosen.name())
    def add_band(self):
        self.options['bands'].append(dict(low=10.,high=30.,color=COLORS[2],visible=True)); self.load(); self.board.plot_settings_changed(self.key,analysis=True)
    def remove_band(self):
        row=self.bands.currentRow()
        if row<0 or len(self.options['bands'])<=1: return
        self.options['bands'].pop(row); self.load(); self.board.plot_settings_changed(self.key,analysis=True)
    def reset(self):
        self.options.clear(); self.options.update(defaults(self.key)); self.load(); self.board.plot_settings_changed(self.key,analysis=True)


def apply_visual_settings(page,axes,spec):
    from scipy.ndimage import gaussian_filter1d
    from matplotlib.collections import QuadMesh
    options=page.plot_board.plot_options(spec.key)
    for ax in axes:
        ax.set_xscale(options['scale'])
        if not options['auto_range'] and options['xmin']<options['xmax']:
            if options['scale']!='log' or options['xmin']>0: ax.set_xlim(options['xmin'],options['xmax'])
        for collection in ax.collections:
            if isinstance(collection,QuadMesh):
                collection.set_cmap(options['cmap'])
                if not options['auto_color'] and options['cmin']<options['cmax']: collection.set_clim(options['cmin'],options['cmax'])
        if options['color'] is not None:
            for patch in ax.patches:
                patch.set_edgecolor(options['color'])
                if patch.get_fill(): patch.set_facecolor(options['color'])
            for collection in ax.collections:
                if not isinstance(collection,QuadMesh) and not hasattr(collection,'_ephys_column'): collection.set_color(options['color'])
        if spec.key in ('raster','amplitude'):
            for collection in ax.collections: collection.set_sizes([options['marker_size']])
        for line in ax.lines:
            if not hasattr(line,'_ephys_column'):
                if not spec.depth:
                    line.set_linewidth(line.get_linewidth()*options['line_width'])
                    if options['color'] is not None: line.set_color(options['color'])
                continue
            line.set_linewidth(line.get_linewidth()*options['line_width'])
            if options['color'] is not None and spec.key not in ('lfp','lfp_gradient'):
                from .desktop_ephys_plots import series_style
                line.set_color(series_style(options['color'],line._ephys_column)['color'])
            if options['smoothing']>0 and getattr(line,'_ephys_role',None)!='transition' and spec.key in ('lfp','rms','correlation_profile','compare_lfp','compare_rms'):
                x,y=line.get_xdata(),line.get_ydata(); x=copy.copy(x)
                import numpy as np
                good=np.isfinite(x)&np.isfinite(y)
                # Never smooth across an unrecorded depth gap.
                starts=np.r_[0,np.flatnonzero(np.diff(good.astype(int)))+1,len(good)]
                for a,b in zip(starts[:-1],starts[1:]):
                    if good[a] and b-a>1: x[a:b]=gaussian_filter1d(x[a:b].astype(float),options['smoothing']/15,mode='nearest')
                line.set_xdata(x)

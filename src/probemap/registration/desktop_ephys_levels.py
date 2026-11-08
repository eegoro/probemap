"""Persistent depth markers shared by anatomy and depth-bearing analysis plots."""
from uuid import uuid4
import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QWidget,QVBoxLayout,QHBoxLayout,QLabel,QTableWidget,QTableWidgetItem,QMenu,QAbstractItemView,QColorDialog
from qtpy.QtGui import QColor
from .desktop_tables import CheckHeader,ColorDotDelegate,TableHeightGrip
from .desktop_widgets import view_button
from .desktop_ephys_probe import probe_geometry


class DepthLevels(QWidget):
    def __init__(self,page):
        super().__init__(); self.page=page; self.artists=[]; self.drag=None; self.backgrounds={}; self.preview=None
        layout=QVBoxLayout(self); layout.setContentsMargins(0,0,0,0)
        head=QHBoxLayout(); head.addWidget(QLabel('Depth levels'),1)
        self.tool=view_button('level','Place or move a depth level',lambda:None); self.tool.setCheckable(True); self.tool.toggled.connect(self.clear_preview); head.addWidget(self.tool)
        self.delete=view_button('clear','Delete selected level',self.remove_selected); head.addWidget(self.delete); layout.addLayout(head)
        self.table=QTableWidget(0,5); t=self.table; t.setProperty('persistent_selection',True)
        t.setHorizontalHeaderLabels(['Name','Depth · µm','Nearest sites','Label','Color']); t.verticalHeader().hide(); t.setShowGrid(False)
        t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows); t.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        t.setFixedHeight(110); t.verticalHeader().setDefaultSectionSize(28)
        h=CheckHeader(t,(0,3)); t.setHorizontalHeader(h); h.toggled.connect(self.bulk)
        for col,width in enumerate((85,85,100,60,45)): t.setColumnWidth(col,width)
        t.setItemDelegateForColumn(4,ColorDotDelegate(t)); t.itemChanged.connect(self.edit); t.cellClicked.connect(self.color)
        t.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu); t.customContextMenuRequested.connect(self.menu)
        layout.addWidget(t); layout.addWidget(TableHeightGrip(t))

    def records(self):
        rec=self.page.current(); return rec.setdefault('depth_levels',[]) if rec else []

    def site_text(self,mark):
        g=probe_geometry(self.page); row=int(np.clip(round((mark['height_um']-g['first_site_center_from_tip_um'])/g['site_pitch_um']),0,g['sites_per_shank']//2-1))
        return f'S{mark["shank"]} · E{row*2}/{row*2+1}'

    def refresh(self):
        t=self.table; previous=t.currentRow(); t.blockSignals(True); t.setRowCount(len(self.records()))
        for row,m in enumerate(self.records()):
            for col,text in enumerate((m['name'],f'{m["height_um"]:.1f}',self.site_text(m),'','')):
                cell=QTableWidgetItem(text); cell.setData(Qt.ItemDataRole.UserRole,m['id'])
                if col in (0,3):
                    cell.setFlags(cell.flags()|Qt.ItemFlag.ItemIsUserCheckable)
                    cell.setCheckState(Qt.CheckState.Checked if m.get('visible' if col==0 else 'label',True) else Qt.CheckState.Unchecked)
                if col in (2,3,4): cell.setFlags(cell.flags() & ~Qt.ItemFlag.ItemIsEditable)
                if col==4: cell.setData(Qt.ItemDataRole.UserRole,m.get('color','#79e7ff'))
                cell.setToolTip('Nearest physical electrode row; level position is not snapped' if col==2 else '')
                t.setItem(row,col,cell)
        if previous>=0 and t.rowCount(): t.selectRow(min(previous,t.rowCount()-1))
        t.blockSignals(False); t.horizontalHeader().viewport().update()

    def selected(self):
        row=self.table.currentRow(); records=self.records()
        return records[row] if 0<=row<len(records) else None

    def edit(self,cell):
        if cell.row()>=len(self.records()): return
        m=self.records()[cell.row()]
        if cell.column()==0: m.update(name=cell.text().strip() or 'Level',visible=cell.checkState()==Qt.CheckState.Checked)
        elif cell.column()==3: m['label']=cell.checkState()==Qt.CheckState.Checked
        elif cell.column()==1:
            try: value=float(cell.text())
            except ValueError: self.refresh(); return
            if not np.isfinite(value) or value<0: self.refresh(); return
            m['height_um']=value
        self.refresh(); self.paint(); self.page.owner.schedule_save()

    def bulk(self,column,checked):
        for m in self.records(): m['visible' if column==0 else 'label']=checked
        self.refresh(); self.paint(); self.page.owner.schedule_save()

    def color(self,row,column):
        if column!=4 or row>=len(self.records()): return
        m=self.records()[row]; chosen=QColorDialog.getColor(QColor(m.get('color','#79e7ff')),self,'Level color')
        if chosen.isValid(): m['color']=chosen.name(); self.refresh(); self.paint(); self.page.owner.schedule_save()

    def menu(self,pos):
        menu=QMenu(self); menu.addAction('Delete selected level',self.remove_selected)
        menu.addAction('Delete all levels',self.clear); menu.exec(self.table.viewport().mapToGlobal(pos))

    def remove_selected(self):
        m=self.selected()
        if m: self.records().remove(m); self.refresh(); self.paint(); self.page.owner.schedule_save()

    def clear(self):
        self.records().clear(); self.refresh(); self.paint(); self.page.owner.schedule_save()

    def axes(self): return [self.page.anatomy_ax,*getattr(self.page,'_depth_axes',[])]

    def decorate(self,axes,animated=False,preview=False):
        artists=[]
        for ax in axes:
            for m in [*self.records(),*([self.preview] if preview and self.preview else [])]:
                if not m.get('visible',True): continue
                color=m.get('color','#79e7ff'); y=m['height_um']
                artists.append(ax.axhline(y,c=color,lw=.9,animated=animated,zorder=20))
                if m.get('label',True):
                    artists.append(ax.text(.99,y,f'{m["name"]} · {y:.0f} µm · {self.site_text(m)}',transform=ax.get_yaxis_transform(),
                        ha='right',va='bottom',color=color,fontsize=7,clip_on=True,zorder=21,animated=animated,
                        bbox=dict(facecolor='#202733',edgecolor='none',alpha=.8,pad=2)))
        return artists

    def paint(self,draw=True):
        for artist in self.artists:
            try: artist.remove()
            except (ValueError,NotImplementedError): pass
        self.artists=self.decorate(self.axes(),animated=bool(self.backgrounds),preview=True)
        if not draw: return
        if self.backgrounds:
            for canvas,bg in self.backgrounds.items():
                canvas.restore_region(bg)
                for artist in self.artists:
                    if artist.figure is canvas.figure: artist.axes.draw_artist(artist)
                canvas.blit(canvas.figure.bbox)
        else:
            for canvas in self.page.plot_board.canvases(): canvas.draw_idle()

    def clear_preview(self,*args):
        self.preview=None; self.backgrounds.clear()
        self.paint()

    def leave(self,event):
        if self.drag is None and self.preview is not None: self.clear_preview()

    def _start_blit(self):
        if self.backgrounds: return
        for artist in self.artists: artist.set_animated(True)
        for canvas in self.page.plot_board.canvases():
            canvas.draw(); self.backgrounds[canvas]=canvas.copy_from_bbox(canvas.figure.bbox)

    def shank_at(self,event):
        if event.inaxes is self.page.anatomy_ax and getattr(event,'xdata',None) is not None:
            g=probe_geometry(self.page)
            return int(np.clip(round(event.xdata/g['shank_pitch_um']),0,g['shanks']-1))
        return self.page.shank.currentData() or 0

    def press(self,event):
        if self.page._busy or event.button!=1 or event.inaxes not in self.axes() or event.ydata is None: return
        if getattr(getattr(event,'canvas',None),'toolbar',None) and event.canvas.toolbar.mode: return
        candidates=[m for m in self.records() if m.get('visible',True)]
        nearby=min(candidates,key=lambda m:abs(event.inaxes.transData.transform((0,m['height_um']))[1]-event.y),default=None)
        if nearby and abs(event.inaxes.transData.transform((0,nearby['height_um']))[1]-event.y)<7: mark=nearby
        elif self.tool.isChecked():
            mark=dict(id=uuid4().hex,name=f'L{len(self.records())+1}',height_um=max(0,float(event.ydata)),shank=self.shank_at(event),visible=True,label=True,color='#79e7ff')
            self.records().append(mark)
        else: return
        self.preview=None
        self.drag=mark; self.refresh(); self.table.selectRow(self.records().index(mark)); self.paint(draw=False)
        self._start_blit(); self.paint()

    def motion(self,event):
        if self.page._busy or event.inaxes not in self.axes() or event.ydata is None:
            if self.drag is None and self.preview: self.clear_preview()
            return
        if getattr(getattr(event,'canvas',None),'toolbar',None) and event.canvas.toolbar.mode: return
        if self.drag is not None:
            self.drag['height_um']=max(0,float(event.ydata))
            if event.inaxes is self.page.anatomy_ax: self.drag['shank']=self.shank_at(event)
            self.paint(); return
        if not self.tool.isChecked(): return
        self._start_blit()
        self.preview=dict(id='preview',name='Level',height_um=max(0,float(event.ydata)),
            shank=self.shank_at(event),visible=True,label=True,color='#79e7ff')
        self.paint()

    def release(self,event):
        if self.drag is None: return
        self.drag=None; self.backgrounds.clear()
        for artist in self.artists: artist.set_animated(False)
        self.refresh(); self.paint(); self.page.owner.schedule_save()

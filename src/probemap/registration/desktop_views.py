"""Fast Qt slice canvases and a direct VisPy volume view. No napari imports."""
from __future__ import annotations

import numpy as np
from qtpy.QtCore import QEvent, QPointF, QRectF, Qt, Signal
from qtpy.QtGui import QColor, QImage, QPainter, QPen
from qtpy.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QGridLayout, QHBoxLayout, QLabel,
                           QPushButton, QSlider, QStackedWidget, QVBoxLayout, QWidget)

from .coarse import transformed_corners
from .desktop_widgets import PopupComboBox, view_button


class SliceCanvas(QWidget):
    pointClicked = Signal(object)
    translated = Signal(object)
    rotated = Signal(float)
    scrolled = Signal(int)
    viewChanged = Signal()

    def __init__(self, axis, parent=None):
        super().__init__(parent)
        self.axis = axis
        self.image = None
        self.spacing = np.ones(3)
        self.origin = np.zeros(3)
        self.index = 0
        self.mode = "view"
        self.points = []
        self.crosshair = None
        self.pivot = np.zeros(3)
        self._last = None
        self._last_pixel = None
        self._start_pixel = None
        self._button = None
        self.zoom = 1.0
        self.view_rotation=0
        self.pan = QPointF()
        self._space_down=False
        self._panning=False
        self.pan_mode=False
        self.navigation_mode=False
        self.left_drag_pans=True
        self.show_axes=True
        self.background_color='#11161e'
        self.smooth=True
        self._transform_drag=False
        self._zooming=False
        self.setMinimumSize(170, 140)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def image_rect(self):
        if self.image is None:
            return QRectF()
        others = [a for a in range(3) if a != self.axis]
        width = self.image.width() * self.spacing[others[1]]
        height = self.image.height() * self.spacing[others[0]]
        if self.view_rotation%180: width,height=height,width
        scale = self.fit_scale() * self.zoom
        w, h = width * scale, height * scale
        return QRectF((self.width()-w)/2 + self.pan.x(), (self.height()-h)/2 + self.pan.y(), w, h)

    def fit_scale(self,rotation=None):
        if self.image is None: return 1.
        others=[a for a in range(3) if a!=self.axis]
        width=self.image.width()*self.spacing[others[1]]
        height=self.image.height()*self.spacing[others[0]]
        if (self.view_rotation if rotation is None else rotation)%180: width,height=height,width
        return min((self.width()-12)/width,(self.height()-12)/height)

    def _world(self, position):
        rect = self.image_rect()
        if self.image is None or rect.isEmpty():
            return None
        others = [a for a in range(3) if a != self.axis]
        p = np.zeros(3)
        p[self.axis] = self.index * self.spacing[self.axis]
        u,v=(position.x()-rect.left())/rect.width(),(position.y()-rect.top())/rect.height()
        for _ in range(self.view_rotation//90): u,v=v,1-u
        p[others[1]] = (u*self.image.width()-0.5)*self.spacing[others[1]]
        p[others[0]] = (v*self.image.height()-0.5)*self.spacing[others[0]]
        return p + self.origin

    def _pixel(self, world):
        others = [a for a in range(3) if a != self.axis]
        rect = self.image_rect()
        world = np.asarray(world)-self.origin
        u=(world[others[1]]/self.spacing[others[1]]+0.5)/self.image.width()
        v=(world[others[0]]/self.spacing[others[0]]+0.5)/self.image.height()
        for _ in range(self.view_rotation//90): u,v=1-v,u
        return QPointF(rect.left()+u*rect.width(),rect.top()+v*rect.height())

    def set_image(self, pixels, spacing, index, origin=(0,0,0)):
        self._display_pixels=np.asarray(pixels,dtype=np.float32)
        rgb = np.ascontiguousarray(np.clip(pixels*255, 0, 255).astype(np.uint8))
        if rgb.ndim == 2:
            self.image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_Grayscale8).copy()
        else:
            self.image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888).copy()
        self.spacing, self.index = np.asarray(spacing), index
        self.origin = np.asarray(origin, dtype=float)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(self.background_color))
        if self.image is None:
            painter.setPen(QColor("#8492a6"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Load images to begin")
            return
        self.paint_image(painter,self.smooth)
        others = [a for a in range(3) if a != self.axis]
        painter.setPen(QColor("#91a5bb"))
        if self.show_axes: painter.drawText(10, 18, f"{'ZYX'[others[1]]} →   {'ZYX'[others[0]]} ↓")
        if self.crosshair is not None:
            p = self._pixel(self.crosshair)
            painter.setPen(QPen(QColor(241, 200, 90, 110), 1))
            painter.drawLine(QPointF(p.x(), 0), QPointF(p.x(), self.height()))
            painter.drawLine(QPointF(0, p.y()), QPointF(self.width(), p.y()))
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        for number, position, active in self.visible_points():
            point = self._pixel(position)
            painter.setPen(QPen(QColor("#ffffff" if active else "#ffc966"), 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(point, 6, 6)
            painter.drawText(point + QPointF(9, -8), str(number))

    def paint_image(self,painter,smooth=True):
        if hasattr(self,'_figure_background'):
            painter.drawImage(QRectF(self.rect()),self._figure_background)
            for path,rgb,alpha in getattr(self,'_figure_region_paths',[]):
                color=QColor(*rgb); color.setAlphaF(alpha)
                painter.fillPath(path,color)
            return
        interpolation=getattr(self,'display_interpolation','linear' if smooth else 'nearest')
        if interpolation not in ('nearest','linear'):
            from .desktop_image_renderer import render_image
            from .desktop_figures import array_image
            key=(self.image.cacheKey(),self.width(),self.height(),self.zoom,self.pan.x(),self.pan.y(),self.view_rotation,interpolation)
            if key!=getattr(self,'_filtered_key',None):
                self._filtered_image=array_image(render_image(self._display_pixels,interpolation,(self.width(),self.height()),self.image_rect(),self.view_rotation))
                self._filtered_key=key
            painter.drawImage(QRectF(self.rect()),self._filtered_image); return
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform,smooth)
        rect=self.image_rect()
        painter.save()
        painter.translate(rect.center()); painter.rotate(self.view_rotation)
        width,height=(rect.height(),rect.width()) if self.view_rotation%180 else (rect.width(),rect.height())
        painter.drawImage(QRectF(-width/2,-height/2,width,height),self.image)
        painter.restore()

    def visible_points(self):
        """Only the current voxel-depth bin, including nonzero grid origins."""
        depth=self.origin[self.axis]+self.index*self.spacing[self.axis]
        return [(number,position,active) for number,position,active in self.points
                if int(np.floor((position[self.axis]-depth)/self.spacing[self.axis]+.5+1e-10))==0]

    def mousePressEvent(self, event):
        self._last = self._world(event.position())
        self._last_pixel = event.position()
        self._start_pixel = event.position()
        self._button = event.button()
        self._transform_drag=(self.navigation_mode and self.mode=='placement'
            and (bool(event.modifiers()&Qt.KeyboardModifier.AltModifier) or event.button()==Qt.MouseButton.MiddleButton)
            and not self._space_down and not bool(event.modifiers()&Qt.KeyboardModifier.ShiftModifier))
        self._zooming=(self.navigation_mode and event.button()==Qt.MouseButton.RightButton and not self._transform_drag)
        self._panning=not (self._transform_drag or self._zooming) and (
                       (self.navigation_mode and self.left_drag_pans and event.button()==Qt.MouseButton.LeftButton)
                       or self.pan_mode or self._space_down or event.button()==Qt.MouseButton.MiddleButton
                       or bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier))
        self.setFocus()
        if self._panning: self.setCursor(Qt.CursorShape.ClosedHandCursor)
        event.accept()

    def mouseMoveEvent(self, event):
        world = self._world(event.position())
        if self._last is None or world is None or not event.buttons():
            return
        # If the full display box expanded during a drag, reinterpret the prior
        # screen pixel on the current grid instead of introducing a false jump.
        previous_world = self._world(self._last_pixel)
        if self._zooming:
            self.zoom_at(self._start_pixel,float(np.exp(np.clip((self._last_pixel.y()-event.position().y())*.01,-1,1))))
        elif self._panning:
            self.pan+=event.position()-self._start_pixel
            self._start_pixel=event.position()
            self.update()
            self.viewChanged.emit()
        elif self.mode == "placement" and self._button in (Qt.MouseButton.LeftButton,Qt.MouseButton.MiddleButton):
            self.translated.emit(world-previous_world)
        elif self.mode == "placement" and self._button == Qt.MouseButton.RightButton:
            a, b = [i for i in range(3) if i != self.axis]
            angle = np.arctan2(world[a]-self.pivot[a], world[b]-self.pivot[b])
            previous = np.arctan2(previous_world[a]-self.pivot[a], previous_world[b]-self.pivot[b])
            delta = (angle-previous + np.pi) % (2*np.pi)-np.pi
            self.rotated.emit(float(np.degrees(delta)*(1 if self.axis != 1 else -1)))
        elif self._button == Qt.MouseButton.MiddleButton or (event.modifiers() & Qt.KeyboardModifier.ShiftModifier):
            delta = event.position()-self._start_pixel
            self.pan += delta
            self._start_pixel = event.position()
            self.update()
            self.viewChanged.emit()
        self._last = world
        self._last_pixel = event.position()

    def mouseReleaseEvent(self, event):
        if (not self._panning and self.mode == "landmarks" and self._button == Qt.MouseButton.LeftButton
                and self._start_pixel is not None and (event.position()-self._start_pixel).manhattanLength() < 4
                and self.image_rect().contains(event.position())):
            self.pointClicked.emit(self._world(event.position()))
        self._last, self._button = None, None
        self._panning=False
        self._zooming=self._transform_drag=False
        if self.pan_mode or self._space_down: self.setCursor(Qt.CursorShape.OpenHandCursor)
        else: self.unsetCursor()

    def wheelEvent(self, event):
        if self.navigation_mode or event.modifiers() & (Qt.KeyboardModifier.ControlModifier|Qt.KeyboardModifier.MetaModifier):
            delta=event.angleDelta().y() or event.pixelDelta().y()
            if delta: self.zoom_at(event.position(),float(1.1**np.clip(delta/120,-4,4)))
        else:
            self.scrolled.emit(1 if event.angleDelta().y() > 0 else -1)
        event.accept()

    def event(self,event):
        # Cocoa sends trackpad pinches as native gestures, not wheel events.
        # Match the VisPy camera's scale change and keep the image coordinate
        # under the pointer fixed. Derive local position from global position:
        # Qt can report the wrong local position for macOS native gestures.
        if event.type()==QEvent.Type.NativeGesture:
            kind=event.gestureType()
            if kind==Qt.NativeGestureType.ZoomNativeGesture:
                scale=1-float(event.value())
                if np.isfinite(scale) and scale>0:
                    self.zoom_at(self.mapFromGlobal(event.globalPosition()),1/scale)
                event.accept()
                return True
            if kind in (Qt.NativeGestureType.BeginNativeGesture,Qt.NativeGestureType.EndNativeGesture):
                event.accept()
                return True
        return super().event(event)

    def zoom_at(self,position,factor):
        before=self.image_rect()
        if before.isEmpty(): return
        u=(position.x()-before.left())/before.width()
        v=(position.y()-before.top())/before.height()
        self.zoom=float(np.clip(self.zoom*factor,.25,40))
        after=self.image_rect()
        self.pan+=position-QPointF(after.left()+u*after.width(),after.top()+v*after.height())
        self.update()
        self.viewChanged.emit()

    def add_zoom_controls(self,layout):
        self.zoom_controls=QWidget()
        row=QHBoxLayout(self.zoom_controls)
        row.setContentsMargins(0,0,0,0)
        amount=QLabel('100%')
        amount.setFixedWidth(52)
        self.viewChanged.connect(lambda:amount.setText(f'{round(self.zoom*100)}%'))
        for text,factor in (('−',1/1.4),('+',1.4)):
            b=QPushButton(text)
            b.setMaximumWidth(36)
            b.clicked.connect(lambda checked=False,f=factor:self.zoom_at(QPointF(self.width()/2,self.height()/2),f))
            row.addWidget(b)
        row.addWidget(amount)
        fit=QPushButton('Fit')
        fit.setMaximumWidth(50)
        fit.clicked.connect(self.fit_image)
        row.addWidget(fit)
        row.addStretch()
        layout.addWidget(self.zoom_controls)
        self.setToolTip('Pinch or Ctrl/⌘ + wheel: zoom · Space + drag or middle-drag: pan · Double-click: reset view')

    def fit_image(self):
        self.zoom,self.pan=1.,QPointF()
        self.view_rotation=0
        self.update()
        self.viewChanged.emit()

    def view_record(self):
        return {'zoom':float(self.zoom),'pan':[self.pan.x(),self.pan.y()],'rotation':self.view_rotation}

    def restore_view(self,record):
        self.zoom=float(record.get('zoom',1))
        self.pan=QPointF(*record.get('pan',[0,0]))
        self.view_rotation=(int(record.get('rotation',0))//90*90)%360
        self.update()
        self.viewChanged.emit()

    def rotate_view(self):
        before=self.fit_scale()
        self.view_rotation=(self.view_rotation+90)%360
        ratio=self.fit_scale()/before
        self.pan=QPointF(-self.pan.y()*ratio,self.pan.x()*ratio)
        self.update(); self.viewChanged.emit()

    def keyPressEvent(self,event):
        if event.key()==Qt.Key.Key_Space:
            self._space_down=True
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        elif event.key() in (Qt.Key.Key_Plus,Qt.Key.Key_Equal,Qt.Key.Key_Minus):
            self.zoom_at(QPointF(self.width()/2,self.height()/2),1/1.4 if event.key()==Qt.Key.Key_Minus else 1.4)
        else: super().keyPressEvent(event)

    def keyReleaseEvent(self,event):
        if event.key()==Qt.Key.Key_Space:
            self._space_down=False
            self.unsetCursor()
        else: super().keyReleaseEvent(event)

    def focusOutEvent(self,event):
        self._space_down=False
        self._panning=False
        self._transform_drag=self._zooming=False
        super().focusOutEvent(event)

    def mouseDoubleClickEvent(self, event):
        self.fit_image()


class SlicePane(QWidget):
    changed = Signal()

    def __init__(self, axis, title="", *, canvas_type=SliceCanvas, compact=False):
        super().__init__()
        self.axis, self.title = axis, title
        layout = QVBoxLayout(self)
        layout.setContentsMargins(3, 3, 3, 3)
        self.label = QLabel(title)
        self.label.setWordWrap(True)
        layout.addWidget(self.label)
        self.canvas = canvas_type(axis)
        if not compact: self.canvas.add_zoom_controls(layout)
        layout.addWidget(self.canvas, 1)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.valueChanged.connect(self.changed)
        self.canvas.scrolled.connect(lambda delta: self.slider.setValue(self.slider.value()+delta))
        layout.addWidget(self.slider)

    def configure(self, shape, spacing, origin=(0,0,0), preserve_world=False):
        old_position = self.world_depth if hasattr(self,"spacing") else None
        self.spacing = np.asarray(spacing)
        self.origin = np.asarray(origin, dtype=float)
        self.slider.blockSignals(True)
        self.slider.setRange(0, shape[self.axis]-1)
        self.slider.setValue(round((old_position-self.origin[self.axis])/self.spacing[self.axis])
                             if preserve_world and old_position is not None else shape[self.axis]//2)
        self.slider.blockSignals(False)

    @property
    def world_depth(self):
        return float(self.origin[self.axis]+self.slider.value()*self.spacing[self.axis])

    def set_world_depth(self, position):
        self.slider.setValue(round((position-self.origin[self.axis])/self.spacing[self.axis]))

    def display(self, pixels):
        index = self.slider.value()
        self.label.setText(f"{self.title}{('XY','XZ','YZ')[self.axis]}  ·  {'ZYX'[self.axis]} {index} / {self.slider.maximum()}  ·  {self.world_depth:.1f} µm")
        self.canvas.set_image(pixels, self.spacing, index, self.origin)


class PlacementSlicePane(SlicePane):
    """An independently switchable overlay or calibrated grayscale image pair."""

    def __init__(self, axis):
        super().__init__(axis)
        self._placement_mode = False
        self._syncing_split=False
        self.slider.installEventFilter(self)
        layout = self.layout()
        layout.removeWidget(self.label)
        header = QHBoxLayout()
        header.addWidget(self.label, 1)
        self.fit_button=view_button('fit','Reset slice view',lambda:self._view_action(False))
        self.rotate_button=view_button('rotate','Rotate view 90° · atlas and scan',lambda:self._view_action(True))
        header.addWidget(self.fit_button); header.addWidget(self.rotate_button)
        from .desktop_figures import figure_button
        self.figure_button=figure_button(self.canvas); header.addWidget(self.figure_button)
        self.mode_selector = PopupComboBox()
        self.mode_selector.addItem("Overlay", "overlay")
        self.mode_selector.addItem("Side by side", "side_by_side")
        self.mode_selector.setToolTip("Compare one physical slice: atlas on the left, placed scan on the right.\n"
                                     "Grayscale panels use full brightness, independent of overlay opacity.")
        self.mode_selector.currentIndexChanged.connect(self._mode_changed)
        header.addWidget(self.mode_selector)
        layout.insertLayout(0, header)
        layout.removeWidget(self.canvas)
        self.views = QStackedWidget()
        self.views.addWidget(self.canvas)
        split = QWidget()
        grid = QGridLayout(split)
        grid.setContentsMargins(0, 0, 0, 0)
        self.reference_canvas, self.scan_canvas = SliceCanvas(axis), SliceCanvas(axis)
        for column,(title,canvas) in enumerate((("Atlas",self.reference_canvas),("Brain scan",self.scan_canvas))):
            canvas.setMinimumSize(90, 100)
            canvas.scrolled.connect(lambda delta: self.slider.setValue(self.slider.value()+delta))
            canvas.viewChanged.connect(lambda canvas=canvas: self._sync_split_view(canvas))
            title_row=QHBoxLayout(); title_row.addWidget(QLabel(title),1); title_row.addWidget(figure_button(canvas))
            grid.addLayout(title_row,0,column)
            grid.addWidget(canvas, 1, column)
            grid.setColumnStretch(column, 1)
        grid.setRowStretch(1, 1)
        self.scan_canvas.setToolTip("Left-drag: move scan · Right-drag: rotate scan\n"
                                    "Wheel: shared slice · Ctrl+wheel: shared zoom · Middle-drag: shared pan")
        self.views.addWidget(split)
        layout.insertWidget(1, self.views, 1)

    @property
    def canvases(self):
        return (self.canvas, self.reference_canvas, self.scan_canvas)

    def eventFilter(self,watched,event):
        if watched is self.slider and event.type()==QEvent.Type.Wheel:
            event.ignore()
            return True
        return super().eventFilter(watched,event)

    def _sync_split_view(self, source):
        if self._syncing_split: return
        other = self.scan_canvas if source is self.reference_canvas else self.reference_canvas
        other.zoom, other.pan = source.zoom, QPointF(source.pan)
        other.view_rotation=source.view_rotation
        other.update()

    def _view_action(self,rotate):
        self._syncing_split=True
        try:
            for canvas in self.canvases:
                canvas.rotate_view() if rotate else canvas.fit_image()
        finally: self._syncing_split=False

    def _mode_changed(self):
        self.views.setCurrentIndex(1 if self._placement_mode and self.mode_selector.currentData()=="side_by_side" else 0)
        self.figure_button.setVisible(self.views.currentIndex()==0)
        self.changed.emit()

    def set_placement_mode(self, enabled):
        self._placement_mode = enabled
        self.mode_selector.setVisible(enabled)
        self.canvas.mode = self.scan_canvas.mode = "placement" if enabled else "view"
        self.canvas.zoom_controls.hide()
        for canvas in self.canvases:
            canvas.navigation_mode=True
            canvas.show_axes=False
            if not enabled: canvas.smooth=True
            canvas.setToolTip('Pinch or wheel: zoom · Drag: pan · Right-drag: zoom · Double-click: reset view' +
                (' · Middle/Option-drag: move scan · Option-right-drag: rotate scan' if enabled and canvas is not self.reference_canvas else ''))
        self.views.setCurrentIndex(1 if enabled and self.mode_selector.currentData()=="side_by_side" else 0)
        self.figure_button.setVisible(self.views.currentIndex()==0)

    def display(self, pixels, reference=None, moving=None):
        super().display(pixels)
        self.label.setText(('XY','XZ','YZ')[self.axis])
        self.slider.setToolTip(f"{'ZYX'[self.axis]} · {self.slider.value()} / {self.slider.maximum()}")
        if self.views.currentIndex() == 1 and reference is not None and moving is not None:
            for canvas,image in ((self.reference_canvas,reference),(self.scan_canvas,moving)):
                canvas.set_image(image, self.spacing, self.slider.value(), self.origin)


class SliceComparison(QWidget):
    """Atlas / unscaled placement / registration on one physical plane."""
    changed = Signal()

    def __init__(self):
        super().__init__()
        self.grid = self.positions = None
        self.plane = PopupComboBox()
        self.plane.addItems(['XY','XZ','YZ'])
        self.plane.currentIndexChanged.connect(self._plane_changed)
        layout=QVBoxLayout(self)
        layout.setContentsMargins(8,8,8,8)
        row=QGridLayout()
        self.headers,self.canvases=[],[]
        self.fit_buttons,self.rotate_buttons=[],[]
        for column,title in enumerate(('Atlas','Before','After')):
            header=QLabel(title)
            heading=QHBoxLayout(); heading.addWidget(header,1)
            row.addLayout(heading,0,column)
            canvas=SliceCanvas(0)
            canvas.navigation_mode=True
            canvas.show_axes=False
            canvas.setToolTip('Pinch or wheel: zoom · Drag: pan · Right-drag: zoom')
            canvas.viewChanged.connect(lambda canvas=canvas:self._sync_view(canvas))
            row.addWidget(canvas,1,column)
            row.setColumnStretch(column,1)
            self.headers.append(header); self.canvases.append(canvas)
            fit=view_button('fit','Reset comparison view',canvas.fit_image)
            rotate=view_button('rotate','Rotate comparison 90°',canvas.rotate_view)
            heading.addWidget(fit); heading.addWidget(rotate)
            from .desktop_figures import figure_button
            heading.addWidget(figure_button(canvas))
            self.fit_buttons.append(fit); self.rotate_buttons.append(rotate)
        self.headers[1].setToolTip('Placement position, rotation and axes; scale removed')
        row.setRowStretch(1,1)
        layout.addLayout(row,1)
        self.slider=QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0,0)
        self.slider.valueChanged.connect(self._slice_changed)
        self.slider.installEventFilter(self)
        layout.addWidget(self.slider)

    def eventFilter(self,watched,event):
        if watched is self.slider and event.type()==QEvent.Type.Wheel:
            event.ignore(); return True
        return super().eventFilter(watched,event)

    @property
    def axis(self): return self.plane.currentIndex()

    def configure(self,grid):
        if self.grid==grid: return
        self.grid=grid
        if self.positions is None:
            self.positions=(np.asarray(grid['origin_um_zyx'])+
                (np.asarray(grid['shape_zyx'])-1)*grid['spacing_um_zyx']/2).tolist()
        self._sync_controls()

    def _sync_controls(self):
        if self.grid is None: return
        axis=self.axis
        origin,spacing=self.grid['origin_um_zyx'][axis],self.grid['spacing_um_zyx'][axis]
        self.slider.blockSignals(True)
        self.slider.setRange(0,self.grid['shape_zyx'][axis]-1)
        self.slider.setValue(round((self.positions[axis]-origin)/spacing))
        self.slider.blockSignals(False)
        self.positions[axis]=origin+self.slider.value()*spacing

    def _plane_changed(self):
        self._sync_controls()
        for canvas in self.canvases:
            canvas.zoom,canvas.pan,canvas.view_rotation=1.,QPointF(),0
        self.changed.emit()

    def _sync_view(self,source):
        for canvas in self.canvases:
            if canvas is not source:
                canvas.zoom,canvas.pan=source.zoom,QPointF(source.pan)
                canvas.view_rotation=source.view_rotation
                canvas.update()

    def _slice_changed(self):
        if self.grid is not None:
            self.positions[self.axis]=(self.grid['origin_um_zyx'][self.axis]+
                                       self.slider.value()*self.grid['spacing_um_zyx'][self.axis])
            self.changed.emit()

    def display(self,images,model='affine'):
        for canvas,pixels in zip(self.canvases,images):
            canvas.axis=self.axis
            canvas.set_image(pixels,self.grid['spacing_um_zyx'],self.slider.value(),self.grid['origin_um_zyx'])

    def state_record(self):
        return dict(axis=self.axis,position_um_zyx=list(self.positions) if self.positions is not None else None,
                    views=[canvas.view_record() for canvas in self.canvases])

    def restore_state(self,record):
        self.grid=None
        positions=record.get('position_um_zyx')
        self.positions=list(positions) if positions is not None else None
        self.plane.blockSignals(True)
        self.plane.setCurrentIndex(max(0,min(2,int(record.get('axis',0)))))
        self.plane.blockSignals(False)
        for canvas,view in zip(self.canvases,record.get('views',[{}]*3)):
            canvas.image=None; canvas.restore_view(view)


def visual_matrix(spacing_zyx, forward_zyx):
    """VisPy uses XYZ coordinates and row-vector MatrixTransform storage."""
    reverse, scale = np.eye(4), np.eye(4)
    reverse[:3, :3] = np.eye(3)[::-1]
    scale[:3, :3] = np.diag(spacing_zyx)
    return (reverse @ forward_zyx @ scale @ reverse).T


class VolumeView(QWidget):
    opacityChanged = Signal(int,int)
    layersRequested = Signal()
    renderSettingsChanged = Signal()
    viewSettingsChanged = Signal()

    def __init__(self, layer_titles=("Atlas","Scan")):
        super().__init__()
        self.layer_titles=layer_titles
        self.canvas = None
        self.reference = self.moving = None
        self.visuals = []
        self.slice_planes = []
        self.slice_grid = self.slice_positions = None
        from .desktop_rendering import default_render_settings
        self.render_settings = default_render_settings()
        self.settings_dialog = None
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(3, 3, 3, 3)
        self.heading=QLabel("3D  ·  drag to orbit, wheel to zoom")
        heading=QHBoxLayout(); heading.addWidget(self.heading,1)
        self.fit_button=view_button('fit','Fit volumes in view',self.reset_view)
        heading.addWidget(self.fit_button); self.layout.addLayout(heading)
        from .desktop_figures import figure_button
        heading.addWidget(figure_button(self))
        from .desktop_layers import VolumeLayers
        self.layers=VolumeLayers(layer_titles)
        self.layers.visibilityChanged.connect(self._apply_layer_visibility)
        self.layers.opacityChanged.connect(self._layer_opacity_changed)
        self.layers.visibilityChanged.connect(self.layersRequested)
        self.layout.addWidget(self.layers)
        self.label_visuals={}
        self._label_meshes=[]
        self._label_styles={}
        self.placeholder = QLabel("3D preview appears after loading")
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.placeholder, 1)
        self.plane_controls=QWidget()
        options=QVBoxLayout(self.plane_controls); options.setContentsMargins(0,0,0,0)
        planes = QHBoxLayout()
        planes.setContentsMargins(0,0,0,0)
        self.show_slice_planes = QCheckBox("Show slice planes")
        self.show_slice_planes.setChecked(True)
        self.show_slice_planes.setToolTip("XY (blue), XZ (green), YZ (red): current atlas-space slice positions.")
        self.fill_slice_planes = QCheckBox("Fill planes")
        for check in (self.show_slice_planes,self.fill_slice_planes):
            check.toggled.connect(self._slice_visibility_changed)
            planes.addWidget(check)
        planes.addStretch()
        options.addLayout(planes)
        self.show_axes=QCheckBox('XYZ axes')
        self.show_axes.setChecked(True)
        self.show_axes.toggled.connect(self._axis_visibility_changed)
        options.addWidget(self.show_axes)
        flip_row=QHBoxLayout()
        flip_row.addWidget(QLabel('Flip 3D'))
        self.flip_checks=[]
        for axis in 'XYZ':
            check=QCheckBox(axis)
            check.setToolTip(f'Mirror the entire 3D view along atlas {axis}. Display only; source coordinates and measurements stay unchanged.')
            check.toggled.connect(self._flip_changed)
            flip_row.addWidget(check); self.flip_checks.append(check)
        flip_row.addStretch()
        options.addLayout(flip_row)
        self.layout.addWidget(self.plane_controls)
        self.reset_button = QPushButton("Fit both volumes")
        self.reset_button.clicked.connect(self.reset_view)
        self.zoom_controls=QWidget()
        zoom_row=QHBoxLayout(self.zoom_controls)
        zoom_row.setContentsMargins(0,0,0,0)
        for title,factor in (('−',1.4),('+',1/1.4)):
            zoom_button=QPushButton(title)
            zoom_button.setMaximumWidth(36)
            zoom_button.clicked.connect(lambda checked=False,f=factor:self.zoom_volume(f))
            zoom_row.addWidget(zoom_button)
        zoom_row.addWidget(self.reset_button,1)
        self.layout.addWidget(self.zoom_controls)
        self.settings_button = QPushButton("3D rendering settings…")
        self.settings_button.clicked.connect(self.open_settings)
        self.layout.addWidget(self.settings_button)
        self.matrix = np.eye(4)
        self._placement_mode=False

    def set_placement_mode(self,enabled,*,compact=False):
        self._placement_mode=enabled
        compact=enabled or compact
        self.heading.setText('3D' if compact else '3D  ·  drag to orbit, wheel to zoom')
        self.layers.show_registration_layers(enabled)
        self.zoom_controls.setVisible(not compact)
        self.settings_button.setVisible(not compact)
        if enabled and self.settings_dialog is not None: self.settings_dialog.hide()
        self._apply_layer_visibility()

    def load(self, reference, moving):
        from vispy import app, scene
        from .desktop_rendering import apply_render_settings
        if hasattr(self,'extra_volume'): del self.extra_volume
        for visual in self.label_visuals.values(): visual.parent=None
        self.label_visuals={}
        self._label_meshes=[]
        app.use_app("pyqt6")
        from .volumes import Volume
        def gpu_preview(volume):
            factors = np.maximum(1,np.ceil(np.array(volume.data.shape)/192).astype(int))
            return Volume(np.ascontiguousarray(volume.data[tuple(slice(None,None,int(n)) for n in factors)]),
                          tuple(np.array(volume.spacing_um_zyx)*factors),volume.path,volume.intensity_limits)
        self.reference, self.moving = gpu_preview(reference), gpu_preview(moving)
        self.render_volumes=(self.reference,self.moving)
        if self.canvas is None:
            self.canvas = scene.SceneCanvas(keys=None, bgcolor="#080c12", size=(450, 350), show=False, parent=self)
            from .desktop_gl import GLPaintGuard
            self.gl_guard=GLPaintGuard(self.canvas.native)
            self.canvas.native.setMinimumSize(170, 140)
            self.layout.replaceWidget(self.placeholder, self.canvas.native)
            self.placeholder.hide()
            self.view = self.canvas.central_widget.add_view()
            self.view.camera = scene.cameras.ArcballCamera(fov=0,flip=tuple(self.flip_record()))
            from vispy.util.quaternion import Quaternion
            self.view.camera.set_state(_quaternion=Quaternion.create_from_euler_angles(20,-25,30,degrees=True))
        for visual in self.visuals:
            visual.parent = None
        self.visuals = []
        for volume,settings in zip((self.reference,self.moving),self.render_settings):
            from .desktop_rendering import volume_visual
            visual = volume_visual(volume,self.view.scene,settings)
            apply_render_settings(visual,settings)
            self.visuals.append(visual)
        # Atlas-aligned XYZ triad. Its geometry stays fixed when the scan moves.
        length = float(max(np.array(reference.data.shape)*reference.spacing_um_zyx))*.12
        points = np.array([[0,0,0],[length,0,0],[0,0,0],[0,length,0],[0,0,0],[0,0,length]])
        self.axis_line = getattr(self, "axis_line", None)
        if self.axis_line is not None:
            self.axis_line.parent = None
            self.axis_text.parent = None
        self.axis_line = scene.visuals.Line(points, color=np.array([[1,.4,.4,1]]*2+[[.4,1,.5,1]]*2+[[.4,.7,1,1]]*2),
                                           connect="segments", width=2, parent=self.view.scene)
        self.axis_text = scene.visuals.Text(["X", "Y", "Z"], pos=points[[1,3,5]], color="white", font_size=10, parent=self.view.scene)
        self._axis_visibility_changed()
        self.slice_grid = self.slice_positions = None
        if not self.slice_planes:
            for color in ((.4,.7,1,1),(.4,1,.5,1),(1,.4,.4,1)):
                mesh = scene.visuals.Mesh(vertices=np.zeros((4,3),dtype=np.float32),
                                          faces=np.array([[0,1,2],[0,2,3]],dtype=np.uint32),
                                          color=(*color[:3],.06), parent=self.view.scene)
                line = scene.visuals.Line(np.zeros((14,3),dtype=np.float32), color=color,
                                          connect="segments", width=1.5, parent=self.view.scene)
                text = scene.visuals.Text("", color=color, font_size=8, anchor_x="center", parent=self.view.scene)
                for order,visual in enumerate((mesh,line,text),start=10):
                    visual.order = order
                    visual.set_gl_state("translucent",depth_test=False)
                self.slice_planes.append((mesh,line,text))
        self._slice_visibility_changed()
        self.update_alignment(np.eye(4), .65, .9)
        self.reset_view()

    def update_slice_planes(self, grid, positions_zyx):
        """Use physical atlas coordinates; the scan transform never rotates these planes."""
        if grid is None or not self.slice_planes:
            return
        positions = np.asarray(positions_zyx,dtype=float)
        if self.slice_grid == grid and np.array_equal(self.slice_positions,positions):
            return
        self.slice_grid, self.slice_positions = grid,positions.copy()
        spacing = np.asarray(grid["spacing_um_zyx"])
        lower = np.asarray(grid["origin_um_zyx"])-spacing/2
        upper = np.asarray(grid["origin_um_zyx"])+(np.asarray(grid["shape_zyx"])-.5)*spacing
        edges = [0,1,1,2,2,3,3,0]
        for axis,(mesh,line,text) in enumerate(self.slice_planes):
            a,b = [i for i in range(3) if i != axis]
            corners = np.tile(positions,(4,1))
            corners[:,a] = [lower[a],lower[a],upper[a],upper[a]]
            corners[:,b] = [lower[b],upper[b],upper[b],lower[b]]
            cross = np.tile(positions,(6,1))
            # Two intersection lines plus a short axis normal through the plane.
            cross[0,a],cross[1,a] = lower[a],upper[a]
            cross[2,b],cross[3,b] = lower[b],upper[b]
            cross[4,axis],cross[5,axis] = positions[axis]-spacing[axis],positions[axis]+spacing[axis]
            xyz = np.ascontiguousarray(corners[:,::-1],dtype=np.float32)
            mesh.set_data(vertices=xyz,faces=np.array([[0,1,2],[0,2,3]],dtype=np.uint32))
            line.set_data(pos=np.ascontiguousarray(np.vstack((corners[edges],cross))[:,::-1],dtype=np.float32))
            text.text = ('XY','XZ','YZ')[axis]
            text.pos = xyz[(1,3,1)[axis]]
        self.show_slice_planes.setToolTip("XY (blue), XZ (green), YZ (red)\n"+"\n".join(
            f"{plane}: {normal} = {depth:.2f} µm"
            for plane,normal,depth in zip(('XY','XZ','YZ'),'ZYX',positions)))
        self._slice_visibility_changed()

    def _axis_visibility_changed(self):
        for name in ('axis_line','axis_text'):
            visual=getattr(self,name,None)
            if visual is not None: visual.visible=self.show_axes.isChecked()
        if self.canvas is not None: self.canvas.update()

    def flip_record(self):
        return [check.isChecked() for check in self.flip_checks]

    def restore_flip(self,record):
        values=record if isinstance(record,(list,tuple)) and len(record)==3 else [False]*3
        for check,value in zip(self.flip_checks,values):
            check.blockSignals(True); check.setChecked(bool(value)); check.blockSignals(False)
        self._apply_flip()

    def _apply_flip(self):
        # Camera reflection includes every world-space visual and its picking
        # transform. Never bake this display preference into registration data.
        if self.canvas is not None:
            self.view.camera.flip=tuple(self.flip_record())
            self.canvas.update()

    def _flip_changed(self):
        self._apply_flip()
        self.viewSettingsChanged.emit()

    def _slice_visibility_changed(self):
        visible = self.show_slice_planes.isChecked() and self.slice_grid is not None
        self.fill_slice_planes.setEnabled(self.show_slice_planes.isChecked())
        for mesh,line,text in self.slice_planes:
            mesh.visible = visible and self.fill_slice_planes.isChecked()
            line.visible = text.visible = visible
        if self.canvas is not None:
            self.canvas.update()

    def slice_settings_record(self):
        return {"visible":self.show_slice_planes.isChecked(), "fill":self.fill_slice_planes.isChecked()}

    def restore_slice_settings(self, record):
        self.show_slice_planes.setChecked(bool(record.get("visible",True)))
        self.fill_slice_planes.setChecked(bool(record.get("fill",False)))

    def update_alignment(self, matrix, reference_opacity, moving_opacity):
        if not self.visuals:
            return
        from vispy.visuals.transforms import MatrixTransform
        self.matrix = np.asarray(matrix).copy()
        for visual, volume, forward, opacity in zip(self.visuals, (self.reference, self.moving),
                                                    (np.eye(4), self.matrix), (reference_opacity, moving_opacity)):
            target=visual_matrix(volume.spacing_um_zyx, forward)
            if isinstance(visual.transform,MatrixTransform):
                if not np.array_equal(visual.transform.matrix,target):
                    visual.transform.matrix=target
            else:
                # Reuse shader transform objects during dragging. Replacing
                # them every frame forces repeated GPU shader recompilation.
                visual.transform=MatrixTransform(target)
            self.layers.sliders[self.visuals.index(visual)].blockSignals(True)
            self.layers.sliders[self.visuals.index(visual)].setValue(round(opacity*100))
            self.layers.sliders[self.visuals.index(visual)].blockSignals(False)
            self.layers.amounts[self.visuals.index(visual)].setText(f'{round(opacity*100)}%')
            visual.opacity = float(opacity)
            visual.visible = opacity > 0 and self.layers.checks[self.visuals.index(visual)].isChecked()
        if len(self.visuals)>2 and hasattr(self,"extra_volume"):
            self.visuals[2].transform=MatrixTransform(visual_matrix(self.extra_volume.spacing_um_zyx,self.matrix))
        self._apply_layer_visibility()
        self.canvas.update()
        if self.settings_dialog is not None:
            self.settings_dialog.sync()

    def open_settings(self):
        if self.settings_dialog is None:
            from .desktop_rendering import RenderSettingsDialog
            self.settings_dialog = RenderSettingsDialog(self)
        self.settings_dialog.sync()
        self.settings_dialog.show()
        self.settings_dialog.raise_()
        self.settings_dialog.activateWindow()

    def zoom_volume(self,factor):
        if self.canvas is None: return
        self.view.camera.scale_factor*=factor
        self.canvas.update()

    def set_layer_titles(self,titles):
        from .desktop_rendering import default_render_settings
        if tuple(titles)==tuple(self.layer_titles): return
        if self.settings_dialog is not None:
            self.settings_dialog.close()
            self.settings_dialog.deleteLater()
            self.settings_dialog=None
        self.layer_titles=tuple(titles)
        if len(self.visuals)>len(titles):
            for visual in self.visuals[len(titles):]: visual.parent=None
            self.visuals=self.visuals[:len(titles)]
        self.layers.set_titles(titles)
        self.render_settings=(self.render_settings+default_render_settings(len(titles))[len(self.render_settings):])[:len(titles)]

    def _layer_opacity_changed(self,index,value):
        self._apply_layer_visibility()
        self.opacityChanged.emit(index,value)

    def _apply_layer_visibility(self):
        for index,visual in enumerate(self.visuals):
            if index<len(self.layers.checks):
                visual.opacity=self.layers.sliders[index].value()/100
                visual.visible=self.layers.checks[index].isChecked() and visual.opacity>0 and (not self._placement_mode or index<2)
        for identifier,visual in self.label_visuals.items():
            style=self._label_styles.get(str(identifier),{})
            visual.visible=self.layers.labels_check.isChecked() and style.get('visible',True) and not self._placement_mode
            source=next(m for m in self._label_meshes if m['id']==identifier)
            color=style.get('color_rgb',source['color_rgb'])
            visual.color=(*[v/255 for v in color],self.layers.labels_opacity.value()/100)
        if self.canvas is not None: self.canvas.update()

    def set_extra_volume(self,volume):
        from vispy import scene
        from .desktop_rendering import apply_render_settings
        from vispy.visuals.transforms import MatrixTransform
        self.extra_volume=volume
        if len(self.visuals)>2: self.visuals[2].parent=None; self.visuals=self.visuals[:2]
        from .desktop_rendering import volume_visual
        visual=volume_visual(volume,self.view.scene,self.render_settings[2])
        apply_render_settings(visual,self.render_settings[2])
        visual.transform=MatrixTransform(visual_matrix(volume.spacing_um_zyx,self.matrix))
        self.visuals.append(visual)
        self.render_volumes=(self.reference,self.moving,volume)
        self._apply_layer_visibility()
        self.renderSettingsChanged.emit()

    def set_label_meshes(self,meshes,styles=None):
        from vispy import scene
        for visual in self.label_visuals.values(): visual.parent=None
        self.label_visuals={}
        self._label_meshes=meshes
        import copy
        self._label_styles=copy.deepcopy(styles or {})
        for item in meshes:
            visual=scene.visuals.Mesh(vertices=item['vertices_zyx_um'][:,::-1],faces=item['faces'],
                                     shading='smooth',color=(1,1,1,.25),parent=self.view.scene)
            visual.order=5
            visual.set_gl_state('translucent',depth_test=False)
            self.label_visuals[item['id']]=visual
        self._apply_layer_visibility()

    def set_label_styles(self,styles):
        if styles==self._label_styles: return
        import copy
        self._label_styles=copy.deepcopy(styles)
        self._apply_layer_visibility()

    def render_settings_record(self):
        import copy
        return copy.deepcopy(self.render_settings)

    def restore_render_settings(self, records):
        from .desktop_rendering import validate_render_settings,apply_render_settings
        previous=self.render_settings
        self.render_settings = validate_render_settings(records,len(self.layer_titles))
        for visual,settings,old in zip(self.visuals,self.render_settings,previous):
            if settings!=old:
                apply_render_settings(visual,settings,old)
        if self.settings_dialog is not None:
            self.settings_dialog.sync()
        if self.canvas is not None:
            self.canvas.update()
        self.renderSettingsChanged.emit()

    def set_render_settings(self,index,settings):
        records=self.render_settings_record()
        records[index]=settings
        self.restore_render_settings(records)

    def reset_view(self):
        if not self.visuals:
            return
        ref = transformed_corners(self.reference.data.shape, self.reference.spacing_um_zyx, np.eye(4))
        mov = transformed_corners(self.moving.data.shape, self.moving.spacing_um_zyx, self.matrix)
        corners = np.vstack((ref, mov))[:, ::-1]
        self.view.camera.set_range(x=(corners[:,0].min(),corners[:,0].max()),
                                   y=(corners[:,1].min(),corners[:,1].max()),
                                   z=(corners[:,2].min(),corners[:,2].max()), margin=.12)
        self.canvas.update()

    def close_canvas(self):
        if self.settings_dialog is not None:
            self.settings_dialog.close()
        if self.canvas is not None:
            self.gl_guard.stopped=True
            # VisPy processes Qt events while closing. Hide the embedded GL
            # widget first so queued paints cannot target a closing context.
            self.canvas.native.setUpdatesEnabled(False)
            self.canvas.native.hide()
            self.canvas.close()

    def camera_record(self):
        if self.canvas is None:
            return {}
        state = self.view.camera.get_state()
        q = state.pop("_quaternion")
        state["quaternion_wxyz"] = [q.w,q.x,q.y,q.z]
        return state

    def restore_camera(self, record):
        if self.canvas is None or not record:
            return
        from vispy.util.quaternion import Quaternion
        state = dict(record)
        state["_quaternion"] = Quaternion(*state.pop("quaternion_wxyz"))
        self.view.camera.set_state(state)

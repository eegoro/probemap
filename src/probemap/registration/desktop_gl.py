"""Keep queued Qt paints away from unprepared or closing GL framebuffers."""
from qtpy.QtCore import QEvent,QObject,QTimer
from qtpy.QtGui import QResizeEvent
from qtpy.QtOpenGLWidgets import QOpenGLWidget


def render_frame(canvas,size):
    """Render the complete logical viewport at a chosen physical resolution."""
    from vispy import gloo
    canvas.set_current()
    fbo=gloo.FrameBuffer(color=gloo.RenderBuffer(size[::-1]),depth=gloo.RenderBuffer(size[::-1]))
    canvas.push_fbo(fbo,(0,0),canvas.size)
    try:
        # VisPy 0.16's render(size=...) leaves the scene in native framebuffer
        # pixels on Retina displays. Map that entire framebuffer to the PNG,
        # without resizing the live widget or changing its camera/layout.
        canvas.transforms.configure(viewport=(0,0,*size),fbo_size=size,
                                    fbo_rect=(0,0,*canvas.physical_size))
        canvas._draw_scene()
        return fbo.read()
    finally:
        canvas.pop_fbo()


class GLPaintGuard(QObject):
    def __init__(self,widget):
        super().__init__(widget)
        self.widget=widget
        self.suspended=self.stopped=self.pending=False
        widget.installEventFilter(self)

    def eventFilter(self,watched,event):
        if event.type()!=QEvent.Type.Paint: return False
        if self.stopped or self.suspended or not watched.isVisible() or not watched.updatesEnabled(): return True
        # Qt initializes the framebuffer in resizeEvent. A queued paint after
        # a grid/page transition can arrive before that event on Cocoa.
        if watched.defaultFramebufferObject()==0:
            if not self.pending:
                self.pending=True
                QTimer.singleShot(0,self.prepare_framebuffer)
            return True
        return False

    def prepare_framebuffer(self):
        self.pending=False
        if self.stopped or self.suspended: return
        widget=self.widget
        if not widget.isVisible() or widget.width()<1 or widget.height()<1: return
        QOpenGLWidget.resizeEvent(widget,QResizeEvent(widget.size(),widget.size()))
        widget.update()

"""Small translucent, icon-only controls over native slice canvases."""
from qtpy.QtCore import QPointF,QRectF,QSize,Qt,Signal
from qtpy.QtGui import QColor,QIcon,QPainter,QPainterPath,QPen,QPixmap
from qtpy.QtWidgets import QFrame,QToolButton,QVBoxLayout


def tool_icon(name,*,color='#e5f4ff'):
    pixmap=QPixmap(48,48)
    pixmap.fill(Qt.GlobalColor.transparent)
    p=QPainter(pixmap)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.scale(2,2)
    p.setPen(QPen(QColor(color),1.6,Qt.PenStyle.SolidLine,Qt.PenCapStyle.RoundCap,Qt.PenJoinStyle.RoundJoin))
    p.setBrush(Qt.BrushStyle.NoBrush)
    def line(a,b): p.drawLine(QPointF(*a),QPointF(*b))
    if name in ('zoom_in','zoom_out'):
        p.drawEllipse(QRectF(3,3,12,12)); line((13,13),(21,21)); line((6,9),(12,9))
        if name=='zoom_in': line((9,6),(9,12))
    elif name=='run':
        path=QPainterPath(); path.moveTo(7,4); path.lineTo(20,12); path.lineTo(7,20); path.closeSubpath(); p.drawPath(path)
    elif name=='fit':
        for x,y,dx,dy in ((3,3,1,1),(21,3,-1,1),(3,21,1,-1),(21,21,-1,-1)):
            line((x,y),(x+5*dx,y)); line((x,y),(x,y+5*dy))
        p.drawRect(QRectF(8,8,8,8))
    elif name in ('reset','rotate'):
        p.drawArc(QRectF(5,5,15,15),35*16,285*16)
        line((17,3),(20,7)); line((20,7),(15,8))
    elif name=='face':
        line((11,3),(11,20)); line((11,20),(14,23)); line((14,23),(17,20)); line((17,20),(17,3))
        for y in (6,10,14): p.drawEllipse(QPointF(14,y),.7,.7)
        line((2,8),(8,8)); line((2,8),(5,5)); line((2,8),(5,11))
        line((8,17),(2,17)); line((8,17),(5,14)); line((8,17),(5,20))
    elif name=='clear':
        line((5,6),(19,6));line((9,3),(15,3))
        line((7,6),(8,21));line((8,21),(16,21));line((16,21),(17,6))
        line((10,10),(10,17));line((14,10),(14,17))
    elif name=='pan':
        line((12,3),(12,21)); line((3,12),(21,12))
        for a,b in (((8,7),(12,3)),((16,7),(12,3)),((8,17),(12,21)),((16,17),(12,21)),
                    ((7,8),(3,12)),((7,16),(3,12)),((17,8),(21,12)),((17,16),(21,12))): line(a,b)
    elif name=='add':
        p.drawEllipse(QRectF(5,5,14,14)); line((12,8),(12,16)); line((8,12),(16,12))
    elif name=='select':
        path=QPainterPath(QPointF(5,3))
        for a in ((5,19),(10,14),(14,21),(17,19),(13,12),(20,12)): path.lineTo(*a)
        path.closeSubpath(); p.drawPath(path)
    elif name in ('draw','track'):
        line((4,19),(19,4)); p.drawEllipse(QPointF(4,19),2,2); p.drawEllipse(QPointF(19,4),2,2)
        if name=='track':
            line((4,6),(10,6)); line((7,3),(7,9)); line((15,18),(21,18)); line((18,15),(18,21))
        else: line((8,20),(20,8))
    elif name=='level':
        line((3,12),(21,12)); line((7,5),(7,9)); line((7,15),(7,19)); line((4,7),(7,4)); line((7,4),(10,7))
    elif name=='measure':
        path=QPainterPath(QPointF(3,17))
        for a in ((17,3),(22,8),(8,22)): path.lineTo(*a)
        path.closeSubpath(); p.drawPath(path)
        for x,y in ((7,13),(11,9),(15,5)): line((x,y),(x+3,y+3))
    elif name=='sidebar':
        p.drawRoundedRect(QRectF(3,4,18,16),2,2); line((9,4),(9,20))
    elif name=='save':
        p.drawRoundedRect(QRectF(4,3,16,18),2,2)
        p.drawRect(QRectF(8,3,8,6)); p.drawRect(QRectF(8,14,8,7))
    elif name=='folder':
        path=QPainterPath(QPointF(3,7))
        for a in ((9,7),(11,10),(21,10),(19,20),(3,20),(3,7)): path.lineTo(*a)
        p.drawPath(path); line((3,12),(20,12))
    elif name=='new':
        p.drawRoundedRect(QRectF(5,3,14,18),2,2)
        line((8,12),(16,12)); line((12,8),(12,16))
    elif name=='settings':
        import math
        p.drawEllipse(QRectF(6,6,12,12)); p.drawEllipse(QRectF(10,10,4,4))
        for angle in range(0,360,45):
            a=math.radians(angle)
            line((12+6*math.cos(a),12+6*math.sin(a)),(12+9*math.cos(a),12+9*math.sin(a)))
    elif name=='adjust':
        for x,y in ((6,8),(12,16),(18,10)):
            line((x,3),(x,y-2)); line((x,y+2),(x,21))
            p.drawEllipse(QPointF(x,y),2,2)
    elif name=='search':
        p.drawEllipse(QRectF(3,3,13,13)); line((14,14),(21,21))
    p.end()
    return QIcon(pixmap)


class SliceTools(QFrame):
    toolSelected=Signal(str)

    def __init__(self,canvas):
        super().__init__(canvas)
        self.canvas=canvas
        self.buttons={}
        self.setObjectName('sliceTools')
        self.setStyleSheet('''QFrame#sliceTools {background:rgba(24,32,43,165);border:1px solid rgba(150,180,200,80);border-radius:7px;}
            QToolButton {background:transparent;border:0;border-radius:4px;padding:3px;}
            QToolButton:hover {background:rgba(110,165,200,150);}
            QToolButton:checked {background:rgba(35,135,185,190);}
            QToolButton:disabled {background:transparent;}''')
        layout=QVBoxLayout(self)
        layout.setContentsMargins(3,4,3,4)
        layout.setSpacing(1)
        labels={'zoom_in':'Zoom in (+)','zoom_out':'Zoom out (−)','fit':'Fit image',
                'pan':'Pan · left-drag / Shift+drag','add':'Add insertion point · step 5',
                'select':'Select insertion points','track':'Select / move track',
                'draw':'Draw 4 mm track · step 6','measure':'Measure distance'}
        for name in labels:
            button=QToolButton(self)
            button.setIcon(tool_icon(name))
            button.setIconSize(QSize(21,21))
            button.setFixedSize(29,29)
            button.setToolTip(labels[name]); button.setAccessibleName(labels[name])
            button.setCheckable(name not in ('zoom_in','zoom_out','fit'))
            if name in ('zoom_in','zoom_out'):
                button.clicked.connect(lambda checked=False,n=name:canvas.zoom_at(QPointF(canvas.width()/2,canvas.height()/2),1.4 if n=='zoom_in' else 1/1.4))
            elif name=='fit': button.clicked.connect(canvas.fit_image)
            else: button.clicked.connect(lambda checked=False,n=name:self.toolSelected.emit(n))
            layout.addWidget(button)
            self.buttons[name]=button
        self.move(7,30)

    def configure(self,mode,index):
        allowed={'pan','measure'}
        if mode==0: allowed.update(('add','select'))
        if mode in (1,3): allowed.add('track')
        if mode==1 and index==0: allowed.add('draw')
        for name,button in self.buttons.items():
            button.setVisible(name in allowed or name in ('zoom_in','zoom_out','fit'))
        self.place()

    def place(self):
        visible=[b for b in self.buttons.values() if not b.isHidden()]
        height=max(16,min(29,int((self.canvas.height()-45-len(visible))/max(1,len(visible)))))
        for button in visible:
            button.setFixedSize(29,height)
            button.setIconSize(QSize(min(21,height-3),min(21,height-3)))
        self.adjustSize()
        self.move(7,30)
        self.raise_()

    def set_active(self,tool):
        for name,button in self.buttons.items():
            if button.isCheckable():
                button.blockSignals(True); button.setChecked(name==tool); button.blockSignals(False)

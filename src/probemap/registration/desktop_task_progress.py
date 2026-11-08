"""Quiet task feedback with explicit percentage, elapsed time and honest ETA."""
from collections import deque
import time

from qtpy.QtCore import QTimer, Qt
from qtpy.QtWidgets import QHBoxLayout, QLabel, QProgressBar, QPushButton, QVBoxLayout, QWidget


def duration_text(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{hours}:{minutes:02}:{seconds:02}' if hours else f'{minutes:02}:{seconds:02}'


class TaskTiming:
    """Estimate from measured progress; never invent progress during blocked IO."""
    def __init__(self, clock=time.monotonic):
        self.clock = clock; self.started = self.phase_started = self.last_reported = clock()
        self.samples = deque(maxlen=40); self.value = 0

    def start(self, continuing=False):
        now = self.clock()
        if not continuing: self.started = now
        self.phase_started = self.last_reported = now; self.value = 0; self.samples.clear()
        self.samples.append((now, 0))

    def report(self, value):
        now = self.clock(); self.last_reported = now; self.value = max(self.value, min(100, max(0, float(value))))
        if self.value != self.samples[-1][1]: self.samples.append((now, self.value))

    def elapsed(self): return self.clock()-self.started

    def remaining(self):
        now = self.clock()
        if self.value >= 100: return 0
        if self.value <= 0 or len(self.samples) < 2 or now-self.phase_started < 2: return None
        recent = [sample for sample in self.samples if sample[0] >= now-30]
        first = recent[0] if len(recent) >= 2 else self.samples[0]
        last = self.samples[-1]
        elapsed, delta = last[0]-first[0], last[1]-first[1]
        if elapsed <= 0 or delta <= 0: return None
        # Preserve the estimate through long IO requests. Without measurable
        # progress its uncertainty grows; never count down towards a fake finish.
        return (100-self.value)*elapsed/delta + max(0, now-last[0])

    def waiting(self):
        return self.value > 0 and self.clock()-self.last_reported > 20


class TaskProgress(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent); self.running = False; self.timing = TaskTiming()
        self.setObjectName('taskProgress'); self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet('''
            QWidget#taskProgress{background:#202936;border:1px solid #354354;border-radius:6px;}
            QLabel{background:transparent;border:0;color:#a8b8cb;font-size:11px;}
            QLabel#taskStage{color:#e5eaf2;font-size:12px;}
            QLabel#taskPercent{color:#8bd0e4;font-size:12px;font-weight:600;}
            QProgressBar{background:#354354;border:0;border-radius:2px;min-height:4px;max-height:4px;}
            QProgressBar::chunk{background:#7ac7e5;border-radius:2px;}
            QPushButton{background:transparent;border:0;padding:2px 4px;color:#a8b8cb;}
            QPushButton:hover{color:#f5f8ff;}
        ''')
        layout = QVBoxLayout(self); layout.setContentsMargins(12, 8, 12, 8); layout.setSpacing(6)
        header = QHBoxLayout(); self.stage = QLabel(); self.stage.setObjectName('taskStage'); header.addWidget(self.stage, 1)
        self.percent = QLabel('0%'); self.percent.setObjectName('taskPercent'); header.addWidget(self.percent)
        self.cancel = QPushButton('Cancel'); self.cancel.setToolTip('Cancel this task'); header.addWidget(self.cancel)
        layout.addLayout(header)
        self.bar = QProgressBar(); self.bar.setRange(0, 100); self.bar.setTextVisible(False); self.bar.setFixedHeight(4); layout.addWidget(self.bar)
        times = QHBoxLayout(); self.elapsed_label = QLabel(); self.remaining_label = QLabel()
        times.addWidget(self.elapsed_label); times.addStretch(); times.addWidget(self.remaining_label); layout.addLayout(times)
        self.timer = QTimer(self); self.timer.setInterval(1000); self.timer.timeout.connect(self.tick)
        self.dismiss = QTimer(self); self.dismiss.setSingleShot(True); self.dismiss.timeout.connect(self.hide)
        self.hide()

    def start(self, title):
        self.dismiss.stop(); self.timing.start(continuing=self.running); self.running = True
        self.stage.setText(title); self.percent.setText('0%'); self.bar.setValue(0)
        self.cancel.show(); self.cancel.setEnabled(True); self.tick(); self.timer.start(); self.show()

    def report(self, value, message):
        self.timing.report(value); self.bar.setValue(int(self.timing.value))
        self.percent.setText(f'{self.timing.value:.1f}%' if 0 < self.timing.value < 1 else f'{int(self.timing.value)}%'); self.stage.setText(message); self.tick()

    def tick(self):
        self.elapsed_label.setText(f'Elapsed {duration_text(self.timing.elapsed())}')
        remaining = self.timing.remaining()
        text = 'Estimating remaining time…' if remaining is None else f'About {duration_text(remaining)} remaining'
        if self.running and self.timing.waiting(): text += ' · Waiting for progress'
        self.remaining_label.setText(text)

    def finish(self, cancelled=False, error=False):
        self.timer.stop(); self.running = False; self.cancel.hide()
        if cancelled or error:
            self.stage.setText('Cancelled' if cancelled else 'Could not complete')
        else:
            self.bar.setValue(100); self.percent.setText('100%'); self.stage.setText('Complete')
        self.elapsed_label.setText(f'Elapsed {duration_text(self.timing.elapsed())}'); self.remaining_label.clear()
        if self.timing.elapsed() < 1 and not (cancelled or error): self.hide()
        else: self.dismiss.start(3500)

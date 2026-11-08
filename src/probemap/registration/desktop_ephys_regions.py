"""The existing atlas region editor, scoped to the electrophysiology display."""
import copy
from types import SimpleNamespace

from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QWidget, QVBoxLayout, QFormLayout

from .desktop_regions import RegionControlsMixin
from .insertion_model import insertion_session


class EphysRegions(RegionControlsMixin, QWidget):
    def __init__(self, page):
        super().__init__(); self.page = page; self.owner = page.owner
        self.labels = self.session = None
        self.report_timer = QTimer(self); self.report_timer.setSingleShot(True)
        self.report_timer.timeout.connect(self.draw_cached)
        layout = QVBoxLayout(self); layout.setContentsMargins(0,0,0,0)
        layout.addWidget(self._region_controls())
        self.atlas_gamma.setToolTip('Image contrast for atlas or scan')
        for form in self.findChildren(QFormLayout):
            label = form.labelForField(self.atlas_gamma)
            if label: label.setText('Image gamma')
        self.region_table.setProperty('persistent_selection', True)
        self.region_table.setFixedHeight(170)

    def _remember(self): pass  # These display settings do not edit the anatomical annotation.

    def set_anatomy(self, anatomy, project):
        if not project: return
        self.session = project.ephys.setdefault('anatomy_display',
            dict(regions=copy.deepcopy(insertion_session(project).get('regions', {})), opacity=38, gamma=1.))
        self.session.setdefault('regions', {})
        self.labels = SimpleNamespace(regions=anatomy.get('label_regions', []) if anatomy else [])
        for widget, value in ((self.region_opacity, self.session.get('opacity',38)),
                              (self.atlas_gamma, self.session.get('gamma',1.))):
            widget.blockSignals(True); widget.setValue(value); widget.blockSignals(False)
        self._populate_regions()

    def draw_cached(self, *args):
        if self.session is None: return
        self.session.update(opacity=self.region_opacity.value(),gamma=self.atlas_gamma.value())
        board=getattr(self.page,'plot_board',None)
        if board:
            board.queue_panel('__anatomy__')
            for key in board.keys():
                if key.startswith('region_'): board.queue_panel(key)
        self.owner.schedule_save()

"""Dispose Qt objects while Python is still running, before plugin shutdown."""
import gc
import sys

import pytest


@pytest.fixture(scope='session', autouse=True)
def close_qt_after_tests():
    yield
    # Core-only runs do not need to import or create a GUI application.
    widgets_module=sys.modules.get('qtpy.QtWidgets')
    if widgets_module is None:
        return
    app=widgets_module.QApplication.instance()
    if app is None:
        return
    from qtpy.QtCore import QCoreApplication,QEvent,QThreadPool
    widgets=list(app.topLevelWidgets())
    for widget in widgets:
        stop=getattr(widget,'stop',None)
        if callable(stop):
            stop()
        widget.close()
    assert QThreadPool.globalInstance().waitForDone(15000), 'Qt workers did not stop'
    app.processEvents()
    for widget in widgets:
        widget.deleteLater()
    QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete)
    app.processEvents()
    widgets.clear()
    gc.collect()
    # Destroy the application before Python unloads Qt and its platform plugin.
    from PyQt6 import sip
    sip.delete(app)

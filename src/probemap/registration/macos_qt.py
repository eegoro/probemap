"""Prepare pip-installed Qt plugins for discovery on macOS."""
from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

_application = None


def prepare_qt() -> Path | None:
    """Clear macOS UF_HIDDEN on Qt plugin dylibs before QApplication starts.

    Some macOS/pip installations repeatedly set UF_HIDDEN on these files. Qt's
    directory scanner then ignores them, causing a fatal 'cocoa plugin not found'
    error. Clearing flags in an earlier notebook cell is not sufficient: use
    initialize_qt() immediately before starting the GUI.
    Only the hidden flag of installed Qt plugins is changed, never their data.
    See https://github.com/pytorch/pytorch/issues/178841 (also affects pure Qt).
    """
    os.environ.setdefault("QT_API", "pyqt6")
    if sys.platform != "darwin" or os.environ["QT_API"].lower() != "pyqt6":
        return None
    from PyQt6.QtCore import QLibraryInfo

    directory = Path(QLibraryInfo.path(QLibraryInfo.LibraryPath.PluginsPath))
    for path in directory.rglob("*.dylib"):
        flags = path.stat().st_flags
        if flags & stat.UF_HIDDEN:
            try:
                os.chflags(path, flags & ~stat.UF_HIDDEN)
            except OSError as error:
                raise RuntimeError(f"Qt plugin is hidden and cannot be repaired: {path}") from error
    os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", str(directory / "platforms"))
    return directory


def initialize_qt():
    """Create and retain QApplication immediately after repairing plugin flags.

    Import QtWidgets *before* clearing flags: imports or image loading can take
    long enough for macOS to hide the plugins again. The two critical operations
    below must stay adjacent. Repeated calls reuse the live application.
    """
    global _application
    os.environ.setdefault("QT_API", "pyqt6")
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QApplication

    if QApplication.instance() is None:
        # Keep embedded GL widgets in a stable share group across windows and
        # page changes. Qt requires this flag before QApplication is created.
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    prepare_qt()
    _application = QApplication.instance()
    if _application is None:
        _application = QApplication(["probemap"])

    # QApplication already exists when IPython connects its Qt event loop.
    # A bare `%gui qt` before this function could abort the kernel on macOS.
    try:
        from IPython import get_ipython
    except ImportError:
        shell = None
    else:
        shell = get_ipython()
    if shell is not None:
        shell.enable_gui("qt")
    return _application

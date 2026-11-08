"""A fresh process is required: an existing QApplication conceals this crash."""

import pytest

pytestmark = pytest.mark.gui
import os
import subprocess
import sys

import pytest


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS UF_HIDDEN / cocoa regression")
def test_hidden_plugins_reappearing_before_application_start():
    code = '''
import os, stat
from PyQt6.QtCore import QDir, Qt
from PyQt6.QtWidgets import QApplication
from probemap.registration.macos_qt import prepare_qt, initialize_qt

directory = prepare_qt()
assert QApplication.instance() is None
# Reproduce the state seen after a pause between notebook cells.
for plugin in (directory / "platforms").glob("*.dylib"):
    os.chflags(plugin, plugin.stat().st_flags | stat.UF_HIDDEN)
assert "libqcocoa.dylib" not in QDir(str(directory / "platforms")).entryList()
application = initialize_qt()
assert application.platformName() == "cocoa"
assert application.testAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
assert initialize_qt() is application
print("QT_STARTUP_OK", flush=True)
'''
    result = subprocess.run([sys.executable, "-c", code], env={**os.environ, "QT_API": "pyqt6"},
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "QT_STARTUP_OK" in result.stdout

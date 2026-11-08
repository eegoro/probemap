"""Directory selection, including multiple mounted network folders on macOS."""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
import sys


def _mac_folders(directory=None):
    """Use public NSOpenPanel APIs without requiring a PyObjC dependency.

    Qt's non-native directory model hides /Volumes on macOS and its directory
    mode does not reliably accept multiple selected folders. Finder's native
    panel supports both directly, including mounted SMB shares and aliases.
    Must be called on the main (UI) thread.
    """
    ctypes.CDLL('/System/Library/Frameworks/AppKit.framework/AppKit')
    objc = ctypes.CDLL('/usr/lib/libobjc.A.dylib')
    objc.objc_getClass.argtypes = [ctypes.c_char_p]
    objc.objc_getClass.restype = ctypes.c_void_p
    objc.sel_registerName.argtypes = [ctypes.c_char_p]
    objc.sel_registerName.restype = ctypes.c_void_p

    def call(receiver, selector, result=ctypes.c_void_p, types=(), args=()):
        function = ctypes.CFUNCTYPE(result, ctypes.c_void_p, ctypes.c_void_p, *types)(('objc_msgSend', objc))
        return function(receiver, objc.sel_registerName(selector.encode()), *args)

    def string(value):
        return call(objc.objc_getClass(b'NSString'), 'stringWithUTF8String:',
                    types=(ctypes.c_char_p,), args=(os.fsencode(value),))

    pool = call(call(objc.objc_getClass(b'NSAutoreleasePool'), 'alloc'), 'init')
    try:
        panel = call(objc.objc_getClass(b'NSOpenPanel'), 'openPanel')
        for selector, value in [('setCanChooseFiles:', False), ('setCanChooseDirectories:', True),
                                ('setAllowsMultipleSelection:', True), ('setResolvesAliases:', True)]:
            call(panel, selector, result=None, types=(ctypes.c_bool,), args=(value,))
        call(panel, 'setTitle:', result=None, types=(ctypes.c_void_p,), args=(string('Add recording folders'),))
        if directory and Path(directory).is_dir():
            url = call(objc.objc_getClass(b'NSURL'), 'fileURLWithPath:',
                       types=(ctypes.c_void_p,), args=(string(directory),))
            call(panel, 'setDirectoryURL:', result=None, types=(ctypes.c_void_p,), args=(url,))
        if call(panel, 'runModal', result=ctypes.c_long) != 1:
            return []
        urls = call(panel, 'URLs')
        count = call(urls, 'count', result=ctypes.c_ulong)
        paths = []
        for index in range(count):
            url = call(urls, 'objectAtIndex:', types=(ctypes.c_ulong,), args=(index,))
            path = call(call(url, 'path'), 'UTF8String', result=ctypes.c_char_p)
            paths.append(os.fsdecode(path))
        return paths
    finally:
        call(pool, 'drain', result=None)


def recording_folders(parent, directory=None):
    if sys.platform == 'darwin':
        return _mac_folders(directory)
    from qtpy.QtCore import QDir
    from qtpy.QtWidgets import QFileDialog, QListView, QTreeView, QAbstractItemView
    dialog = QFileDialog(parent, 'Add recording folders', str(directory or Path.home()))
    dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)
    dialog.setFileMode(QFileDialog.FileMode.Directory)
    dialog.setOption(QFileDialog.Option.ShowDirsOnly, True)
    dialog.setFilter(QDir.Filter.AllDirs | QDir.Filter.NoDotAndDotDot | QDir.Filter.Hidden)
    for view in dialog.findChildren(QListView) + dialog.findChildren(QTreeView):
        view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
    return dialog.selectedFiles() if dialog.exec() else []

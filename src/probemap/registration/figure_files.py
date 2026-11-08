"""Visible, atomic figure publication (including macOS hidden-file flags)."""
import os
import stat
from pathlib import Path
from uuid import uuid4


def visible_figure_path(path, suffix):
    path = Path(path).with_suffix(suffix)
    name = path.name.lstrip('.')
    return path.with_name(name if name and name != suffix.lstrip('.') else 'figure' + suffix)


def temporary_figure_path(path):
    path = Path(path)
    return path.with_name('probemap-export-' + uuid4().hex + path.suffix)


def publish_figure(temporary, path):
    temporary, path = Path(temporary), Path(path)
    # Finder can retain UF_HIDDEN when a dot-prefixed temporary file is renamed.
    if hasattr(os, 'chflags') and hasattr(stat, 'UF_HIDDEN'):
        flags = temporary.stat().st_flags
        if flags & stat.UF_HIDDEN:
            os.chflags(temporary, flags & ~stat.UF_HIDDEN)
    temporary.replace(path)

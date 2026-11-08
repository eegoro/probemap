"""Locks, mapped-file lifetimes and Windows API selection."""
import errno
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from probemap.platform_io import close_memmap,exclusive_file_lock


def test_lock_excludes_another_process_and_releases_after_error(tmp_path):
    path=tmp_path/'папка с пробелами.lock'
    env=dict(os.environ)
    env['PYTHONPATH']=str(Path(__file__).resolve().parents[1]/'src')+os.pathsep+env.get('PYTHONPATH','')
    code='''import sys
from probemap.platform_io import exclusive_file_lock
try:
    with exclusive_file_lock(sys.argv[1]): print('free')
except RuntimeError: print('locked')
'''
    def child():
        result=subprocess.run([sys.executable,'-c',code,str(path)],capture_output=True,text=True,env=env,timeout=20)
        assert result.returncode==0,result.stderr
        return result.stdout.strip()
    with pytest.raises(ValueError):
        with exclusive_file_lock(path):
            assert child()=='locked'
            raise ValueError('cancel')
    assert child()=='free'


def test_windows_backend_initializes_and_locks_byte_zero(tmp_path,monkeypatch):
    import probemap.platform_io as module
    calls=[]
    def locking(fd,mode,length):
        calls.append((mode,length,os.lseek(fd,0,os.SEEK_CUR)))
    monkeypatch.setattr(module,'_WINDOWS',True)
    monkeypatch.setitem(sys.modules,'msvcrt',SimpleNamespace(locking=locking,LK_NBLCK=1,LK_UNLCK=2))
    path=tmp_path/'recording.lock'
    with exclusive_file_lock(path):
        assert path.read_bytes()==b'\0'
    assert calls==[(1,1,0),(2,1,0)]


def test_windows_lock_conflict_is_not_mistaken_for_permission_to_write(tmp_path,monkeypatch):
    import probemap.platform_io as module
    def locking(*args): raise OSError(errno.EACCES,'busy')
    monkeypatch.setattr(module,'_WINDOWS',True)
    monkeypatch.setitem(sys.modules,'msvcrt',SimpleNamespace(locking=locking,LK_NBLCK=1,LK_UNLCK=2))
    with pytest.raises(RuntimeError,match='already being prepared'):
        with exclusive_file_lock(tmp_path/'recording.lock'): pytest.fail('Lock was not acquired')


def test_mapped_output_can_be_removed_even_when_a_view_is_still_referenced(tmp_path):
    path=tmp_path/'partial.npy'
    mapped=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(4,4))
    mapped[:]=1; mapped.flush()
    view=mapped[:,0]
    close_memmap(mapped)
    assert view._mmap.closed
    close_memmap(view)  # Closing an already released owner is harmless.
    target=path.with_name('complete.npy'); path.rename(target); target.unlink()
    assert not target.exists()


def test_windows_startup_does_not_touch_macos_plugin_flags(monkeypatch):
    from probemap.registration import macos_qt
    monkeypatch.setattr(macos_qt,'sys',SimpleNamespace(platform='win32'))
    monkeypatch.setenv('QT_API','pyqt6')
    assert macos_qt.prepare_qt() is None


def test_portable_names_avoid_windows_devices():
    from probemap.registration.ephys_cache import safe_name
    for name in ('CON','nul','AUX.txt','COM1','LPT9'):
        assert safe_name(name).split('.')[0].upper() not in {'CON','NUL','AUX','COM1','LPT9'}


def test_entry_point_selects_qt_before_importing_desktop(monkeypatch):
    from probemap.__main__ import main
    monkeypatch.delenv('QT_API',raising=False)
    def desktop(args):
        assert os.environ['QT_API']=='pyqt6'
        assert args==['--version']
        return 0
    monkeypatch.setitem(sys.modules,'probemap.registration.desktop_app',SimpleNamespace(main=desktop))
    assert main(['--version'])==0

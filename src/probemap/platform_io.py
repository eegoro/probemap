"""Cross-platform file locking and explicit mapped-file lifetime management."""
from contextlib import contextmanager
import errno
import os
from pathlib import Path


_WINDOWS = os.name == 'nt'


@contextmanager
def exclusive_file_lock(path, message='This recording is already being prepared in another ProbeMap window.'):
    """Use OS-managed nonblocking locks, released even if a process crashes."""
    with Path(path).open('a+b') as stream:
        acquired=False
        try:
            if _WINDOWS:
                import msvcrt
                # Windows locks a byte range, not the whole file. Every writer
                # uses byte zero, including when the lock file starts empty.
                stream.seek(0,os.SEEK_END)
                if stream.tell()==0: stream.write(b'\0'); stream.flush()
                stream.seek(0)
                try: msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
                except OSError as error:
                    if error.errno in (errno.EACCES,errno.EAGAIN,errno.EDEADLK) or getattr(error,'winerror',None) in (32,33):
                        raise RuntimeError(message) from None
                    raise
            else:
                import fcntl
                try: fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError: raise RuntimeError(message) from None
            acquired=True
            yield
        finally:
            if acquired:
                if _WINDOWS:
                    import msvcrt
                    stream.seek(0); msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
                else:
                    import fcntl
                    fcntl.flock(stream,fcntl.LOCK_UN)


def close_memmap(array):
    """Close an owned mapping before rename/delete; do not use its views after."""
    seen=set()
    while array is not None and id(array) not in seen:
        seen.add(id(array))
        mapping=getattr(array,'_mmap',None)
        if mapping is not None:
            if not mapping.closed: mapping.close()
            return
        array=getattr(array,'base',None)

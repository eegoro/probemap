"""A small local list of recently opened project files."""
from pathlib import Path

from .project_paths import project_file_path


class RecentProjects:
    """Use application preferences when supplied, otherwise keep state in memory."""
    KEY='recentProjects'
    LIMIT=5

    def __init__(self,settings=None):
        self.settings=settings
        values=settings.value(self.KEY,[]) if settings is not None else []
        self._paths=values if isinstance(values,list) else [values] if isinstance(values,str) else []

    def entries(self):
        paths=[]
        for value in self._paths:
            path=self._available(value)
            if path is not None and path not in paths: paths.append(path)
            if len(paths)==self.LIMIT: break
        return paths

    @staticmethod
    def _available(value):
        if not isinstance(value,(str,Path)) or not str(value).strip(): return None
        try:
            path=project_file_path(value)
            return path if path.is_file() else None
        except (OSError,ValueError,RuntimeError):
            return None

    def _store(self,paths):
        values=[str(path) for path in paths]
        if values==self._paths: return
        self._paths=values
        if self.settings is not None:
            self.settings.setValue(self.KEY,values)
            self.settings.sync()

    def remember(self,path):
        path=self._available(path)
        if path is not None:
            self._store([path,*[entry for entry in self.entries() if entry!=path]][:self.LIMIT])

    def forget(self,path):
        path=project_file_path(path)
        self._store([entry for entry in self.entries() if entry!=path])

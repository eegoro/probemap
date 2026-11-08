"""Recent project preferences without touching project contents."""
from probemap.recent_projects import RecentProjects


class Settings:
    def __init__(self,values=None):
        self.values={} if values is None else values
        self.writes=0

    def value(self,key,default): return self.values.get(key,default)
    def setValue(self,key,value):
        self.values[key]=value
        self.writes+=1
    def sync(self): pass


def saved_project(folder):
    folder.mkdir(parents=True)
    path=folder/'project.json'
    path.write_text('project contents are not a preferences store')
    return path


def test_recent_order_deduplication_limit_and_persistence(tmp_path):
    settings=Settings()
    recent=RecentProjects(settings)
    paths=[saved_project(tmp_path/f'Animal {i}'/'outputs') for i in range(7)]
    originals=[path.read_bytes() for path in paths]
    for path in paths: recent.remember(path)
    assert recent.entries()==paths[::-1][:5]
    recent.remember(paths[3].parent)
    assert recent.entries()==[paths[3],paths[6],paths[5],paths[4],paths[2]]
    writes=settings.writes
    recent.remember(paths[3])
    assert settings.writes==writes
    assert RecentProjects(settings).entries()==recent.entries()
    assert [path.read_bytes() for path in paths]==originals


def test_missing_moved_and_aliased_projects(tmp_path):
    a=saved_project(tmp_path/'A')
    b=saved_project(tmp_path/'B')
    alias=tmp_path/'alias'
    try: alias.symlink_to(a.parent,target_is_directory=True)
    except OSError: alias=a.parent  # Windows can prohibit symlinks without Developer Mode.
    settings=Settings({'recentProjects':[str(a),str(alias),str(tmp_path/'missing'),{},str(b)]})
    recent=RecentProjects(settings)
    assert recent.entries()==[a,b]
    a.parent.rename(tmp_path/'Moved A')
    assert recent.entries()==[b]
    recent.remember(tmp_path/'Moved A')
    moved=tmp_path/'Moved A'/'project.json'
    assert recent.entries()==[moved,b]
    recent.forget(b)
    assert RecentProjects(settings).entries()==[moved]


def test_empty_invalid_or_legacy_single_path_preferences(tmp_path):
    for value in (None,123,{},['',None,{},'/nonexistent/ProbeMap/project.json']):
        assert RecentProjects(Settings({'recentProjects':value})).entries()==[]
    path=saved_project(tmp_path/'Legacy')
    assert RecentProjects(Settings({'recentProjects':str(path)})).entries()==[path]
    recent=RecentProjects()
    recent.remember(tmp_path/'not saved')
    assert recent.entries()==[]

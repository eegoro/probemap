"""Resolve saved image links when a project and its data are moved together."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def _base(project_file):
    parent=project_file_path(project_file).parent
    return parent.parent if parent.name=='.probemap_history' else parent


def project_file_path(path):
    """Accept either the results folder or a project JSON file."""
    path=Path(path).expanduser().resolve()
    return path/'project.json' if path.is_dir() else path


def _paths(record):
    for key in ('reference','moving','track'):
        if record.get(key): yield record[key],'path'
    for key in ('labels_dir','output_dir'):
        if record.get(key): yield record,key
    for export in record.get('exports',[]):
        if export.get('path'): yield export,'path'
    for recording in record.get('ephys',{}).get('recordings',[]):
        for key in ('folder','meta_path','bin_path','cache_path','cache_root'):
            if recording.get(key): yield recording,key
        for sorting in recording.get('sortings',[]):
            for key in ('path','cache_path'):
                if sorting.get(key): yield sorting,key
    for session in record.get('track_analysis',{}).get('sessions',[]):
        for spec in session.get('sources',{}).values():
            if isinstance(spec,dict) and spec.get('path'): yield spec,'path'


def relative_path_links(record,project_file):
    links={}
    for owner,key in _paths(record):
        try: links[owner[key]]=Path(os.path.relpath(owner[key],_base(project_file))).as_posix()
        except ValueError: pass  # Different Windows drives cannot share a relative path.
    return links


def relocate_project_paths(record,project_file):
    """Repair path strings only; preserve transforms, annotations and geometry."""
    base=_base(project_file)
    links=record.get('relative_paths',{})

    def resolve(value,allow_missing=False):
        path=Path(str(value).replace('\\','/')).expanduser()
        relative=base/Path(str(links[value]).replace('\\','/')) if value in links else (base/path if not path.is_absolute() else None)
        if relative is not None and relative.exists(): return str(relative.resolve())
        # Legacy projects used absolute paths. Match their shared data-tree
        # ancestor against the opened project's actual location, without a
        # workstation-specific fallback or a recursive filesystem search.
        for ancestor in (base,*base.parents):
            for index,part in enumerate(path.parts[:-1]):
                if part.casefold()==ancestor.name.casefold():
                    candidate=ancestor.joinpath(*path.parts[index+1:])
                    if candidate.exists(): return str(candidate.resolve())
        if path.is_absolute() and path.exists(): return str(path.resolve())
        if allow_missing and relative is not None: return str(relative.resolve())
        return str((path if path.is_absolute() else base/path).resolve())

    for owner,key in _paths(record):
        owner[key]=resolve(owner[key],key=='output_dir')
    # The opened file owns its results directory. A stale saved output_dir
    # must never send subsequent saves back to a previous study location.
    record['output_dir']=str(base)
    # Session identity includes image paths. A relocation must retain the
    # existing annotations rather than create an empty analysis session.
    for session in record.get('track_analysis',{}).get('sessions',[]):
        sources=session.get('sources',{})
        if 'signature' in session and all(key in sources for key in ('track','reference','matrix')):
            signature={key:sources[key] for key in ('track','reference','matrix')}
            session['signature']=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
    return record

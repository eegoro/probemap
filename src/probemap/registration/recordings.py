"""Session recording library; assignment preserves embedded channel maps."""
from pathlib import Path

from .insertion_model import alphabetic_insertion_name
from .neuropixels_model import add_recording


def recording_entries(session):
    entries=[(None,r) for r in session.get('unassigned_recordings',[])]
    entries.extend((item,r) for item in session['insertions'] for r in item.get('recordings',[]))
    order=session.setdefault('recording_order',[])
    known=set(order)
    for _,record in entries:
        if record['id'] not in known:
            order.append(record['id']); known.add(record['id'])
    ranks={identifier:i for i,identifier in enumerate(order)}
    return sorted(entries,key=lambda entry:ranks[entry[1]['id']])


def import_recordings(session,paths):
    """Import files or folders in stable order, ignoring unrelated folder files."""
    candidates={}
    for value in paths:
        path=Path(value).expanduser()
        if path.is_dir():
            for child in sorted(path.rglob('*')):
                if child.is_file() and child.suffix.lower() in ('.meta','.imro','.txt'):
                    candidates.setdefault(child.resolve(),False)
        else: candidates[path.resolve()]=True
    entries=recording_entries(session)
    existing={str(Path(rec['path']).resolve()) for _,rec in entries}
    names={rec['name'].casefold() for _,rec in entries}
    added=[]; errors=[]
    for path,explicit in candidates.items():
        if str(path) in existing: continue
        index=0
        while alphabetic_insertion_name(index).lower() in names: index+=1
        name=alphabetic_insertion_name(index).lower()
        holder={}
        try: rec=add_recording(holder,path,name)
        except (OSError,UnicodeError,ValueError) as error:
            if explicit or path.suffix.lower()!='.txt': errors.append(f'{path.name}: {error}')
            continue
        rec['opacity']=1/(1+.3*len(entries))
        rec['color']=None  # Inherit the assigned insertion's current colour.
        session.setdefault('unassigned_recordings',[]).append(rec)
        session['recording_order'].append(rec['id'])
        entries.append((None,rec)); names.add(name); existing.add(str(path)); added.append(rec)
    return added,errors


def assign_recording(session,identifier,insertion_id):
    owner,record=next((item,r) for item,r in recording_entries(session) if r['id']==identifier)
    target=next(item for item in session['insertions'] if item['id']==insertion_id) if insertion_id is not None else None
    if target is not None and target.get('probe',{}).get('geometry',{}).get('model','NP2013') not in ('NP2013','NP2014','Legacy NP2'):
        raise ValueError('This is an NP2 four-shank recording map. Assign it to a matching NP2 probe.')
    if target is owner: return record
    remove_recordings(session,[identifier],keep_order=True)
    (target.setdefault('recordings',[]) if target is not None else session.setdefault('unassigned_recordings',[])).append(record)
    return record


def remove_recordings(session,identifiers,*,keep_order=False):
    identifiers=set(identifiers)
    session['unassigned_recordings']=[r for r in session.get('unassigned_recordings',[]) if r['id'] not in identifiers]
    for item in session['insertions']:
        item['recordings']=[r for r in item.get('recordings',[]) if r['id'] not in identifiers]
    if not keep_order: session['recording_order']=[i for i in session.get('recording_order',[]) if i not in identifiers]

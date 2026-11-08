"""Portable probe templates; library files never own a project's geometry.

Local axes: X across shanks, Y from physical tip to base, Z normal to face.
Contact X is relative to its shank centre; all lengths are micrometres.
"""
from __future__ import annotations
import copy
import csv
import io
import json
import math
import os
from pathlib import Path
from uuid import uuid4

SCHEMA_VERSION = 1
MAX_CONTACTS = 20000
NP_SOURCE = 'https://github.com/SpikeInterface/probeinterface/blob/main/src/probeinterface/resources/neuropixels_probe_features.json'
NN_SOURCE = 'https://www.neuronexus.com/files/catalog/NNxPenetratingProbes.pdf'


def library_path():
    override = os.environ.get('PROBEMAP_PROBES_LIBRARY')
    if override: return Path(override).expanduser()
    import sys
    if sys.platform == 'darwin': return Path.home()/'Library'/'Application Support'/'ProbeMap'/'probes_library'
    if sys.platform == 'win32': return Path(os.environ.get('APPDATA', str(Path.home())))/'ProbeMap'/'probes_library'
    return Path(os.environ.get('XDG_DATA_HOME', str(Path.home()/'.local/share')))/'probemap'/'probes_library'


def grid_geometry(name='My probe', *, shanks=1, rows=16, columns=1, layers=1,
                  row_pitch=50., column_pitch=20., layer_pitch=20., shank_pitch=250.,
                  first_site=100., stagger=0., length=5000., width=70., thickness=24.,
                  taper=50., site_width=12., site_height=None, shape='square', body='tapered'):
    counts = [shanks, rows, columns, layers]
    if any(int(n) != n or n < 1 for n in counts): raise ValueError('Counts must be positive whole numbers.')
    if math.prod(counts) > MAX_CONTACTS: raise ValueError(f'Use at most {MAX_CONTACTS:,} contacts.')
    g = dict(model=name.strip(), shanks=int(shanks), sites_per_shank=int(rows*columns*layers),
             shank_pitch_um=float(shank_pitch), shank_length_um=float(length), shank_width_um=float(width),
             shank_thickness_um=float(thickness), taper_length_um=float(taper),
             first_site_center_from_tip_um=float(first_site), site_pitch_um=float(row_pitch),
             column_pitch_um=float(column_pitch), left_site_center_um=(width-(columns-1)*column_pitch)/2,
             site_size_um=float(site_width), site_height_um=float(site_height if site_height is not None else site_width),
             contact_shape=shape, body_shape=body, contacts=[],
             generator=dict(shanks=int(shanks),rows=int(rows),columns=int(columns),layers=int(layers),
                row_pitch=row_pitch,column_pitch=column_pitch,layer_pitch=layer_pitch,shank_pitch=shank_pitch,
                first_site=first_site,stagger=stagger,length=length,width=width,thickness=thickness,
                taper=taper,site_width=site_width,site_height=site_height,shape=shape,body=body))
    for s in range(int(shanks)):
        for z in range(int(layers)):
            for r in range(int(rows)):
                for c in range(int(columns)):
                    electrode = (z*int(rows)+r)*int(columns)+c
                    g['contacts'].append(dict(id=f's{s}e{electrode}',shank=s,electrode=electrode,
                        x_um=(c-(columns-1)/2)*column_pitch+(stagger if r%2 else 0),
                        y_um=first_site+r*row_pitch,z_um=(z-(layers-1)/2)*layer_pitch if layers>1 else thickness/2))
    return g


def contacts_for(g):
    if 'contacts' in g: return g['contacts']
    # Exact legacy NP2 convention; materialise only at the boundary.
    return [dict(id=f's{s}e{e}',shank=s,electrode=e,
                 x_um=g['left_site_center_um']+(e%2)*g['column_pitch_um']-g['shank_width_um']/2,
                 y_um=g['first_site_center_from_tip_um']+(e//2)*g['site_pitch_um'],z_um=g['shank_thickness_um']/2)
            for s in range(g['shanks']) for e in range(g['sites_per_shank'])]


def validate_geometry(g):
    if not isinstance(g,dict) or not isinstance(g.get('model'),str) or not g['model'].strip():
        raise ValueError('Give the probe a name.')
    n=g.get('shanks')
    if isinstance(n,bool) or not isinstance(n,int) or not 1 <= n <= 64: raise ValueError('Use 1–64 shanks.')
    for key in ('shank_length_um','shank_width_um','shank_thickness_um','site_pitch_um','site_size_um','site_height_um'):
        value=g.get(key,g.get('site_size_um') if key=='site_height_um' else None)
        if not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:
            raise ValueError(f'{key.replace("_um", "").replace("_", " ").capitalize()} must be positive and finite.')
    pitch=g.get('shank_pitch_um',0)
    if not isinstance(pitch,(int,float)) or not math.isfinite(pitch) or pitch<0 or (n>1 and pitch<g['shank_width_um']):
        raise ValueError('Shank spacing must be at least the shank width.')
    taper=g.get('taper_length_um',0)
    if not isinstance(taper,(int,float)) or not math.isfinite(taper) or not 0<=taper<g['shank_length_um']:
        raise ValueError('Tip length must be shorter than the shank.')
    if g.get('contact_shape','square') not in ('square','rectangle','circle','ellipse'): raise ValueError('Unknown contact shape.')
    if g.get('body_shape','tapered') not in ('tapered','rectangle','wire','cylinder','polygon','prism','none'): raise ValueError('Unknown body shape.')
    contacts=contacts_for(g)
    if not 1<=len(contacts)<=MAX_CONTACTS: raise ValueError(f'Use 1–{MAX_CONTACTS:,} contacts.')
    ids=set(); electrodes=set(); positions=set()
    for c in contacts:
        if not isinstance(c,dict): raise ValueError('Each contact needs an ID, shank and coordinates.')
        identifier=c.get('id')
        if not isinstance(identifier,str) or not identifier or identifier in ids: raise ValueError('Contact IDs must be unique and non-empty.')
        ids.add(identifier)
        s=c.get('shank'); e=c.get('electrode')
        if not isinstance(s,int) or not 0<=s<n or not isinstance(e,int) or e<0: raise ValueError('Invalid shank or electrode ID.')
        if (s,e) in electrodes: raise ValueError('Electrode IDs must be unique within each shank.')
        electrodes.add((s,e))
        xyz=[c.get(k) for k in ('x_um','y_um','z_um')]
        if any(not isinstance(v,(int,float)) or not math.isfinite(v) for v in xyz): raise ValueError('Contact coordinates must be finite numbers.')
        if g.get('body_shape')!='none' and not 0<=xyz[1]<=g['shank_length_um']: raise ValueError(f'Contact {identifier} is beyond the tip or base.')
        if g.get('geometry_version',1)<2 and abs(xyz[0])+g['site_size_um']/2>g['shank_width_um']/2+1e-6:
            raise ValueError(f'Contact {identifier} is outside the shank width. Increase the width or move the contact.')
        if g.get('geometry_version',1)<2 and abs(xyz[2])>g['shank_thickness_um']/2+1e-6:
            raise ValueError(f'Contact {identifier} is outside the body thickness. Increase thickness or reduce Z spacing.')
        position=(s,*[round(v,6) for v in xyz],c.get('surface','patch'),c.get('angle_deg',0),c.get('span_deg',360))
        if position in positions: raise ValueError('Two contacts occupy the same position.')
        positions.add(position)
    if g.get('geometry_version',1)>=2:
        from .probe_geometry import validate_physical
        validate_physical(g)
    return g


def builtin_probes():
    from .registration.neuropixels_model import geometry_for
    result=[]
    def add(identifier,g,manufacturer,note,source):
        result.append(dict(schema_version=SCHEMA_VERSION,id=identifier,name=('Neuropixels 2.0 · '+g['model'] if g['model'] in ('NP2013','NP2014') else g['model']),manufacturer=manufacturer,
            builtin=True,notes=note,sources=[source],geometry=g))
    for model in ('NP2013','NP2014','Legacy NP2'):
        g=geometry_for(model)
        add(model,g,'imec','Neuropixels 2.0 · four shanks. Tip-to-first-centre offset follows the ProbeInterface convention; verify the physical revision.',NP_SOURCE)
    g=grid_geometry('Neuropixels 1.0 · NP1000',rows=480,columns=2,row_pitch=20,column_pitch=32,
                    first_site=220,length=10000,width=70,thickness=24,taper=209)
    for c in g['contacts']:
        r=c['electrode']//2; col=c['electrode']%2
        c['x_um']=(27 if r%2==0 else 11)+32*col-35
    g.pop('generator',None)
    add('NP1000',g,'imec','960 physical sites, staggered rows. Recording channel selection is supplied separately. First centre = 209 µm tip + 11 µm.',NP_SOURCE)
    nhp=copy.deepcopy(g); nhp['model']='Neuropixels 1.0 · NHP short NP1010'; nhp['shank_thickness_um']=132.
    for c in nhp['contacts']: c['z_um']=66.
    add('NP1010',nhp,'imec','NHP short staggered probe. 10 mm shank, 132 µm substrate. Recording selection is separate from physical geometry.',NP_SOURCE)
    g=grid_geometry('Neuropixels 2.0 · single shank NP2000',rows=640,columns=2,row_pitch=15,column_pitch=32,
                    first_site=217,length=10000,width=70,thickness=24,taper=206)
    for c in g['contacts']: c['x_um']+=8
    g.pop('generator',None)
    add('NP2000',g,'imec','1280 physical sites. Verify part number and tip revision against the physical probe.',NP_SOURCE)
    # Catalog site coordinates. Body represented as a simplified constant-width
    # envelope, not the manufacturer's continuously tapered outline.
    for count,pitch,length,width in ((16,50,5000,100),(16,100,10000,100),(32,25,5000,145),(32,50,5000,145)):
        name=f'A1x{count}-{int(length/1000)}mm-{pitch}-177'
        g=grid_geometry(name,rows=count,row_pitch=pitch,first_site=50,length=length,width=width,
                        thickness=15,taper=35,site_width=15,shape='circle')
        add('neuronexus-'+name,g,'NeuroNexus','A-series linear silicon probe. 15 µm substrate variant; 177 µm² sites (≈15 µm diameter). Simplified body envelope. Tip-to-first-centre = 50 µm is a template assumption; verify it and the outline against your probe drawing before anatomical use.',NN_SOURCE)
    g=grid_geometry('Tetrode · editable example',rows=1,columns=2,layers=2,column_pitch=12.5,layer_pitch=12.5,
        first_site=12.5,length=5000,width=40,thickness=12.5,taper=0,site_width=10,shape='circle',body='wire')
    add('example-tetrode',g,'Generic','Starting geometry only. Wire spacing, exposed contact size and tip offsets depend on fabrication; enter your measured values. Not a manufacturer specification.','')
    g=grid_geometry('Linear depth electrode · editable example',rows=16,row_pitch=150,first_site=150,
        length=15000,width=150,thickness=150,taper=100,site_width=40,shape='circle',body='wire')
    add('example-depth',g,'Generic','Starting geometry for custom penetrating depth electrodes. Replace all dimensions with your physical electrode measurements.','')
    return result


class ProbeLibrary:
    def __init__(self,path=None):
        self.path=Path(path) if path is not None else library_path()
        self.errors=[]

    def entries(self):
        entries=builtin_probes(); self.errors=[]
        for path in sorted(self.path.glob('*.json')):
            try:
                item=self.read(path); item['builtin']=False; entries.append(item)
            except (OSError,ValueError,KeyError,TypeError) as error: self.errors.append(f'{path.name}: {error}')
        return entries

    @staticmethod
    def read(path):
        item=json.loads(Path(path).read_text(encoding='utf-8-sig'))
        if not isinstance(item,dict) or item.get('schema_version')!=SCHEMA_VERSION: raise ValueError('Unsupported ProbeMap probe file.')
        validate_geometry(item.get('geometry'))
        if not isinstance(item.get('id'),str) or not item['id'].startswith('custom-') or not item['id'][7:].isalnum():
            raise ValueError('Invalid custom probe ID.')
        item['name']=item['geometry']['model']; item['builtin']=False
        return item

    def save(self,item):
        item=copy.deepcopy(item); validate_geometry(item['geometry'])
        item.update(schema_version=SCHEMA_VERSION,builtin=False,name=item['geometry']['model'])
        if not str(item.get('id','')).startswith('custom-'): item['id']='custom-'+uuid4().hex
        if not item['id'][7:].isalnum(): raise ValueError('Invalid custom probe ID.')
        if any(e['geometry']['model'].casefold()==item['geometry']['model'].casefold() and e['id']!=item['id'] for e in self.entries()):
            raise ValueError('A probe with this name already exists. Choose another name.')
        self.path.mkdir(parents=True,exist_ok=True)
        self.export(item,self.path/(item['id']+'.json'))
        return item

    @staticmethod
    def export(item,path):
        validate_geometry(item['geometry'])
        path=Path(path); temp=path.with_name(path.name+'.'+uuid4().hex+'.tmp')
        try:
            temp.write_text(json.dumps(item,indent=2,allow_nan=False)+'\n',encoding='utf-8')
            temp.replace(path)
        finally: temp.unlink(missing_ok=True)

    def delete(self,item):
        if item.get('builtin'): raise ValueError('Built-in probes cannot be removed.')
        identifier=item['id']
        if not identifier.startswith('custom-') or not identifier[7:].isalnum(): raise ValueError('Invalid custom probe ID.')
        (self.path/(identifier+'.json')).unlink(missing_ok=True)


def contacts_csv(g):
    stream=io.StringIO(); writer=csv.DictWriter(stream,fieldnames=['id','shank','electrode','x_um','y_um','z_um'])
    writer.writeheader(); writer.writerows({k:c[k] for k in writer.fieldnames} for c in contacts_for(g))
    return stream.getvalue()


def parse_contacts_csv(text):
    if not text.strip(): raise ValueError('The coordinate file is empty.')
    reader=csv.DictReader(io.StringIO(text.lstrip('\ufeff')),delimiter='\t' if '\t' in text.splitlines()[0] else ',')
    if not {'x_um','y_um'}<=set(reader.fieldnames or []): raise ValueError('CSV needs x_um and y_um columns; optional: id, shank, electrode, z_um.')
    contacts=[]; next_ids={}
    for row in reader:
        s=int(row.get('shank') or 0); e=int(row.get('electrode') or next_ids.get(s,0)); next_ids[s]=e+1
        if any(not math.isfinite(float(row.get(k) or 0)) for k in ('x_um','y_um','z_um')): raise ValueError('Contact coordinates must be finite.')
        contacts.append(dict(id=row.get('id') or f's{s}e{e}',shank=s,electrode=e,
            x_um=float(row['x_um']),y_um=float(row['y_um']),z_um=float(row.get('z_um') or 0)))
        if len(contacts)>MAX_CONTACTS: raise ValueError('Too many contacts.')
    return contacts

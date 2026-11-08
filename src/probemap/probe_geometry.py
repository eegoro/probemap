"""Physical bodies and individually oriented recording surfaces, in local XYZ µm.

Y runs along an insertion from the chosen distal reference towards the base.
Bodies and contacts are independent; a contact may be terminal, planar or curved.
"""
from __future__ import annotations
import math
import numpy as np


def body_polygon(g):
    if g.get('body_shape') in ('polygon','prism'):
        return np.asarray(g['body_polygon_um'],float)
    w=g['shank_width_um']/2; length=g['shank_length_um']
    if g.get('body_shape','tapered')=='tapered':
        return np.array([[0,0],[-w,g['taper_length_um']],[-w,length],[w,length],[w,g['taper_length_um']]])
    return np.array([[-w,0],[-w,length],[w,length],[w,0]])


def polygon_area(poly):
    p=np.asarray(poly,float);return float(np.sum(p[:,0]*np.roll(p[:,1],-1)-p[:,1]*np.roll(p[:,0],-1))/2)


def polygon_valid(poly):
    p=np.asarray(poly,float)
    if p.ndim!=2 or p.shape[1]!=2 or not 3<=len(p)<=256 or not np.isfinite(p).all(): return False
    if abs(polygon_area(p))<1e-8 or np.any(np.linalg.norm(p-np.roll(p,1,axis=0),axis=1)<1e-8): return False
    def orient(a,b,c):
        u,v=b-a,c-a;return float(u[0]*v[1]-u[1]*v[0])
    def on(a,b,c): return np.all(c>=np.minimum(a,b)-1e-8) and np.all(c<=np.maximum(a,b)+1e-8)
    for i in range(len(p)):
        a,b=p[i],p[(i+1)%len(p)]
        for j in range(i+1,len(p)):
            if j==i+1 or (i==0 and j==len(p)-1):continue
            c,d=p[j],p[(j+1)%len(p)];o=[orient(a,b,c),orient(a,b,d),orient(c,d,a),orient(c,d,b)]
            if o[0]*o[1]<0 and o[2]*o[3]<0:return False
            if any(abs(v)<1e-8 and on(x,y,z) for v,x,y,z in [(o[0],a,b,c),(o[1],a,b,d),(o[2],c,d,a),(o[3],c,d,b)]):return False
    return True


def point_in_polygon(point,poly):
    p=np.asarray(poly,float);q=np.asarray(point,float);inside=False
    for a,b in zip(p,np.roll(p,-1,axis=0)):
        d=b-a;t=np.clip(np.dot(q-a,d)/max(np.dot(d,d),1e-20),0,1)
        if np.linalg.norm(q-a-t*d)<1e-6:return True
        if (a[1]>q[1])!=(b[1]>q[1]) and q[0]<(b[0]-a[0])*(q[1]-a[1])/(b[1]-a[1])+a[0]:inside=not inside
    return inside


def triangulate(poly):
    """Ear clipping, including concave simple contours; no fan across empty space."""
    p=np.asarray(poly,float);remaining=list(range(len(p))); result=[];sign=1 if polygon_area(p)>0 else -1
    def cross(a,b,c):
        u,v=b-a,c-a;return float(u[0]*v[1]-u[1]*v[0])*sign
    for _ in range(len(p)**2):
        if len(remaining)<=3:break
        found=False
        for i,b in enumerate(remaining):
            a,c=remaining[i-1],remaining[(i+1)%len(remaining)]
            if cross(p[a],p[b],p[c])<=1e-10:continue
            if any(all(cross(x,y,p[k])>=-1e-10 for x,y in [(p[a],p[b]),(p[b],p[c]),(p[c],p[a])]) for k in remaining if k not in (a,b,c)):continue
            result.append([a,b,c]);remaining.pop(i);found=True;break
        if not found:raise ValueError('The contour cannot be triangulated. Check crossing or repeated vertices.')
    if len(remaining)==3:result.append(remaining)
    return np.asarray(result,int).reshape(-1,3)


def body_mesh(g,sections=None):
    """Body-only mesh. Optional sections clip each shaft along native Y."""
    kind=g.get('body_shape','tapered'); vertices=[];faces=[]
    if kind=='none':return np.empty((0,3)),np.empty((0,3),int)
    if sections is None:sections=[(s,0.,g['shank_length_um']) for s in range(g['shanks'])]
    for s,lo,hi in sections:
        lo,hi=np.clip([lo,hi],0,g['shank_length_um'])
        if hi<=lo:continue
        shift=s*g['shank_pitch_um'];offset=len(vertices)
        if kind in ('wire','cylinder'):
            angles=np.linspace(0,2*np.pi,32,endpoint=False);n=len(angles)
            rings=[lo,hi]
            if kind=='wire' and lo<g.get('taper_length_um',0)<hi:rings.insert(1,g['taper_length_um'])
            for y in rings:
                factor=min(1.,max(.00001,y/max(g.get('taper_length_um',0),1e-8))) if kind=='wire' and g.get('taper_length_um',0)>0 else 1.
                vertices.extend([[shift+factor*g['shank_width_um']/2*np.cos(a),y,factor*g['shank_thickness_um']/2*np.sin(a)] for a in angles])
            for r in range(len(rings)-1):
                for i in range(n):
                    j=(i+1)%n;a=offset+r*n+i;b=offset+r*n+j;c=offset+(r+1)*n+j;d=offset+(r+1)*n+i
                    faces.extend([[a,b,c],[a,c,d]])
            for i in range(1,n-1):faces.extend([[offset,offset+i+1,offset+i],[offset+(len(rings)-1)*n,offset+(len(rings)-1)*n+i,offset+(len(rings)-1)*n+i+1]])
            continue
        poly=body_polygon(g)
        if kind!='prism' and (lo>poly[:,1].min() or hi<poly[:,1].max()):
            d=np.roll(poly,-1,axis=0)-poly;e=np.roll(d,-1,axis=0);turns=d[:,0]*e[:,1]-d[:,1]*e[:,0]
            if np.any(turns>1e-8) and np.any(turns<-1e-8):
                for triangle in triangulate(poly):
                    piece=dict(g,body_shape='polygon',body_polygon_um=poly[triangle].tolist())
                    points,indices=body_mesh(piece,[(s,lo,hi)]);faces.extend((indices+len(vertices)).tolist());vertices.extend(points)
                continue
        if kind!='prism':
            # Sutherland–Hodgman clipping against the two Y planes.
            for boundary,lower in [(lo,True),(hi,False)]:
                clipped=[]
                for a,b in zip(poly,np.roll(poly,-1,axis=0)):
                    ia=a[1]>=boundary if lower else a[1]<=boundary;ib=b[1]>=boundary if lower else b[1]<=boundary
                    if ia:clipped.append(a)
                    if ia!=ib:clipped.append(a+(boundary-a[1])/(b[1]-a[1])*(b-a))
                poly=np.asarray(clipped)
                if len(poly)<3:break
            if len(poly)<3:continue
            poly=np.asarray([point for i,point in enumerate(poly) if i==0 or not np.allclose(point,poly[i-1])])
            if len(poly)>1 and np.allclose(poly[0],poly[-1]):poly=poly[:-1]
            if len(poly)<3 or abs(polygon_area(poly))<1e-8:continue
        n=len(poly)
        if kind=='prism':vertices.extend([[x+shift,y,z] for y in (lo,hi) for x,z in poly])
        else:vertices.extend([[x+shift,y,z] for z in (-g['shank_thickness_um']/2,g['shank_thickness_um']/2) for x,y in poly])
        triangles=triangulate(poly)
        faces.extend((triangles+offset).tolist());faces.extend((triangles[:,::-1]+offset+n).tolist())
        for i in range(n):
            j=(i+1)%n;faces.extend([[offset+i,offset+j,offset+n+j],[offset+i,offset+n+j,offset+n+i]])
    return np.asarray(vertices,float).reshape(-1,3),np.asarray(faces,int).reshape(-1,3)


def contact_center(c,g):
    return np.array([c['x_um']+c['shank']*g['shank_pitch_um'],c['y_um'],c['z_um']],float)


def contact_mesh_local(c,g):
    center=contact_center(c,g);surface=c.get('surface','patch')
    width=c.get('width_um',g['site_size_um']);height=c.get('height_um',g.get('site_height_um',width))
    if surface=='cylindrical_patch':
        radius=g['shank_width_um']/2;angle=np.arctan2(c['z_um'],c['x_um'])
        shape=c.get('shape',g.get('contact_shape','circle'))
        if shape in ('circle','ellipse'):
            theta=np.linspace(0,2*np.pi,32,endpoint=False);xy=np.column_stack([np.cos(theta),np.sin(theta)])
        else:xy=np.array([[-1,-1],[1,-1],[1,1],[-1,1]])
        theta=angle+xy[:,0]*width/(2*radius)
        vertices=np.column_stack([c['shank']*g['shank_pitch_um']+radius*np.cos(theta),center[1]+xy[:,1]*height/2,radius*np.sin(theta)])
        return vertices,np.array([[0,i,i+1] for i in range(1,len(vertices)-1)],int)
    if surface in ('ring','sector','exposed_tip'):
        radius=c.get('radius_um',g['shank_width_um']/2)
        start=c.get('angle_deg',0);span=360 if surface in ('ring','exposed_tip') else c.get('span_deg',90)
        angles=np.deg2rad(np.linspace(start,start+span,max(4,int(abs(span)/10)+1)))
        ys=[center[1]-height/2,center[1]+height/2]
        if surface=='exposed_tip':
            ys=[center[1],center[1]+height]
            taper=g.get('taper_length_um',0)
            if g.get('body_shape')=='wire' and ys[0]<taper<ys[-1]:ys.insert(1,taper)
        def exposed_radius(y):
            if surface=='exposed_tip' and g.get('body_shape')=='wire' and g.get('taper_length_um',0)>0:
                return radius*min(1.,max(.00001,y/g['taper_length_um']))
            return radius
        vertices=np.array([[center[0]+exposed_radius(y)*np.cos(a),y,center[2]+exposed_radius(y)*np.sin(a)] for y in ys for a in angles])
        n=len(angles);faces=[]
        for r in range(len(ys)-1):
            for i in range(n-1):
                a=r*n+i;b=a+1;c=(r+1)*n+i+1;d=(r+1)*n+i
                faces.extend([[a,b,c],[a,c,d]])
        if surface=='exposed_tip':
            vertices=np.vstack([vertices,center]);faces.extend([[len(vertices)-1,i+1,i] for i in range(n-1)])
        return vertices,np.array(faces,int)
    u=np.array(c.get('u_xyz',[1,0,0]),float);v=np.array(c.get('v_xyz',[0,1,0]),float)
    u/=np.linalg.norm(u);v/=np.linalg.norm(v)
    shape=c.get('shape',g.get('contact_shape','square'))
    if shape in ('circle','ellipse'):
        angles=np.linspace(0,2*np.pi,24,endpoint=False);xy=np.column_stack([np.cos(angles),np.sin(angles)])
    else:xy=np.array([[-1,-1],[1,-1],[1,1],[-1,1]])
    vertices=center+xy[:,0,None]*width/2*u+xy[:,1,None]*height/2*v
    return vertices,np.array([[0,i,i+1] for i in range(1,len(vertices)-1)],int)


def new_contact(g,xyz,*,shank=0,**properties):
    used={c['electrode'] for c in g['contacts'] if c['shank']==shank};e=0
    while e in used:e+=1
    ids={c['id'] for c in g['contacts']};identifier=f's{shank}e{e}'
    while identifier in ids:identifier+='a'
    return dict(id=identifier,shank=shank,electrode=e,x_um=float(xyz[0]),y_um=float(xyz[1]),z_um=float(xyz[2]),**properties)


def perimeter_contacts(g,count,*,shank=0,face='front',edge=None,offset=0.):
    """Even arc-length distribution on a closed polygon or one selected edge."""
    poly=body_polygon(g); pairs=list(zip(poly,np.roll(poly,-1,axis=0)))
    if edge is not None:pairs=[pairs[int(edge)]]
    lengths=np.array([np.linalg.norm(b-a) for a,b in pairs]);total=lengths.sum()
    values=(np.arange(count)+.5)*total/count if edge is None else np.linspace(0,total,count+2)[1:-1]
    contacts=[]; cumulative=np.r_[0,np.cumsum(lengths)]
    winding=1 if polygon_area(poly)>0 else -1
    for value in values:
        i=min(len(pairs)-1,np.searchsorted(cumulative,value,side='right')-1);a,b=pairs[i];tangent=(b-a)/lengths[i]
        p=a+tangent*(value-cumulative[i]);outward=winding*np.array([tangent[1],-tangent[0]])
        if g.get('body_shape')=='prism':
            xyz=[p[0],g.get('contact_level_um',100),p[1]];u=[tangent[0],0,tangent[1]];v=[0,1,0]
        elif face=='edge':xyz=[p[0],p[1],0];u=[tangent[0],tangent[1],0];v=[0,0,1]
        else:
            p-=offset*outward;xyz=[p[0],p[1],g['shank_thickness_um']/2*(1 if face=='front' else -1)]
            u=[tangent[0],tangent[1],0];v=[-tangent[1],tangent[0],0]
        contacts.append(dict(xyz=xyz,u_xyz=u,v_xyz=v,surface='patch'))
    return contacts


def validate_physical(g):
    kind=g.get('body_shape','tapered')
    if kind in ('polygon','prism'):
        if not polygon_valid(g.get('body_polygon_um',[])):raise ValueError('Draw a closed contour with at least three vertices and no crossings.')
        if kind=='polygon':
            ys=np.asarray(g['body_polygon_um'])[:,1]
            if min(ys)<0 or max(ys)>g['shank_length_um']+1e-6:raise ValueError('The flat contour must lie between the distal reference (Y = 0) and the body length.')
    for c in g['contacts']:
        width=c.get('width_um',g['site_size_um']);height=c.get('height_um',g.get('site_height_um',width))
        if not np.isfinite([width,height]).all() or min(width,height)<=0:raise ValueError('Contact dimensions must be positive and finite.')
        surface=c.get('surface','patch')
        if surface not in ('patch','terminal','ring','sector','exposed_tip','cylindrical_patch'):raise ValueError('Unknown recording surface.')
        if surface in ('ring','sector','exposed_tip'):
            if kind not in ('wire','cylinder','none'):raise ValueError('Curved contacts need a cylindrical body, wire, or unknown body.')
            values=[c.get('radius_um',g['shank_width_um']/2),c.get('angle_deg',0),c.get('span_deg',360)]
            if not np.isfinite(values).all() or values[0]<=0 or not 0<values[2]<=360:raise ValueError('Check contact radius and angular span (0–360°).')
        else:
            if surface=='cylindrical_patch' and kind not in ('wire','cylinder'):raise ValueError('A cylindrical patch needs a cylindrical body.')
            u=np.asarray(c.get('u_xyz',[1,0,0]),float);v=np.asarray(c.get('v_xyz',[0,1,0]),float)
            if u.shape!=(3,) or v.shape!=(3,) or not np.isfinite([u,v]).all() or min(np.linalg.norm(u),np.linalg.norm(v))<1e-8 or abs(np.dot(u,v)/(np.linalg.norm(u)*np.linalg.norm(v)))>1e-5:raise ValueError('Contact surface axes must be finite, nonzero and perpendicular.')
            if c.get('shape',g.get('contact_shape','square')) not in ('square','rectangle','circle','ellipse'):raise ValueError('Unknown contact shape.')
        if kind=='none':continue
        local,_=contact_mesh_local(c,g);local[:,0]-=c['shank']*g['shank_pitch_um']
        if len(local) and (local[:,1].min() < -1e-5 or local[:,1].max()>g['shank_length_um']+1e-5):
            raise ValueError(f'Contact {c["id"]} extends past the distal reference or body base. Adjust its height or position.')
        if kind in ('polygon','tapered','rectangle'):
            if any(not point_in_polygon(pt[:2],body_polygon(g)) or abs(pt[2])>g['shank_thickness_um']/2+1e-5 for pt in local):
                raise ValueError(f'Contact {c["id"]} extends outside the body. Increase the inset or reduce its dimensions.')
        if kind=='prism' and any(not point_in_polygon(pt[[0,2]],g['body_polygon_um']) for pt in local):
            raise ValueError(f'Contact {c["id"]} extends outside the cross-section.')
        if kind in ('wire','cylinder') and c.get('surface') in ('terminal','ring','sector','exposed_tip'):
            radial=(local[:,0]/(g['shank_width_um']/2))**2+(local[:,2]/(g['shank_thickness_um']/2))**2
            if len(radial) and radial.max()>1.00001:raise ValueError(f'Contact {c["id"]} exceeds the body diameter.')
        xyz=np.array([c['x_um'],c['y_um'],c['z_um']])
        if kind=='polygon' and not point_in_polygon(xyz[:2],g['body_polygon_um']):raise ValueError(f'Contact {c["id"]} is outside the contour.')
        if kind=='prism' and not point_in_polygon(xyz[[0,2]],g['body_polygon_um']):raise ValueError(f'Contact {c["id"]} is outside the cross-section.')
        if kind in ('wire','cylinder'):
            radius=(xyz[0]/(g['shank_width_um']/2))**2+(xyz[2]/(g['shank_thickness_um']/2))**2
            if radius>1.00001:raise ValueError(f'Contact {c["id"]} is outside the cylindrical body.')
    return g

"""Display-only intersections of registered shanks with an atlas envelope."""
from dataclasses import dataclass

import numpy as np

from .landmarks import transform_points
from .neuropixels_model import probe_coordinates


@dataclass
class AtlasSurface:
    vertices_zyx_um: np.ndarray
    faces: np.ndarray
    spacing_um_zyx: tuple

    def __post_init__(self):
        triangles=np.asarray(self.vertices_zyx_um,dtype=float)[self.faces]
        self._extent=float(np.max(np.ptp(self.vertices_zyx_um,axis=0)))
        self._origin=triangles[:,0]
        self._edge1=triangles[:,1]-triangles[:,0]
        self._edge2=triangles[:,2]-triangles[:,0]
        lower,upper=triangles.min(axis=1),triangles.max(axis=1)
        centers=(lower+upper)/2
        self._tree=[]
        def build(indices):
            index=len(self._tree); self._tree.append(None)
            bounds=(lower[indices].min(0).tolist(),upper[indices].max(0).tolist())
            children=indices
            if len(indices)>48:
                axis=int(np.argmax(np.ptp(centers[indices],axis=0))); middle=len(indices)//2
                ordered=indices[np.argpartition(centers[indices,axis],middle)]
                children=(build(ordered[:middle]),build(ordered[middle:]))
            self._tree[index]=(*bounds,children)
            return index
        build(np.arange(len(triangles)))

    def _ray_faces(self,tip,direction):
        """Traverse a cached bounding-volume tree, keeping orbit interaction fast."""
        stack=[0]; leaves=[]
        origin=tip.tolist(); delta=direction.tolist()
        while stack:
            lower,upper,children=self._tree[stack.pop()]
            near,far=0.,1.
            for axis in range(3):
                if abs(delta[axis])<1e-12:
                    if not lower[axis]-1e-7<=origin[axis]<=upper[axis]+1e-7: break
                else:
                    a,b=(lower[axis]-origin[axis])/delta[axis],(upper[axis]-origin[axis])/delta[axis]
                    near=max(near,min(a,b)); far=min(far,max(a,b))
                    if near>far+1e-10: break
            else:
                if isinstance(children,np.ndarray): leaves.append(children)
                else: stack.extend(children)
        return np.concatenate(leaves) if leaves else np.empty(0,dtype=int)

    def entry(self,tip,base):
        """Surface hit closest to the base of the finite, fully mapped shank.

        Both ends are already in atlas micrometres. Never substitute a nearest
        surface vertex or extend a shank that does not intersect the atlas.
        """
        tip,base=np.asarray(tip,float),np.asarray(base,float)
        direction=base-tip
        if not np.isfinite([tip,base]).all() or np.linalg.norm(direction)<1e-9: return None
        fractions=self._fractions(tip,base)
        if not len(fractions): return None
        fraction=float(fractions[-1])
        return tip+fraction*direction,fraction

    def _fractions(self,tip,base):
        tip,base=np.asarray(tip,float),np.asarray(base,float)
        direction=base-tip
        candidate=self._ray_faces(tip,direction)
        origin,e1,e2=self._origin[candidate],self._edge1[candidate],self._edge2[candidate]
        if not len(origin): return np.empty(0)
        cross=np.cross(direction,e2)
        determinant=np.einsum('ij,ij->i',e1,cross)
        scale=np.linalg.norm(e1,axis=1)*np.linalg.norm(cross,axis=1)
        valid=np.abs(determinant)>np.maximum(scale*1e-10,1e-15)
        inverse=np.zeros_like(determinant); inverse[valid]=1/determinant[valid]
        offset=tip-origin
        u=np.einsum('ij,ij->i',offset,cross)*inverse
        q=np.cross(offset,e1)
        v=q@direction*inverse
        t=np.einsum('ij,ij->i',e2,q)*inverse
        valid&=(u>=-1e-8)&(v>=-1e-8)&(u+v<=1+1e-8)&(t>=0)&(t<=1)
        hits=np.sort(t[valid])
        return hits[np.r_[True,np.diff(hits)>1e-9]] if len(hits) else hits

    def contains(self,point):
        """Odd crossings toward a guaranteed exterior endpoint of the envelope."""
        point=np.asarray(point,float)
        if not np.isfinite(point).all(): return False
        end=point+np.array([self._extent*3+1,0.,0.])
        hits=self._fractions(point,end)
        return bool(np.count_nonzero(hits>1e-8)%2)

    def segment_intervals(self,tip,base):
        """Finite line sections classified against the complete atlas envelope."""
        tip,base=np.asarray(tip,float),np.asarray(base,float)
        if not np.isfinite([tip,base]).all() or np.linalg.norm(base-tip)<1e-9: return []
        hits=self._fractions(tip,base)
        cuts=np.r_[0.,hits[(hits>1e-9)&(hits<1-1e-9)],1.]
        result=[]
        for start,stop in zip(cuts[:-1],cuts[1:]):
            inside=self.contains(tip+(start+stop)/2*(base-tip))
            if result and result[-1][2]==inside:
                result[-1]=(result[-1][0],float(stop),inside)
            else: result.append((float(start),float(stop),inside))
        return result



def atlas_surface_mesh(volume,max_dimension=256,*,max_voxels=None):
    """A bounded, filled tissue envelope independent of display quality/gamma."""
    from scipy import ndimage
    from skimage.measure import marching_cubes
    from .insertion_model import brain_support
    from .volumes import Volume,normalize_intensity
    factors=np.maximum(1,np.ceil(np.asarray(volume.data.shape)/max_dimension).astype(int))
    if max_voxels is not None:
        while np.prod(np.ceil(np.asarray(volume.data.shape)/factors))>max_voxels:
            factors[np.argmax(np.asarray(volume.data.shape)/factors)]+=1
    data=np.asarray(volume.data[tuple(slice(None,None,int(n)) for n in factors)])
    spacing=np.asarray(volume.spacing_um_zyx)*factors
    preview=Volume(normalize_intensity(data),tuple(spacing))
    # A low foreground level keeps dim cortex in the outer envelope; a global
    # Otsu threshold can instead select only the bright interior of an atlas.
    mask,_=brain_support(preview,.05)
    if not mask.any(): raise ValueError('No atlas surface could be found.')
    # Filling the envelope excludes ventricular walls from entry detection.
    field=ndimage.gaussian_filter(np.pad(mask.astype(np.float32),2),.65)
    if field.max()<=.5: raise ValueError('The atlas is too thin to build a surface.')
    vertices,faces,_,_=marching_cubes(field,.5,spacing=tuple(spacing),allow_degenerate=False)
    vertices-=2*spacing
    return vertices.astype(np.float32),faces,tuple(spacing)


def build_atlas_surface(volume,max_dimension=256):
    return AtlasSurface(*atlas_surface_mesh(volume,max_dimension))


def probe_surface_entries(insertion,matrix,surface):
    """Keep native poses, source surface depths and analysis annotations intact."""
    settings=insertion.get('probe') or {}
    if not settings.get('pose') or settings.get('geometry',{}).get('body_shape')=='none': return []
    length=settings['geometry']['shank_length_um']
    tips=transform_points(probe_coordinates(settings,np.arange(settings['geometry']['shanks']),0.),matrix)
    bases=transform_points(probe_coordinates(settings,np.arange(settings['geometry']['shanks']),length),matrix)
    result=[]
    for shank,(tip,base) in enumerate(zip(tips,bases)):
        hit=surface.entry(tip,base)
        if hit is None: continue
        point,fraction=hit
        result.append(dict(insertion_id=insertion['id'],name=insertion['name'],
            shank=shank,color=insertion['color'],atlas_zyx_um=point.tolist(),
            height_from_tip_um=fraction*length))
    return result


def probe_shank_sections(insertion,matrix,surface):
    """Split registered shafts at the atlas surface; retain native height units."""
    settings=insertion['probe']; length=settings['geometry']['shank_length_um']
    if settings['geometry'].get('body_shape')=='none': return {True:[],False:[]}
    tips=transform_points(probe_coordinates(settings,np.arange(settings['geometry']['shanks']),0.),matrix)
    bases=transform_points(probe_coordinates(settings,np.arange(settings['geometry']['shanks']),length),matrix)
    sections={True:[],False:[]}
    for shank,(tip,base) in enumerate(zip(tips,bases)):
        for start,stop,inside in surface.segment_intervals(tip,base):
            sections[inside].append((shank,start*length,stop*length))
    return sections

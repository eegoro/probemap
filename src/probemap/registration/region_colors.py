"""Deterministic atlas colors with strong separation between nearby regions."""
import numpy as np
from scipy.spatial import cKDTree
from skimage.color import rgb2lab


def region_palette(data,regions):
    identifiers=sorted(int(r['id']) for r in regions)
    factors=np.maximum(1,np.ceil(np.asarray(data.shape)/128).astype(int))
    small=np.asarray(data[tuple(slice(None,None,int(f)) for f in factors)])
    neighbors={i:set() for i in identifiers}
    for axis in range(3):
        for offset in (1,2):
            a=[slice(None)]*3; b=a.copy(); a[axis]=slice(None,-offset); b[axis]=slice(offset,None)
            left,right=small[tuple(a)],small[tuple(b)]
            active=(left!=right)&(left>0)&(right>0)
            for i,j in np.unique(np.column_stack((left[active],right[active])),axis=0):
                if int(i) in neighbors and int(j) in neighbors:
                    neighbors[int(i)].add(int(j)); neighbors[int(j)].add(int(i))
    present=[i for i in identifiers if (small==i).any()]
    centers=np.array([np.argwhere(small==i).mean(0) for i in present])
    if len(present)>1:
        for index,indices in enumerate(cKDTree(centers).query(centers,k=min(4,len(present)))[1]):
            for other in np.atleast_1d(indices)[1:]:
                i,j=present[index],present[int(other)]; neighbors[i].add(j); neighbors[j].add(i)
    rgb=np.array(np.meshgrid(*([np.linspace(.08,.95,12)]*3),indexing='ij')).reshape(3,-1).T
    lab=rgb2lab(rgb[None])[0]
    usable=(lab[:,0]>40)&(lab[:,0]<85)&(np.linalg.norm(lab[:,1:],axis=1)>35)
    rgb,lab=rgb[usable],lab[usable]; assigned={}; used=set()
    for identifier in sorted(identifiers,key=lambda i:(-len(neighbors[i]),i)):
        near=[assigned[i] for i in sorted(neighbors[identifier]) if i in assigned]
        references=near or list(assigned.values())
        score=np.min(np.linalg.norm(lab[:,None]-lab[references][None],axis=2),axis=1) if references else np.linalg.norm(lab-[60,0,0],axis=1)
        if len(used)<len(rgb): score[list(used)]=-1
        chosen=int(np.argmax(score)); assigned[identifier]=chosen; used.add(chosen)
    return {str(i):np.round(rgb[c]*255).astype(int).tolist() for i,c in assigned.items()}

"""Adapt the run to the bound long feet while preserving airborne intervals."""
import sys
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from animation.surface import Surface
from roster.write_face_clips import write_clips
from arms import rotation


def adjust(path):
    s = Surface(path)
    sc = s.scene
    clip = 'running'
    start, end = sc.interval(clip)
    times = np.linspace(start, end, round((end-start)*120)+1)
    # Use the bound source foot surface on the existing split body mesh.
    p, neutral = next((p,v) for p,v in zip(s.parts,s.rest)
                      if sc.names[p.node]=='body' and v[np.unique(p.faces),1].max()<.4)
    used = np.unique(p.faces)
    angles = np.deg2rad(np.arange(0.,-51.,-1.))
    tracks = []
    for side,sign in [('L',1),('R',-1)]:
        j = sc.joint('foot.'+side)
        ids = used[neutral[used,0]*sign>0]
        angles_used, parents, original = [], [], []
        for t in times:
            pose,sh = sc.sample(clip,t)
            w = sc.world(pose)
            ankle = w[j,1:3,3]
            points = p.points(w,sh.get(p.node,sc.morph_rest(p.node)),ids)[:,1:3] - ankle
            hull = points[ConvexHull(points).vertices]
            heights = (ankle[0] + np.cos(angles[:,None])*hull[:,0]
                       - np.sin(angles[:,None])*hull[:,1]).min(axis=1)
            # Small remaining ground error is handled by the common placement track.
            ok = np.flatnonzero(heights >= -.006)
            k = ok[0] if len(ok) else np.argmax(heights)
            angles_used.append(angles[k])
            parents.append(rotation(w[sc.parents[j]]))
            original.append(rotation(w[j]))
        smooth = gaussian_filter1d(angles_used[:-1],1.2,mode='wrap')
        smooth = np.r_[smooth,smooth[0]]
        values=[]
        for a,parent,rot in zip(smooth,parents,original):
            q=(parent.inv()*Rotation.from_rotvec([a,0,0])*rot).as_quat()
            if values and np.dot(values[-1],q)<0: q=-q
            values.append(q.tolist())
        tracks.append(dict(joint=j,property='rotation',times=times.tolist(),values=values))
        print(side,'peak ankle pitch correction (deg)',np.rad2deg(abs(smooth)).max())
    write_clips(path,[dict(name=clip,tracks=tracks,morph_tracks=[])],extend=True)


if __name__=='__main__':
    adjust(Path(sys.argv[1]))

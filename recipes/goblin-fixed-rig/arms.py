"""Apply a clip-local outward rotation to both upper arms on the fixed rig."""
import sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from animation.scene import Scene
from roster.write_face_clips import write_clips


def rotation(matrix):
    basis = matrix[:3, :3]
    return Rotation.from_matrix(basis / np.linalg.norm(basis, axis=0))


def adjust(path, angle):
    scene = Scene(path)
    clip = 'running'
    tracks = []
    for side, sign in [('L', 1), ('R', -1)]:
        joint = scene.joint('upperarm01.' + side)
        parent = scene.parents[joint]
        times = scene.clips[clip][joint, 'rotation'][0]
        values = []
        for t in times:
            pose, _ = scene.sample(clip, t)
            world = scene.world(pose)
            torso = rotation(world[scene.joint('spine01')])
            axis = torso.apply(rotation(scene.world(scene.rest)[scene.joint('spine01')]).inv().apply([0, 0, 1]))
            offset = Rotation.from_rotvec(axis * np.deg2rad(sign * angle))
            q = (rotation(world[parent]).inv() * offset * rotation(world[joint])).as_quat()
            if values and np.dot(values[-1], q) < 0:
                q = -q
            values.append(q.tolist())
        tracks.append(dict(joint=joint, property='rotation', times=times.tolist(), values=values))
    write_clips(path, [dict(name=clip, tracks=tracks, morph_tracks=[])], extend=True)


if __name__ == '__main__':
    adjust(Path(sys.argv[1]), float(sys.argv[2]))

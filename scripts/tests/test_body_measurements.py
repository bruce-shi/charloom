"""Independent geometric checks for the reusable measurement commands."""

from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "skills/charloom-body-measure/scripts"))
from measurement_common import bound_point
from mesh_sections import cut_triangles, section_basis, section_loop, reference_span
from outline_points import required_landmarks
from bind_landmarks import embedded_binding
from surface_tape import guide_centres, ray_hit, silhouette_hit, inner_upperarm_rays, validate_inner_endpoints, length


def box():
    points = np.array([
        [-.1, -.3, -.05], [.1, -.3, -.05], [.1, .3, -.05], [-.1, .3, -.05],
        [-.1, -.3, .05], [.1, -.3, .05], [.1, .3, .05], [-.1, .3, .05],
    ])
    faces = np.array([
        [0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
        [0, 4, 5], [0, 5, 1], [3, 2, 6], [3, 6, 7],
        [0, 3, 7], [0, 7, 4], [1, 5, 6], [1, 6, 2],
    ])
    return points, faces


def camera():
    return dict(id="front", center=[0, 0, 0], right=[1, 0, 0], up=[0, 1, 0],
                forward=[0, 0, -1], width=100, height=100, meters_per_pixel=.01)


class BodyMeasurementsTest(unittest.TestCase):
    def test_embedded_centre_follows_affine_skin_edit(self):
        vertices,faces=box()
        point=np.array([.02,-.01,.01])
        binding=embedded_binding(vertices,np.arange(len(vertices)),point,8,.008)
        coeff=np.array(binding['coefficients']);ids=binding['vertices']
        np.testing.assert_allclose(coeff@vertices[ids],point,atol=1e-10)
        transform=Rotation.from_euler('x',30,degrees=True).as_matrix()@np.diag([.8,1.2,.9])
        offset=np.array([.2,-.3,.7])
        np.testing.assert_allclose(coeff@(vertices[ids]@transform.T+offset),point@transform.T+offset,atol=1e-10)

    def test_inner_arm_starts_at_axilla_not_shoulder(self):
        vertices,faces=box()
        joints=np.array([[0,.3,0],[0,-.3,0]])
        for side,sign in [('l',-1),('r',1)]:
            endpoints=np.array([[sign*.1,.15,0],[sign*.1,-.25,0]])
            points=np.array([ray_hit(origin,direction,vertices,faces,.2)[0]
                             for _,origin,direction in inner_upperarm_rays(joints,endpoints,side,45)])
            np.testing.assert_allclose(points[[0,-1]],endpoints,atol=1e-12)
            self.assertAlmostEqual(length(points),400)
            validate_inner_endpoints(points,endpoints,.01)
            with self.assertRaisesRegex(ValueError,'marked axilla'):
                validate_inner_endpoints(points,endpoints+np.array([0,.02,0]),1)

    def test_inner_arm_preserves_endpoint_depth_instead_of_joint_depth(self):
        vertices,faces=box();joints=np.array([[0,.3,0],[0,-.3,0]])
        endpoints=np.array([[-.1,.15,.02],[-.1,-.25,-.02]])
        points=np.array([ray_hit(origin,direction,vertices,faces,.2)[0]
                         for _,origin,direction in inner_upperarm_rays(joints,endpoints,'l',45)])
        np.testing.assert_allclose(points[[0,-1]],endpoints,atol=1e-12)
        self.assertGreater(length(points),400)

    def test_triangle_cut_is_rotation_invariant(self):
        vertices, faces = box()
        for angles in ([0, 0, 0], [21, 37, -44], [78, -19, 16]):
            rotation = Rotation.from_euler("xyz", angles, degrees=True).as_matrix()
            center = np.array([.7, 1.2, -.2])
            transformed = vertices @ rotation.T + center
            axis = rotation @ [0, 1, 0]
            basis = section_basis(center-axis, center+axis, rotation @ [1, 0, 0], rotation @ [0, 0, 1])
            segments, _ = cut_triangles(transformed, faces, center, basis)
            polygon = section_loop(segments)
            np.testing.assert_allclose(np.ptp(polygon, axis=0), [.2, .1], atol=2e-6)

    def test_section_requires_a_real_closed_surface(self):
        vertices, faces = box()
        basis = section_basis(np.array([0, -1, 0]), np.array([0, 1, 0]), [1, 0, 0])
        segments, _ = cut_triangles(vertices, faces[:4], np.zeros(3), basis)
        with self.assertRaises(ValueError):
            section_loop(segments)

    def test_surface_ray_returns_face_bound_point_and_miss(self):
        vertices, faces = box()
        point, face, bary = ray_hit(np.zeros(3), np.array([1., 0, 0]), vertices, faces, .2)
        np.testing.assert_allclose(point, [.1, 0, 0], atol=1e-12)
        np.testing.assert_allclose(bary @ vertices[faces[face]], point, atol=1e-12)
        self.assertIsNone(ray_hit(np.zeros(3), np.array([1., 0, 0]), vertices, faces, .09))

    def test_no_depth_from_a_single_camera(self):
        guide = {"landmarks": {"shoulder_l": [{"view": "front", "px": [50, 50]}]}}
        with self.assertRaisesRegex(ValueError, "nonparallel"):
            guide_centres(guide, {"front": camera()}, np.eye(4), ["shoulder_l"])

    def test_image_ray_preserves_projected_axis(self):
        mask = np.zeros((100, 100), bool)
        mask[20:81, 40:61] = True
        direction = np.array([.6, 0., .8])
        span, px = reference_span(np.zeros(3), direction, camera(), np.eye(4), mask, .3, .001)
        # A 0.20 m image span corresponds to about 0.35 m along this slanted 3D ray.
        self.assertAlmostEqual(float(np.diff(span)[0]), .35, delta=.003)
        self.assertAlmostEqual(float(px[1, 1]-px[0, 1]), 0)
        with self.assertRaisesRegex(ValueError, "cannot observe"):
            reference_span(np.zeros(3), [0, 0, 1], camera(), np.eye(4), mask, .3, .001)

    def test_missing_silhouette_boundary_is_not_an_estimate(self):
        mask = np.ones((100, 100), bool)
        with self.assertRaisesRegex(ValueError, "No reference boundary"):
            silhouette_hit(np.zeros(3), np.array([1., 0, 0]), camera(), np.eye(4), mask, .001, .2)

    def test_invalid_surface_binding_fails(self):
        vertices, faces = box()
        with self.assertRaises(ValueError):
            bound_point(vertices, faces, {"triangle": 0, "barycentric": [-.2, .6, .6]})

    def test_basic_landmark_contract_is_not_dense_body_coverage(self):
        names = required_landmarks()
        self.assertEqual(len(names), 26)
        self.assertIn("ankle_r", names)
        self.assertIn("eye_l_bottom", names)
        self.assertNotIn("palm_l", names)
        self.assertFalse(any(n.startswith("outline_") for n in names))


if __name__ == "__main__":
    unittest.main()

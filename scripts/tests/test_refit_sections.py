"""Numerical invariants of native-body anatomical station mapping."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'skills/charloom-body-fit/scripts'))
from refit_sections import map_sections

class SectionRefitTest(unittest.TestCase):
    def rows(self):
        return [dict(native_axis_m=h,target_axis_m=h*2,native_bounds_m=[[-1,-2],[1,2]],target_bounds_m=[[1,-1],[5,1]]) for h in [0,1,2]]
    def test_anatomical_order_and_both_transverse_edges(self):
        pts=np.array([[0,-1,-2],[1,1,2],[2,0,0]])
        actual=map_sections(pts,[0,0,0],np.eye(3),[0,0,0],np.eye(3),self.rows())
        np.testing.assert_allclose(actual,[[0,1,-1],[2,5,1],[4,3,0]])
    def test_rotated_limb_stays_in_its_own_frame(self):
        basis=np.array([[0,-1,0],[1,0,0],[0,0,1]])
        point=np.array([[1,1,2]])@basis.T+[3,4,5]
        actual=map_sections(point,[3,4,5],basis,[1,2,3],basis,self.rows())
        np.testing.assert_allclose(actual,np.array([[2,5,1]])@basis.T+[1,2,3])
    def test_reversed_anatomical_stations_rejected(self):
        rows=self.rows();rows[1]['target_axis_m']=5
        with self.assertRaises(ValueError):
            map_sections(np.zeros((1,3)),[0,0,0],np.eye(3),[0,0,0],np.eye(3),rows)

if __name__=='__main__': unittest.main()

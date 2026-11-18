"""Compute the pipeline's calibrated working frame before any body fitting.
PY working_frame.py SCAN.glb --prior PRIOR.glb --guide GUIDE --out FRAME.json
"""
import argparse
from pathlib import Path
from measurement_common import digest, write
from roster.mesh import Glb
from transfer.scan import Scan
from transfer.guide import load_annotations

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path);p.add_argument('--prior',type=Path,required=True)
    p.add_argument('--guide',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists(): raise FileExistsError(a.out)
    scan=Scan(a.source);guide,_,_=load_annotations(scan,a.guide/'guidance.json')
    matrix=scan.working_frame(Glb(a.prior),guide['front_view'])
    write(a.out,dict(source_to_working=matrix.tolist(),source_hash=scan.source_hash,
          source_sha256=digest(a.source),prior_sha256=digest(a.prior),
          scale_basis='Template body height divided by rotated full scan bounds; working-model units, not measured real stature'))
    print(a.out)

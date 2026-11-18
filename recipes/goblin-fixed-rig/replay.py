"""Reproduce the two Goblin GLBs from the registered bound models and the running FBX."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

here = Path(__file__).resolve().parent
repo = Path.cwd()
out = Path(sys.argv[1]).resolve()
out.mkdir(parents=True, exist_ok=False)
env = dict(os.environ, PYTHONPATH=str(repo / 'scripts'))
config_dir = Path(os.environ.get('CHARLOOM_REPLAY_CONFIG', here))
inputs = json.loads((config_dir / 'inputs.json').read_text())
base = Path(os.environ.get('CHARLOOM_BOUND_CHARACTER', repo / 'out/transfers/final/goblin'))
motion = repo / inputs['motion_source']['path']


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def run(script, *args):
    subprocess.run([sys.executable, script, *args], env=env, check=True)


if sha(motion) != inputs['motion_source']['sha256']:
    raise ValueError('Running FBX hash differs from the recorded source')
variants = {}
for layer, rel in [('body', '3-skin/body.glb'), ('character', '4-garments/character.glb')]:
    if sha(base / rel) != inputs[layer]['baseline_sha256']:
        raise ValueError(f'Baseline {layer} GLB has changed')
    folder = out / layer
    run('scripts/animate.py', 'retarget', motion, base / rel, '--config', config_dir / 'running.json',
        '--clip', 'Armature|mixamo.com|Layer0', '--out', folder)
    asset = folder / 'character.glb'
    run(here / 'arms.py', asset, '26')
    run(here / 'feet_pose.py', asset)
    variants[layer + '.glb'] = str(asset)
config = dict(reference=variants['character.glb'], variants=variants, contact_meshes=['body'],
              root='root', clip='running', motion_kind='running', sample_hz=90, tolerance_mm=.6)
(out / 'ground.json').write_text(json.dumps(config))
run('scripts/ground-walk.py', out / 'ground.json', '--out', out / 'final')
for layer in ('body', 'character'):
    digest = sha(out / 'final' / (layer + '.glb'))
    if digest != inputs[layer]['final_sha256']:
        raise ValueError(f'Final {layer} GLB differs from the delivery')
print('Both output hashes match the delivery')

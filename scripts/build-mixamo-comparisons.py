#!/usr/bin/env python3
"""Retarget the local Mixamo takes to network and local characters, without editing sources."""
import argparse
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
import os
CV_PYTHON = os.getenv('CHARLOOM_CV_PYTHON', str(ROOT/'out/tools/cv/.venv/bin/python'))

def cli(*args):
    return json.loads(subprocess.check_output([CV_PYTHON, str(ROOT/'scripts/animate.py'), *map(str, args)], cwd=ROOT))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cases = []
    for character, original, coverage in [
        ('quaternius', ROOT / 'fixtures/quaternius/superhero-male.glb', .9),
        ('myth-crossbow', ROOT / 'fixtures/local-myth/crossbow-guard.glb', .8),
    ]:
        target = original
        for motion in ['walking', 'running', 'waving']:
            source = ROOT / f'fixtures/mixamo/{motion}.fbx'
            config = output / f'{character}-{motion}.config.json'
            config.write_text(json.dumps({'fps':60, 'name':f'mixamo_{motion}', 'min_target_coverage':coverage, 'align_reference_chains':True}, indent=2)+'\n')
            folder = output / character / motion
            result = cli('retarget',source,target,'--clip','mixamo.com','--config',config,'--out',folder)
            target = folder / 'character.glb'
            report = result
            cases.append({'character':character,'motion':motion,'source':str(source),'input_character':str(original),'output':str(target),'retarget':report,'structural_validation':cli('validate',target),'visual_review':'pending'})
    report = {'cases':cases,'quality_status':'needs_review','notes':['Source take duration is preserved; short takes are not extended to invent missing movement.','The myth comparison uses the body motion only; retained weapon channels are not automatically characterized as grips.']}
    (output / 'matrix.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'output':str(output),'cases':[{'character':c['character'],'motion':c['motion'],'coverage':c['retarget']['coverage'],'output':c['output']}for c in cases]},indent=2))

if __name__ == '__main__':
    main()

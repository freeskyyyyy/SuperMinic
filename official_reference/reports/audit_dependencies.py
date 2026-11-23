from pathlib import Path
import json, subprocess, tomllib
root = Path(__file__).resolve().parents[1]
source = root / 'src'
lock = tomllib.loads((source/'supermimic/uv.lock').read_text())
locked = {p['name']: p for p in lock['package']}
mj = tomllib.loads((source/'mjlab_research/pyproject.toml').read_text())
report = {
    'supermimic_commit': subprocess.check_output(['git','-C',str(source/'supermimic'),'rev-parse','HEAD'],text=True).strip(),
    'mjlab_candidate_commit': subprocess.check_output(['git','-C',str(source/'mjlab_research'),'rev-parse','HEAD'],text=True).strip(),
    'mjlab_declared_version': mj['project']['version'],
    'mjlab_locked_version': locked['mjlab']['version'],
    'mjlab_declared_dependencies': mj['project']['dependencies'],
    'original_locked_versions': {n:locked[n]['version'] for n in ['torch','mujoco','mujoco-warp','warp-lang','rsl-rl-lib']},
    'original_lock_matches_mjlab_version': mj['project']['version'] == locked['mjlab']['version'],
    'runtime_verified': False,
}
(root/'reports/dependency-audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k != 'mjlab_declared_dependencies'},ensure_ascii=False,indent=2))

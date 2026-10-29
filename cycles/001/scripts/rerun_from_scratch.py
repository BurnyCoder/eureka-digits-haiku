"""Reproducibility check: retrain every registered arm in a scratch copy and compare with the cycle's records.

Copies only what study.py needs (the fixture, the registered protocol and registration, the pilot publication,
the pilot record and its checkpoint) into a fresh temporary directory outside the cycle, runs
`study.py experiments` there, then compares each new record with results/<id>.json on the checkpoint SHA-256,
the stored test labels and predictions, the example IDs, the accuracy and the per-epoch history.
Sources: tempfile.mkdtemp https://docs.python.org/3/library/tempfile.html
         subprocess.run https://docs.python.org/3/library/subprocess.html
         shutil.copy2 https://docs.python.org/3/library/shutil.html
Outputs: results/rerun_from_scratch_repeat.log (timestamped run log), research/rerun_comparison.json (per-record match flags).
"""
from pathlib import Path
import json
import shutil
import subprocess
import sys
import tempfile

root = Path(__file__).resolve().parents[1]
scratch = Path(tempfile.mkdtemp(prefix='eureka_rerun_'))  # outside the cycle; removed at the end
try:
    for sub in ('research', 'results', 'models'):
        (scratch/sub).mkdir()
    # Only the inputs study.py reads before training; the pilot checkpoint is reused for regularized_s0.
    for name in ('study.py', 'research/protocol.json', 'research/registration.json', 'research/pilot-publication.json',
                 'results/pilot_regularized_s0.json', 'models/regularized-s0.safetensors'):
        shutil.copy2(root/name, scratch/name)
    with (root/'results/rerun_from_scratch_repeat.log').open('w', encoding='utf-8') as log:
        subprocess.run([sys.executable, 'study.py', 'experiments'], cwd=scratch, stdout=log, stderr=subprocess.STDOUT, check=True)
    fields = ['accuracy', 'example_ids', 'y_true', 'y_pred', 'history', 'checkpoint_sha256']
    report = {}
    for path in sorted((scratch/'results').glob('*.json')):
        new = json.loads(path.read_text(encoding='utf-8'))
        old = json.loads((root/'results'/path.name).read_text(encoding='utf-8'))
        report[path.stem] = {field: new[field] == old[field] for field in fields}
    out = {'records': report, 'all_match': all(all(flags.values()) for flags in report.values())}
    (root/'research/rerun_comparison.json').write_text(json.dumps(out, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(out, indent=2))
finally:
    shutil.rmtree(scratch)

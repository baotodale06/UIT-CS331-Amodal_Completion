"""Use the local GPU after the frozen predecessor, only while Kaggle is queued."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import sha256, write_json
from amodal_benchmark.parallel import output_lock
from scripts.collect_shape_probability_training import parse_status

STUDY = ROOT / 'output/shape-probability-local-20261123'
PREVIOUS = ROOT / 'output/shape-local-20261123'
KERNEL = 'ez4kacchan/shape-only-probability-training-20261123/1'
CLI = Path.home() / '.cache/shape-kaggle-cli/Scripts/kaggle.exe'
WORK = '/home/hungvudao/.cache/shape-local-gpu-20261123'
SOURCE = WORK + '/probability-source'
WSL_STUDY = '/mnt/c/Users/daovu/Documents/Project/cv_nangcao/output/shape-probability-local-20261123'
PROBE = r'''
import json, os
from pathlib import Path
prefix = '/home/hungvudao/.cache/shape-local-gpu-20261123/'
found = []
for path in Path('/proc').iterdir():
    if not path.name.isdigit() or int(path.name) == os.getpid():
        continue
    try:
        args = path.joinpath('cmdline').read_bytes().decode().split('\0')
    except (OSError, UnicodeError):
        continue
    if any(arg.startswith(prefix + 'training-source-v2/') or arg.startswith(prefix + 'probability-source/')
           or arg.endswith('/output/shape-local-20261123/launch_local.py') for arg in args):
        found.append(dict(pid=int(path.name), args=args))
print(json.dumps(found))
'''


def decision(execution, active, observation, kaggle=None):
    """A missing observation never authorizes another training process."""
    if active:
        return 'waiting_predecessor'
    commands = execution.get('commands', [])
    if not execution.get('complete'):
        return 'needs_review'
    if (len(commands) != 7 or any(row.get('exit_code') != 0 for row in commands)
            or [row['command'][-1] for row in commands[-4:]] != ['calibrate', 'preflight', 'train', 'native']):
        return 'needs_review'
    if observation.get('disposition') != 'verified':
        return 'waiting_predecessor_replay'
    if kaggle is None:
        return 'check_kaggle'
    if kaggle == 'QUEUED':
        return 'start_local'
    if kaggle in ('RUNNING', 'COMPLETE'):
        return 'external_training_available'
    return 'needs_review'


def verify_operations():
    manifest = json.loads((STUDY / 'operational-manifest.json').read_bytes())
    for name, expected in manifest['files'].items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT.resolve()) or sha256(path) != expected:
            raise ValueError('Prepared fallback source changed: ' + name)
    return sha256(STUDY / 'operational-manifest.json')


def live_processes():
    result = subprocess.run(['wsl', '-e', 'python3', '-'], input=PROBE.encode(),
                            capture_output=True, check=True, timeout=45)
    return json.loads(result.stdout)


def verify_staged_source():
    program = r'''
import hashlib, json
from pathlib import Path
study = Path('/mnt/c/Users/daovu/Documents/Project/cv_nangcao/output/shape-probability-local-20261123')
source = Path('/home/hungvudao/.cache/shape-local-gpu-20261123/probability-source')
manifest = json.loads((study / 'training-source-manifest.json').read_bytes())
for name, expected in manifest['files'].items():
    path = (source / name).resolve()
    assert path.is_relative_to(source.resolve())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name
print(json.dumps(dict(verified=True, files=len(manifest['files']))))
'''
    result = subprocess.run(['wsl', '-e', 'python3', '-'], input=program.encode(),
                            capture_output=True, check=True, timeout=45)
    return json.loads(result.stdout)


def run_chain():
    path = STUDY / 'execution.json'
    if path.exists():
        raise FileExistsError('Inspect the existing local launch; never duplicate or automatically resume')
    if live_processes():
        raise ValueError('A predecessor or probability process is still alive')
    receipt = dict(started=datetime.now(timezone.utc).isoformat(), windows_pid=os.getpid(),
        operational_manifest_sha256=verify_operations(), source=SOURCE, commands=[],
        staged_source=verify_staged_source(), complete=False, benchmark_success=False)
    write_json(path, receipt)
    environment = dict(os.environ, PYTHONIOENCODING='utf-8')
    common = ['wsl', '-e', 'env', 'PYTHONNOUSERSITE=1', 'PYTHONPATH=' + SOURCE,
        'OMP_NUM_THREADS=2', 'MKL_NUM_THREADS=2', 'OPENBLAS_NUM_THREADS=2',
        'LD_LIBRARY_PATH=' + WORK + '/env/lib:/usr/lib/wsl/lib', 'nice', '-n', '10',
        WORK + '/env/bin/python', '-u', SOURCE + '/scripts/train_shape_probability_local.py',
        '--data', WORK + '/inputs/data', '--sam-training', WORK + '/inputs/sam-training',
        '--sam-validation', WORK + '/inputs/sam-validation',
        '--initial-checkpoint', WORK + '/inputs/initial/best.pt', '--study', WSL_STUDY + '/run']

    def execute(command, label):
        verify_operations()
        verify_staged_source()
        row = dict(command=command, started=datetime.now(timezone.utc).isoformat(), log=label + '.log')
        receipt['commands'].append(row)
        write_json(path, receipt)
        with (STUDY / row['log']).open('xb') as stream:
            process = subprocess.Popen(command, env=environment, stdout=stream, stderr=subprocess.STDOUT)
            row['windows_process_id'] = process.pid
            write_json(path, receipt)
            row['exit_code'] = process.wait()
        row['finished'] = datetime.now(timezone.utc).isoformat()
        write_json(path, receipt)
        if row['exit_code']:
            raise RuntimeError('Local fallback command failed; inspect ' + row['log'])

    execute(common + ['--mode', 'calibrate'], 'calibrate')
    verifier = ROOT / 'scripts/verify_shape_probability_local.py'
    execute([sys.executable, str(verifier), '--calibration-only', '--output', str(STUDY / 'calibration-replay.json')],
            'calibration-replay')
    for mode in ('preflight', 'train', 'native'):
        execute(common + ['--mode', mode], mode)
    execute([sys.executable, str(verifier), '--output', str(STUDY / 'native-replay.json')], 'native-replay')
    execute([sys.executable, str(ROOT / 'scripts/build_shape_probability_local_report.py')], 'report')
    receipt.update(complete=True, finished=datetime.now(timezone.utc).isoformat())
    write_json(path, receipt)


def poll():
    verify_operations()
    execution = json.loads((PREVIOUS / 'execution-v2.json').read_bytes())
    active = live_processes()
    observed = json.loads((PREVIOUS / 'collection-status.json').read_bytes())
    result = dict(at=datetime.now(timezone.utc).isoformat(), active_processes=active,
                  predecessor_complete=execution.get('complete'), benchmark_success=False)
    state = decision(execution, active, observed)
    if state == 'check_kaggle':
        # Recompute the completed predecessor's native evidence before using the GPU.
        from scripts.verify_shape_local import replay
        if replay(PREVIOUS / 'run-v2') != json.loads((PREVIOUS / 'native-replay.json').read_bytes()):
            raise ValueError('Predecessor independent replay changed')
        checked = subprocess.run([str(CLI), 'kernels', 'status', KERNEL], capture_output=True,
            check=True, timeout=60, text=True, env=dict(os.environ, PYTHONIOENCODING='utf-8'))
        result['kaggle_state'] = parse_status(checked.stdout)
        result['kaggle_kernel'] = KERNEL
        state = decision(execution, active, observed, result['kaggle_state'])
    result['disposition'] = state
    write_json(STUDY / 'queue-status.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    control = STUDY / 'queue-control'
    control.mkdir(parents=True, exist_ok=True)
    with output_lock(control):
        if (STUDY / 'launch-request.json').exists() or (STUDY / 'execution.json').exists():
            raise FileExistsError('Inspect the existing fallback attempt before any restart')
        while True:
            try:
                result = poll()
            except (subprocess.SubprocessError, OSError, json.JSONDecodeError) as error:
                result = dict(at=datetime.now(timezone.utc).isoformat(), disposition='observation_error',
                              error=type(error).__name__, benchmark_success=False)
                write_json(STUDY / 'queue-last-error.json', result)
            print(json.dumps(result), flush=True)
            if result['disposition'] == 'start_local':
                write_json(STUDY / 'launch-request.json', result)
                try:
                    write_json(STUDY / 'queue-status.json', dict(result, disposition='running_local'))
                    run_chain()
                except (OSError, subprocess.SubprocessError, RuntimeError, ValueError) as error:
                    write_json(STUDY / 'queue-status.json', dict(result, disposition='needs_review', error=str(error)))
                    raise
                replay = json.loads((STUDY / 'native-replay.json').read_bytes())
                write_json(STUDY / 'queue-status.json', dict(result, disposition='verified',
                    selected_epoch=replay['selected_epoch'], new_checkpoint_eligible=replay['new_checkpoint_eligible'],
                    finished=datetime.now(timezone.utc).isoformat()))
                return
            if not args.watch or result['disposition'] in ('needs_review', 'external_training_available'):
                return
            time.sleep(60)


if __name__ == '__main__':
    main()

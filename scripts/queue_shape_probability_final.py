"""Run the fixed diagnostic only after successful parent audit and process exit."""
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
from scripts.shape_probability_final_contract import CONFIG, LOGS, load_parent, read, verify_sources

STUDY = ROOT / 'output/shape-probability-final-diagnostic-20261123'
PARENT = ROOT / 'output/shape-probability-workers4-20261123'
WORK = '/home/hungvudao/.cache/shape-local-gpu-20261123'
SOURCE = WORK + '/probability-final-diagnostic-source'
WSL_ROOT = '/mnt/c/Users/daovu/Documents/Project/cv_nangcao'
WSL_STUDY = WSL_ROOT + '/output/shape-probability-final-diagnostic-20261123'
WSL_PARENT = WSL_ROOT + '/output/shape-probability-workers4-20261123'
MANIFEST = STUDY / 'diagnostic-source-manifest.json'
PROBE = r'''
import json, os
from pathlib import Path
found = []
prefix = '/home/hungvudao/.cache/shape-local-gpu-20261123/'
for path in Path('/proc').iterdir():
    if not path.name.isdigit() or int(path.name) == os.getpid():
        continue
    try:
        args = path.joinpath('cmdline').read_bytes().decode().split('\0')
    except (OSError, UnicodeError):
        continue
    if args and (args[0].startswith(prefix + 'env/bin/python') or any(
            arg.startswith(prefix + 'probability-source-workers4/') or
            arg.startswith(prefix + 'probability-final-diagnostic-source/') for arg in args)):
        found.append(dict(pid=int(path.name), args=args))
print(json.dumps(found))
'''


def decision(execution, status, active):
    if active:
        return 'waiting_parent_processes'
    commands = execution.get('commands', [])
    if not execution.get('complete'):
        return 'needs_review'
    if ([row.get('log') for row in commands] != LOGS or
            any(row.get('exit_code') != 0 for row in commands)):
        return 'needs_review'
    if status.get('disposition') != 'verified':
        return 'needs_review'
    return 'ready'


def live_processes():
    result = subprocess.run(['wsl', '-e', 'python3', '-'], input=PROBE.encode(),
                            capture_output=True, check=True, timeout=45)
    active = json.loads(result.stdout)
    command = ['powershell', '-NoProfile', '-Command',
        "Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python*' -and "
        "$_.CommandLine -like '*queue_shape_probability_workers4.py*' } | "
        "Select-Object ProcessId,CreationDate | ConvertTo-Json -Compress"]
    result = subprocess.run(command, capture_output=True, check=True, timeout=45)
    windows = json.loads(result.stdout) if result.stdout.strip() else []
    return dict(wsl=active, windows=windows if isinstance(windows, list) else [windows])


def staged_source():
    program = r'''
import hashlib, json
from pathlib import Path
study = Path('/mnt/c/Users/daovu/Documents/Project/cv_nangcao/output/shape-probability-final-diagnostic-20261123')
source = Path('/home/hungvudao/.cache/shape-local-gpu-20261123/probability-final-diagnostic-source')
manifest = json.loads((study / 'diagnostic-source-manifest.json').read_bytes())
for name, expected in manifest['files'].items():
    path = (source / name).resolve()
    if not path.is_relative_to(source.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError('Staged diagnostic source changed: ' + name)
print(json.dumps(dict(verified=True, files=len(manifest['files']))))
'''
    result = subprocess.run(['wsl', '-e', 'python3', '-'], input=program.encode(),
                            capture_output=True, check=True, timeout=45)
    return json.loads(result.stdout)


def validate_preparation():
    manifest_hash = verify_sources(MANIFEST)
    staging = read(STUDY / 'staging.json')
    if (staging.get('verified') is not True or staging.get('gpu_validation_performed') is not False
            or staging.get('manifest_sha256') != manifest_hash
            or not staging.get('commands') or any(row.get('exit_code') != 0 for row in staging['commands'])):
        raise ValueError('Completed isolated CPU staging required')
    checks = read(STUDY / 'cpu-validation.json')
    if (checks.get('passed') is not True or checks.get('manifest_sha256') != manifest_hash
            or checks.get('exit_code') != 0 or sha256(STUDY / 'cpu-check.log') != checks['log_sha256']):
        raise ValueError('Full CPU validation required')
    return manifest_hash


def run_chain():
    if (STUDY / 'execution.json').exists() or (STUDY / 'launch-request.json').exists():
        raise FileExistsError('Preserve existing diagnostic attempt; inspect before any recovery')
    manifest_hash = validate_preparation()
    processes = live_processes()
    if processes['wsl'] or processes['windows']:
        raise ValueError('Parent or GPU environment process is still alive')
    _, bindings = load_parent(PARENT, read(ROOT / CONFIG))
    staged = staged_source()
    receipt = dict(started=datetime.now(timezone.utc).isoformat(), windows_pid=os.getpid(),
                   source_manifest_sha256=manifest_hash, parent=bindings, staged_source=staged,
                   commands=[], complete=False, benchmark_success=False)
    with (STUDY / 'launch-request.json').open('x', encoding='utf8') as stream:
        json.dump(receipt, stream, indent=2)
    path = STUDY / 'execution.json'
    write_json(path, receipt)
    prefix = ['wsl', '-e', 'env', 'PYTHONNOUSERSITE=1', 'PYTHONPATH=' + SOURCE,
              'OMP_NUM_THREADS=2', 'MKL_NUM_THREADS=2', 'OPENBLAS_NUM_THREADS=2',
              'LD_LIBRARY_PATH=' + WORK + '/env/lib:/usr/lib/wsl/lib', 'nice', '-n', '10',
              WORK + '/env/bin/python', '-u']
    common = ['--parent', WSL_PARENT, '--data', WORK + '/inputs/data',
              '--sam-validation', WORK + '/inputs/sam-validation',
              '--source-manifest', WSL_STUDY + '/diagnostic-source-manifest.json']

    def execute(command, label):
        if validate_preparation() != manifest_hash or load_parent(PARENT, read(ROOT / CONFIG))[1] != bindings:
            raise ValueError('Parent or diagnostic sources changed')
        staged_source()
        row = dict(command=command, started=datetime.now(timezone.utc).isoformat(), log=label + '.log')
        receipt['commands'].append(row)
        write_json(path, receipt)
        with (STUDY / row['log']).open('xb') as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                       env=dict(os.environ, PYTHONIOENCODING='utf-8'))
            row['windows_process_id'] = process.pid
            write_json(path, receipt)
            row['exit_code'] = process.wait()
        row['finished'] = datetime.now(timezone.utc).isoformat()
        write_json(path, receipt)
        if row['exit_code'] != 0:
            raise RuntimeError('Diagnostic command failed; inspect ' + row['log'])

    for label, flags in [('preflight', ['--preflight-objects', '3']), ('full', [])]:
        result = WSL_STUDY + '/' + label
        execute(prefix + [SOURCE + '/scripts/diagnose_shape_probability_final.py'] + common +
                ['--initial-checkpoint', WORK + '/inputs/initial/best.pt', '--output', result] + flags, label)
        execute(prefix + [SOURCE + '/scripts/verify_shape_probability_final.py'] + common +
                ['--result', result, '--output', WSL_STUDY + '/' + label + '-replay.json'] +
                (['--preflight'] if flags else []), label + '-replay')
    receipt.update(complete=True, finished=datetime.now(timezone.utc).isoformat(),
                   replay_sha256=sha256(STUDY / 'full-replay.json'))
    write_json(path, receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    control = STUDY / 'queue-control'
    control.mkdir(exist_ok=True)
    with output_lock(control):
        if (STUDY / 'execution.json').exists() or (STUDY / 'launch-request.json').exists():
            raise FileExistsError('Existing diagnostic attempt must be reviewed, never automatically restarted')
        validate_preparation()
        staged_source()
        while True:
            result = dict(at=datetime.now(timezone.utc).isoformat(), benchmark_success=False)
            try:
                validate_preparation()
                processes = live_processes()
                state = decision(read(PARENT / 'execution.json'), read(PARENT / 'queue-status.json'),
                                 processes['wsl'] or processes['windows'])
                result.update(disposition=state, processes=processes)
                if state == 'ready':
                    # Successful complete artifacts alone do not prove process termination.
                    load_parent(PARENT, read(ROOT / CONFIG))
            except (subprocess.SubprocessError, OSError, json.JSONDecodeError) as error:
                result.update(disposition='observation_error', error=type(error).__name__)
            write_json(STUDY / 'queue-status.json', result)
            print(json.dumps(result), flush=True)
            if result['disposition'] == 'ready':
                try:
                    run_chain()
                except Exception as error:
                    write_json(STUDY / 'queue-status.json', dict(result, disposition='needs_review', error=str(error)))
                    raise
                replay = read(STUDY / 'full-replay.json')
                write_json(STUDY / 'queue-status.json', dict(result, disposition='verified',
                    finished=datetime.now(timezone.utc).isoformat(), retained_offsets=replay['retained_offsets']))
                return
            if not args.watch or result['disposition'] == 'needs_review':
                return
            time.sleep(30)


if __name__ == '__main__':
    main()

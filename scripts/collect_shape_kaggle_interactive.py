"""Observe the owned Kaggle execution, recover terminal outputs, and audit locally."""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'output/shape-kaggle-interactive-20261123'
TRANSPORT = ROOT / 'output/shape-monitoring-20261123'
SESSION_ID = 373165445
SOURCE_HASH = 'f3621020ba9120c6476683445d3af14c0594796a62fbead8c0b459b3aa8fab16'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TRANSPORT))
from amodal_benchmark.parallel import output_lock


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf8')
    temporary.replace(path)


def classify(observation, launch):
    execution = observation.get('execution', {})
    if (observation.get('session_id') != SESSION_ID or
            execution.get('session_id') != SESSION_ID or
            execution.get('source_sha256') != SOURCE_HASH or
            execution.get('wrapper_pid') != launch['wrapper_pid'] or
            execution.get('execution_mode') != 'interactive' or
            execution.get('benchmark_success') is not False):
        raise ValueError('Execution identity changed')
    child = observation.get('processes', {}).get('child_pid')
    same_child = bool(child and child['start_ticks'] == execution['child_start_ticks'] and child['state'] != 'Z')
    wrapper = observation.get('processes', {}).get('wrapper_pid')
    live_wrapper = bool(wrapper and wrapper['state'] != 'Z')
    if execution.get('complete') is True:
        if type(execution.get('exit_code')) is not int:
            raise ValueError('Terminal exit code absent')
        return 'finishing' if same_child or live_wrapper else 'terminal'
    if same_child and live_wrapper:
        return 'running'
    return 'needs_review'


PROBE = r'''
import json, subprocess
from pathlib import Path
from datetime import datetime, timezone
root = Path('/kaggle/working')
execution = json.loads((root/'interactive-execution.json').read_bytes())
processes = {}
for key in ('wrapper_pid', 'child_pid'):
    pid = execution.get(key)
    try:
        stat = (Path('/proc')/str(pid)/'stat').read_text().rsplit(')', 1)[1].split()
        processes[key] = dict(pid=pid, state=stat[0], start_ticks=stat[19])
    except FileNotFoundError:
        pass
documents = {}
for name in ('training/training.json', 'diagnostic/result.json', 'preflight/preflight.json',
             'native-validation/result.json', 'completion.json', 'shape-interactive-export/export.json'):
    path = root/name
    if path.exists():
        value = json.loads(path.read_bytes())
        if name.startswith('training/'):
            value = {key: value.get(key) for key in ('complete', 'epochs_completed', 'selected_epoch',
                'new_checkpoint_eligible', 'initial_validation')} | {'latest_epoch': value.get('history', [])[-1:]}
        elif name.startswith('native-validation/'):
            value = {key: value.get(key) for key in ('complete', 'selected_epoch', 'geometry', 'objects', 'images')}
        elif name.startswith('shape-interactive-export/'):
            value = {key: value.get(key) for key in ('session_id', 'archive', 'archive_sha256', 'bytes', 'exit_code')}
        documents[name] = value
logs = {}
for path in root.glob('*.log'):
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size-3500))
        logs[path.name] = stream.read().decode(errors='replace')
gpu = subprocess.run(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu','--format=csv,noheader'], capture_output=True, text=True)
print(json.dumps(dict(at=datetime.now(timezone.utc).isoformat(), session_id=373165445,
    execution=execution, processes=processes, documents=documents, logs=logs,
    gpu=gpu.stdout, gpu_exit_code=gpu.returncode)))
'''


def verify_operations():
    manifest = read(STUDY / 'observer-manifest.json')
    for name, expected in manifest['files'].items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT.resolve()) or digest(path) != expected:
            raise ValueError('Observer source changed: ' + name)
    return digest(STUDY / 'observer-manifest.json')


def start_export(session):
    payload = (ROOT / 'scripts/export_shape_kaggle_interactive.py').read_bytes()
    checksum = hashlib.sha256(payload).hexdigest()
    # Exact bytes; Jupyter Contents text normalization must not alter the source.
    code = '''import base64, hashlib, json, subprocess, sys
from pathlib import Path
root = Path('/kaggle/working')
path = root/'shape-interactive-exporter.py'
payload = base64.b64decode(PAYLOAD)
if path.exists():
    assert path.read_bytes() == payload, 'Exporter source changed'
else:
    path.write_bytes(payload)
directory = root/'shape-interactive-export'
directory.mkdir(exist_ok=True)
request = directory/'launch.json'
if request.exists():
    result = json.loads(request.read_bytes())
else:
    with (directory/'export.log').open('xb') as stream:
        child = subprocess.Popen([sys.executable,'-u',str(path)],stdin=subprocess.DEVNULL,
            stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    result = dict(pid=child.pid, start_ticks=Path('/proc/'+str(child.pid)+'/stat').read_text().rsplit(')',1)[1].split()[19],
        session_id=373165445, exporter_sha256=hashlib.sha256(payload).hexdigest())
    request.write_text(json.dumps(result)+'\\n')
print(json.dumps(result))
'''.replace('PAYLOAD', repr(base64.b64encode(payload).decode()))
    result = json.loads(session.execute(code))
    if result['session_id'] != SESSION_ID or result['exporter_sha256'] != checksum:
        raise ValueError('Exporter identity changed')
    save(STUDY / 'export-launch.json', result)


def check_export(session):
    result = session.request('GET', 'api/contents/shape-interactive-export/export.json')
    if result.status_code == 404:
        return None
    if result.status_code != 200:
        raise RuntimeError('Export observation HTTP ' + str(result.status_code))
    body = result.json()
    payload = body['content'] if body['format'] == 'text' else base64.b64decode(body['content'])
    receipt = json.loads(payload)
    if (receipt['session_id'] != SESSION_ID or receipt['archive'] != 'outputs.zip' or
            receipt['exporter_sha256'] != digest(ROOT / 'scripts/export_shape_kaggle_interactive.py')):
        raise ValueError('Export binding changed')
    save(STUDY / 'export.json', receipt)
    return receipt


def extract_verified(archive, receipt, destination):
    if archive.stat().st_size != receipt['bytes'] or digest(archive) != receipt['archive_sha256']:
        raise ValueError('Export archive hash or length mismatch')
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        if len(set(names)) != len(names) or set(names) != set(receipt['files']):
            raise ValueError('Export inventory differs from archive')
        for name in names:
            logical = PurePosixPath(name)
            target = (destination / name).resolve()
            if (logical.is_absolute() or '..' in logical.parts or '\\' in name or ':' in name or
                    not target.is_relative_to(destination.resolve())):
                raise ValueError('Unsafe export member')
            expected = receipt['files'][name]
            if bundle.getinfo(name).file_size != expected['bytes']:
                raise ValueError('Export file length mismatch')
            if target.exists():
                if target.stat().st_size != expected['bytes'] or digest(target) != expected['sha256']:
                    raise ValueError('Existing recovered evidence differs')
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + '.partial')
            checksum = hashlib.sha256()
            with bundle.open(name) as source, temporary.open('wb') as stream:
                for block in iter(lambda: source.read(1024 * 1024), b''):
                    checksum.update(block)
                    stream.write(block)
            if checksum.hexdigest() != expected['sha256']:
                raise ValueError('Export file hash mismatch')
            temporary.replace(target)


def recover(session, receipt):
    archive = STUDY / 'outputs.zip'
    if not archive.exists():
        response = session.request('GET', 'files/shape-interactive-export/outputs.zip', stream=True)
        if response.status_code != 200:
            raise RuntimeError('Export download HTTP ' + str(response.status_code))
        temporary = archive.with_suffix('.partial.zip')
        try:
            with response, temporary.open('wb') as stream:
                for block in response.iter_content(1024 * 1024):
                    stream.write(block)
        except Exception as error:
            raise RuntimeError('Export download failed: ' + type(error).__name__) from None
        if temporary.stat().st_size != receipt['bytes'] or digest(temporary) != receipt['archive_sha256']:
            raise ValueError('Downloaded export differs')
        temporary.replace(archive)
    evidence = STUDY / 'evidence'
    extract_verified(archive, receipt, evidence)
    execution = read(evidence / 'interactive-execution.json')
    if digest(evidence / 'interactive-execution.json') != receipt['execution_sha256']:
        raise ValueError('Recovered execution binding differs')
    if execution.get('exit_code') != 0:
        return dict(disposition='failed_execution_recovered', recovered=True, verified=False,
                    exit_code=execution.get('exit_code'), benchmark_success=False)
    report = STUDY / 'replay.json'
    if not report.exists():
        command = ['wsl', '-e', 'env', 'PYTHONNOUSERSITE=1',
            '/home/hungvudao/.cache/cv-nangcao-evaluator-venv/bin/python',
            '/mnt/c/Users/daovu/Documents/Project/cv_nangcao/scripts/replay_shape_kaggle_interactive.py']
        with (STUDY / 'replay.log').open('wb') as stream:
            completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        if completed.returncode:
            return dict(disposition='replay_failed', recovered=True, verified=False,
                        exit_code=completed.returncode, benchmark_success=False)
    return read(report) | dict(disposition='verified', recovered=True)


def main():
    from kaggle_interactive_client import Session

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    control = STUDY / 'observer-control'
    control.mkdir(exist_ok=True)
    with output_lock(control):
        while True:
            try:
                manifest_hash = verify_operations()
                session = Session()
                if session.session_id != SESSION_ID:
                    raise ValueError('Wrong session')
                observed = json.loads(session.execute(PROBE))
                state = classify(observed, read(STUDY / 'launch.json'))
                save(STUDY / 'observation.json', observed)
                status = dict(at=observed['at'], session_id=SESSION_ID, disposition=state,
                              observer_pid=os.getpid(), manifest_sha256=manifest_hash,
                              training=observed['documents'].get('training/training.json'), benchmark_success=False)
                if state == 'terminal':
                    start_export(session)
                    exported = check_export(session)
                    if exported:
                        status.update(recover(session, exported))
                    else:
                        status['disposition'] = 'exporting'
                save(STUDY / 'observer-status.json', status)
                print(json.dumps(status), flush=True)
                if status['disposition'] not in {'running', 'finishing', 'exporting'} or not args.watch:
                    return
            except (RuntimeError, OSError) as error:
                # Observation errors neither prove termination nor authorize a retry.
                status = dict(at=datetime.now(timezone.utc).isoformat(), session_id=SESSION_ID,
                              disposition='observation_error', error_type=type(error).__name__,
                              benchmark_success=False)
                save(STUDY / 'observer-status.json', status)
                print(json.dumps(status), flush=True)
                if not args.watch:
                    raise RuntimeError('Interactive observation unavailable') from None
            time.sleep(60)


if __name__ == '__main__':
    main()

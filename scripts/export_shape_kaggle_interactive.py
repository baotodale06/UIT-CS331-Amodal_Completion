"""Export terminal outputs from the owned Kaggle session without changing training."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

SESSION_ID = 373165445
ROOT = Path('/kaggle/working')
EXPORT = ROOT / 'shape-interactive-export'
FILES = {
    'interactive-execution.json', 'interactive-main.log', 'interactive-wrapper.log',
    'shape-interactive-main.py', 'shape-interactive-wrapper.py', 'bundle.json',
    'source.bin', 'probability-source.bin', 'probability-source-manifest.json',
    'runtime-packages.json', 'installed-packages.txt', 'completion.json',
    'conda-install.log', 'pinned-install.log', 'diagnostic.log', 'geometry-tests.log',
    'kaggle-contract-tests.log', 'probability-loss-tests.log', 'training-preflight.log',
    'training.log', 'native-validation.log',
}
DIRECTORIES = {'diagnostic', 'preflight', 'training', 'native-validation'}


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def process(pid):
    path = Path('/proc') / str(pid)
    try:
        stat = (path / 'stat').read_text().rsplit(')', 1)[1].split()
        return dict(pid=pid, state=stat[0], start_ticks=stat[19])
    except FileNotFoundError:
        return None


def terminal(receipt):
    if receipt.get('session_id') != SESSION_ID or receipt.get('complete') is not True:
        raise ValueError('Owned terminal execution required')
    if type(receipt.get('exit_code')) is not int:
        raise ValueError('Terminal exit code required')
    child = process(receipt['child_pid'])
    if child and child['start_ticks'] == receipt['child_start_ticks'] and child['state'] != 'Z':
        raise ValueError('Training child remains alive')
    wrapper = process(receipt['wrapper_pid'])
    if wrapper and wrapper['state'] != 'Z':
        raise ValueError('Wrapper remains alive')


def main():
    execution = json.loads((ROOT / 'interactive-execution.json').read_bytes())
    terminal(execution)
    EXPORT.mkdir(exist_ok=True)
    final = EXPORT / 'export.json'
    if final.exists():
        existing = json.loads(final.read_bytes())
        if (existing['session_id'] != SESSION_ID or
                existing['execution_sha256'] != digest(ROOT / 'interactive-execution.json') or
                existing['archive_sha256'] != digest(EXPORT / existing['archive'])):
            raise ValueError('Existing export changed')
        return
    inventory = {}
    paths = sorted(p for p in ROOT.rglob('*') if
                   p.relative_to(ROOT).parts[0] in DIRECTORIES or
                   p.relative_to(ROOT).as_posix() in FILES)
    archive = EXPORT / 'outputs.zip'
    temporary = EXPORT / 'outputs.partial.zip'
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as bundle:
        for path in paths:
            if path.is_symlink():
                raise ValueError('Output symlink is not exportable')
            if not path.is_file():
                continue
            name = path.relative_to(ROOT).as_posix()
            before = path.stat()
            checksum = digest(path)
            bundle.write(path, name)
            after = path.stat()
            if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
                raise ValueError('Output changed during terminal export')
            inventory[name] = dict(sha256=checksum, bytes=after.st_size)
    terminal(execution)
    if inventory['interactive-execution.json']['sha256'] != digest(ROOT / 'interactive-execution.json'):
        raise ValueError('Execution changed during export')
    temporary.replace(archive)
    receipt = dict(at=datetime.now(timezone.utc).isoformat(), session_id=SESSION_ID,
                   archive=archive.name, archive_sha256=digest(archive), bytes=archive.stat().st_size,
                   execution_sha256=inventory['interactive-execution.json']['sha256'],
                   exporter_sha256=digest(Path(__file__)), files=inventory,
                   exit_code=execution['exit_code'], benchmark_success=False)
    temporary_receipt = final.with_suffix('.tmp')
    temporary_receipt.write_text(json.dumps(receipt, indent=2) + '\n')
    temporary_receipt.replace(final)


if __name__ == '__main__':
    main()

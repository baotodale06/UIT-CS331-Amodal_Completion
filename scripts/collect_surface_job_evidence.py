#!/usr/bin/env python3
"""Watch the frozen Surface-PMC jobs and recover finished evidence without remote writes."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
import tempfile
import time


HOST = 'daoqm@slurm.csse.rose-hulman.edu'
PROJECT = 'Vu_Hung/cv_nangcao'
CONFIRMATION = 'output/surface-confirmation128-20261110'
DESIGN = CONFIRMATION + '-design'
FOLLOWUP = 'output/surface-followup-20261110'
TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY',
            'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'}

REMOTE_STATUS = r'''
import hashlib, json, subprocess
from pathlib import Path
from datetime import datetime, timezone
root = Path.home() / 'Vu_Hung/cv_nangcao'
raw = subprocess.check_output(['sacct', '-j', '37812,37814,37843', '-X', '-n', '-P',
    '--format=JobID,State,Elapsed,ExitCode'], text=True)
jobs = {}
for line in raw.splitlines():
    fields = line.strip().split('|')
    if len(fields) >= 4:
        jobs[fields[0]] = dict(state=fields[1].split()[0].rstrip('+'),
            elapsed=fields[2], exit_code=fields[3])
runs = ['pdmc-matched-control-20261110', 'pdmc-surface-progressive-20261110']
record = dict(as_of_utc=datetime.now(timezone.utc).isoformat(), jobs=jobs,
    target_records={name:sum(1 for p in (root/'output'/name).glob('targets/*/*/result.json'))
                    for name in runs})
record['confirmation_candidate_records'] = sum(1 for p in
    (root/'output/surface-confirmation128-20261110/full/candidates').glob('*/seed-*.json'))
print(json.dumps(record))
'''


def status_program(progressive_job, finalizer_job, confirmation_job):
    identifiers = (progressive_job, finalizer_job, confirmation_job)
    if any(type(value) is not int or value <= 0 for value in identifiers):
        raise ValueError('Positive integer Slurm job identifiers required')
    return REMOTE_STATUS.replace('37812,37814,37843', ','.join(map(str, identifiers)))


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def ssh(host, command, **kwargs):
    return subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
        '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=3', host, command],
        check=True, **kwargs)


def remote_json(host, program):
    result = ssh(host, 'python3 -', input=program.encode(), capture_output=True)
    return json.loads(result.stdout)


def valid_member(name, prefixes):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name:
        raise ValueError('Unsafe archive path: ' + name)
    if not any(path == PurePosixPath(prefix) or PurePosixPath(prefix) in path.parents
               for prefix in prefixes):
        raise ValueError('Unexpected archive path: ' + name)
    return path


def recover_archive(archive, destination, prefixes, expected_files=None):
    """Verify all members before extracting; never overwrite different existing bytes."""
    destination = destination.resolve()
    manifest = {}
    with tarfile.open(archive, 'r:gz') as bundle:
        for member in bundle:
            relative = valid_member(member.name, prefixes)
            if member.isdir():
                continue
            if not member.isfile() or member.name in manifest:
                raise ValueError('Nonregular or duplicate archive member: ' + member.name)
            target = (destination / str(relative)).resolve()
            if not target.is_relative_to(destination):
                raise ValueError('Archive destination escapes recovery directory')
            with bundle.extractfile(member) as stream:
                checksum = hashlib.sha256()
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    checksum.update(block)
                value = checksum.hexdigest()
            if expected_files is not None and expected_files.get(member.name) != value:
                raise ValueError('Remote member checksum mismatch: ' + member.name)
            if target.exists() and (not target.is_file() or digest(target) != value):
                raise ValueError('Existing recovery evidence differs: ' + member.name)
            manifest[member.name] = value
        if expected_files is not None and set(manifest) != set(expected_files):
            raise ValueError('Archive membership differs from remote manifest')
    with tarfile.open(archive, 'r:gz') as bundle:
        for member in bundle:
            if not member.isfile():
                continue
            target = destination / member.name
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.extractfile(member) as source, tempfile.NamedTemporaryFile(
                    dir=target.parent, suffix='.partial', delete=False) as output:
                temporary = Path(output.name)
                for block in iter(lambda: source.read(1024 * 1024), b''):
                    output.write(block)
            if digest(temporary) != manifest[member.name]:
                raise ValueError('Extracted checksum mismatch: ' + member.name)
            temporary.replace(target)
    return manifest


def collect_confirmation(host, recovery):
    relative = DESIGN + '/confirmation-evidence.tar.gz'
    expected = remote_json(host, 'import hashlib,json\nfrom pathlib import Path\n'
        f'p=Path.home()/{PROJECT!r}/{relative!r}\n'
        'h=hashlib.sha256()\nwith p.open("rb") as f:\n'
        ' for b in iter(lambda:f.read(1048576),b""):h.update(b)\n'
        'print(json.dumps(dict(sha256=h.hexdigest(),bytes=p.stat().st_size)))\n')
    archive = recovery / 'confirmation-evidence.tar.gz'
    if not archive.exists():
        partial = archive.with_suffix('.gz.partial')
        with partial.open('wb') as output:
            ssh(host, 'cat ~/' + PROJECT + '/' + relative, stdout=output, stderr=subprocess.PIPE)
        if digest(partial) != expected['sha256'] or partial.stat().st_size != expected['bytes']:
            raise ValueError('Confirmation archive checksum mismatch')
        partial.replace(archive)
    if digest(archive) != expected['sha256']:
        raise ValueError('Existing confirmation archive differs from remote')
    members = recover_archive(archive, recovery / 'confirmation', [CONFIRMATION, DESIGN])
    result = json.loads((recovery / 'confirmation' / CONFIRMATION / 'analysis/confirmation.json').read_text())
    if not result['replay_passed'] or result['images'] != 128:
        raise ValueError('Confirmation lacks complete replay')
    frozen = recovery / 'confirmation' / DESIGN / 'selection-only-freeze.json'
    if result['freeze_sha256'] != digest(frozen) or digest(frozen) != '711f49db26e19d0655c64b59682d9ae50c0841d38071096ad9dc7f83d733d4c4':
        raise ValueError('Confirmation does not match the immutable pre-generation freeze')
    write_json(recovery / 'confirmation-receipt.json', dict(remote_archive=relative,
        **expected, member_sha256=members, result=result,
        recovered_at=datetime.now(timezone.utc).isoformat()))


def collect_progressive(host, recovery):
    report = FOLLOWUP + '/progressive-comparison'
    runs = ['output/pdmc-matched-control-20261110', 'output/pdmc-surface-progressive-20261110']
    program = 'import hashlib,json\nfrom pathlib import Path\n' + f'root=Path.home()/{PROJECT!r}\n'
    program += f'files=list((root/{report!r}).rglob("*"))\n'
    program += f'for name in {runs!r}:\n'
    program += ' files += list((root/name).rglob("*.json"))\n'
    program += 'files.append(root/"output/surface-followup-20261110-design/progressive-experiments.csv")\n'
    program += 'print(json.dumps({p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()}))\n'
    expected = remote_json(host, program)
    if report + '/comparison.json' not in expected or report + '/REPORT.md' not in expected:
        raise ValueError('Completed finalizer lacks comparison artifacts')
    prefixes = [report, *runs, 'output/surface-followup-20261110-design/progressive-experiments.csv']
    for name in expected:
        valid_member(name, prefixes)
    archive = recovery / 'progressive-evidence.tar.gz'
    if not archive.exists():
        partial = archive.with_suffix('.gz.partial')
        with partial.open('wb') as output:
            ssh(host, 'cd ~/' + PROJECT + ' && tar --null -czf - -T -',
                input=b'\0'.join(name.encode() for name in sorted(expected)) + b'\0',
                stdout=output, stderr=subprocess.PIPE)
        recover_archive(partial, recovery / 'progressive', prefixes, expected)
        partial.replace(archive)
    else:
        recover_archive(archive, recovery / 'progressive', prefixes, expected)
    write_json(recovery / 'progressive-receipt.json', dict(archive_sha256=digest(archive),
        member_sha256=expected, recovered_at=datetime.now(timezone.utc).isoformat(),
        scope='Complete comparison reports and run JSON metadata; native PNGs remain on the cluster.'))


def successful(jobs, job):
    row = jobs.get(job, {})
    return row.get('state') == 'COMPLETED' and row.get('exit_code') == '0:0'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default=HOST)
    parser.add_argument('--recovery-dir', type=Path, default=Path('output/surface-job-recovery-20261110'))
    parser.add_argument('--poll-seconds', type=int, default=60)
    parser.add_argument('--progressive-job', type=int, default=37812)
    parser.add_argument('--finalizer-job', type=int, default=37814)
    parser.add_argument('--confirmation-job', type=int, default=37843)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if args.poll_seconds < 60:
        parser.error('Poll interval must be at least 60 seconds')
    program = status_program(args.progressive_job, args.finalizer_job, args.confirmation_job)
    recovery = args.recovery_dir.resolve()
    recovery.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            record = remote_json(args.host, program)
            record['watched_jobs'] = dict(progressive=args.progressive_job,
                finalizer=args.finalizer_job, confirmation=args.confirmation_job)
            for job, name, collector in [(str(args.confirmation_job), 'confirmation', collect_confirmation),
                                         (str(args.finalizer_job), 'progressive', collect_progressive)]:
                if successful(record['jobs'], job) and not (recovery / (name + '-receipt.json')).exists():
                    collector(args.host, recovery)
            record['recovered'] = {name:(recovery / (name + '-receipt.json')).exists()
                                   for name in ('confirmation', 'progressive')}
            failed = {job:row for job,row in record['jobs'].items()
                      if row['state'] in TERMINAL and not successful(record['jobs'], job)}
            record['failed_jobs_requiring_review'] = failed
            record['complete'] = all(record['recovered'].values())
            write_json(recovery / 'status.json', record)
            print(json.dumps(record), flush=True)
            if record['complete']:
                return
            if failed:
                raise SystemExit('A Slurm job failed; scientific retry requires review. Remote jobs were not changed.')
        except (OSError, ValueError, subprocess.SubprocessError, tarfile.TarError) as error:
            record = dict(as_of_utc=datetime.now(timezone.utc).isoformat(),
                observation_or_collection_error=str(error), remote_jobs_changed=False)
            write_json(recovery / 'last-error.json', record)
            print(json.dumps(record), flush=True)
            if args.once or isinstance(error, (ValueError, tarfile.TarError)):
                raise
        if args.once:
            return
        time.sleep(args.poll_seconds)


if __name__ == '__main__':
    main()

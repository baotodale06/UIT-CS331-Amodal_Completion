#!/usr/bin/env python3
"""Build the annotated report only after checksum-verified complete exports arrive."""
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--once', action='store_true', help='Check readiness once, without waiting')
    parser.add_argument('--poll-seconds', type=float, default=45)
    args = parser.parse_args()
    if args.poll_seconds < 10:
        parser.error('Poll interval must be at least ten seconds')
    design = ROOT/'output/surface-annotated-20261112-design'
    active = json.loads((design/'active-study.json').read_text())
    study = ROOT/active['study']
    recovery = ROOT/'output/surface-annotated-20261112-recovery'
    state_path = recovery/'report-workflow-status.json'
    requirements = [recovery/(study.name+'-'+tag)/'recovery-receipt.json'
                    for tag in ('cohort-native', 'pool-metadata')]
    started = datetime.now(timezone.utc).isoformat()
    state = dict(started_at=started, process_id=os.getpid(), study=active['study'], stages=[],
                 source_sha256=sha256(Path(__file__)))
    while True:
        missing = [str(path.relative_to(ROOT)) for path in requirements
                   if not path.is_file() or not json.loads(path.read_text()).get('verified')]
        state.update(phase='waiting_for_verified_exports', missing=missing,
                     updated_at=datetime.now(timezone.utc).isoformat())
        write_json(state_path, state)
        if not missing:
            break
        if args.once:
            print(json.dumps(dict(phase=state['phase'], missing=missing)))
            return
        time.sleep(args.poll_seconds)
    # The independent audit rehashes native artifacts and recomputes selection,
    # correction and paired statistics. Receipt existence alone is not acceptance.
    replay = '/home/hungvudao/.cache/surface-replay-py310/bin/python'
    if sys.platform != 'win32':
        raise RuntimeError('This local workflow requires Windows plus the pinned WSL replay environment')
    wsl_root = '/mnt/'+ROOT.drive[0].lower()+'/'+ROOT.as_posix().split(':',1)[1].lstrip('/')
    stages = [
        ('science-freeze', [sys.executable, 'scripts/check_annotated_surface_freeze.py', '--design', str(design)]),
        ('reference-amendment', [sys.executable, 'scripts/check_annotated_freshrefs.py']),
        ('cohort-audit', ['wsl.exe', '--cd', wsl_root, '-e', replay, 'scripts/audit_annotated_surface.py']),
        ('pool-audit', ['wsl.exe', '--cd', wsl_root, '-e', replay, 'scripts/audit_annotated_surface_pools.py']),
        ('build', [sys.executable, 'scripts/build_annotated_surface_report.py']),
        ('validate', [sys.executable, 'scripts/validate_annotated_surface_report.py']),
        ('package', [sys.executable, 'scripts/package_annotated_surface.py'])]
    logs = recovery/'report-workflow-logs'
    logs.mkdir(exist_ok=True)
    state['recovery_receipts'] = {str(path.relative_to(ROOT)):sha256(path) for path in requirements}
    for name, command in stages:
        stage = dict(name=name, command=command, started_at=datetime.now(timezone.utc).isoformat())
        state.update(phase='running_'+name, updated_at=stage['started_at'])
        state['stages'].append(stage)
        write_json(state_path,state)
        print(stage['started_at'],name,flush=True)
        log = logs/(name+'.log')
        with log.open('wb') as stream:
            result = subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        stage.update(exit_code=result.returncode,finished_at=datetime.now(timezone.utc).isoformat(),
                     log=str(log.relative_to(ROOT)),log_sha256=sha256(log))
        if result.returncode:
            state.update(phase='failed_'+name,updated_at=stage['finished_at'])
            write_json(state_path,state)
            raise RuntimeError('Report workflow failed at '+name+'; inspect '+str(log))
    state.update(phase='validated_pending_visual_review',updated_at=datetime.now(timezone.utc).isoformat(),
                 report=str((study/'report/REPORT.pdf').relative_to(ROOT)),
                 archive=str((study/'annotated-surface-results.zip').relative_to(ROOT)))
    write_json(state_path,state)
    print(json.dumps(dict(phase=state['phase'],report=state['report'],archive=state['archive'])))


if __name__ == '__main__':
    main()

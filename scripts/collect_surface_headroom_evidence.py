#!/usr/bin/env python3
"""Recover independently hashed saved-pool headroom evidence after Slurm completion."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

from collect_surface_job_evidence import HOST, PROJECT, TERMINAL, digest, recover_archive, remote_json, ssh, write_json

RUN = 'output/surface-candidate-headroom-20261111'
DESIGN = RUN+'-design'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=int, required=True)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if args.job <= 0:
        raise ValueError('Positive Slurm job required')
    recovery = Path(RUN+'-recovery')
    freeze_sha = digest(Path(DESIGN)/'freeze.json')
    status_program = f'''from pathlib import Path
import subprocess,json
root=Path.home()/{PROJECT!r}
raw=subprocess.check_output(['sacct','-j',{str(args.job)!r},'-X','-n','-P','--format=JobID,State,ExitCode,Elapsed'],text=True)
fields=next(line.split('|') for line in raw.splitlines() if line.startswith({str(args.job)+'|'!r}))
print(json.dumps(dict(job_id=int(fields[0]),state=fields[1].split()[0].rstrip('+'),exit_code=fields[2],elapsed=fields[3],records=sum(1 for p in (root/{RUN!r}).glob('targets/*/*/headroom.json')))))
'''
    while True:
        try:
            state = remote_json(HOST, status_program)
            state['as_of_utc'] = datetime.now(timezone.utc).isoformat()
            state['complete'] = False
            if state['state'] == 'COMPLETED' and state['exit_code'] == '0:0':
                program = f'''import hashlib,json
from pathlib import Path
root=Path.home()/{PROJECT!r}
p=root/{(DESIGN+'/evidence.tar.gz')!r}
manifest=json.loads((root/{(DESIGN+'/evidence-manifest.json')!r}).read_text())
print(json.dumps(dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size,member_sha256=manifest)))
'''
                expected = remote_json(HOST, program)
                recovery.mkdir(parents=True, exist_ok=True)
                archive = recovery/'evidence.tar.gz'
                if not archive.exists():
                    partial = recovery/'evidence.tar.gz.partial'
                    with partial.open('wb') as stream:
                        ssh(HOST, 'cat ~/'+PROJECT+'/'+DESIGN+'/evidence.tar.gz', stdout=stream, stderr=subprocess.PIPE)
                    if digest(partial) != expected['sha256']:
                        raise ValueError('Headroom archive checksum differs')
                    partial.replace(archive)
                if digest(archive) != expected['sha256']:
                    raise ValueError('Existing headroom archive changed')
                members = recover_archive(archive, recovery/'evidence',
                    [RUN, DESIGN, 'scripts', 'amodal_benchmark'], expected['member_sha256'])
                summary = json.loads((recovery/'evidence'/RUN/'analysis/summary.json').read_text())
                if summary['freeze_sha256'] != freeze_sha or summary['targets'] != 2565 or summary['scored_candidates'] != 7664:
                    raise ValueError('Incomplete or changed headroom evidence')
                write_json(recovery/'receipt.json', dict(**expected, checked_members=len(members),
                    recovered_at=state['as_of_utc'], result=summary, native_pngs_recovered=False))
                state['complete'] = True
            write_json(recovery/'status.json', state)
            print(json.dumps(state), flush=True)
            if state['complete']:
                return
            if state['state'] in TERMINAL:
                raise SystemExit('Headroom job failed; preserve evidence: '+json.dumps(state))
        except (OSError, subprocess.SubprocessError) as error:
            write_json(recovery/'last-error.json', dict(as_of_utc=datetime.now(timezone.utc).isoformat(), error=str(error)))
            print(str(error), flush=True)
            if args.once:
                raise
        if args.once:
            return
        time.sleep(60)


if __name__ == '__main__':
    main()

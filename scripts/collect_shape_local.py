"""Observe the isolated local run and audit completed results without restarting it."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from amodal_benchmark.dataset import sha256,write_json
from amodal_benchmark.parallel import output_lock
from scripts.verify_shape_local import STUDY,replay


def collect():
    execution = json.loads((STUDY/'execution-v2.json').read_bytes())
    result = dict(at=datetime.now(timezone.utc).isoformat(),benchmark_success=False,
        execution_sha256=sha256(STUDY/'execution-v2.json'))
    commands = execution['commands']
    failed = [row for row in commands if row.get('exit_code',0)!=0]
    if failed:
        result.update(disposition='failed',command=failed[-1])
    elif execution['complete']:
        if (len(commands)!=7 or any(row.get('exit_code')!=0 for row in commands)
                or [row['command'][-1] for row in commands[-4:]]!=['calibrate','preflight','train','native']):
            raise ValueError('Execution chain is incomplete')
        verified = replay(STUDY/'run-v2')
        path = STUDY/'native-replay.json'
        if path.exists():
            if json.loads(path.read_bytes())!=verified:
                raise ValueError('Preserve differing completed audit')
        else:
            write_json(path,verified)
        result.update(verified,disposition='verified',native_replay_sha256=sha256(path))
    else:
        result.update(disposition='waiting',last_command=commands[-1]['command'] if commands else None)
        training = STUDY/'run-v2/training/training.json'
        if training.exists():
            try:
                receipt = json.loads(training.read_bytes())
            except json.JSONDecodeError:
                result['training_receipt_state'] = 'write_in_progress'
            else:
                result.update(epochs_completed=receipt.get('epochs_completed',0),
                    selected_epoch=receipt.get('selected_epoch',0),training_complete=receipt.get('complete',False))
    write_json(STUDY/'collection-status.json',result)
    print(json.dumps({key:value for key,value in result.items()
        if key not in ('original_quadro_pixel_differences','geometry_vs_local','geometry_vs_quadro','calibration_geometry')}),flush=True)
    return result['disposition']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch',action='store_true')
    args = parser.parse_args()
    control = STUDY/'collector-lock'
    control.mkdir(parents=True,exist_ok=True)
    with output_lock(control):
        while True:
            try:
                state = collect()
            except (OSError,json.JSONDecodeError) as exc:
                if not args.watch:
                    raise
                print(json.dumps(dict(at=datetime.now(timezone.utc).isoformat(),observation_error=str(exc))),flush=True)
            else:
                if state=='verified' and not (STUDY/'report').exists():
                    from scripts.build_shape_local_report import build
                    print(str(build(STUDY/'report')),flush=True)
                if not args.watch or state in ('verified','failed'):
                    return
            time.sleep(60)


if __name__=='__main__':
    main()

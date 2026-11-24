"""Independently replay recovered interactive outputs with the established CPU runtime."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import sha256, write_json
from scripts.collect_shape_probability_training import verify_sources, replay
from scripts.verify_shape_v2_metadata import equal

STUDY = ROOT / 'output/shape-kaggle-interactive-20261123'


def main():
    evidence = STUDY / 'evidence'
    read = lambda path: json.loads(path.read_bytes())
    exported = read(STUDY / 'export.json')
    launch = read(STUDY / 'launch.json')
    request = read(STUDY / 'launch-request.json')
    for name, expected in exported['files'].items():
        path = (evidence / name).resolve()
        if not path.is_relative_to(evidence.resolve()):
            raise ValueError('Unsafe recovered path')
        equal(path.stat().st_size, expected['bytes'], 'recovered length')
        equal(sha256(path), expected['sha256'], 'recovered bytes')
    execution = read(evidence / 'interactive-execution.json')
    equal(execution['session_id'], 373165445, 'interactive session')
    equal([execution['complete'], execution['exit_code'], execution['execution_mode']], [True, 0, 'interactive'], 'terminal execution')
    equal(execution['wrapper_pid'], launch['wrapper_pid'], 'owned wrapper')
    equal(sha256(evidence / 'shape-interactive-main.py'), launch['source_sha256'], 'frozen notebook')
    equal(sha256(evidence / 'shape-interactive-wrapper.py'), request['wrapper_sha256'], 'executed wrapper')
    equal(sha256(evidence / 'interactive-execution.json'), exported['execution_sha256'], 'execution export binding')
    completion = read(evidence / 'completion.json')
    equal(completion['run'], 'ez4kacchan/shape-only-probability-training-20261123:v1', 'logical notebook run')
    equal(completion['complete'], True, 'completed training')
    equal(completion['probability_source_sha256'], sha256(ROOT / 'output/shape-proper-loss-20261123/probability-source.bin'), 'completion sources')
    equal(completion['benchmark_success'], False, 'validation scope')
    equal(completion['training_receipt_sha256'], sha256(evidence / 'training/training.json'), 'completion training')
    equal(completion['selected_checkpoint_sha256'], sha256(evidence / 'training/best.pt'), 'completion checkpoint')
    equal(completion['native_result_sha256'], sha256(evidence / 'native-validation/result.json'), 'completion native')
    verify_sources(evidence)
    result = replay(evidence)
    equal(completion['selected_epoch'], result['selected_epoch'], 'completion epoch')
    equal(completion['new_checkpoint_eligible'], result['new_checkpoint_eligible'], 'completion eligibility')
    result.update(at=datetime.now(timezone.utc).isoformat(), session_id=373165445,
                  execution_mode='interactive', archive_sha256=exported['archive_sha256'],
                  independent_replay_source_sha256=sha256(Path(__file__)), benchmark_success=False)
    write_json(STUDY / 'replay.json', result)
    (STUDY / 'REPORT.md').write_text(
        '# Kaggle interactive shape-only probability training\n\n'
        f"Session 373165445 completed all 20 epochs on two T4 GPUs. Selected epoch: {result['selected_epoch']}. "
        f"Eligible learned checkpoint: {result['new_checkpoint_eligible']}.\n\n"
        'Independent source, completion and native geometry replay passed for 1,264 validation objects '
        'in 217 images. Export hashes bind all recovered artifacts to this interactive execution.\n\n'
        'This is iterative, validation-selected research. The queued batch version is a separate execution. '
        'No full-benchmark improvement is established, and the four-metric 3% goal remains unachieved.\n', encoding='utf8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()

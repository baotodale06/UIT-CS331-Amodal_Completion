"""Bindings for a fixed final-epoch diagnostic, separate from model selection."""
import hashlib
import json
from pathlib import Path
import zipfile

from amodal_benchmark.dataset import sha256

ROOT = Path(__file__).resolve().parents[1]
CONFIG = 'configs/research/shape-probability-final-diagnostic-20261123.json'
POLICY_SHA256 = 'f28370d1bd703b147a83c0ab710cdd43b062c1ea2ae626f1195201c59388fe29'
LOGS = ['calibrate.log', 'calibration-replay.log', 'preflight.log', 'train.log',
        'native.log', 'native-replay.log', 'report.log']
read = lambda path: json.loads(Path(path).read_bytes())


def validate_parent(execution, training, replay, policy):
    if (execution.get('complete') is not True
            or [row.get('log') for row in execution.get('commands', [])] != LOGS
            or any(row.get('exit_code') != 0 for row in execution['commands'])):
        raise ValueError('All seven parent execution phases must finish successfully')
    required = dict(schema=8, method='shape-probability-workers4', platform='LocalWSL',
                    complete=True, epochs_completed=20, holdout_used=False, benchmark_used=False)
    if any(training.get(key) != value for key, value in required.items()):
        raise ValueError('Completed isolated schema8 parent training required')
    if (policy['fixed_epoch'] != 20 or policy['logit_biases'] != [0., .5, 1., 1.5, 2.]
            or [row['epoch'] for row in training['history']] != list(range(1, 21))):
        raise ValueError('The fixed final epoch or offset grid changed')
    if (replay.get('verified') is not True or replay.get('benchmark_success') is not False
            or replay.get('source_archive_sha256') != policy['parent_source_archive_sha256']
            or replay.get('selected_epoch') != training['selected_epoch']
            or replay.get('new_checkpoint_eligible') != training['new_checkpoint_eligible']
            or replay.get('checkpoint_sha256') != training['best_sha256']):
        raise ValueError('Completed independent parent replay required')


def validate_checkpoint(saved, training):
    if (saved.get('epoch') != 20 or saved.get('contract') != training['contract']
            or saved.get('history') != training['history'] or not saved.get('model')):
        raise ValueError('The checkpoint must be the fixed completed epoch20, not the selected checkpoint')


def verify_sources(manifest_path):
    manifest_path = Path(manifest_path)
    manifest = read(manifest_path)
    archive = manifest_path.parent / 'diagnostic-source.zip'
    if sha256(archive) != manifest['zip_sha256']:
        raise ValueError('Diagnostic source archive changed')
    required = {CONFIG, 'scripts/shape_probability_final_contract.py',
                'scripts/diagnose_shape_probability_final.py',
                'scripts/verify_shape_probability_final.py',
                'scripts/queue_shape_probability_final.py',
                'tests/test_shape_probability_final.py'}
    if not required.issubset(manifest['files']):
        raise ValueError('Diagnostic source inventory incomplete')
    with zipfile.ZipFile(archive) as bundle:
        if sorted(bundle.namelist()) != sorted(manifest['files']):
            raise ValueError('Diagnostic source archive inventory differs')
        for name, expected in manifest['files'].items():
            path = (ROOT / name).resolve()
            if (not path.is_relative_to(ROOT.resolve()) or sha256(path) != expected
                    or hashlib.sha256(bundle.read(name)).hexdigest() != expected):
                raise ValueError('Diagnostic source differs: ' + name)
    if sha256(ROOT / CONFIG) != POLICY_SHA256:
        raise ValueError('The prespecified diagnostic policy changed')
    policy = read(ROOT / CONFIG)
    observed = ROOT / 'output/shape-probability-final-diagnostic-20261123/observed-training.json'
    if sha256(observed) != policy['observed_training_receipt_sha256']:
        raise ValueError('The pre-diagnostic observation changed')
    return sha256(manifest_path)


def load_parent(parent, policy):
    parent = Path(parent)
    training_path = parent / 'run/training/training.json'
    training = read(training_path)
    replay = read(parent / 'native-replay.json')
    validate_parent(read(parent / 'execution.json'), training, replay, policy)
    if read(parent / 'queue-status.json')['disposition'] != 'verified':
        raise ValueError('The parent observer has not completed its audit')
    bindings = {
        'parent_training_sha256': sha256(training_path),
        'parent_native_replay_sha256': sha256(parent / 'native-replay.json'),
        'parent_execution_sha256': sha256(parent / 'execution.json'),
        'final_checkpoint_sha256': sha256(parent / 'run/training/last.pt'),
        'parent_source_archive_sha256': sha256(parent / 'training-source.zip'),
        'parent_operational_manifest_sha256': sha256(parent / 'operational-manifest.json'),
        'parent_report_receipt_sha256': sha256(parent / 'report/receipt.json'),
    }
    if (bindings['parent_training_sha256'] != replay['training_sha256']
            or bindings['final_checkpoint_sha256'] != training['last_sha256']
            or bindings['parent_source_archive_sha256'] != policy['parent_source_archive_sha256']
            or bindings['parent_operational_manifest_sha256'] != policy['parent_operational_manifest_sha256']
            or sha256(parent / 'run/native/result.json') != replay['native_sha256']):
        raise ValueError('Completed parent evidence or final checkpoint changed')
    report = read(parent / 'report/receipt.json')
    operations = read(parent / 'operational-manifest.json')
    if (report['replay_sha256'] != bindings['parent_native_replay_sha256']
            or report['report_sha256'] != sha256(parent / 'report/REPORT.md')
            or report['source_sha256'] != operations['files']['scripts/build_shape_probability_workers4_report.py']
            or report.get('benchmark_success') is not False):
        raise ValueError('Completed parent report binding changed')
    return training, bindings


def variant_name(bias):
    return 'final_bias_' + format(bias, 'g')


def pixel_hash(prediction):
    return hashlib.sha256(prediction.astype('uint8').tobytes()).hexdigest()


def retained_offsets(summaries, floor, biases):
    return [bias for bias in biases
            if all(summaries[variant_name(bias)][key] >= floor[key]
                   for key in ('full_iou', 'hidden_iou'))]

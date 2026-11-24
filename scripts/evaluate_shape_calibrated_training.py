"""Export and independently repeat the selected Kaggle-native shape masks."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amodal_benchmark.dataset import sha256, write_json
from amodal_benchmark.shape_v2 import make_network
from scripts.shape_calibrated_training import evaluate_native
from scripts.shape_robust_data import aggregate


def main():
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('training', 'data', 'sam-validation', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    read = lambda path: json.loads(path.read_bytes())
    training = read(args.training/'training.json')
    selected = read(args.training/'best-validation-native.json')
    contract = training['contract']
    if (training.get('schema') != 9 or training.get('complete') is not True
            or training['epochs_completed'] != 20 or training.get('preflight') is not False
            or training['holdout_used'] or training['benchmark_used']
            or selected['contract'] != contract or selected['epoch'] != training['selected_epoch']):
        raise ValueError('Completed separate Kaggle training required')
    for name, value in contract['sources'].items():
        if sha256(ROOT/name) != value:
            raise ValueError('Training source changed: '+name)
    if (sha256(args.training/'best.pt') != training['best_sha256']
            or sha256(args.training/'best-validation-native.json') != training['best_validation_sha256']
            or sha256(args.data/'manifest.json') != contract['data_sha256']
            or sha256(args.sam_validation/'result.json') != contract['validation_sam_result_sha256']):
        raise ValueError('Frozen training or input binding changed')
    if not torch.cuda.is_available() or 'T4' not in torch.cuda.get_device_name():
        raise ValueError('Selected repeat must use Kaggle T4')
    torch.set_num_threads(2)
    checkpoint = torch.load(args.training/'best.pt', map_location='cpu', weights_only=False)
    if checkpoint['contract'] != contract or checkpoint['epoch'] != training['selected_epoch']:
        raise ValueError('Selected checkpoint changed')
    model = make_network(pretrained=False).cuda()
    model.load_state_dict(checkpoint['model'], strict=True)
    manifest = read(args.data/'manifest.json')
    records = [row for row in manifest['rows'] if row['split'] == 'validation']
    args.output.mkdir(exist_ok=False)
    rows = evaluate_native(model, args.data, records, args.sam_validation, 'cuda',
                           mask_archive=args.output/'selected-masks.zip')
    if rows != selected['rows'] or aggregate(rows) != selected['aggregate']:
        raise ValueError('Selected native pixel hashes or scores differ from training')
    result = dict(complete=True, verified=True, platform='Kaggle', gpu=torch.cuda.get_device_name(),
        selected_epoch=selected['epoch'], new_checkpoint_eligible=training['new_checkpoint_eligible'],
        rows=rows, geometry=aggregate(rows), contract=contract,
        selected_masks_sha256=sha256(args.output/'selected-masks.zip'),
        training_sha256=sha256(args.training/'training.json'), checkpoint_sha256=training['best_sha256'],
        selected_native_sha256=sha256(args.training/'best-validation-native.json'),
        exact_training_pixel_replay=True, benchmark_admitted=False, benchmark_success=False)
    write_json(args.output/'result.json', result)
    print(json.dumps({key: result[key] for key in ('complete','selected_epoch','geometry','benchmark_success')}))


if __name__ == '__main__':
    main()

"""Report the separately verified Kaggle probability-loss study."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from amodal_benchmark.dataset import sha256,write_json


def main():
    study = ROOT/'output/shape-proper-loss-20261123'
    replay = json.loads((study/'attempt-1-replay.json').read_bytes())
    if replay.get('verified') is not True or replay.get('disposition')!='verified':
        raise ValueError('Independent native replay must complete before reporting')
    evidence = study/'attempt-1'
    training = json.loads((evidence/'training/training.json').read_bytes())
    if sha256(evidence/'training/training.json')!=replay['training_sha256']:
        raise ValueError('Verified training evidence changed')
    report = study/'report'
    report.mkdir(exist_ok=False)
    lines = ['# Shape-only probability-loss Kaggle study','',
        'Separate two-T4 experiment; no Surface ranker, appearance loss, RGB correction or combined method.',
        'Loss: per-object Bernoulli negative log likelihood outside observable input support, using true amodal labels.',
        'Initialization, split, seed, augmentation, 20 epochs, learning rate and selection floors match the constrained Kaggle protocol.','',
        f"Completed epochs: {training['epochs_completed']}; selected epoch: {replay['selected_epoch']}; eligible learned checkpoint: {replay['new_checkpoint_eligible']}.",
        'Native validation uses all 1,264 objects / 217 images. Selected masks replayed pixelwise, then were independently rescored locally.','',
        '| Native endpoint | Original v2 | Selected probability model | Absolute change |',
        '| --- | ---: | ---: | ---: |']
    for metric,values in replay['geometry'].items():
        lines.append(f"| {metric} | {values['v2_sam']:.9f} | {values['probability_sam']:.9f} | {values['absolute_change']:+.9f} |")
    lines += ['', 'Epoch zero cannot be promoted. An eligible learned checkpoint still requires separate inference admission, '
        'validation on the inference GPU, a candidate-selection smoke and the complete 2,565-target benchmark.',
        'Validation was reused during method development; this is not untouched confirmation. No benchmark or reserved holdout assets entered training.',
        'The >=3% relative improvement goal on CLIP, LPIPS, VGG Feature Similarity and SSIM remains unachieved.']
    (report/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write_json(report/'provenance.json',dict(replay_sha256=sha256(study/'attempt-1-replay.json'),
        training_sha256=replay['training_sha256'],report_source_sha256=sha256(Path(__file__)),benchmark_success=False))
    print(str(report/'REPORT.md'))


if __name__=='__main__':
    main()

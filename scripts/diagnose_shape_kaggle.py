"""Compare native preprocessing and cuDNN arithmetic without optimizing weights."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
from PIL import Image, features
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from amodal_benchmark.dataset import sha256,write_json
from amodal_benchmark.shape_v2 import make_network,observation_tensor,visible_box
from amodal_benchmark.shape_v2_runtime import training_receipt
from scripts.shape_robust_data import native_canvas,geometry


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--sam-validation',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--input-only',action='store_true')
    args=p.parse_args()
    manifest=json.loads((args.data/'manifest.json').read_bytes())
    baseline=json.loads((args.sam_validation/'result.json').read_bytes())
    training_receipt(args.checkpoint)
    selected=[row for row in manifest['rows'] if row['split']=='validation']
    images=set(sorted({row['image_id'] for row in selected})[:12])
    groups=defaultdict(list)
    for row in selected:
        if row['image_id'] in images:
            groups[row['image_id']].append(row)
    reference={row['id']:row['sam_input'] for row in baseline['rows']}
    args.output.mkdir(exist_ok=False)
    frames={}
    for image_id,objects in sorted(groups.items()):
        with Image.open(args.data/objects[0]['image']) as image:
            rgb=np.array(image.convert('RGB'))
        for row in objects:
            with Image.open(args.sam_validation/'objects'/row['id']/'sam-visible.png') as image:
                support=np.array(image)>0
            box=visible_box(support,2.)
            observed=np.array(Image.fromarray(rgb).crop(box))
            cropped=np.array(Image.fromarray(support).crop(box))
            inputs=observation_tensor(observed,cropped)
            frames[row['id']]=dict(id=row['id'],image_id=image_id,image_sha256=sha256(args.data/row['image']),
                rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),input_sha256=hashlib.sha256(inputs.tobytes()).hexdigest())
    write_json(args.output/'inputs.json',dict(rows=frames,jpeg=features.version_codec('jpg'),
        turbo=features.version_feature('libjpeg_turbo'),torch=str(torch.__version__),
        cudnn=torch.backends.cudnn.version(),numpy=np.__version__,torch_config=torch.__config__.show()))
    if args.input_only:
        return
    assert torch.cuda.is_available()
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    model=make_network(pretrained=False).cuda().eval()
    model.load_state_dict(saved['model'],strict=True)
    runs=[]
    for label,deterministic in (('original',False),('repeat',False),('deterministic',True)):
        torch.backends.cudnn.deterministic=deterministic
        records=[]
        with torch.inference_mode(),zipfile.ZipFile(args.output/(label+'-masks.zip'),'x',zipfile.ZIP_DEFLATED) as archive:
            for image_id,objects in sorted(groups.items()):
                with Image.open(args.data/objects[0]['image']) as image:
                    rgb=np.array(image.convert('RGB'))
                for row in objects:
                    with Image.open(args.sam_validation/'objects'/row['id']/'sam-visible.png') as image:
                        support=np.array(image)>0
                    with Image.open(args.data/row['mask']) as image:
                        encoded=np.array(image)
                    truth,visible=(encoded&1)>0,(encoded&2)>0
                    box=visible_box(support,2.)
                    observed=np.array(Image.fromarray(rgb).crop(box))
                    cropped=np.array(Image.fromarray(support).crop(box))
                    inputs=torch.from_numpy(observation_tensor(observed,cropped)).unsqueeze(0).cuda()
                    logits=model(inputs)
                    probability=torch.nn.functional.interpolate(logits,cropped.shape,mode='bilinear',align_corners=False).sigmoid()
                    prediction=(native_canvas(probability[0,0].cpu().numpy(),box,truth.shape)>=.5)|support
                    metrics=geometry(prediction,truth,visible)
                    import io
                    stream=io.BytesIO()
                    Image.fromarray(prediction.astype(np.uint8)*255).save(stream,format='PNG')
                    archive.writestr(row['id']+'.png',stream.getvalue())
                    records.append(dict(id=row['id'],geometry=metrics,baseline=reference[row['id']],
                        different_geometry=metrics!=reference[row['id']],prediction_sha256=hashlib.sha256(stream.getvalue()).hexdigest()))
        runs.append(dict(label=label,cudnn_deterministic=deterministic,rows=records,
            objects=len(records),different_objects=sum(row['different_geometry'] for row in records)))
    equal_repeat=all(a['prediction_sha256']==b['prediction_sha256'] for a,b in zip(runs[0]['rows'],runs[1]['rows']))
    write_json(args.output/'result.json',dict(gpu=torch.cuda.get_device_name(),cuda=torch.version.cuda,
        checkpoint_sha256=sha256(args.checkpoint),runs=runs,repeat_pixelwise_identical=equal_repeat,
        source_sha256=sha256(Path(__file__)),training_performed=False,benchmark_success=False))
    print(json.dumps(dict(objects=len(frames),runs={row['label']:row['different_objects'] for row in runs},repeat_identical=equal_repeat)))


if __name__=='__main__':
    main()

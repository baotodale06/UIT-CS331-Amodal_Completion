"""Kiểm thử CPU các contract tránh leakage, chọn candidate và căn tọa độ."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
import numpy as np
from PIL import Image
from main import parse_args, QueryObject, run_iteration
from pmc.c2f_inputs import visible_crop_bounds
from pmc.geometry import create_canvas, crop
from pmc.metrics import mask_metrics, hidden_rgb_metrics
from pmc.runtime import ModelManager
from pmc.scoring import ScoreContext, shape_score, select_candidate, target_mask
from pmc.shape_predictor import save_json


class Contracts(unittest.TestCase):
    def test_crop_and_canvas(self):
        visible = np.zeros((13, 17), np.uint8)
        visible[0:3, 0:2] = 1
        bounds = visible_crop_bounds(visible)
        self.assertEqual(bounds[0], 0)
        self.assertEqual(bounds[2], 0)
        canvas = create_canvas(visible, 6, 0)
        arrays, r0, r1, c0, c1 = crop([canvas, canvas.copy()], visible, canvas)
        np.testing.assert_array_equal(*arrays)
        self.assertGreaterEqual(min(r0,c0), 0)
        self.assertEqual(arrays[0].sum(), visible.sum())

    def test_score_and_invalid(self):
        full = np.ones((8,8), bool)
        visible = full.copy(); visible[4:] = False
        ctx = ScoreContext(full, visible, full)
        a, b = shape_score(full, ctx), shape_score(visible, ctx)
        self.assertEqual(select_candidate([b,a]), (1,None))
        self.assertEqual(select_candidate([{'score': None}]), (0,'all_candidates_invalid'))
        self.assertEqual(target_mask(None, visible)[2], 'no_detection')

    def test_support_excludes_padding(self):
        prior = np.zeros((8,8), bool); prior[2:6,2:6] = True
        ctx = ScoreContext(prior, prior, prior)
        self.assertEqual(shape_score(np.ones_like(prior),ctx)['leak'], 0)

    def test_metrics_hidden(self):
        gt = np.ones((8,8), bool); v = gt.copy(); v[4:] = False
        scores = mask_metrics(v,gt,v)
        self.assertEqual(scores['amodal_iou'], .5)
        self.assertEqual(scores['hidden_iou'], 0)
        self.assertIsNone(mask_metrics(gt,gt,gt)['hidden_iou'])
        rgb = np.zeros((8,8,3),np.uint8)
        self.assertEqual(hidden_rgb_metrics(rgb,rgb+255,gt)['hidden_mae'], 1)

    def test_atomic_json_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'config.json'
            save_json(path,{'path': Path('/tmp')})
            self.assertEqual(json.loads(path.read_text())['path'], '/tmp')

    def test_manager_cleanup_exception_and_exclusion(self):
        torch=SimpleNamespace(inference_mode=nullcontext, cuda=SimpleNamespace(is_available=lambda:False))
        model=SimpleNamespace(to=lambda device: moves.append(device))
        moves=[]
        manager=ModelManager('cpu',torch_module=torch)
        manager.register('one',lambda:model)
        with self.assertRaises(ValueError):
            with manager.use('one'):
                with self.assertRaises(RuntimeError):
                    with manager.use('one'): pass
                raise ValueError('inference error')
        self.assertIsNone(manager.active)
        self.assertEqual(moves,['cpu','cpu'])

    def test_metrics_only_cli(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            image=np.zeros((64,64,3),np.uint8)
            gt=np.zeros((64,64),np.uint8); gt[16:48,16:48]=255
            v=gt.copy(); v[32:]=0
            for name,arr in [('image',image),('gt',gt),('visible',v)]:
                Image.fromarray(arr).save(root/(name+'.png'))
            manifest=root/'samples.jsonl'
            manifest.write_text(json.dumps({'image':'image.png','visible_mask':'visible.png',
                'full_mask':'gt.png','prediction_mask':'gt.png','prediction_rgb':'image.png','gt_rgb':'image.png'})+'\n')
            subprocess.run([sys.executable,'scripts/evaluate.py','--manifest',str(manifest),
                '--metrics-only','--full','--eval_output',str(root/'output')],check=True,capture_output=True)
            report=json.loads(next((root/'output').glob('*/metrics_summary.json')).read_text())
            self.assertEqual(report['metrics']['amodal_iou']['mean'],1)
            self.assertEqual(report['attempted'],1)
            self.assertTrue(next((root/'output').glob('*/sample_*/comparison.png')).is_file())

    def test_iteration_updates_only_selected(self):
        # Kiểm thử luồng thật run_iteration với model giả, không kiểm tra lại công thức scorer.
        from unittest.mock import patch
        from main import save_mask
        with tempfile.TemporaryDirectory() as temp:
            image=Image.fromarray(np.zeros((100,100,3),np.uint8))
            visible=np.zeros((100,100),np.uint8); visible[40:60,40:60]=1
            query=QueryObject(image,visible,0,Path(temp)/'object')
            query.query_class='cat'
            query.img_canvas=create_canvas(np.asarray(image),6,255)
            for name in ('query_mask_canvas','original_visible_canvas','shape_prior_canvas'):
                setattr(query,name,create_canvas(visible,6,0))
            query.occ_mask_canvas=create_canvas(np.zeros_like(visible),6,0)
            query.outpaint_mask_canvas=create_canvas(np.zeros_like(visible),6,1)
            query.prior_valid_canvas=create_canvas(np.ones_like(visible),6,0)
            args=parse_args(['--input_dir',temp,'--standard_inpainting','--num_candidates','2'])
            def segment(images,classes):
                return [(np.array([np.asarray(query.original_visible_canvas)[230:370,230:370]],bool),[],[]) for _ in images]
            engine=SimpleNamespace(generate=lambda inputs,seeds,size,prompt:[image,image],segment=segment)
            def chosen(im,current,q,bounds,engine,classes):
                q.status='completed'; q.run_iter=False
            # Mask crop lấy chính hàm crop để không phụ thuộc bounds hardcode.
            anchor=crop([query.original_visible_canvas],visible,query.original_visible_canvas)[0][0].astype(bool)
            engine.segment=lambda images,classes:[(np.array([anchor]),[],[]) for _ in images]
            with patch('main.check_occlusion',side_effect=chosen) as check:
                run_iteration(query,engine,args,['cat'])
            self.assertEqual(query.iter_id,1)
            self.assertEqual(check.call_count,1)
            self.assertTrue((query.output_img_dir/'iteration_00/selection.json').exists())


if __name__=='__main__':
    unittest.main()

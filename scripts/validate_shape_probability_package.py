"""Execute the exact source-overlay code offline before private GPU submission."""
import ast
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from amodal_benchmark.dataset import sha256, write_json


def constant(module, name):
    return ast.literal_eval(next(node.value for node in module.body if isinstance(node,ast.Assign)
        and any(isinstance(target,ast.Name) and target.id==name for target in node.targets)))


def main():
    study = ROOT/'output/shape-proper-loss-20261123'
    parent = ROOT/'output/shape-kaggle-20261123'
    path = study/'kernel/main.py'
    program = ast.parse(path.read_text(encoding='utf-8'))
    parent_program = ast.parse((parent/'training-conda/main.py').read_text(encoding='utf-8'))
    assert constant(program,'packages')==constant(parent_program,'packages')
    assert len(constant(program,'packages'))==138
    manifest = json.loads((study/'probability-source-manifest.json').read_bytes())
    assert constant(program,'probability_manifest')==manifest
    assert sha256(study/'probability-source.bin')==manifest['sha256']
    bundle = json.loads((parent/'dataset/bundle.json').read_bytes())
    assert constant(program,'EXPECTED_BUNDLE_SHA256')==sha256(parent/'dataset/bundle.json')
    start = next(i for i,node in enumerate(program.body) if isinstance(node,ast.Import)
        and any(alias.name=='base64' for alias in node.names))
    end = next(i for i,node in enumerate(program.body) if isinstance(node,ast.Assign)
        and any(isinstance(target,ast.Name) and target.id=='env' for target in node.targets))
    assert start<end
    overlay = compile(ast.Module(body=program.body[start:end],type_ignores=[]),str(path),'exec')
    with TemporaryDirectory(prefix='shape-probability-package-') as work:
        source, output = Path(work)/'source',Path(work)/'output'
        source.mkdir()
        output.mkdir()
        with zipfile.ZipFile(parent/'dataset/source.bin') as archive:
            assert set(archive.namelist())==set(bundle['sources'])
            for name,expected in bundle['sources'].items():
                payload = archive.read(name)
                assert hashlib.sha256(payload).hexdigest()==expected
                target = (source/name).resolve()
                assert target.is_relative_to(source.resolve())
                target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes(payload)
        scope = dict(source=source,output=output,hashlib=hashlib,json=json,zipfile=zipfile)
        exec(overlay,scope)
        for name,expected in {**bundle['sources'],**manifest['sources']}.items():
            assert sha256(source/name)==expected
        assert sha256(output/'probability-source.bin')==manifest['sha256']
        assert json.loads((output/'probability-source-manifest.json').read_bytes())==manifest
        try:
            exec(overlay,scope)
        except AssertionError:
            overwrite_rejected = True
        else:
            raise AssertionError('Overlay must refuse overwriting existing sources')
    config = json.loads((ROOT/'configs/research/shape-proper-loss-kaggle-20261123.json').read_bytes())
    original = json.loads((ROOT/'configs/research/shape-sam-kaggle-20261123.json').read_bytes())
    for key in ('seed','initial_checkpoint_sha256','validation_diagnostic_result_sha256','inference_geometry'):
        assert config[key]==original[key]
    for key in ('world_size','batch_size','workers_per_rank','size','epochs','learning_rate','weight_decay',
                'precision','selection','input_mix','augmentation','distributed_sampling'):
        assert config['training'][key]==original['training'][key]
    trainer = (ROOT/'scripts/train_shape_probability_kaggle.py').read_text(encoding='utf-8')
    assert 'loss = probability_loss(logits.float(), y, x[:,3:4])' in trainer and 'shape_loss(' not in trainer
    assert 'schema=7, method="shape-probability-kaggle"' in trainer
    assert config['schema']==7 and config['training']['probability_loss_version']==1
    metadata = json.loads((study/'kernel/kernel-metadata.json').read_bytes())
    assert metadata['is_private'] is True and metadata['enable_gpu'] is True and metadata['enable_tpu'] is False
    assert metadata['dataset_sources']==['ez4kacchan/shape-only-training-inputs-20261123']
    result = dict(verified=True,inherited_source_files=len(bundle['sources']),additional_source_files=len(manifest['sources']),
        overlay_sha256=manifest['sha256'],kernel_source_sha256=sha256(path),runtime_packages=138,
        inherited_sources_unchanged=True,existing_source_overwrite_rejected=overwrite_rejected,
        native_replay_and_selection_unchanged_or_stronger=True,benchmark_success=False)
    write_json(study/'offline-package-validation.json',result)
    package = json.loads((study/'package.json').read_bytes())
    package['kernel_source_sha256'] = result['kernel_source_sha256']
    write_json(study/'package.json',package)
    print(json.dumps(result))


if __name__=='__main__':
    main()

"""Independently replay all 1,024 masks, raw-image errors and choices from native PNGs."""
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from amodal_benchmark.dataset import write_json, sha256


def read(path):
    return json.loads(path.read_text())


def mask(path):
    return np.array(Image.open(path).convert("L"))>0


def iou(x,y):
    return np.count_nonzero(x & y)/max(1,np.count_nonzero(x | y))


def main():
    out=ROOT/"output/shape-20261121"
    source=ROOT/"output/surface-confirmation128-20261110"
    summary=read(out/"controlled-summary.json")
    means={}
    n=0
    for case in read(source/"cases.json"):
        cid=case["case_id"]
        row=read(out/"controlled"/cid/"result.json")
        prior=mask(out/"controlled"/cid/"prior.png")
        v=mask(source/"inputs"/cid/"visible.png")
        o=mask(source/"inputs"/cid/"occlusion.png")
        gt=mask(source/"evaluation_only"/cid/"full.png")
        core=mask(source/"evaluation_only"/cid/"hidden_core.png")
        truth=np.asarray(Image.open(source/"evaluation_only"/cid/"truth.png"),float)
        assert (prior[v]).all() and not (prior & ~v & ~o).any()
        pool=source/"full"/"candidates"/cid
        scores=[]
        weighted=[]
        quality=[]
        for i in range(8):
            name=f"seed-{11*(i+1)}"
            m=mask(pool/(name+"-mask.png"))
            im=np.asarray(Image.open(pool/(name+".png")),float)
            z=iou(m,prior)
            scores.append(z if m.any() else float("-inf"))
            weighted.append(.6*z+.25*(m & v).sum()/v.sum()-.15*(m & ~prior).sum()/max(1,m.sum()) if m.any() else float("-inf"))
            mae=np.abs(im[core]-truth[core]).mean()/255
            measured=row["candidate_metrics"][i]
            quality.append(iou(m,gt))
            np.testing.assert_allclose([measured["iou"],measured["hidden_iou"],measured["hidden_core_mae"]],
                                       [quality[-1],iou(m&o,gt&o),mae],rtol=0,atol=1e-12)
            n+=1
        indices=row["indices"]
        for k in (4,8):
            assert indices["shape"+str(k)]==int(np.argmax(scores[:k]))
            assert indices["shape_weighted"+str(k)]==int(np.argmax(weighted[:k]))
            assert indices["oracle"+str(k)]==int(np.argmax(quality[:k]))
        losses=read(source/"inference"/"decisions"/(cid+".json"))["losses"]
        valid=[i for i in range(8) if np.isfinite(scores[i]) and losses[i] is not None]
        ranks={i: (sum(scores[j]>scores[i] for j in valid) + sum(losses[j]<losses[i] for j in valid)
                   +.5*sum(scores[j]==scores[i] for j in valid)+.5*sum(losses[j]==losses[i] for j in valid)) for i in valid}
        assert indices["combined8"]==(min(ranks,key=ranks.get) if ranks else 0)
        for method,result in row["metrics"].items():
            expected=quality[indices[method]] if method in indices else np.mean(quality[:int(method[-1])])
            assert abs(result["iou"]-expected)<1e-12
            means.setdefault(method,[]).append(expected)
    assert n==1024
    for method,values in means.items():
        versus="uniform4" if method.endswith("4") else "uniform8"
        assert abs(np.mean(values)-summary["summary"][versus][method]["iou"]["method"])<1e-12
    write_json(out/"controlled-audit.json",dict(passed=True,candidates=n,cases=len(values),
               tests=["native full/hidden IoU","native raw hidden-core RGB MAE","all predicted and oracle indices",
                      "equal-midrank combined index","all reported mean IoUs"],summary_sha256=sha256(out/"controlled-summary.json")))


if __name__=="__main__":
    main()


"""Summarize the paired Surface comparison and render the two Week 8 examples."""
import csv
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from train_shape import OUT, TEST, read_json
from amodal_benchmark.dataset import sha256, write_json
from analyze_surface_followup import paired_interval

HIGHER = {"iou", "hidden_iou", "object_ssim", "clip_text_cosine"}


def main():
    rows = [read_json(OUT/"controlled"/r["case_id"]/"result.json") for r in read_json(TEST/"cases.json")]
    assert len({r["target_id"] for r in rows}) == len(rows) == 128
    sources = [r["source"] for r in rows]
    metrics = ["iou","hidden_iou","hidden_core_mae","object_lpips","hidden_spatial_lpips","dreamsim","clip_text_cosine","object_ssim"]
    summary = {}
    csv_rows = []
    for method in rows[0]["metrics"]:
        for control in ["uniform4" if method.endswith("4") else "uniform8", "surface8"]:
            for metric in metrics:
                method_values = [r["metrics"][method][metric] for r in rows]
                control_values = [r["metrics"][control][metric] for r in rows]
                result = paired_interval(control_values,method_values,sources,higher=metric in HIGHER)
                summary.setdefault(control,{}).setdefault(method,{})[metric] = result
                csv_rows.append(dict(comparator=control,selector=method,metric=metric,**result))
    with (OUT/"controlled-comparisons.csv").open("w",newline="") as f:
        w = csv.DictWriter(f,fieldnames=list(csv_rows[0]))
        w.writeheader()
        w.writerows(csv_rows)
    assert abs(summary["uniform8"]["surface8"]["object_lpips"]["relative_gain_percent"]-11.328)<.001
    write_json(OUT/"controlled-summary.json",dict(summary=summary,
               protocol_sha256=sha256("configs/research/shape-20261121.json"),
               checkpoint_sha256=sha256(OUT/"prior.pt"),prior_iou=np.mean([r["prior_iou"] for r in rows]),
               prior_hidden_iou=np.mean([r["prior_hidden_iou"] for r in rows]),
               empty_masks=sum(s is None for r in rows for s in r["scores"]),
               fallbacks={m:sum(r["fallback"][m] for r in rows) for m in rows[0]["fallback"]}))
    chosen = []
    for label,first,surface in [("umbrella",.388,.095),("bus",.166,.174)]:
        matches = [r for r in rows if label in r["label"].lower() and
                   abs(r["metrics"]["first"]["object_lpips"]-first)<.00051 and
                   abs(r["metrics"]["surface8"]["object_lpips"]-surface)<.00051]
        assert len(matches)==1, (label,[r["case_id"] for r in matches])
        chosen.extend(matches)
    board = Image.new("RGB",(6*224,2*282+44),"white")
    draw = ImageDraw.Draw(board)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",17)
    headings = ["Observed","Pre-occlusion RGB","PMC (seed 11)","Shape","Surface","Combined"]
    for col, head in enumerate(headings):
        draw.text((col*224+5,8),head,font=font,fill="black")
    for n,r in enumerate(chosen):
        cid = r["case_id"]
        paths = [TEST/"inputs"/cid/"observed.png", TEST/"evaluation_only"/cid/"truth.png"]
        methods = ["first","shape8","surface8","combined8"]
        paths += [TEST/"full"/"candidates"/cid/f'seed-{11*(1+r["indices"][m])}.png' for m in methods]
        for col,path in enumerate(paths):
            y=40+282*n
            with Image.open(path) as im:
                board.paste(im.convert("RGB").resize((224,224),Image.Resampling.LANCZOS),(224*col,y))
            caption = r["label"] if col==0 else ""
            if col>=2:
                value=r["metrics"][methods[col-2]]
                caption=f'IoU {value["iou"]:.3f}    LPIPS {value["object_lpips"]:.3f}'
            draw.text((224*col+4,y+230),caption,font=font,fill="black")
    board.save(OUT/"slide-17.png")
    write_json(OUT/"slide-17-cases.json",[r["target_id"] for r in chosen])


if __name__ == "__main__":
    main()


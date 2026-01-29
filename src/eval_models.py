# src/eval_models.py

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import csv
import json
import random

import numpy as np
from PIL import Image, ImageDraw

import torch
import torchvision
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional as F

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

# paths
ROOT = Path(__file__).resolve().parents[1]
DATA_YOLO = ROOT / "data" / "processed" / "yolov8"
DATA_COCO = ROOT / "data" / "processed" / "coco"
RUNS = ROOT / "runs"
OUT = RUNS / "eval_vis"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# tryby dla Tourchvision
# "rgb" -> weights in runs/torchvision/{ds}__{method}/model_final.pt
# "gray" -> weights in runs/torchvision/{ds}__{method}__gray/model_final.pt
TV_MODES = ["rgb", "gray"]
TV_GRAY_SUFFIX = "__gray"


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def pick_images(folder: Path, k: int = 12, seed: int = 0) -> list[Path]:
    imgs = [p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS]
    imgs.sort()
    if not imgs:
        return []
    rng = random.Random(seed)
    if len(imgs) <= k:
        return imgs
    return rng.sample(imgs, k)


def draw_boxes(
    img: Image.Image,
    boxes_xyxy: list[list[float]],
    labels: list[str],
    scores: list[float] | None = None,
    color: str = "red",
) -> Image.Image:
    im = img.copy().convert("RGB")
    draw = ImageDraw.Draw(im)
    for i, b in enumerate(boxes_xyxy):
        x1, y1, x2, y2 = map(float, b)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        tag = labels[i] if i < len(labels) else "obj"
        if scores is not None and i < len(scores):
            tag = f"{tag} {float(scores[i]):.2f}"
        draw.text((x1 + 2, y1 + 2), tag, fill=color)
    return im


def write_csv(rows: list[dict], out_csv: Path) -> None:
    ensure_dir(out_csv.parent)
    if not rows:
        return
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# funkcje pomocnicze do przekształcania danych COCO
# Annotacje
def find_ann(split_dir: Path) -> Path:
    for name in ["_annotations.coco.json", "annotations.coco.json"]:
        p = split_dir / name
        if p.exists():
            return p
    cand = sorted(split_dir.rglob("*.json"))
    return cand[0] if cand else split_dir / "_annotations.coco.json"


# Ścieżki do obrazów 
def coco_paths(dataset: str) -> tuple[Path, Path, Path, Path]:
    base = DATA_COCO / dataset
    train_dir = base / "train"
    valid_dir = base / "valid"
    return valid_dir, find_ann(valid_dir), train_dir, find_ann(train_dir)


def contig_to_name_from_train_json(train_ann: Path) -> dict[int, str]:
    obj = json.loads(train_ann.read_text(encoding="utf-8"))
    cats = obj.get("categories", [])
    cats_sorted = sorted(cats, key=lambda c: int(c["id"]))  # 1..K
    return {i + 1: str(c["name"]) for i, c in enumerate(cats_sorted)}


# funkcja pomocnicza do przekształcania danych YOLO
def yolo_valid_images_dir(dataset: str) -> Path:
    cand = DATA_YOLO / dataset / "valid" / "images"
    if cand.exists():
        return cand
    cand = DATA_YOLO / dataset / "val" / "images"
    if cand.exists():
        return cand
    return DATA_YOLO / dataset / "valid" / "images"


# klasa przechowójąca rezultaty
@dataclass
class EvalRow:
    framework: str
    dataset: str
    model: str
    input_mode: str  # "rgb" / "gray" / "n/a" (for yolo)
    n_images: int
    score_thr: float
    topk: int
    avg_dets_per_img: float
    avg_score: float
    out_dir: str


# YOLO ewaluacja
def eval_yolo(
    dataset: str,
    model_stem: str,  # "yolov8n" / "yolov8s"
    n_images: int = 12,
    conf: float = 0.5,
    seed: int = 0,
    color: str = "red",
    topk: int = 10,
    image_paths: list[Path] | None = None,
) -> EvalRow:
    if YOLO is None:
        raise RuntimeError("Ultralytics nie jest zainstalowany (pip install ultralytics).")

    run_dir = RUNS / "ultralytics" / f"{dataset}__{model_stem}"
    weights = run_dir / "weights" / "best.pt"
    if not weights.exists():
        raise FileNotFoundError(f"Brak wag: {weights}")

    out_dir = OUT / "ultralytics" / f"{dataset}__{model_stem}"
    ensure_dir(out_dir)

    model = YOLO(str(weights))
    names = model.names if hasattr(model, "names") else {}

    if image_paths is None:
        img_dir = yolo_valid_images_dir(dataset)
        if not img_dir.exists():
            raise FileNotFoundError(f"Nie znaleziono katalogu obrazów valid: {img_dir}")
        imgs = pick_images(img_dir, k=n_images, seed=seed)
    else:
        imgs = [Path(p) for p in image_paths][:n_images]

    det_counts, scores_all = [], []
    for p in imgs:
        r = model.predict(source=str(p), conf=conf, verbose=False)[0]
        boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
        scores = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.zeros((0,))
        clss = r.boxes.cls.cpu().numpy() if r.boxes is not None else np.zeros((0,))

        if len(scores) > 0 and topk > 0:
            order = np.argsort(-scores)[:topk]
            boxes = boxes[order]
            scores = scores[order]
            clss = clss[order]

        labels = [str(names.get(int(c), int(c))) for c in clss.tolist()]
        det_counts.append(len(boxes))
        scores_all.extend(scores.tolist())

        im = Image.open(p).convert("RGB")
        vis = draw_boxes(im, boxes.tolist(), labels, scores.tolist(), color=color)
        vis.save(out_dir / f"{p.stem}__pred.jpg", quality=95)

    return EvalRow(
        framework="ultralytics",
        dataset=dataset,
        model=model_stem,
        input_mode="n/a",
        n_images=len(imgs),
        score_thr=float(conf),
        topk=int(topk),
        avg_dets_per_img=float(np.mean(det_counts)) if det_counts else 0.0,
        avg_score=float(np.mean(scores_all)) if scores_all else 0.0,
        out_dir=str(out_dir),
    )


# Wybór modelu z torchvision
def load_torchvision_model(method: str, num_classes: int) -> torch.nn.Module:
    if method == "fasterrcnn":
        m = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
        in_features = m.roi_heads.box_predictor.cls_score.in_features
        m.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            in_features, num_classes
        )
        return m

    if method == "retinanet":
        m = torchvision.models.detection.retinanet_resnet50_fpn_v2(weights="DEFAULT")
        in_features = m.head.classification_head.cls_logits.in_channels
        num_anchors = m.head.classification_head.num_anchors
        m.head.classification_head = torchvision.models.detection.retinanet.RetinaNetClassificationHead(
            in_features, num_anchors, num_classes
        )
        return m

    raise ValueError(f"Unknown method: {method}")

def tv_run_dir(dataset: str, method: str, mode: str) -> Path:
    if mode == "gray":
        return RUNS / "torchvision" / f"{dataset}__{method}{TV_GRAY_SUFFIX}"
    return RUNS / "torchvision" / f"{dataset}__{method}"


def load_image_for_tv(p: Path, mode: str) -> tuple[Image.Image, torch.Tensor]:
    img_pil = Image.open(p).convert("RGB")
    if mode == "gray":
        img_pil = img_pil.convert("L").convert("RGB")
    img_t = F.to_tensor(img_pil)
    return img_pil, img_t


def eval_torchvision_on_images(
    dataset: str,
    method: str,  # "fasterrcnn" / "retinanet"
    image_paths: list[Path],
    score_thr: float = 0.3,
    device: str = "cpu",
    color: str | None = None,
    topk: int = 10,
    mode: str = "rgb",  # "rgb" / "gray"
) -> EvalRow:
    # bierzemy nazwy klas z train coco (mapping 1..K)
    _, _, _, train_ann = coco_paths(dataset)
    contig_to_name = contig_to_name_from_train_json(train_ann)
    num_classes = len(contig_to_name) + 1  # + background

    run_dir = tv_run_dir(dataset, method, mode)
    weights = run_dir / "model_final.pt"
    if not weights.exists():
        raise FileNotFoundError(f"Brak wag torchvision ({mode}): {weights}")

    out_dir = OUT / "torchvision" / f"{dataset}__{method}__{mode}"
    ensure_dir(out_dir)

    if color is None:
        color = "blue" if method == "fasterrcnn" else "green"

    model = load_torchvision_model(method, num_classes=num_classes)
    state = torch.load(weights, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    det_counts, scores_all = [], []
    with torch.inference_mode():
        for p in image_paths:
            img_pil, img_t = load_image_for_tv(p, mode)
            out = model([img_t.to(device)])[0]

            boxes = out["boxes"].detach().cpu().numpy()
            scores = out["scores"].detach().cpu().numpy()
            labels = out["labels"].detach().cpu().numpy()

            keep = scores >= float(score_thr)
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]

            if len(scores) > 0 and topk > 0:
                order = np.argsort(-scores)[:topk]
                boxes = boxes[order]
                scores = scores[order]
                labels = labels[order]

            lab_text = [contig_to_name.get(int(l), f"class_{int(l)}") for l in labels.tolist()]

            det_counts.append(len(boxes))
            scores_all.extend(scores.tolist())

            vis = draw_boxes(img_pil, boxes.tolist(), lab_text, scores.tolist(), color=color)
            vis.save(out_dir / f"{p.stem}__pred.jpg", quality=95)

    return EvalRow(
        framework="torchvision",
        dataset=dataset,
        model=method,
        input_mode=mode,
        n_images=len(image_paths),
        score_thr=float(score_thr),
        topk=int(topk),
        avg_dets_per_img=float(np.mean(det_counts)) if det_counts else 0.0,
        avg_score=float(np.mean(scores_all)) if scores_all else 0.0,
        out_dir=str(out_dir),
    )


def existing_tv_modes(dataset: str, method: str) -> list[str]:
    # zwróć tylko te z wagami
    out = []
    for m in TV_MODES:
        w = tv_run_dir(dataset, method, m) / "model_final.pt"
        if w.exists():
            out.append(m)
    return out


def main():
    datasets = ["cats_dogs", "traffic_signs", "vehicles"]
    rows: list[dict] = []

    for ds in datasets:
        # wspólne obrazki (raz) z YOLO valid/images
        shared_dir = yolo_valid_images_dir(ds)
        shared_imgs = pick_images(shared_dir, k=12, seed=120867)

        # różne kolory dla YOLO8n i YOLO8s
        r = eval_yolo(ds, "yolov8n", conf=0.5, seed=0, color="red", topk=10, image_paths=shared_imgs)
        rows.append(asdict(r))
        print("OK", r)

        r = eval_yolo(ds, "yolov8s", conf=0.5, seed=0, color="orange", topk=10, image_paths=shared_imgs)
        rows.append(asdict(r))
        print("OK", r)

        # uruchom dla istniejących trybów
        for method in ["fasterrcnn", "retinanet"]:
            modes = existing_tv_modes(ds, method)
            if not modes:
                print(f"[WARN] brak wag torchvision dla {ds} {method} (ani rgb ani gray) -> skip")
                continue

            for mode in modes:
                r = eval_torchvision_on_images(
                    ds,
                    method,
                    shared_imgs,
                    score_thr=0.3,
                    device="cpu",
                    color=("blue" if method == "fasterrcnn" else "green"),
                    topk=10,
                    mode=mode,
                )
                rows.append(asdict(r))
                print("OK", r)

    write_csv(rows, OUT / "quick_eval_all.csv")
    print("Saved to:", OUT)


if __name__ == "__main__":
    main()

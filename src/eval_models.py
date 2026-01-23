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


# -------- paths --------
ROOT = Path(__file__).resolve().parents[1]
DATA_YOLO = ROOT / "data" / "processed" / "yolov8"
DATA_COCO = ROOT / "data" / "processed" / "coco"
RUNS = ROOT / "runs"
OUT = RUNS / "eval_vis"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


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


# -------- COCO helpers --------
def find_ann(split_dir: Path) -> Path:
    # Roboflow COCO najczęściej ma _annotations.coco.json [train_torchvision.py]
    for name in ["_annotations.coco.json", "annotations.coco.json"]:
        p = split_dir / name
        if p.exists():
            return p
    cand = sorted(split_dir.rglob("*.json"))
    return cand[0] if cand else split_dir / "_annotations.coco.json"


def coco_paths(dataset: str) -> tuple[Path, Path, Path, Path]:
    base = DATA_COCO / dataset
    train_dir = base / "train"
    valid_dir = base / "valid"
    return valid_dir, find_ann(valid_dir), train_dir, find_ann(train_dir)


def contig_to_name_from_train_json(train_ann: Path) -> dict[int, str]:
    obj = json.loads(train_ann.read_text(encoding="utf-8"))
    cats = obj.get("categories", [])
    # zgodnie z train_torchvision.py: sort po category_id i nadaj 1..K [file:5]
    cats_sorted = sorted(cats, key=lambda c: int(c["id"]))
    return {i + 1: str(c["name"]) for i, c in enumerate(cats_sorted)}


# -------- YOLO helpers --------
def yolo_valid_images_dir(dataset: str) -> Path:
    # u Ciebie YOLO ma valid/images [file:6][file:7]
    cand = DATA_YOLO / dataset / "valid" / "images"
    if cand.exists():
        return cand
    cand = DATA_YOLO / dataset / "val" / "images"
    if cand.exists():
        return cand
    return DATA_YOLO / dataset / "valid" / "images"


# -------- result row --------
@dataclass
class EvalRow:
    framework: str
    dataset: str
    model: str
    n_images: int
    score_thr: float
    topk: int
    avg_dets_per_img: float
    avg_score: float
    out_dir: str


# -------- YOLO eval --------
def eval_yolo(
    dataset: str,
    model_stem: str,  # "yolov8n" / "yolov8s"
    n_images: int = 12,
    conf: float = 0.25,
    seed: int = 0,
    color: str = "red",
    topk: int = 3,
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

        # top-k
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
        n_images=len(imgs),
        score_thr=float(conf),
        topk=int(topk),
        avg_dets_per_img=float(np.mean(det_counts)) if det_counts else 0.0,
        avg_score=float(np.mean(scores_all)) if scores_all else 0.0,
        out_dir=str(out_dir),
    )


# -------- Torchvision helpers --------
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


class CocoDet(CocoDetection):
    def __init__(self, imgfolder: Path, annfile: Path):
        super().__init__(str(imgfolder), str(annfile))

    def __getitem__(self, idx):
        img, anns = super().__getitem__(idx)
        imgid = self.ids[idx]
        img_t = F.to_tensor(img)
        return img, img_t, anns, imgid


# --- Torchvision eval: na losowych COCO indexach (stara wersja) ---
def eval_torchvision(
    dataset: str,
    method: str,  # "fasterrcnn" / "retinanet"
    n_images: int = 12,
    score_thr: float = 0.5,
    seed: int = 0,
    device: str = "cpu",
    color: str | None = None,
    topk: int = 3,
) -> EvalRow:
    valid_dir, valid_ann, train_dir, train_ann = coco_paths(dataset)

    run_dir = RUNS / "torchvision" / f"{dataset}__{method}"
    weights = run_dir / "model_final.pt"
    if not weights.exists():
        raise FileNotFoundError(f"Brak wag torchvision: {weights}")

    out_dir = OUT / "torchvision" / f"{dataset}__{method}"
    ensure_dir(out_dir)

    contig_to_name = contig_to_name_from_train_json(train_ann)
    num_classes = len(contig_to_name) + 1  # + background

    model = load_torchvision_model(method, num_classes=num_classes)
    state = torch.load(weights, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    ds = CocoDet(valid_dir, valid_ann)

    rng = random.Random(seed)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    idxs = idxs[: min(n_images, len(idxs))]

    if color is None:
        color = "blue" if method == "fasterrcnn" else "green"

    det_counts, scores_all = [], []

    with torch.inference_mode():
        for idx in idxs:
            img_pil, img_t, anns, imgid = ds[idx]
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
            vis.save(out_dir / f"img_{int(imgid)}__pred.jpg", quality=95)

    return EvalRow(
        framework="torchvision",
        dataset=dataset,
        model=method,
        n_images=len(idxs),
        score_thr=float(score_thr),
        topk=int(topk),
        avg_dets_per_img=float(np.mean(det_counts)) if det_counts else 0.0,
        avg_score=float(np.mean(scores_all)) if scores_all else 0.0,
        out_dir=str(out_dir),
    )


# --- Torchvision eval: na dokładnie tych samych plikach co YOLO (NOWE) ---
def eval_torchvision_on_images(
    dataset: str,
    method: str,  # "fasterrcnn" / "retinanet"
    image_paths: list[Path],
    score_thr: float = 0.5,
    device: str = "cpu",
    color: str | None = None,
    topk: int = 3,
) -> EvalRow:
    # bierzemy nazwy klas z train COCO (mapping 1..K jak w treningu) [file:5]
    _, _, _, train_ann = coco_paths(dataset)
    contig_to_name = contig_to_name_from_train_json(train_ann)
    num_classes = len(contig_to_name) + 1  # + background

    run_dir = RUNS / "torchvision" / f"{dataset}__{method}"
    weights = run_dir / "model_final.pt"
    if not weights.exists():
        raise FileNotFoundError(f"Brak wag torchvision: {weights}")

    out_dir = OUT / "torchvision" / f"{dataset}__{method}"
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
            img_pil = Image.open(p).convert("RGB")
            img_t = F.to_tensor(img_pil).to(device)

            out = model([img_t])[0]

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
            # zapis pod tą samą bazową nazwą co YOLO (łatwe porównanie)
            vis.save(out_dir / f"{p.stem}__pred.jpg", quality=95)

    return EvalRow(
        framework="torchvision",
        dataset=dataset,
        model=method,
        n_images=len(image_paths),
        score_thr=float(score_thr),
        topk=int(topk),
        avg_dets_per_img=float(np.mean(det_counts)) if det_counts else 0.0,
        avg_score=float(np.mean(scores_all)) if scores_all else 0.0,
        out_dir=str(out_dir),
    )


def main():
    datasets = ["cats_dogs", "traffic_signs", "vehicles"]
    rows: list[dict] = []

    for ds in datasets:
        # wspólne obrazki (raz) z YOLO valid/images [file:6]
        shared_dir = yolo_valid_images_dir(ds)
        shared_imgs = pick_images(shared_dir, k=12, seed=0)

        # YOLO: różne kolory dla 8n i 8s
        r = eval_yolo(ds, "yolov8n", conf=0.25, seed=0, color="red", topk=3, image_paths=shared_imgs)
        rows.append(asdict(r))
        print("OK", r)

        r = eval_yolo(ds, "yolov8s", conf=0.25, seed=0, color="orange", topk=3, image_paths=shared_imgs)
        rows.append(asdict(r))
        print("OK", r)

        # Torchvision na tych samych plikach
        r = eval_torchvision_on_images(ds, "fasterrcnn", shared_imgs, score_thr=0.5, device="cpu", color="blue", topk=3)
        rows.append(asdict(r))
        print("OK", r)

        r = eval_torchvision_on_images(ds, "retinanet", shared_imgs, score_thr=0.5, device="cpu", color="green", topk=3)
        rows.append(asdict(r))
        print("OK", r)

    write_csv(rows, OUT / "quick_eval_all.csv")
    print("Saved to:", OUT)


if __name__ == "__main__":
    main()

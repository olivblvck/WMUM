from __future__ import annotations

import json
import math
import re
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
REPORTS = OUT / "_reports"


DATASETS = ["cats_dogs", "fruits_vegetables", "traffic_signs", "vehicles", "people"]


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class RepairStats:
    dataset: str
    fmt: str  # "yolov8" / "coco"
    copied_images: int = 0
    copied_labels: int = 0
    missing_images: int = 0
    missing_labels: int = 0
    fixed_labels: int = 0
    removed_labels: int = 0
    removed_boxes: int = 0
    clipped_boxes: int = 0
    coco_images_total: int = 0
    coco_annotations_total: int = 0
    coco_images_removed: int = 0
    coco_annotations_removed: int = 0
    notes: List[str] = None

    def __post_init__(self):
        if self.notes is None:
            self.notes = []


def ensure_empty_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS


def find_data_yaml(ds_yolo_dir: Path) -> Path:
    candidates = list(ds_yolo_dir.glob("*.yaml")) + list(ds_yolo_dir.glob("*.yml"))
    for c in candidates:
        if c.name.lower() in {"data.yaml", "data.yml"}:
            return c
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"Nie znaleziono pliku YAML w {ds_yolo_dir}")


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

# zwróć wyjątek dla wyników niepoprawnych (infinity, NaN)
def safe_float(s: str) -> Optional[float]:
    try:
        x = float(s)
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    except Exception:
        return None


def repair_yolo_label_file(label_path: Path) -> Tuple[bool, int, int]:
    lines = label_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    out_lines = []
    changed = False
    removed_boxes = 0
    clipped_boxes = 0

    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        parts = re.split(r"\s+", ln)
        if len(parts) < 5:
            removed_boxes += 1
            changed = True
            continue

        cls = parts[0]
        xc = safe_float(parts[1])
        yc = safe_float(parts[2])
        w = safe_float(parts[3])
        h = safe_float(parts[4])

        if xc is None or yc is None or w is None or h is None:
            removed_boxes += 1
            changed = True
            continue

        # czyszczenie
        if w <= 0 or h <= 0:
            removed_boxes += 1
            changed = True
            continue

        # klip do [0,1]
        xc2, yc2, w2, h2 = clamp01(xc), clamp01(yc), clamp01(w), clamp01(h)
        if (xc2, yc2, w2, h2) != (xc, yc, w, h):
            clipped_boxes += 1
            changed = True

        # jeśli po klipowaniu zrobiły się 0 - usuń
        if w2 <= 0 or h2 <= 0:
            removed_boxes += 1
            changed = True
            continue

        out_lines.append(f"{cls} {xc2:.6f} {yc2:.6f} {w2:.6f} {h2:.6f}")

    if out_lines != [ln.strip() for ln in lines if ln.strip()]:
        changed = True

    if changed:
        label_path.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")

    return changed, removed_boxes, clipped_boxes


def repair_yolov8_dataset(ds_name: str):
    src = RAW / "yolov8" / ds_name
    if not src.exists():
        return

    dst = OUT / "yolov8" / ds_name
    ensure_empty_dir(dst)

    stats = RepairStats(dataset=ds_name, fmt="yolov8")

    # kopiujemy całą strukturę, potem naprawiamy labelki w miejscu w dst
    shutil.copytree(src, dst, dirs_exist_ok=True)

    # usuwanie cache
    for cache in dst.rglob("*.cache"):
        cache.unlink(missing_ok=True)
        stats.notes.append(f"Removed cache: {cache.relative_to(dst)}")

    # naprawa label files (train/valid/test)
    label_dirs = []
    for split in ["train", "valid", "val", "test"]:
        p = dst / split / "labels"
        if p.exists():
            label_dirs.append(p)

    if not label_dirs:
        stats.notes.append("No <split>/labels directories found. Check Roboflow export.")
        return stats

    for ld in label_dirs:
        for lp in ld.glob("*.txt"):
            changed, removed_boxes, clipped_boxes = repair_yolo_label_file(lp)
            if changed:
                stats.fixed_labels += 1
            stats.removed_boxes += removed_boxes
            stats.clipped_boxes += clipped_boxes

    return stats


def load_coco_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_coco_json(path: Path, obj: dict):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def coco_find_images_dir(ds_coco_dir: Path) -> Optional[Path]:
    for cand in ["images", "train", "valid", "val", "test"]:
        p = ds_coco_dir / cand
        if p.exists() and any(is_image(x) for x in p.rglob("*")):
            return p
    return None


def repair_coco_annotation(coco: dict, images_dir: Path, stats: RepairStats) -> dict:
    # map image_id -> (file_name, width, height)
    img_by_id = {}
    for im in coco.get("images", []):
        img_by_id[im["id"]] = im

    # usuń images bez pliku na dysku
    kept_images = []
    removed_image_ids = set()
    for im in coco.get("images", []):
        fp = images_dir / im["file_name"]
        if fp.exists():
            kept_images.append(im)
        else:
            removed_image_ids.add(im["id"])
            stats.coco_images_removed += 1
    coco["images"] = kept_images

    # napraw bboxy
    kept_anns = []
    for ann in coco.get("annotations", []):
        if ann.get("image_id") in removed_image_ids:
            stats.coco_annotations_removed += 1
            continue

        im = img_by_id.get(ann.get("image_id"))
        if im is None:
            stats.coco_annotations_removed += 1
            continue

        w_img, h_img = im.get("width"), im.get("height")

        if not w_img or not h_img:
            fp = images_dir / im["file_name"]
            if fp.exists():
                with Image.open(fp) as img:
                    w_img, h_img = img.size
                im["width"], im["height"] = w_img, h_img
            else:
                stats.coco_annotations_removed += 1
                continue

        bbox = ann.get("bbox", None)
        if not bbox or len(bbox) != 4:
            stats.coco_annotations_removed += 1
            continue

        x, y, bw, bh = bbox
        # usuń złe
        if bw <= 0 or bh <= 0:
            stats.coco_annotations_removed += 1
            continue

        # klipowanie do obrazu (COCO: x,y = lewy-górny róg) [x,y,width,height]
        x2 = max(0.0, min(float(x), w_img - 1))
        y2 = max(0.0, min(float(y), h_img - 1))
        bw2 = max(0.0, min(float(bw), w_img - x2))
        bh2 = max(0.0, min(float(bh), h_img - y2))

        if (x2, y2, bw2, bh2) != (x, y, bw, bh):
            stats.clipped_boxes += 1
            ann["bbox"] = [x2, y2, bw2, bh2]

        if bw2 <= 0 or bh2 <= 0:
            stats.coco_annotations_removed += 1
            continue

        kept_anns.append(ann)

    coco["annotations"] = kept_anns
    return coco


def repair_coco_dataset(ds_name: str):
    src = RAW / "coco" / ds_name
    if not src.exists():
        return

    dst = OUT / "coco" / ds_name
    ensure_empty_dir(dst)
    shutil.copytree(src, dst, dirs_exist_ok=True)

    stats = RepairStats(dataset=ds_name, fmt="coco")

    # każdy split ma folder z obrazami i plik _annotations.coco.json
    splits = ["train", "valid", "test"]
    found_any = False

    for split in splits:
        split_dir = dst / split
        ann_path = split_dir / "_annotations.coco.json"
        if not split_dir.exists() or not ann_path.exists():
            continue

        found_any = True
        images_dir = split_dir  # obrazy są w tym samym folderze co JSON

        coco = load_coco_json(ann_path)
        stats.coco_images_total += len(coco.get("images", []))
        stats.coco_annotations_total += len(coco.get("annotations", []))

        coco = repair_coco_annotation(coco, images_dir=images_dir, stats=stats)
        save_coco_json(ann_path, coco)

    if not found_any:
        stats.notes.append("No split folders with _annotations.coco.json found (expected train/valid/test).")

    return stats



def main():
    REPORTS.mkdir(parents=True, exist_ok=True)

    all_stats = []

    for ds in DATASETS:
        s1 = repair_yolov8_dataset(ds)
        if s1:
            all_stats.append(s1)
            (REPORTS / f"{ds}_yolov8_repair.json").write_text(
                json.dumps(asdict(s1), ensure_ascii=False, indent=2), encoding="utf-8"
            )

        s2 = repair_coco_dataset(ds)
        if s2:
            all_stats.append(s2)
            (REPORTS / f"{ds}_coco_repair.json").write_text(
                json.dumps(asdict(s2), ensure_ascii=False, indent=2), encoding="utf-8"
            )

    print("Done. Reports in:", REPORTS)
    for s in all_stats:
        print(f"- {s.dataset} [{s.fmt}] removed_boxes={s.removed_boxes} clipped_boxes={s.clipped_boxes} "
              f"coco_images_removed={s.coco_images_removed} coco_ann_removed={s.coco_annotations_removed}")


if __name__ == "__main__":
    main()

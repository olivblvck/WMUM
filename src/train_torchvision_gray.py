from __future__ import annotations

from pathlib import Path
import csv
import json
from typing import List, Dict, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import torchvision
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional as F

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "coco"
RUNS = ROOT / "runs" / "torchvision"

# trening
EPOCHS = 10
BATCH_SIZE = 1
LR = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0
DEVICE = "cpu"  # "mps"
SAVE_EVERY_EPOCH = 1

# True: gdy jest model_final.pt to skip
# False: używamy checkpoint_last.pth (resume)
SKIP_IF_FINAL_EXISTS = False

# suffix, żeby nie nadpisywać RGB
RUN_SUFFIX = "__gray"

TASKS = [
    {"dataset": "cats_dogs", "method": "fasterrcnn", "enabled": True},
    {"dataset": "cats_dogs", "method": "retinanet", "enabled": True},
    # {"dataset": "fruits_vegetables", "method": "fasterrcnn", "enabled": True},
    # {"dataset": "fruits_vegetables", "method": "retinanet", "enabled": True},
    {"dataset": "traffic_signs", "method": "fasterrcnn", "enabled": True},
    {"dataset": "traffic_signs", "method": "retinanet", "enabled": True},
    {"dataset": "vehicles", "method": "fasterrcnn", "enabled": True},
    {"dataset": "vehicles", "method": "retinanet", "enabled": True},
]


def coco_paths(ds: str) -> Tuple[Path, Path, Path, Path]:
    train_dir = DATA / ds / "train"
    val_dir = DATA / ds / "valid"

    train_json = train_dir / "_annotations.coco.json"
    val_json = val_dir / "_annotations.coco.json"

    return train_dir, train_json, val_dir, val_json


def load_categories(train_json: Path) -> List[dict]:
    obj = json.loads(train_json.read_text(encoding="utf-8"))
    return obj.get("categories", [])


def build_cat_mapping(categories: List[dict]) -> Dict[int, int]:
    # COCO category_id -> contiguous 1...K
    cat_ids = sorted([c["id"] for c in categories])
    return {cid: i + 1 for i, cid in enumerate(cat_ids)}


class CocoDet(CocoDetection):
    def __init__(
        self,
        img_folder: str | Path,
        ann_file: str | Path,
        cat_id_to_contig: Dict[int, int],
        drop_empty: bool = False,
        min_box_size: float = 2.0,
        max_empty_tries: int = 25,
    ):
        super().__init__(str(img_folder), str(ann_file))
        self.cat_id_to_contig = cat_id_to_contig
        self.drop_empty = drop_empty
        self.min_box_size = float(min_box_size)
        self.max_empty_tries = int(max_empty_tries)

    def __getitem__(self, idx):
        tries = 0
        cur_idx = idx

        while True:
            img, anns = super().__getitem__(cur_idx)
            img_id = self.ids[cur_idx]

            boxes = []
            labels = []
            areas = []
            iscrowd = []

            for a in anns:
                x, y, w, h = a["bbox"]
                if w < self.min_box_size or h < self.min_box_size:
                    continue

                boxes.append([x, y, x + w, y + h])
                labels.append(self.cat_id_to_contig[a["category_id"]])
                areas.append(float(a.get("area", w * h)))
                iscrowd.append(int(a.get("iscrowd", 0)))

            # Jeśli trening i brak boxów -> skip
            if self.drop_empty and len(boxes) == 0 and tries < self.max_empty_tries:
                tries += 1
                cur_idx = (cur_idx + 1) % len(self.ids)
                continue

            boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
            labels_t = torch.as_tensor(labels, dtype=torch.int64)
            areas_t = (
                torch.as_tensor(areas, dtype=torch.float32)
                if len(areas)
                else torch.zeros((0,), dtype=torch.float32)
            )
            iscrowd_t = (
                torch.as_tensor(iscrowd, dtype=torch.int64)
                if len(iscrowd)
                else torch.zeros((0,), dtype=torch.int64)
            )

            target = {
                "boxes": boxes_t,
                "labels": labels_t,
                "image_id": torch.tensor([img_id]),
                "area": areas_t,
                "iscrowd": iscrowd_t,
            }

            # BW input (L -> RGB, żeby dalej było 3 kanały)
            img = img.convert("L").convert("RGB")
            img = F.to_tensor(img)

            return img, target


def collate_fn(batch):
    return tuple(zip(*batch))


def make_model(method: str, num_classes: int) -> torch.nn.Module:
    if method == "fasterrcnn":
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            in_features, num_classes
        )
        return model

    if method == "retinanet":
        model = torchvision.models.detection.retinanet_resnet50_fpn_v2(weights="DEFAULT")
        in_features = model.head.classification_head.cls_logits.in_channels
        num_anchors = model.head.classification_head.num_anchors
        model.head.classification_head = torchvision.models.detection.retinanet.RetinaNetClassificationHead(
            in_features, num_anchors, num_classes
        )
        return model

    raise ValueError(f"Unknown method: {method}")


def train_one_epoch(model, loader, optimizer, device, epoch, max_batches=None):
    model.train()
    losses_avg = 0.0
    n = 0

    pbar = tqdm(loader, desc=f"epoch {epoch+1}", leave=False)
    for images, targets in pbar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        losses_avg += float(loss.item())
        n += 1
        pbar.set_postfix(loss=f"{losses_avg / max(n, 1):.4f}")

        if max_batches is not None and n >= int(max_batches):
            break

    return losses_avg / max(n, 1)


@torch.inference_mode()
def coco_eval_bbox(
    model,
    ann_file: Path,
    dataset: CocoDet,
    device,
    out_json: Path,
    contig_to_cat_id: Dict[int, int],
):
    model.eval()
    coco_gt = COCO(str(ann_file))

    preds = []
    for i in tqdm(range(len(dataset)), desc="infer", leave=False):
        img, target = dataset[i]
        img = img.to(device)
        out = model([img])[0]

        boxes = out["boxes"].detach().cpu()
        scores = out["scores"].detach().cpu()
        labels = out["labels"].detach().cpu()

        img_id = int(target["image_id"].item())

        for box, score, lab in zip(boxes, scores, labels):
            if float(score) < 0.01:
                continue

            x1, y1, x2, y2 = box.tolist()
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)

            contig = int(lab.item())
            coco_cat = int(contig_to_cat_id.get(contig, contig))

            preds.append(
                {
                    "image_id": img_id,
                    "category_id": coco_cat,
                    "bbox": [x1, y1, w, h],
                    "score": float(score.item()),
                }
            )

    out_json.write_text(json.dumps(preds), encoding="utf-8")

    # pycocotools.loadRes crashuje jeśli preds == []
    if len(preds) == 0:
        print("[WARN] Empty predictions -> COCOeval skipped (AP=0).")
        return 0.0, 0.0, 0.0

    coco_dt = coco_gt.loadRes(str(out_json))
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    ap = float(coco_eval.stats[0])
    ap50 = float(coco_eval.stats[1])
    ap75 = float(coco_eval.stats[2])
    return ap, ap50, ap75


def save_checkpoint(path: Path, epoch: int, model: torch.nn.Module, optimizer: torch.optim.Optimizer, avg_loss: float):
    # zapis do tmp i dopiero potem replace (żeby nie było uszkodzonego .pth po przerwaniu)
    ckpt = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "avg_loss": float(avg_loss),
    }

    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(ckpt, tmp)
    tmp.replace(path)


def try_load_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, device: str):
    if not path.exists():
        return 0, None

    try:
        ckpt = torch.load(path, map_location=device)
    except Exception as e:
        print(f"[WARN] Bad checkpoint {path}: {e} -> starting from scratch")
        return 0, None

    try:
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    except Exception as e:
        print(f"[WARN] Checkpoint incompatible {path}: {e} -> starting from scratch")
        return 0, None

    start_epoch = int(ckpt.get("epoch", -1)) + 1
    last_loss = ckpt.get("avg_loss", None)
    return start_epoch, last_loss


def main():
    RUNS.mkdir(parents=True, exist_ok=True)

    # osobny CSV, żeby nie nadpisać RGB
    out_csv = RUNS / "summary_gray.csv"
    rows = []

    for t in TASKS:
        if not t.get("enabled", True):
            continue

        ds = t["dataset"]
        method = t["method"]

        train_dir, train_json, val_dir, val_json = coco_paths(ds)

        categories = load_categories(train_json)
        cat_id_to_contig = build_cat_mapping(categories)
        contig_to_cat_id = {v: k for k, v in cat_id_to_contig.items()}

        num_classes = len(categories) + 1

        exp_name = f"{ds}__{method}{RUN_SUFFIX}"
        out_dir = RUNS / exp_name
        out_dir.mkdir(parents=True, exist_ok=True)

        ckpt_path = out_dir / "checkpoint_last.pth"
        model_final_path = out_dir / "model_final.pt"
        preds_json = out_dir / "predictions.json"

        ds_train = CocoDet(train_dir, train_json, cat_id_to_contig, drop_empty=True)
        ds_val = CocoDet(val_dir, val_json, cat_id_to_contig, drop_empty=False)

        train_loader = DataLoader(
            ds_train,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            collate_fn=collate_fn,
        )

        model = make_model(method, num_classes=num_classes).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        if SKIP_IF_FINAL_EXISTS and model_final_path.exists():
            print(f"[SKIP] {exp_name} final exists: {model_final_path}")
            model.load_state_dict(torch.load(model_final_path, map_location=DEVICE))
        else:
            start_epoch, last_loss = try_load_checkpoint(ckpt_path, model, optimizer, DEVICE)
            if start_epoch > 0:
                print(f"[RESUME] {exp_name} from epoch {start_epoch} (last avg_loss={last_loss})")

            for epoch in range(start_epoch, EPOCHS):
                avg_loss = train_one_epoch(model, train_loader, optimizer, DEVICE, epoch, max_batches=20)
                print(f"{exp_name} epoch {epoch+1}/{EPOCHS} loss={avg_loss:.4f}")

                if SAVE_EVERY_EPOCH and ((epoch + 1) % SAVE_EVERY_EPOCH == 0):
                    save_checkpoint(ckpt_path, epoch, model, optimizer, avg_loss)

            torch.save(model.state_dict(), model_final_path)

        ap, ap50, ap75 = coco_eval_bbox(model, val_json, ds_val, DEVICE, preds_json, contig_to_cat_id)

        rows.append(
            {
                "dataset": ds,
                "method": method,
                "input_mode": "gray",
                "device": DEVICE,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "AP": ap,
                "AP50": ap50,
                "AP75": ap75,
                "run_dir": str(out_dir.resolve()),
            }
        )

        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)

    print("Saved:", out_csv)
    print("Done:", out_csv)


if __name__ == "__main__":
    main()

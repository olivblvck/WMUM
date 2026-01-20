from pathlib import Path
import csv
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "yolov8"
RUNS = ROOT / "runs" / "ultralytics"

EPOCHS = 20
IMGSZ = 512
BATCH = 8

TRAIN_DEVICE = "mps"
VAL_DEVICE = "cpu"

SKIP_DONE = True  # pomija, jeśli istnieje weights/best.pt

# 6 linijek: dataset x model (możesz każdą "wyłączyć" enabled=False)
TASKS = [
    {"dataset": "cats_dogs", "model": "yolov8n.pt", "enabled": True},
    {"dataset": "cats_dogs", "model": "yolov8s.pt", "enabled": True},
    {"dataset": "fruits_vegetables", "model": "yolov8n.pt", "enabled": True},
    {"dataset": "fruits_vegetables", "model": "yolov8s.pt", "enabled": True},
    {"dataset": "traffic_signs", "model": "yolov8n.pt", "enabled": True},
    {"dataset": "traffic_signs", "model": "yolov8s.pt", "enabled": True},
]

def metric_get(obj, path, default=None):
    cur = obj
    for part in path.split("."):
        if cur is None:
            return default
        cur = getattr(cur, part, None)
    return default if cur is None else cur

def to_scalar(x, default=0.0):
    if x is None:
        return float(default)
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, (list, tuple)):
        vals = [float(v) for v in x if isinstance(v, (int, float))]
        return float(sum(vals) / len(vals)) if vals else float(default)
    m = getattr(x, "mean", None)
    if callable(m):
        try:
            y = m()
            item = getattr(y, "item", None)
            return float(item()) if callable(item) else float(y)
        except Exception:
            pass
    item = getattr(x, "item", None)
    if callable(item):
        try:
            return float(item())
        except Exception:
            return float(default)
    try:
        return float(x)
    except Exception:
        return float(default)

def extract_metrics(val_res):
    rd = getattr(val_res, "results_dict", None)
    if isinstance(rd, dict) and rd:
        def pick(keys):
            for k in keys:
                if k in rd:
                    return to_scalar(rd[k], 0.0)
            return 0.0
        precision = pick(["metrics/precision(B)", "metrics/precision", "precision"])
        recall = pick(["metrics/recall(B)", "metrics/recall", "recall"])
        map50 = pick(["metrics/mAP50(B)", "metrics/mAP50", "mAP50"])
        map50_95 = pick(["metrics/mAP50-95(B)", "metrics/mAP50-95", "mAP50-95", "map"])
        return precision, recall, map50, map50_95

    p = to_scalar(metric_get(val_res, "box.p", 0.0), 0.0)
    r = to_scalar(metric_get(val_res, "box.r", 0.0), 0.0)
    map50 = to_scalar(metric_get(val_res, "box.map50", 0.0), 0.0)
    map50_95 = to_scalar(metric_get(val_res, "box.map", 0.0), 0.0)
    return p, r, map50, map50_95

def write_summary(rows, out_csv):
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["dataset", "model", "precision", "recall", "map50", "map50_95", "run_dir"],
        )
        w.writeheader()
        w.writerows(rows)

def main():
    RUNS.mkdir(parents=True, exist_ok=True)
    out_csv = RUNS / "summary.csv"
    rows = []

    for t in TASKS:
        if not t.get("enabled", True):
            continue

        ds = t["dataset"]
        model_ckpt = t["model"]
        exp_name = f"{ds}__{Path(model_ckpt).stem}"
        run_dir = RUNS / exp_name
        best_pt = run_dir / "weights" / "best.pt"

        data_yaml = DATA / ds / "data.yaml"
        if not data_yaml.exists():
            raise FileNotFoundError(f"Brak {data_yaml}")

        if SKIP_DONE and best_pt.exists():
            print(f"[SKIP] {exp_name} already done: {best_pt}")
        else:
            model = YOLO(model_ckpt)
            model.train(
                data=str(data_yaml),
                epochs=EPOCHS,
                imgsz=IMGSZ,
                batch=BATCH,
                device=TRAIN_DEVICE,
                project=str(RUNS),
                name=exp_name,
                exist_ok=True,
                val=False,
            )

        # Walidacja (CPU) i zapis wyników w RUNS, nie w scripts/runs/detect/...
        model_for_val = YOLO(str(best_pt)) if best_pt.exists() else YOLO(model_ckpt)
        val_res = model_for_val.val(
            data=str(data_yaml),
            imgsz=IMGSZ,
            device=VAL_DEVICE,
            project=str(RUNS),
            name=f"{exp_name}__val",
            exist_ok=True,
        )

        precision, recall, map50, map50_95 = extract_metrics(val_res)

        rows.append({
            "dataset": ds,
            "model": Path(model_ckpt).stem,
            "precision": precision,
            "recall": recall,
            "map50": map50,
            "map50_95": map50_95,
            "run_dir": str(run_dir.resolve()),
        })
        write_summary(rows, out_csv)

    print("Done:", out_csv)

if __name__ == "__main__":
    main()

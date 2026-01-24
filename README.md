## Struktura repozytorium

```text
WMUM/
├── .venv/                       # Virtualenv (lokalnie, ignorowane przez git)
├── data/
│   ├── raw/                     # Dane wejściowe (np. eksporty z Roboflow) – duże, zwykle poza Gitem
│   └── processed/               # Dane przetworzone używane przez trening/ewaluację
│       ├── coco/                # Format COCO dla Torchvision
│       │   ├── cats_dogs/{train,valid,test}/...
│       │   ├── traffic_signs/{train,valid,test}/...
│       │   └── vehicles/{train,valid,test}/...
│       ├── yolov8/              # Format YOLOv8 (train/valid/test + labels/images + data.yaml)
│       │   ├── cats_dogs/...
│       │   ├── traffic_signs/...
│       │   └── vehicles/...
│       └── _reports/            # Raporty z prepare_data (np. naprawy labeli)
│
├── notebooks/                   # Notebooki krok-po-kroku (pipeline projektu)
│   ├── 01_prepare_data.ipynb
│   ├── 02_train_yolo.ipynb
│   ├── 03_train_torchvision.ipynb
│   ├── 04_eval_models.ipynb
│   ├── yolov8n.pt               # Bazowe wagi YOLOv8 (checkpoint startowy)
│   └── yolov8s.pt               # Bazowe wagi YOLOv8 (checkpoint startowy)
│
├── runs/
│   ├── eval_vis/                # Lekkie wyniki do porównań (obrazy z bbox + CSV)
│   │   ├── ultralytics/{dataset}__{model}/...__pred.jpg
│   │   ├── torchvision/{dataset}__{method}/...__pred.jpg
│   │   └── quick_eval_all.csv
│   ├── ultralytics/             # Ciężkie artefakty treningu YOLO (wagi, wykresy) – ignorowane w .gitignore
│   └── torchvision/             # Ciężkie artefakty treningu Torchvision (checkpointy) – ignorowane w .gitignore
│
├── src/                         # Kod źródłowy (skrypty używane także w notebookach)
│   ├── prepare_data.py          # Przygotowanie/naprawa danych (YOLOv8 + COCO)
│   ├── train_ultralytics.py     # Trening YOLOv8 (Ultralytics)
│   ├── train_torchvision.py     # Trening modeli Torchvision (Faster R-CNN / RetinaNet)
│   └── eval_models.py           # Ewaluacja + wizualizacje predykcji (siatki obrazków)
│
├── requirements.txt
├── .gitignore
└── README.md

## Struktura repozytorium

```text
├── .venv/                       
├── data/
│   ├── raw/                  #.gitignore         
│   └── processed/            #.gitignore         
│       ├── coco/                
│       │   ├── cats_dogs/{train,valid,test}/...
│       │   ├── traffic_signs/{train,valid,test}/...
│       │   └── vehicles/{train,valid,test}/...
│       ├── yolov8/              
│       │   ├── cats_dogs/...
│       │   ├── traffic_signs/...
│       │   └── vehicles/...
│       └── _reports/            
│
├── notebooks/                   
│   ├── 01_prepare_data.ipynb
│   ├── 02_train_yolo.ipynb
│   ├── 03_train_torchvision.ipynb
│   ├── 04_eval_models.ipynb
│   ├── yolov8n.pt               
│   └── yolov8s.pt              
│
├── runs/
│   ├── eval_vis/                
│   │   ├── ultralytics/{dataset}__{model}/...__pred.jpg
│   │   ├── torchvision/{dataset}__{method}/...__pred.jpg
│   │   └── quick_eval_all.csv
│   ├── ultralytics/                #.gitignore                 
│   │     └── summary.csv
│   └── torchvision/                #.gitignore               
│   │     └── summary.csv
│
├── src/                         
│   ├── prepare_data.py         
│   ├── train_ultralytics.py     
│   ├── train_torchvision.py    
│   ├── train_torchvision_gray.py   
│   ├── EvalPlots.py 
│   └── eval_models.py           
│
├── requirements.txt
├── .gitignore
└── README.md


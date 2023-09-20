노면/기상 데이터 라벨링 전처리 코드
==============================

## Installation

### Requirements

- Anaconda
- python 3.8
- pip

### Steps

Conda
```
conda install pytorch torchvision torchaudio cpuonly -c pytorch
pip install ultralytics
```

## Usage

다음은 간단한 예시입니다

python
```
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
results = model.predict(source='<image.jpg>', save=True, save_txt=True)
```
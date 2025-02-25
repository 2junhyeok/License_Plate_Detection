# Performance enhancement method for license plate recognition

YOLOv11n을 활용하여 차량 번호판 탐지를 수행합니다.

성능을 고도화하기 위한 두 가지 방법
- 이미지를 4분할하여 멀리 있는 번호판을 탐지
- 이미지에서 자동차를 탐지하여 crop한 다음 번호판을 탐지

## Requirements
```
pip install -r requirements.txt
```

## Environment
```
conda create -n odenv python==3.11.11
conda activate odenv
```

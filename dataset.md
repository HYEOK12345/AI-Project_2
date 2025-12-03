## Dataset Download 경로 및 방법

이 프로젝트는 Roboflow에서 제공하는 Varroa Mite Dataset을 사용함.

Dataset URL:
https://universe.roboflow.com/varroa-virus-detection/varroa-mites-detector/dataset/2

다운로드 후 YOLO 형식으로 export한 zip 파일을 다음 명령으로 구성함:
```
mkdir -p datasets/varroa_mite
cd datasets/varroa_mite

unzip varroa_mite_yolo.zip -d
```
데이터 구조
```
datasets/
└── varroa_mite/
    ├── images/train
    ├── images/val
    ├── images/test
    ├── labels/train
    ├── labels/val
    └── labels/test
```

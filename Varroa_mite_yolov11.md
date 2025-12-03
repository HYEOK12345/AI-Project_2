# 전이 학습을 적용한 YOLOv11 경량 모델의 꿀벌응애 자동 탐지 성능 분석

---

## 1. 프로젝트 개요 및 제목

본 프로젝트는 YOLOv11 모델을 활용하여 꿀벌에 기생하는 응애(Varroa Mite)를 자동으로 탐지하는 연구입니다.  
특히 경량 모델(n/s/m)에 전이 학습을 적용하여 실제 양봉 현장에서 활용 가능한 모델인지 분석하는 것을 목표로 합니다.

---

## 2. Framework 구조도

<img width="1283" height="638" alt="image" src="https://github.com/user-attachments/assets/d89ed929-499a-4f32-8555-0e7c0c154f85" />

---

## 3. Abstract

꿀벌은 농업 생산과 생태계 유지에 중요한 역할을 하지만, 꿀벌응애(Varroa Mite)는 꿀벌의 생존과 양봉 산업에 큰 피해를 주는 주요 요인으로 알려져 있다. 기존의 응애 탐지는 대부분 사람이 직접 육안으로 확인하는 방식이어서 정확도와 효율성이 떨어진다.  

본 프로젝트에서는 YOLOv11의 경량 모델을 이용하여 응애를 자동으로 탐지하고, 전이 학습을 적용하여 모델이 제한된 데이터에서도 안정적으로 학습할 수 있는지 확인하였다. 또한 프레임워크 전반에 다양한 데이터 증강 기법을 적용하여 소형 객체(응애) 탐지 성능을 개선하였다.  

이 연구는 실시간 양봉 관리 시스템에 활용할 수 있는 기초 모델 구축을 목표로 한다.

---

## 4. data.yaml 파일

YOLOv11 학습을 위한 데이터 설정 파일은 아래와 같이 구성함.

```yaml
# data.yaml

path: ./datasets/varroa_mite

train: images/train
val: images/val
test: images/test

nc: 1
names:
  0: varroa_mite
```


---

## 5. Dataset Download 경로 및 방법

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
## 6. 기본 실행 코드
1) 패키지 설치
```
git clone https://github.com/USER/varroa-yolov11.git
cd varroa-yolov11

pip install -r requirements.txt
```
2) 모델 학습 실행
```
yolo train \
  model=yolo11s.pt \
  data=data.yaml \
  epochs=50 \
  imgsz=640 \
  batch=16 \
  project=runs/train \
  name=varroa_y11s
```
2-1) YOLOv11의 경량 모델 비교 실험
```
yolo train model=yolo11n.pt data=data.yaml epochs=50 imgsz=640 batch=16
yolo train model=yolo11s.pt data=data.yaml epochs=50 imgsz=640 batch=16
yolo train model=yolo11m.pt data=data.yaml epochs=50 imgsz=640 batch=16
```
3) 모델 평가
```
yolo val \
  model=runs/train/varroa_y11s/weights/best.pt \
  data=data.yaml \
  imgsz=640
```
4) 이미지/폴더 추론 실행
```
yolo predict \
  model=runs/train/varroa_y11s/weights/best.pt \
  source=examples/ \
  imgsz=640 \
  save=True
```

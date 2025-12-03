## data.yaml 파일

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


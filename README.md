# keras 를 위한 cuda tensorflow 세팅

하기 url 참고
https://altiai.atlassian.net/wiki/spaces/VOIC/pages/274137089/ML+Icon+Recognition

# 빠른 실행
$ conda env create -f tensorflow.yml -m tensorflow

$ conda activate tensorflow

*iconDataset을 project root directory로 이동시키거나 링크를 겁니다*

$ python imageConverter.py // data를 변환하여 jpgImages를 생성합니다.

$ python classification.py // 학습하여 saved_models 에 model 생성

$ python evaluation.py // 학습한 model을 사용하여 dataset으로 검증합니다

$ python prediction.py // 원하는 라벨에 대해 테스트해볼 수 있습니다.


# 데이터셋

dataset은 deepth01 컴퓨터의 하기 경로에 있습니다.

/home/resource/iconDataset


# 모듈 설명

1) *이미지 정제* : imageConverter.py

refineImageData() : 

python 을 통해 실행하면 불리는 함수로, 동일 경로의 dataset directory(./iconDataset)를 찾아 데이터 정제작업을 합니다.

디렉토리의 구조는 
datasetDirectory/[labelName]/[fileName]
의 형태로 되어야합니다.

png 이미지 데이터를 대상으로 배경색을 달리하는 augmentation을 실시하며, datasetDirectory와 동일한 구조의 동일 갯수의 이미지 파일들을 포함하는 jpgImages를 형성합니다.

classification.py에서 호출하는 함수로
정제된 데이터 Directory -"jpgImages"로부터 keras에 feeding할 데이터를 생성합니다.

output data의 형태는 (x_train, y_train), (x_test, y_test)로 
각각은 numpy array입니다. x는 image data y는 숫자로 변환된 label(index) 이며, 같은 index의 대칭하는 x와 y가 data와 그의 label 입니다.


2) 학습 : 정제된 데이터를 바탕으로 model에 feeding하여 학습을 진행합니다.
학습된 모델은 ./saved_models에 저장됩니다.

$ python classification.py


3) 학습 모델에 대한 평가 : 무작위 데이터를 선발해 이를 통해 학습 모델에 대한 evaluation을 진행합니다.

$ python evaluation.py


4) 학습 모델을 활용한 예측 : 원하는 이미지에 대해 인식 결과를 확인할 수 있습니다.

$ python prediction.py

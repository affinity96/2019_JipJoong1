2019 IT집중교육 1 프로젝트
# 당신이 살찌는 이유 : 음식사진 기반 칼로리 계산 서비스

## 주제 선정
 우리는 수업시간에 배웠던 머신 러닝을 활용하여 분류 혹은 예측할 수 있는 주제를 생각하다 이전 주제 선정 시 보류해 두었던 ‘음식 사진 데이터 셋을 활용한 음식의 분류 및 칼로리 등의 서비스 제공’을 주제로 선정하였다. 이미지 데이터를 활용하여 모델을 학습하는데 부담이 될 수 있을 것 같았지만, 분류에 괜찮은 정확도를 보여주기만 한다면 간단할 수 있는 주제이기에 선정하였다. 
체중과 다이어트 문제는 현대 사람들에게 많은 이슈를 불러 일으키고 있다. ‘먹은만큼 칼로리를 소비하지 않는다.’라는 어떻게 보면 굉장히 간단한 문제 때문에 살이 찌는 것인데, 일일이 오늘 먹은 음식들을 기록하고 칼로리를 찾아보고 어떤 운동을 얼만큼 해야 칼로리가 소모되는지를 찾는다면, 여간 귀찮은 일이 아닐 것이다. 따라서 우리는 섭취한 음식의 사진으로 무슨 음식인지 파악하고, 해당 음식의 칼로리와 그에 따른 알맞은 운동량을 계산하여 보다 직관적으로 현대인들의 체중감량에 도움이 되고자 하였다. 

## 데이터 셋 및 전처리
### 데이터 수집
 우리는 초기에 AI-HUB(http://www.aihub.or.kr/content/140)에서 제공되는 3000장씩, 30종의 한식 요리 샘플 이미지 데이터를 이용하기로 하고, 추후에 KIST에서 제공되는 150종의 연구목적 요리 이미지 데이터베이스를 신청하여, 더 많은 종류의 요리 이미지 데이터셋을 통해 모델링에 사용할 수 있었다.

#### 깻잎 장아찌 폴더의 이미지 파일
![그림1](https://user-images.githubusercontent.com/53653160/111269674-42cc2180-8672-11eb-838b-4cc6f947a037.png)

### 데이터 전처리
#### 이미지 리사이즈
 수집한 데이터의 크기가 제 각각이었기 때문에, 우리는 이미지의 크기를 가로 128픽셀, 세로 128픽셀로 똑같이 resize하였다. 다음은 이미지를 resize 하는 코드이다.
 
![그림2](https://user-images.githubusercontent.com/53653160/111269678-43fd4e80-8672-11eb-9b8f-60de161ffa0f.png)

 우리는 이미지의 크기를 보다 더 크게 resize한다면, 보다 높은 accuracy를 얻을 수 있을 것이라 기대하였으며, 이후 학습에서는 가로 224픽셀, 세로 224픽셀로 resize한 뒤 진행하였다.
 
![그림3](https://user-images.githubusercontent.com/53653160/111269681-4495e500-8672-11eb-80b7-14396acc8c57.png)

#### Numpy array 변환
 우리가 가지고 있는 컴퓨터 자원으로는 10,000장에 달하는 이미지를 학습하는데 무리가 있다고 판단하여, 모델 학습은 구글 Colab을 활용하여 진행하였다. 이때, 이미지 전체를 드라이브에 올리면 용량 문제가 발생하므로, 이미지를 numpy array로 변환하여 구글 드라이브에 올리기로 하였다. 초기 제공받은 9천여장의 음식 사진으로 모델 선정을 하기로 하고 추후에 수집한 15만여 장의 데이터로 모델과 분류 량을 더 늘여 나가고자 하였다.
 추후 제공 받은 15만여장의 데이터는 150 종의 음식 종류마다 천 여장씩 사진이 존재했는데, 이 사진들을 모두 가로 세로 128 크기로 resize 하여도 여전히 방대한 크기였기에, 하나의 음식에 몇 장의 이미지 데이터가 있을 때 성능이 최적인지 비교하기로 하였다. 이를 위해 한 종류의 음식당 각 100, 200, 00, 500장의 사진을 이용한 데이터 셋을 각각 만들어 모델 학습에 사용하였다.

## 모델 형성 및 비교

### Simple CNN 모델 구성

![그림4](https://user-images.githubusercontent.com/53653160/111269682-452e7b80-8672-11eb-992b-6192b4d44433.png)

 우리가 고안한 SImple CNN의 모델은 위와 같이 9개의 convolution layer로 구성된다.  각 레이어의 활성함수로 relu함수를 사용하였다. 우리는 각 layer 마다 Batch Normalization을 사용하였는데, layer의 수가 많아지면 생기는 vanishing/gradient 문제를 해결하기 위해 layer를 지날 때마다 정규화를 해주어 더 좋은 정확도가 나올 수 있게 했다. 4개의 layer을 지날 때마다 Dropout을 하였으며, 이를 조금씩 증가시키자 정확도가 좋게 나타났음을 볼 수 있었다.  Simple CNN모델에 대해서는 30가지의 음식 종류에 대한 분류를 했으므로, Convolutional layer를 모두 거친 뒤에는 Flatten()을 통해 모델을 1차원으로 줄이고 Dense를 통해 전연결층으로 만들어준 뒤 마지막으로 결과분류를 위한 30개의 아웃풋을 가지는 전연결층을 만들어주고 softmax를 통해 출력을 0~1사이로 정규화해서 분류 결과를 나타낼 수 있게 했다. 

![그림5](https://user-images.githubusercontent.com/53653160/111269685-452e7b80-8672-11eb-8ace-178bceb2b059.png)

 아래의 함수는 Data augmentation 함수로 기존의 training set image을 변형하여 추가적인 데이터를 추가하여 모델의 정확도를 높이기 위해 수행한다. 기존의 이미지를 30도 기울이거나, 0.2배 만큼 움직이게 하고, 이미지를 반전시키는 등의 augmentation을 거침으로써 정확도를 올리고자 하였다.
 
![그림6](https://user-images.githubusercontent.com/53653160/111269688-45c71200-8672-11eb-95da-5ab935d2e2cd.png)

 Overfitting을 방지하기 위해 학습 중 20 epoch동안 더 이상 정확도의 발전 이 없다면 설정한 200 epoch만큼 학습을 하지 않아도 먼저 끝내도록 하였다. 또한 checkpoint call back 함수를 이용해서 지정한 지점에서 학습을 재개하거나 완료된 모델을 불러와 사용할 수 있도록 설정하였다. Adam optimizer와 위의 checkpoint, early stopping 콜백함수를 이용하였는데 checkpoint를 이용하여 validation data에 대한 정확도가 갱신될 때마다 모델을 저장하였고, early stopping을 이용하여 과대적합을 방지하였다. 또한 ReduceLRonPlateau를 이용하여 validation accuracy가 epoch진행시 개선이 없을 경우 learning rate를 0.5배씩 줄이는 기법을 진행하여 학습효율을 증대하였다. 무작위로 선정된 validation data를 사용하여 검증 후 테스트 데이터에 대한 정확도를 출력하게 하였다.

### 타 모델과의 비교 결과

![그림9](https://user-images.githubusercontent.com/53653160/111269692-465fa880-8672-11eb-84ee-742476a2813a.png)

30종의 sample 이미지 데이터셋에서 우리가 고안한 Simple CNN 모델이 다른 모델에 비해 좋은 accuracy를 나타냄을 확인하였고, 150종의 요리사진 데이터셋으로부터 100, 200, 300, 500개씩의 사진파일이 들어 있는 데이터셋 과 128*128, 224*224 pixel사이즈로 resizing한 데이터 셋에 대해서 최적의 모델을 비교하였고 다음과 같이 300개씩의 사진파일을 사용하고, 128X128 pixel사이즈로 resizing하였을때 성능의 최적 모델을 생성할 수 있었다. 아래와 같이 최적 모델은 79번째 epoch에서 early stopping에 의해 조기 종료되었고 최고 성능은 57.967%의 validation accuracy를. test accuracy는 57.027% test loss는 2.078(categorical cross entropy loss function)의 성능을 출력하였다. 이때 model checkpoint를 이용하여 저장한 최고 validation accuracy 성능을 보인 모델을 이용하여 서비스를 구현하였다.

![그림10](https://user-images.githubusercontent.com/53653160/111269693-465fa880-8672-11eb-99b1-62ff44f653d2.png)

## 웹 페이지 화면

### 초기 웹페이지
![그림11](https://user-images.githubusercontent.com/53653160/111269697-46f83f00-8672-11eb-80f8-1d7fba40cac5.png)

### 단일 음식 출력
![그림12](https://user-images.githubusercontent.com/53653160/111269701-46f83f00-8672-11eb-9e0a-7002047e5659.png)

### 다중 음식 출력
![그림13](https://user-images.githubusercontent.com/53653160/111269703-4790d580-8672-11eb-961d-d38b90ee4e92.png)

### 운동량 추천
![그림14](https://user-images.githubusercontent.com/53653160/111269704-48296c00-8672-11eb-855b-f48a5e52ae44.png)



# 팀: 운전만해~~♬♪(김수은,박선후,윤다빈,황종원)

# 호남 지역의 '안전'지킴이
 # "Driver"
### Safety Driving Manager
> ### AI를 이용해 운전자들의 운전 습관을 측정함으로써 교통안전을 촉진하는 서비스

## * Background
> #### **호남 지역의 인구 대비 교통 사고 발생 건수 전국 2위!!**
![](https://images.velog.io/images/parksh089g/post/d3134ec7-11ab-4673-9039-9df9246085ad/image.png)

통계청(KOSIS)의 데이터 분석 결과 호남지역(광주, 전북, 전남)이 2019년 인구 대비 교통사고 비율 2위를 차지하고 있음을 확인
이러한 도시교통문제의 근본적 해결방안으로 운전자의 잘못된 운전 습관 해결을 위한 서비스의 필요성을 느끼고 본 프로젝트를 시작

![](https://images.velog.io/images/parksh089g/post/7c84ab8e-61cf-400a-81a3-3363cae44e05/image.png)
## * Abstract
**-AI(인공지능)를 이용한 운전습관 분석 웹서비스 "Driver"-**

사전에 학습된 객체(신호등, 도보, 장애물)와 차선 등을 바탕으로 교통법규에 의거하여 사용자의 위험한 운전 습관에 대한 피드백을 웹을 통해 제공, 간결한 UI & UX 구성으로 서비스 위치와 사용 방법을 쉽게 이해 가능
운전자는 주행 영상이 담긴 블랙박스 영상을 업로드 한 후 분석을 진행할 수 있으며, ‘분석 결과 조회’ 페이지를 활용하여 잘못된 운전 습관을 파악하고 객체와 차선이 인식된 분석 결과 동영상도 확인할 수 있음

![image](https://user-images.githubusercontent.com/74482398/129962321-374d59dc-50fc-4ddf-9b90-18daf0d5f639.png)

착한 마일리지와 같은 개념으로 좋은 점수를 획득한 운전자에게 마일리지를 제공하여 추후에 벌점 감면 등의 혜택을 제공하는 것을 목표로 함

![](https://images.velog.io/images/parksh089g/post/ca6ff051-b7e5-404f-b9d9-4b7e27ecb059/image.png)

![flowchart](https://user-images.githubusercontent.com/74482398/130003193-e978342b-87bb-4f44-8735-18d7a91366d5.jpg)

## * Object Detection
pytorch 기반의 객체 인식 모델인 yolov5을 사용

**1.Dataset(데이터셋)**
약 17만여 장의 이미지 셋(xml)을 전처리 과정을 통해 학습에 알맞게 변환
>17만여 장 이미지 클래스
>* 장애물(맨홀뚜껑, 로드크랙 등)
>* 차량(트럭, 승용차 등)
>* 보행자
>* 차선
>* 신호등(빨간불,초록불,노란불/직진 및 좌회전신호 등)
>* 교통 표지판(제한속도, 어린이보호구역 등)
>* 로드마크(정지선, 횡단보도, 화살표, 유턴 등)

**2.Evaluation and Validation**
>* epoch 24/ batch-size 6 / input-size 1280 / RTX 3090 / pretrained weight yolov5x6
>* train set: 167,621
>* val set: 10,371
>* mAP: 0.69

**3.결과**

![](https://images.velog.io/images/parksh089g/post/7865a370-d8a3-447b-8020-fa9c6b255d41/image.png)
![](https://images.velog.io/images/parksh089g/post/a2057c7a-7e90-47e2-b43c-c5a2bcc7c33d/%EC%BA%A1%EC%B2%982.JPG)

## * Lane Detection
딥러닝 기반 차선 감지로 openCV만을 이용한 차선 인식 보다 정확도를 향상 시켜 개선

**1. 개발 환경** 

![](https://images.velog.io/images/parksh089g/post/0218ef96-37d5-458a-a8ae-0e625f143764/image.png)

**2.CNN(Convolution Neural Network) 기반 구조**

![](https://images.velog.io/images/parksh089g/post/cbd128d3-701d-4d31-9d59-4437af5b7ab7/image.png)

4개의 Convolution Layers – 각 층 거듭할수록 Filter 수 감소
4개의 Fully-Connected Layers – 마지막 FC 층은 6개의 ouput(차선 라인 6개 계수)
Dropout Layer – Overfitting, Robustness 증가 막기 위함
Crop Layer – 상위 세개 이미지 제거
mean-squared-error for loss – 다양한 커브길 데이터 다루기 위함
shuffled data – 특정 비디오에 대한 Overfit 막기 위함

**3.Evaluation Metrics(평가지표)**

우리의 목표는 모델에 학습시킨 차선을 토대로 새로운 영상에 차선을 그려 넣는 것
따라서 훈련 중 Loss(각 차선들의 실제 계수들과 모델의 예측 사이 차이) 최소화를 위해 MSE(Mean-Squared-Error)를 사용

**4.Dataset(데이터셋)**

Youtube 영상들 활용(1280*720, 30fps) -> (80*160)로 사이즈 줄여 훈련 시간 감소
6개 계수값 레이블 계산을 위해 Computer Vision 기술 적용
-OpenCV Image Calibration : 카메라 왜곡 바로 잡기
-Perspective Transformation : 차선을 평평한 평면 위에 올리기

**5.Evaluation and Validation**

훈련 0.0046, 평가 0.0048 MSE 수치

**6.결과**

![](https://images.velog.io/images/parksh089g/post/f7d0d7f1-234d-4cc3-b612-95b48e3771f4/image.png)
![](https://images.velog.io/images/parksh089g/post/d8b4d972-3665-4d6a-933e-f6c52476a7f1/image.png)

## * Distance Calculate and Relative Speed

![](https://images.velog.io/images/parksh089g/post/90231343-1f22-4911-a2ad-be4d5ac2a31a/image.png)

다음 그림은 차량이 영상 평면에 투영될 때의 모습을 나타낸 것이다. W 는 차량의 실제 폭을 나타내고, D는 카메라와 차량이 떨어진 실제 거리, w는 차량이 영상 평면에 투영되었을 때의 차량 폭(width), 그리고 f는 화소로 환산했을 때의 초점 거리이다. 이 때 식 (1)과 같은 비례식이 성립한다. 여기서 실제 차량과의 거리를 구하는 식은 (2)가 된다.
![](https://images.velog.io/images/parksh089g/post/ad4bc35b-e4a4-460a-83c7-8c208db36c1c/image.png)![](https://images.velog.io/images/parksh089g/post/ce5838c2-06a2-43bc-a78c-7453d2a5cf3e/image.png) 

항상 같은 카메라로 촬영하므로 초점거리 f는 일정한 상수값이 되고, 차량의 실제 폭이 항상 동일하다고 가정했을 때, W도 상수가 되어 다시 식 (3)과 같이 표현할 수 있다.

![](https://images.velog.io/images/parksh089g/post/7c52dca3-57bc-4a8e-b660-603b526282bc/image.png) 

R은 결국 특정한 차량에 대해 차량의 실제 폭과 카메라의 초점 거리의 곱으로서, 동일한 차량, 동일한 카메라일 경우 차량의 거리와 상관없이 항상 일정한 값을 가진다.

frame 단위로 객체의 위치를 찾는 yolov5를 활용하여 위와 같은 방법을 통해 차간 거리를 계산하였다. 그리고 이전 프레임과 현재 프레임의 거리차를 구한 뒤, fps를 곱해 주면 프레임 단위로 상대속도를 알 수 있다.
## * Web
**실행**

- python start_flask.py 로 실행

- __init__.py 파일로 각 페이지 및 인공지능 연동

  

**mainpage 화면**

![img_title](https://github.com/2021-oasis-hackathon/WeRide/blob/main/images/%EB%A9%94%EC%9D%B8%ED%99%94%EB%A9%B4.png)

- mainpage.html, mainpage.css

  > 메인화면



**SERVICE 화면**

![img_title](https://github.com/2021-oasis-hackathon/WeRide/blob/main/images/SERVICE%ED%99%94%EB%A9%B4.png)

![image-20210819025505360](https://github.com/2021-oasis-hackathon/WeRide/blob/main/images/SERVICE%ED%99%94%EB%A9%B42.png)

- servicepage.html, servicepage.css

  > SERVICE 화면
  >
  > 서비스 시작 버튼으로 파일업로드 화면으로 이동

  

**UPLOADING 화면**

![image-20210819025830674](https://github.com/2021-oasis-hackathon/WeRide/blob/main/images/%EC%97%85%EB%A1%9C%EB%93%9C%ED%99%94%EB%A9%B4.png)

- upload.html, upload.css

  > 파일 선택을 통해 영상 업로드
  >
  > 업로드된 영상이 detect파일과 연동



**분석중 화면**

![image-20210819025915051](https://github.com/2021-oasis-hackathon/WeRide/blob/main/images/%EB%B6%84%EC%84%9D%EC%A4%91%ED%99%94%EB%A9%B4.png)

![image-20210819030005016](https://github.com/2021-oasis-hackathon/WeRide/blob/main/images/%EB%B6%84%EC%84%9D%EC%A4%91%ED%99%94%EB%A9%B42.png)

- loading.html, loading.css

  > 선택한 파일을 분석 진행



##### 분석완료 화면

![image-20210819030122823](https://github.com/2021-oasis-hackathon/WeRide/blob/main/images/%EB%B6%84%EC%84%9D%EC%99%84%EB%A3%8C%ED%99%94%EB%A9%B4.png)

- success.html, success.css

  > 분석 완료 시 뜨는 화면



**SCORE점수 화면**

![image-20210819030138455](https://github.com/2021-oasis-hackathon/WeRide/blob/main/images/SCORE%ED%99%94%EB%A9%B4.png)

- score.html, score.css

  > 영상을 분석하여 점수파악 후 출력화면
  >
  > Click하면 분석결과 조회화면으로 넘어감



**분석결과 조회화면**



![image-20210819030405191](https://github.com/2021-oasis-hackathon/WeRide/blob/main/images/%EB%B6%84%EC%84%9D%EA%B2%B0%EA%B3%BC%20%ED%99%94%EB%A9%B4.png)

- analysis.html, analysis.css

  > 영상 분석 후 감점 항목을 파악하여 출력
  >
  > 분석결과 영상을 보여줌
## * Score Calculate
객체 인식, 차선 인식, 거리 계산, 상대속도 정보를 바탕으로 다음 시나리오에 대한 감점 요소 적용
(프로토타입에는 구현이 안되어있는 부분도 있습니다.)

**1. 신호등**(일부 미구현)
> - 빨간불에서 초록불로 바뀔 때, 이동 가능하다.
> - 초록불에서 황색불로 바뀔 때, 교차로에 진입한 차량은 교차로를 빠져나가야 하며, 진입하기 전이라면 정지한다.
> - 황색불에서 빨간불로 바뀔 때, 정차한다.
> - 빨간불일 때, 정차한다
> - 초록불일 때, 이동가능하다.
> - 황색불일 때, 교차로에 진입한 차량은 교차로를 빠져나가야 하며, 진입하기 전이라면 정지한다.
> - 황색 점멸 신호, 서행하면서 통행한다.
> - 적색 점멸 신호, 잠시 정차 후 좌우를 살핀 뒤 지나간다.
> - 비보호 표지판이 있을 경우, 녹색 신호일 때 맞은편 직진하는 차량이 없을 시 좌회전이 가능하다. 
> - 비보호 좌회전 교차로에서 신호등이 빨간불이면 정차해야한다.
> - 빨간색 ← 신호등에 불이 들어왔다면 정지해야한다.
> - 노란색 ← 신호등이 켜지면 교차로에 진입한 차량은 교차로를 빠져나가야 하며, 진입하기 전이라면 정지한다.
> - 노란색 ← 신호등이 켜지면 좌회전 차량도 정지할 준비를 한다.
> - 녹색 ← 신호등이 켜지면 좌회전 차량은 좌회전을, 녹색 원형 등에 불이 들어오면 직진 차량은 직진한다. 
> - 빨간색 ← 신호등이 켜지고, 녹색 원형 등에 불이 들어오면, 좌회전 차량은 정지하고 직진차량만 진행한다.
> - 녹색 ← 신호등이 켜지고, 빨간 원형 등에 불이 들어오면, 직진 차량은 정지하고 좌회전차량만 진행한다. 
> - 비보호 겸용 좌회전 신호등의 경우, 좌회전 신호등에서는 ‘좌회전 신호, 녹색 직진 신호’ 모두 좌회전이 가능하다.

**2. 표지판** (일부 미구현)
>**속도제한 표지판**
>- 숫자 아래에 검은 줄이 있을 경우, 최저 속도를 제한하는 표지판임을 의미한다. 반드시 주어진 속도 이상으로 운행해야 한다.
>- 숫자 아래에 검은 줄이 없을 경우, 최고 속도를 제한하는 표지판임을 의미한다. 반드시 주어진 속도 이하로 운행해야 한다.

**3. 로드마크** (일부 미구현)
>
> **노면표시_정지선**
>- 정지 신호에 따라 정차해야한다.
>- 비보호 좌회전은 적색 신호시 정지선에 멈췄다가 신호등이 녹색으로 바뀌면 교차로 중간까지 서행으로 나간 후 대기해 맞은편 차량이 모두 지나간 뒤에 좌회전한다.
> **노면표시_횡단보도**
>- 보행자가 도로를 건널 수 있으므로 서행한다.
> **노면표시_글자노면표시**
>- 반드시 주어진 속도 이하로 운행해야 한다.
>- 흰색 바탕에 빨간색 원일 경우, 어린이보호구역, 노인보호구역, 장애인보호구역을 의미하며 주어진 속도 이하로 운행해야 한다.
> **노면표시_글자노면표시**
>- 천천히 : 서행할 것을 지시한다.
>- 정지 : 일시정지할 것을 지시한다.
>- 버스 : 버스정차구획을 지시한다.
>- 어린이보호구역 : 어린이보호구역을 지시한다.
>- 노인보호구역 : 노인보호구역을 지시한다.
>- 장애인보호구역 : 장애인보호구역을 지시한다.
>- 화살표 아래에 글자가 쓰여진 것은, 화살표대로 진행하면 그 장소가 나온다는 뜻일 뿐 그 화살표대로 진행하라는 의미가 아니다. 
> **노면화살표_직진**
> - 직진 화살표가 있을 경우 직진 가능하다
> - 직진 화살표가 있을 경우 직진을 하기 위한 신호대기가 가능하다
> - 직진 화살표가 있을 경우 좌회전 신호대기가 불가능하다.
> - 직진 화살표가 있을 경우 좌/우회전이 가능하다
> **노면화살표_좌회전**
> - 좌회전 화살표가 있을 경우 직진이 가능하다.
> - 좌회전 화살표가 있을 경우 좌회전이 가능하다
> - 좌회전 화살표가 있을 경우 직진 신호 대기는 불가능하다
> - 좌회전 화살표가 있을 경우 좌회전 신호 대기는 가능하다
> **노면화살표_우회전**
> - 우회전 화살표가 있는 경우 직진이 가능하다
> - 우회전 화살표가 있는 경우 직진을 위한 신호 대기는 불가능하다
> - 우회전 화살표가 있는 경우 우회전이 가능하다
> **노면화살표_직진&좌회전**
> - 직진&좌회전화살표가 있을 경우 직진이 가능하다
> - 직진&좌회전화살표가 있을 경우 직진 신호대기가 가능하다
> - 직진&좌회전화살표가 있을 경우 좌회전이 가능하다
> - 직진&좌회전화살표가 있을 경우 좌회전 신호대기가 가능하다
> **노면화살표_직진&우회전**
> - 직진&우회전화살표가 있을 경우 직진이 가능하다
> - 직진&우회전화살표가 있을 경우 직진 신호대기가 가능하다
> - 직진&우회전화살표가 있을 경우 우회전이 가능하다
> **노면화살표_유턴**
> - 유턴 화살표가 있을 경우 직진이 가능하다
> - 유턴 화살표가 있을 경우 유턴이 가능하다
> **노면화살표_기타노면화살표**
> - 직진 화살표 + X : 직진 금지
> - 직진 및 우회전 화살표 + X : 직진 및 우회전 금지
> - 좌회전 화살표 + X : 좌회전 금지
> - 우회전 화살표 + X : 우회전 금지
> - 좌우회전 화살표 + X : 좌우회전 금지
> - 유턴화살표 + X : 유턴금지
> - 노면 색깔 화살표 : 주행 방향을 안내한다
> - 화살표 아래에 글자가 쓰여진 것은, 화살표대로 진행하면 그 장소가 나온다는 뜻일 뿐 그 화살표대로 진행하라는 의미가 아니다. 
> - 비보호 글씨와 좌회전 화살표의 경우 비보호 좌회전이 허용된다

**4. 차간 거리 유지**

**5. 차량 급감속**

**6. 보행자 발견시 서행**

**7. 차선이탈**

**8. 정지선 이전에 정차 유무** (미구현)

## * Differentiation from existing services
기존 운전 습관에 관한 선행 서비스들은 급 출발과 급 정지, 급 가속, 급 감속, 과속 등의 요소만을 파악할 수 있다는 단점이 존재

이에 본 서비스는 기술을 확장함으로써 기본적인 요소뿐만 아니라 운전자들이 잘 모르는 교통 법규 위반 사실과 직접적인 체감 상으로 느낄 수 없는 차간 거리 확보와 상대 속도 측정 등 다양한 기능의 분석이 가능

## * Benefit
* 운전자의 입장에서 문제점을 확인함으로써 호남 지역 교통 문제의 근본적 해결책을 찾을 수 있음
* 확대된 운전 습관 개선 서비스로 인하여 새로운 고객층 확대 및 시장 진출 가능성으로 가짐
* 점수의 범위에 따라 개런티를 부과하는 positive 방식을 채택하여 사용자들의 이용 촉진 기능

* 추후 운전자 마일리지 제도를 새로 도입하여 고객의 유입 활성화
* 추후 위 기능을 토대로 실시간 분석과 더 정밀한 분석이 가능한 어플리케이션으로 확장함으로써 편의성을 높인 서비스 제작 가능
* 정부, 기업 등의 참여 유도

## * Reference paper
* Lane Detection with Deep Learning-Michael Virgo
* 영상 기반의 차량 검출 및 차간 거리 추정 방법( Vision-based Vehicle Detection and Inter-Vehicle Distance Estimation ) - 김기석, 조재수
* 차량 장착 블랙박스 카메라를 이용한 효과적인 도로의 거리 예측방법( Effective Road Distance Estimation Using a Vehicle-attached Black Box Camera ) - 김진수
* 비전센서를 이용한 차선인식 및 차간거리 추정 방법( Vision-based Lane Detection and Inter-Vehicle Distance Estimation ) - 박 종 섭, 김 기 석, 노 수 장, 박 요 한, 조 재 수
* End-to-end Learning for Inter-Vehicle Distance and Relative Velocity Estimation in ADAS with a Monocular Camera - Zhenbo Song, Jianfeng Lu, Tong Zhang, Hongdong Li

# 프로젝트 개요

## 목적
- 강아지의 얼굴, 코, 눈을 검출하여 이미지를 저장
- 검출되어진 이미지를 통해서 Embedding Data 를 보관 및 메타 데이터 보관
- 저장되어진 Embedding Data 를 통해서 추후에 검출 되어진 이미지에 대해서 유사 개체 조회 기능

## 기능
- YOLO V4 를 통해서 Face, Nose, Eye Detect
- Embedding Model 을 통해서 Face, Nose, Eye 특징 추출
- Cosine Distance Calc, Similarity Calc

## 모델의 종류
- Detect: YoloV4
- Embedding: Dino(Facebook), mars-small128.pb(DeepSORT 논문에 사용되어진 Model)

## 서비스 흐름도
- 얼굴 등록
1. Client 에서 Canvas 를 통해서 이미지를 Base64 WS 방식으로 전송
2. Server 에서는 해당 이미지를 CV 로 변환하여 YoloV4 를 통해서 객체 검출(얼굴, 눈, 코)
3. 검출 되어진 이미지를 확인하여서 사용하기 적법한 정보인지 확인(얼굴, 코, 눈 이 정확하게 있는지 확인)
4. 얼굴, 코, 눈을 Embedding Model 을 통해서 특징을 추출
5. 얼굴, 코, 눈에 대한 이미지와 메타 데이터 및 Embedding 데이터를 DB에 저장

- 인증
1. Client 에서 Canvas 를 통해서 이미지를 Base64 WS 방식으로 전송
2. Server 에서는 해당 이미지를 CV 로 변환하여 YoloV4 를 통해서 객체 검출(얼굴, 눈, 코)
3. 검출 되어진 이미지를 확인하여서 사용하기 적법한 정보인지 확인(얼굴, 코, 눈 이 정확하게 있는지 확인)
4. 얼굴, 코, 눈을 Embedding Model 을 통해서 특징을 추출
5. DB 에 저장되어 있는 기존 얼굴, 눈, 코에 대한 데이터를 반려동물 기준으로 조회 진행
6. 얼굴, 눈, 코 검출되어진 데이터(Embedding) 로 Cosine Distance Calc 진행
7. 각각의 얼굴, 눈, 코 Distance 에 대하여 가중치 적용하여 Similarity Calc
8. 일정 Threshold 를 넘으면 같이 개체라고 판별하여 WS 를 통해서 개체 데이터 제공

## 특이사항
- DeepSORT 논문에 사용되어진 Embedding Model 의 pre-trained 되어진 성능이 그렇게 뛰어나지 않아서 Similarity Calc 진행시 너무 낮은 성능을 보여주고 있음
- DinoV2의 기본 성능이 괜찮은 것으로 보이고 ViT 기반 및 무라벨링 학습이다보니 추후에 더욱 더 괜찮은 학습성능이 기대되어짐
- 실시간성을 보장하기 위해서 WS 방식을 채택했으나 이 부분에 대해서는 다른 방식을 차용해도 문제없을 것으로 보임


## 그외 기능관련 코드
- train.py => yoloV11 을 통해서 학습을 위해 있는 코드
- make_animal Command 존재, 초기에 시연을 위해서 데이터를 셋업하는 코드, training_data 폴더를 기준으로 해서 생성 자세한 부분은 ai/utils/management/commands/make_animal 참고(cinerama74-5e998b30d5380746c61d5cb7-wwgcl8myeovqepif3eao-Miniature_Pinscher-true-FEMALE-20140801 파일이름 예시)
- yolo_eval Command 존재, YoloV4 에 대한 성능 평가 및 지표 제공을 위해 존재하는 코드 ai/utils/management/commands/yolo_eval 코드 제공
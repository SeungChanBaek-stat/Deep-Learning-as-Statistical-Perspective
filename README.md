<!-- # Deep-Learning-as-Statistical-Perspective
## 딥러닝의 통계적 이해 프로젝트 관련자료입니다.
GradCAM_FCDrop_OriginalGradCAM_v3 는 기존 GradCAM의 주목영역 표시방법을 기반으로 한 GradCAM의 불확실성 측정에 대한 파이썬 코드입니다.
GradCAM_FCDrop_ModifiedGradCAM_v6 는 Segmentation style GradCAM의 주목영역 표시방법을 기반으로 한 GradCAM의 불확실성 측정에 대한 파이썬 코드입니다. -->
# Deep Learning as Statistical Perspective

## 프로젝트 개요
딥러닝의 통계적 이해 프로젝트 관련 자료입니다. 이 프로젝트는 GradCAM을 이용하여 딥러닝 모델의 주목 영역을 시각화하고, 그 불확실성을 측정하는 방법을 연구합니다.

### 주요 파일 및 디렉토리 설명

- **GradCAM_FCDrop_OriginalGradCAM_v3**: 기존 GradCAM의 주목 영역 표시 방법을 기반으로 한 GradCAM의 불확실성 측정에 대한 파이썬 코드입니다.
- **GradCAM_FCDrop_ModifiedGradCAM_v6**: Segmentation 스타일 GradCAM의 주목 영역 표시 방법을 기반으로 한 GradCAM의 불확실성 측정에 대한 파이썬 코드입니다.

## 프로젝트 구조

Deep-Learning-as-Statistical-Perspective

|-- GradCAM_FCDrop_ModifiedGradCAM_v6.ipynb
|-- GradCAM_FCDrop_OriginalGradCAM_v3.ipynb
|-- GradCAM_test.py
|-- Instructions
|   `-- colab_instructions.txt
|-- README.md
|-- data
|   |-- gradcam_original_test_large
|   |   |-- test_or
|   |   |-- test_or_results
|   |   |-- test_tf
|   |   `-- test_tf_results
|   `-- gradcam_original_test_small
|       |-- results
|       |   |-- result_1.png
|       |   |-- result_2.png
|       |   |-- result_3.png
|       |   `-- result_4.png
|       `-- test
|           |-- img_1.JPEG
|           |-- img_2.JPEG
|           |-- img_3.JPEG
|           `-- img_4.png
|-- gradcam
|   |-- GradCAM_original.py
|   `-- GradCAM_original_test.py
|-- prerequisites
|   `-- prerequisites.py
|-- requirements.txt
<!-- `-- \261\327\267\241\265\345\304\267\300\307 \272\322\310\256\275\307\274\272 \303\370\301\244.pdf -->



### 설명

- **data/gradcam_original_test_small**: 이미지 데이터 4장의 파일이 저장되는 디렉토리입니다.
- **data/gradcam_original_test_large**: ILSVRC 2012 이미지넷 데이터 202장의 파일이 저장되는 디렉토리입니다.
- **prerequisites**: VGG-16의 클래스 인덱스 지정 및 시드 고정하는 함수가 들어있는 디렉토리입니다.
- **GradCAM_FCDrop_OriginalGradCAM_v3.ipynb**: 기존 GradCAM을 이용한 불확실성 측정 주피터 노트북입니다.
- **GradCAM_FCDrop_ModifiedGradCAM_v6.ipynb**: Segmentation 스타일 GradCAM을 이용한 불확실성 측정 주피터 노트북입니다.
- **GradCAM_test.py**: 기존 GradCAM을 이용한 불확실성 측정 코드입니다.
- **requirements.txt**: 프로젝트 의존성 목록입니다.
- **README.md**: 프로젝트 개요 및 설명 파일입니다.

## 사용 방법

1. 필요한 라이브러리를 설치합니다:
    ```bash
    pip install -r requirements.txt
    ```

2. GradCAM을 이용한 불확실성 측정을 수행합니다:
    ```bash
    python GradCAM_test.py
    ```

## 기여 방법

프로젝트에 기여하고 싶다면, 이슈를 생성하거나 풀 리퀘스트를 제출해 주세요. 모든 기여는 환영합니다!

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참고하세요.
# ML Study Examples

머신러닝과 딥러닝 학습을 위한 실습용 예제 코드 모음입니다.

## 📁 파일 구조

```
ml_study_example/
├── README.md
├── CLAUDE.md
├── 01_basic_regression.py      # 기본 회귀 분석
├── 02_classification_basics.py # 기본 분류 분석
├── 03_clustering_example.py    # 클러스터링
├── 04_data_preprocessing.py    # 데이터 전처리
└── 05_deep_learning_basics.py  # 딥러닝 기초
```

## 🎯 학습 목표

각 예제는 학부생이 머신러닝과 딥러닝의 핵심 개념을 이해하고 실습할 수 있도록 설계되었습니다.

## 📚 예제 설명

### 1. 기본 회귀 분석 (`01_basic_regression.py`)
- **내용**: 선형 회귀와 다항 회귀 구현
- **주요 개념**: 
  - 최소제곱법
  - 과적합과 과소적합
  - 모델 성능 평가 (MSE, R²)
- **시각화**: 회귀선, 예측 vs 실제값, 다항차수별 비교

### 2. 기본 분류 분석 (`02_classification_basics.py`)
- **내용**: 로지스틱 회귀와 결정 트리 분류
- **주요 개념**:
  - 이진 분류와 다중 클래스 분류
  - 결정 경계
  - 혼동 행렬
  - 예측 확률 분석
- **시각화**: 결정 경계, 혼동 행렬, 특성 중요도

### 3. 클러스터링 (`03_clustering_example.py`)
- **내용**: K-Means와 계층적 클러스터링
- **주요 개념**:
  - 엘보우 방법
  - 실루엣 점수
  - 덴드로그램
  - 클러스터링 평가 지표
- **시각화**: 클러스터 결과, 엘보우 그래프, 덴드로그램

### 4. 데이터 전처리 (`04_data_preprocessing.py`)
- **내용**: 완전한 데이터 전처리 파이프라인
- **주요 개념**:
  - 결측값 처리 (평균, 중앙값, 삭제)
  - 특성 스케일링 (표준화, 정규화)
  - 범주형 인코딩 (Label, One-Hot)
  - 특성 선택 (단변량, RFE, 중요도 기반)
- **시각화**: 전처리 전후 비교, 스케일링 효과, 특성 중요도

### 5. 딥러닝 기초 (`05_deep_learning_basics.py`)
- **내용**: 신경망, CNN, RNN 기본 구현
- **주요 개념**:
  - NumPy로 구현한 기본 신경망
  - Keras를 이용한 MLP, CNN, RNN
  - 활성화 함수 비교
  - 신경망 아키텍처 비교
- **시각화**: 훈련 과정, 예측 결과, 활성화 함수, 네트워크 구조

## 🚀 실행 방법

### 필수 라이브러리 설치
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### 딥러닝 예제를 위한 추가 설치 (선택사항)
```bash
pip install tensorflow
```

### 예제 실행
```bash
# 각 예제를 개별적으로 실행
python 01_basic_regression.py
python 02_classification_basics.py
python 03_clustering_example.py
python 04_data_preprocessing.py
python 05_deep_learning_basics.py
```

## 📖 학습 가이드

### 추천 학습 순서
1. **01_basic_regression.py** - 머신러닝의 기본 개념 이해
2. **02_classification_basics.py** - 분류 문제와 평가 방법 학습
3. **04_data_preprocessing.py** - 실제 데이터 처리 방법 익히기
4. **03_clustering_example.py** - 비지도 학습 개념 이해
5. **05_deep_learning_basics.py** - 딥러닝 기초 개념 학습

### 각 예제별 학습 포인트

#### 회귀 분석
- 선형 관계와 비선형 관계의 차이점
- 과적합을 방지하는 방법
- 모델 성능을 평가하는 다양한 지표

#### 분류 분석
- 확률 기반 분류의 이해
- 결정 경계의 의미
- 분류 성능 평가 방법

#### 클러스터링
- 최적 클러스터 수 결정 방법
- 클러스터링 결과 해석
- 다양한 클러스터링 알고리즘의 특징

#### 데이터 전처리
- 실제 데이터의 문제점과 해결 방법
- 전처리 순서의 중요성
- 각 전처리 기법의 적용 상황

#### 딥러닝
- 신경망의 동작 원리
- 다양한 네트워크 아키텍처의 특징
- 활성화 함수의 역할과 선택 기준

## 🔧 문제 해결

### 일반적인 오류와 해결방법

1. **ImportError: No module named 'tensorflow'**
   ```bash
   pip install tensorflow
   ```

2. **메모리 부족 오류**
   - 데이터 크기를 줄이거나 배치 크기를 조정하세요

3. **그래프가 표시되지 않는 경우**
   ```python
   import matplotlib
   matplotlib.use('Agg')  # GUI 없이 그래프 저장
   ```

## 📊 예제에서 다루는 주요 알고리즘

| 알고리즘 | 파일 | 용도 |
|---------|------|------|
| Linear Regression | 01 | 연속값 예측 |
| Polynomial Regression | 01 | 비선형 관계 모델링 |
| Logistic Regression | 02 | 이진/다중 분류 |
| Decision Tree | 02 | 해석 가능한 분류 |
| K-Means | 03 | 중심 기반 클러스터링 |
| Hierarchical Clustering | 03 | 계층적 클러스터링 |
| Neural Network | 05 | 복잡한 패턴 학습 |
| CNN | 05 | 이미지 처리 |
| RNN | 05 | 시계열 데이터 처리 |

## 🎓 추가 학습 자료

각 예제 실행 후 다음 사항들을 추가로 학습해보세요:

- **이론 심화**: 각 알고리즘의 수학적 배경
- **하이퍼파라미터 튜닝**: 성능 최적화 방법
- **교차 검증**: 모델 성능의 신뢰성 확보
- **앙상블 방법**: 여러 모델 조합으로 성능 향상
- **실제 데이터셋**: Kaggle, UCI ML Repository 활용

## 📝 과제 및 실습 아이디어

1. **회귀**: 다른 데이터셋으로 선형/비선형 관계 탐색
2. **분류**: 다양한 분류기 성능 비교 및 앙상블 구현
3. **클러스터링**: 다른 클러스터링 알고리즘 (DBSCAN, GMM) 적용
4. **전처리**: 실제 결측값이 많은 데이터셋으로 파이프라인 구축
5. **딥러닝**: 더 복잡한 네트워크 아키텍처 설계 및 실험

---

**주의사항**: 
- 모든 예제는 교육 목적으로 작성되어 실제 프로덕션 환경에서는 추가적인 최적화가 필요할 수 있습니다.
- 딥러닝 예제는 TensorFlow 설치가 필요하며, 설치되지 않은 경우 NumPy 구현만 실행됩니다.
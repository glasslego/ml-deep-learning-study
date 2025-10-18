"""
기본 분류 분석 예제
학부생을 위한 로지스틱 회귀와 결정 트리 분류 실습 코드
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정 (Mac의 경우)
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

def logistic_regression_example():
    """로지스틱 회귀 이진 분류 예제"""
    print("=== 로지스틱 회귀 이진 분류 예제 ===")
    
    # 이진 분류 데이터 생성
    X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, 
                             n_informative=2, n_clusters_per_class=1, random_state=42)
    
    # 훈련/테스트 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 모델 학습
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # 예측
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    
    # 성능 평가
    accuracy = accuracy_score(y_test, y_pred)
    print(f"정확도: {accuracy:.3f}")
    print("\n분류 보고서:")
    print(classification_report(y_test, y_pred))
    
    # 시각화
    plt.figure(figsize=(15, 5))
    
    # 데이터 분포
    plt.subplot(1, 3, 1)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.7)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Original Data Distribution')
    plt.colorbar(scatter)
    
    # 결정 경계 시각화
    plt.subplot(1, 3, 2)
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    
    Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, levels=50, alpha=0.8, cmap='RdBu')
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='black')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary')
    plt.colorbar()
    
    # 혼동 행렬
    plt.subplot(1, 3, 3)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    plt.tight_layout()
    plt.show()
    
    return model

def multiclass_classification_example():
    """다중 클래스 분류 예제 (아이리스 데이터셋)"""
    print("\n=== 다중 클래스 분류 예제 (아이리스 데이터셋) ===")
    
    # 아이리스 데이터셋 로드
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    
    # 훈련/테스트 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 로지스틱 회귀 모델
    lr_model = LogisticRegression(max_iter=200)
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    lr_accuracy = accuracy_score(y_test, lr_pred)
    
    # 결정 트리 모델
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)
    dt_pred = dt_model.predict(X_test)
    dt_accuracy = accuracy_score(y_test, dt_pred)
    
    print(f"로지스틱 회귀 정확도: {lr_accuracy:.3f}")
    print(f"결정 트리 정확도: {dt_accuracy:.3f}")
    
    # 시각화 (처음 두 특성만 사용)
    plt.figure(figsize=(15, 5))
    
    # 원본 데이터
    plt.subplot(1, 3, 1)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.7)
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title('Iris Dataset (First 2 Features)')
    plt.colorbar(scatter)
    
    # 로지스틱 회귀 결과
    plt.subplot(1, 3, 2)
    X_2d = X[:, :2]  # 처음 두 특성만 사용
    X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(
        X_2d, y, test_size=0.2, random_state=42)
    
    lr_2d = LogisticRegression(max_iter=200)
    lr_2d.fit(X_train_2d, y_train_2d)
    
    h = 0.02
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    
    Z = lr_2d.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.8, cmap='viridis')
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='viridis', edgecolors='black')
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title('Logistic Regression Decision Boundary')
    
    # 특성 중요도 (결정 트리)
    plt.subplot(1, 3, 3)
    importances = dt_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importance (Decision Tree)')
    
    plt.tight_layout()
    plt.show()
    
    return lr_model, dt_model

def compare_classifiers():
    """분류기 성능 비교"""
    print("\n=== 분류기 성능 비교 ===")
    
    # 데이터 생성
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=5,
                             n_redundant=2, n_classes=3, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 다양한 분류기
    classifiers = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
    }
    
    results = {}
    
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
        
        print(f"{name:20} - 정확도: {accuracy:.3f}")
    
    # 결과 시각화
    plt.figure(figsize=(10, 6))
    names = list(results.keys())
    accuracies = list(results.values())
    
    bars = plt.bar(names, accuracies, color=['skyblue', 'lightgreen'])
    plt.ylabel('Accuracy')
    plt.title('Classifier Performance Comparison')
    plt.ylim(0, 1)
    
    # 막대 위에 정확도 값 표시
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

def probability_analysis():
    """예측 확률 분석"""
    print("\n=== 예측 확률 분석 ===")
    
    # 이진 분류 데이터
    X, y = make_classification(n_samples=200, n_features=2, n_redundant=0,
                             n_informative=2, n_clusters_per_class=1, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # 예측 확률
    y_prob = model.predict_proba(X_test)
    
    # 확률 분포 시각화
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.hist(y_prob[y_test == 0, 1], alpha=0.7, label='Class 0', bins=20)
    plt.hist(y_prob[y_test == 1, 1], alpha=0.7, label='Class 1', bins=20)
    plt.xlabel('Predicted Probability for Class 1')
    plt.ylabel('Frequency')
    plt.title('Probability Distribution by True Class')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.scatter(range(len(y_test)), y_prob[:, 1], c=y_test, cmap='viridis', alpha=0.7)
    plt.axhline(y=0.5, color='red', linestyle='--', label='Decision Threshold')
    plt.xlabel('Sample Index')
    plt.ylabel('Predicted Probability for Class 1')
    plt.title('Prediction Probabilities')
    plt.legend()
    plt.colorbar(label='True Class')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 예제 실행
    logistic_regression_example()
    multiclass_classification_example()
    compare_classifiers()
    probability_analysis()
    
    print("\n실습 완료! 다음 사항들을 확인해보세요:")
    print("1. 로지스틱 회귀의 결정 경계와 확률 해석")
    print("2. 다중 클래스 분류에서의 성능 평가")
    print("3. 혼동 행렬의 해석 방법")
    print("4. 각 분류기의 장단점")
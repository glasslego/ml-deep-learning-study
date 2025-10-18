"""
기본 회귀 분석 예제
학부생을 위한 선형 회귀와 다항 회귀 실습 코드
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정 (Mac의 경우)
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

def create_sample_data():
    """샘플 데이터 생성"""
    np.random.seed(42)
    X = np.linspace(0, 10, 100)
    y = 2 * X + 1 + np.random.normal(0, 2, 100)  # 선형 관계 + 노이즈
    return X.reshape(-1, 1), y

def linear_regression_example():
    """선형 회귀 예제"""
    print("=== 선형 회귀 예제 ===")
    
    # 데이터 생성
    X, y = create_sample_data()
    
    # 훈련/테스트 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 모델 학습
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # 예측
    y_pred = model.predict(X_test)
    
    # 성능 평가
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"회귀 계수 (기울기): {model.coef_[0]:.2f}")
    print(f"절편: {model.intercept_:.2f}")
    print(f"평균 제곱 오차 (MSE): {mse:.2f}")
    print(f"결정 계수 (R²): {r2:.2f}")
    
    # 시각화
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test, y_test, alpha=0.7, label='실제 값')
    plt.plot(X_test, y_pred, 'r-', label='예측 값')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Linear Regression Example')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return model

def polynomial_regression_example():
    """다항 회귀 예제"""
    print("\n=== 다항 회귀 예제 ===")
    
    # 비선형 데이터 생성
    np.random.seed(42)
    X = np.linspace(0, 4, 100)
    y = 0.5 * X**3 - 2 * X**2 + X + 1 + np.random.normal(0, 1, 100)
    X = X.reshape(-1, 1)
    
    # 훈련/테스트 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 다양한 차수의 다항 회귀 비교
    degrees = [1, 2, 3, 4]
    plt.figure(figsize=(15, 10))
    
    for i, degree in enumerate(degrees):
        # 다항 특성 생성
        poly_features = PolynomialFeatures(degree=degree)
        X_train_poly = poly_features.fit_transform(X_train)
        X_test_poly = poly_features.transform(X_test)
        
        # 모델 학습
        model = LinearRegression()
        model.fit(X_train_poly, y_train)
        
        # 예측
        y_pred = model.predict(X_test_poly)
        
        # 성능 평가
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"차수 {degree} - MSE: {mse:.2f}, R²: {r2:.2f}")
        
        # 시각화를 위한 부드러운 곡선 생성
        X_plot = np.linspace(0, 4, 300).reshape(-1, 1)
        X_plot_poly = poly_features.transform(X_plot)
        y_plot = model.predict(X_plot_poly)
        
        # 서브플롯
        plt.subplot(2, 2, i+1)
        plt.scatter(X_test, y_test, alpha=0.7, label='실제 값')
        plt.plot(X_plot, y_plot, 'r-', label=f'차수 {degree} 예측')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.title(f'Polynomial Regression (degree {degree})')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def compare_models():
    """모델 비교 예제"""
    print("\n=== 모델 성능 비교 ===")
    
    # 실제 데이터셋 사용 (보스턴 주택 가격 대신 간단한 예제)
    from sklearn.datasets import make_regression
    
    X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {}
    
    # 선형 회귀
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    linear_pred = linear_model.predict(X_test)
    models['Linear'] = {
        'model': linear_model,
        'mse': mean_squared_error(y_test, linear_pred),
        'r2': r2_score(y_test, linear_pred)
    }
    
    # 2차 다항 회귀
    poly_features = PolynomialFeatures(degree=2)
    X_train_poly = poly_features.fit_transform(X_train)
    X_test_poly = poly_features.transform(X_test)
    
    poly_model = LinearRegression()
    poly_model.fit(X_train_poly, y_train)
    poly_pred = poly_model.predict(X_test_poly)
    models['Polynomial (degree 2)'] = {
        'model': poly_model,
        'mse': mean_squared_error(y_test, poly_pred),
        'r2': r2_score(y_test, poly_pred)
    }
    
    # 결과 출력
    for name, metrics in models.items():
        print(f"{name:20} - MSE: {metrics['mse']:.2f}, R²: {metrics['r2']:.3f}")

if __name__ == "__main__":
    # 예제 실행
    linear_regression_example()
    polynomial_regression_example()
    compare_models()
    
    print("\n실습 완료! 다음 사항들을 확인해보세요:")
    print("1. 선형 회귀의 계수와 절편의 의미")
    print("2. 다항 회귀에서 차수가 성능에 미치는 영향")
    print("3. 과적합(overfitting)과 과소적합(underfitting)의 개념")
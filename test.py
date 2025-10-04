# 1. 필요한 라이브러리 불러오기
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def main():
    # 2. 예제 데이터 생성
    # 공부 시간 (독립 변수 X)
    X = np.array([2, 4, 5, 6, 7, 9]).reshape(-1, 1)
    # 시험 점수 (종속 변수 y)
    y = np.array([60, 75, 77, 85, 90, 98])

    # Scikit-learn은 입력(X)으로 2D 배열을 기대하므로,
    # 1D 배열인 X를 .reshape(-1, 1)을 통해 2D 배열로 변환합니다.

    # 3. 선형 회귀 모델 생성 및 학습
    model = LinearRegression() # 모델 객체 생성
    model.fit(X, y) # 모델 학습 (최소제곱법을 통해 최적의 계수를 찾음)

    # 4. 학습 결과 확인
    # 기울기(Coefficient)와 절편(Intercept) 출력
    print(f"기울기 (β₁): {model.coef_[0]:.2f}")
    print(f"y절편 (β₀): {model.intercept_:.2f}")

    # 5. 새로운 데이터에 대한 예측
    # 예: 8시간 공부했을 때의 예상 점수는?
    new_x = [[8]] # 예측할 데이터도 2D 배열 형태여야 함
    predicted_y = model.predict(new_x)
    print(f"8시간 공부했을 때 예상 점수: {predicted_y[0]:.2f}점")

    # 6. 결과 시각화
    plt.figure(figsize=(8, 6))
    # 원본 데이터 산점도
    plt.scatter(X, y, color='blue', label='Actual Data')
    # 학습된 회귀선
    plt.plot(X, model.predict(X), color='red', linewidth=2, label='Regression Line')

    plt.title('Study Hours vs Exam Score')
    plt.xlabel('Study Hours')
    plt.ylabel('Exam Score')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
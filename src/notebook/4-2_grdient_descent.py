import matplotlib.pyplot as plt
import numpy as np

# 한글 폰트 설정 (Mac의 경우)
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False


def generate_data():
    """간단한 선형 데이터 생성 (y = 4 + 3x + noise)"""
    np.random.seed(42)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)
    return X, y


# 1. Batch Gradient Descent
def batch_gradient_descent(X, y, learning_rate=0.01, iterations=1000):
    m = len(y)
    X_b = np.c_[np.ones((m, 1)), X]  # 절편 추가
    theta = np.random.randn(2, 1)  # 파라미터 초기화

    history = []

    for i in range(iterations):
        gradients = 2 / m * X_b.T.dot(X_b.dot(theta) - y)
        theta = theta - learning_rate * gradients

        # 손실 계산
        mse = np.mean((X_b.dot(theta) - y) ** 2)
        history.append(mse)

    return theta, history


# 2. Stochastic Gradient Descent (SGD)
def stochastic_gradient_descent(X, y, learning_rate=0.01, epochs=50):
    m = len(y)
    X_b = np.c_[np.ones((m, 1)), X]
    theta = np.random.randn(2, 1)

    history = []

    for epoch in range(epochs):
        for i in range(m):
            random_index = np.random.randint(m)
            xi = X_b[random_index : random_index + 1]
            yi = y[random_index : random_index + 1]

            gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
            theta = theta - learning_rate * gradients

        # 에포크마다 손실 계산
        mse = np.mean((X_b.dot(theta) - y) ** 2)
        history.append(mse)

    return theta, history


# 3. Mini-batch Gradient Descent
def mini_batch_gradient_descent(X, y, learning_rate=0.01, epochs=50, batch_size=20):
    m = len(y)
    X_b = np.c_[np.ones((m, 1)), X]
    theta = np.random.randn(2, 1)

    history = []

    for epoch in range(epochs):
        shuffled_indices = np.random.permutation(m)
        X_b_shuffled = X_b[shuffled_indices]
        y_shuffled = y[shuffled_indices]

        for i in range(0, m, batch_size):
            xi = X_b_shuffled[i : i + batch_size]
            yi = y_shuffled[i : i + batch_size]

            gradients = 2 / batch_size * xi.T.dot(xi.dot(theta) - yi)
            theta = theta - learning_rate * gradients

        # 에포크마다 손실 계산
        mse = np.mean((X_b.dot(theta) - y) ** 2)
        history.append(mse)

    return theta, history


def run_gradient_descent_comparison():
    """Gradient Descent 알고리즘 비교 실행"""
    print("=== Gradient Descent 알고리즘 비교 ===")
    print()

    # 데이터 생성
    X, y = generate_data()

    # 각 알고리즘 실행
    theta_batch, history_batch = batch_gradient_descent(
        X, y, learning_rate=0.1, iterations=50
    )
    theta_sgd, history_sgd = stochastic_gradient_descent(
        X, y, learning_rate=0.01, epochs=50
    )
    theta_mini, history_mini = mini_batch_gradient_descent(
        X, y, learning_rate=0.1, epochs=50, batch_size=20
    )

    # 결과 출력
    print("=== 결과 비교 ===")
    print("실제 값: θ0 = 4, θ1 = 3")
    print(f"Batch GD:      θ0 = {theta_batch[0][0]:.4f}, θ1 = {theta_batch[1][0]:.4f}")
    print(f"SGD:           θ0 = {theta_sgd[0][0]:.4f}, θ1 = {theta_sgd[1][0]:.4f}")
    print(f"Mini-batch GD: θ0 = {theta_mini[0][0]:.4f}, θ1 = {theta_mini[1][0]:.4f}")
    print()

    # 시각화
    plt.figure(figsize=(12, 5))

    # 손실 함수 비교
    plt.subplot(1, 2, 1)
    plt.plot(history_batch, label="Batch GD", linewidth=2)
    plt.plot(history_sgd, label="SGD", alpha=0.7)
    plt.plot(history_mini, label="Mini-batch GD", linewidth=2)
    plt.xlabel("Iterations/Epochs")
    plt.ylabel("MSE Loss")
    plt.title("손실 함수 수렴 비교")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 데이터와 예측선
    plt.subplot(1, 2, 2)
    plt.scatter(X, y, alpha=0.5, label="Data")
    X_new = np.array([[0], [2]])
    X_new_b = np.c_[np.ones((2, 1)), X_new]

    plt.plot(X_new, X_new_b.dot(theta_batch), "r-", linewidth=2, label="Batch GD")
    plt.plot(X_new, X_new_b.dot(theta_sgd), "g--", linewidth=2, label="SGD")
    plt.plot(X_new, X_new_b.dot(theta_mini), "b:", linewidth=3, label="Mini-batch GD")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.title("예측 결과 비교")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def main():
    """메인 함수: Gradient Descent 알고리즘 비교 실행"""
    print("📊 Gradient Descent 알고리즘 완전 가이드 📊")
    print("=" * 60)
    print()

    run_gradient_descent_comparison()

    print("=" * 60)
    print("🎉 Gradient Descent 비교가 완료되었습니다! 🎉")
    print("=" * 60)


if __name__ == "__main__":
    main()

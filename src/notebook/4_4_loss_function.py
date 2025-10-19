import matplotlib.pyplot as plt
import numpy as np

# 한글 폰트 설정 (Mac의 경우)
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False


def visualize_binary_cross_entropy():
    """이진 교차 엔트로피 손실 함수 시각화"""
    print("=== 이진 교차 엔트로피 손실 함수 ===")
    print("Log Loss 시각화: -log(p) 형태의 손실 함수")
    print()

    # 예측 확률 범위
    y_pred = np.linspace(0.01, 0.99, 100)

    # y=1일 때 손실: -log(ŷ)
    loss_y1 = -np.log(y_pred)

    # y=0일 때 손실: -log(1-ŷ)
    loss_y0 = -np.log(1 - y_pred)

    plt.figure(figsize=(12, 5))

    # y=1일 때
    plt.subplot(1, 2, 1)
    plt.plot(y_pred, loss_y1, linewidth=3, color="#e74c3c")
    plt.xlabel("예측 확률 (ŷ)", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("실제 레이블 y = 1일 때", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.axvline(x=1.0, color="green", linestyle="--", alpha=0.5, label="완벽한 예측")
    plt.legend()

    # y=0일 때
    plt.subplot(1, 2, 2)
    plt.plot(y_pred, loss_y0, linewidth=3, color="#3498db")
    plt.xlabel("예측 확률 (ŷ)", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("실제 레이블 y = 0일 때", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.axvline(x=0.0, color="green", linestyle="--", alpha=0.5, label="완벽한 예측")
    plt.legend()

    plt.tight_layout()
    plt.show()

    print("✅ 시각화 완료!")
    print("- 실제값이 1일 때: 예측 확률이 1에 가까울수록 손실이 0에 가까워짐")
    print("- 실제값이 0일 때: 예측 확률이 0에 가까울수록 손실이 0에 가까워짐")
    print("- 잘못 예측할수록 손실이 급격히 증가함")
    print()


# PyTorch 예제는 호환성 문제로 주석 처리
# import torch
# import torch.nn as nn


def pytorch_explanation():
    """PyTorch 손실 함수 설명"""
    print("=" * 50)
    print("PyTorch Loss Functions (코드 예제)")
    print("=" * 50)

    print("\n💡 PyTorch 주요 손실 함수:")
    print("   1. BCEWithLogitsLoss: 이진 분류 (Sigmoid + BCE)")
    print("   2. CrossEntropyLoss: 다중 분류 (Softmax + CE)")
    print("   3. MSELoss: 회귀 (평균 제곱 오차)")

    print("\n📝 PyTorch 이진 분류 예제 코드:")
    print("```python")
    print("import torch")
    print("import torch.nn as nn")
    print()
    print("# BCEWithLogitsLoss 사용")
    print("criterion = nn.BCEWithLogitsLoss()")
    print("logits = torch.tensor([2.5, -1.0, 0.5, -3.0])")
    print("labels = torch.tensor([1.0, 0.0, 1.0, 0.0])")
    print("loss = criterion(logits, labels)")
    print("```")

    print("\n📝 PyTorch 다중 분류 예제 코드:")
    print("```python")
    print("# CrossEntropyLoss 사용")
    print("criterion = nn.CrossEntropyLoss()")
    print("logits = torch.tensor([[2.0, 1.0, 0.1], [0.5, 2.5, 1.0]])")
    print("labels = torch.tensor([0, 1])  # 클래스 인덱스")
    print("loss = criterion(logits, labels)")
    print("```")


def compare_frameworks():
    """프레임워크 비교"""
    print("\n" + "=" * 50)
    print("PyTorch vs TensorFlow 손실 함수")
    print("=" * 50)

    print("\n📊 이진 분류 (Binary Classification)")
    print("   PyTorch:")
    print("   → nn.BCEWithLogitsLoss()")
    print("   → 로짓 + 레이블 직접 입력")
    print()
    print("   TensorFlow:")
    print("   → BinaryCrossentropy(from_logits=True)")
    print("   → from_logits 파라미터로 제어")

    print("\n📊 다중 분류 (Multi-class Classification)")
    print("   PyTorch:")
    print("   → nn.CrossEntropyLoss()")
    print("   → 정수 레이블 (0, 1, 2, ...)")
    print()
    print("   TensorFlow:")
    print("   → SparseCategoricalCrossentropy(from_logits=True)")
    print("   → 정수 레이블 or CategoricalCrossentropy (one-hot)")


# TensorFlow는 AVX 지원 문제로 주석 처리
# import tensorflow as tf
# from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy


def tensorflow_explanation():
    """TensorFlow 손실 함수 설명"""
    print("=" * 50)
    print("TensorFlow Loss Functions (코드 예제)")
    print("=" * 50)

    print("\n💡 TensorFlow 주요 손실 함수:")
    print("   1. BinaryCrossentropy: 이진 분류")
    print("   2. CategoricalCrossentropy: 다중 분류 (one-hot)")
    print("   3. SparseCategoricalCrossentropy: 다중 분류 (정수)")
    print("   4. MeanSquaredError: 회귀")

    print("\n📝 TensorFlow 이진 분류 예제 코드:")
    print("```python")
    print("import tensorflow as tf")
    print("from tensorflow.keras.losses import BinaryCrossentropy")
    print()
    print("# BinaryCrossentropy 사용")
    print("loss_fn = BinaryCrossentropy(from_logits=True)")
    print("logits = tf.constant([2.5, -1.0, 0.5, -3.0])")
    print("labels = tf.constant([1.0, 0.0, 1.0, 0.0])")
    print("loss = loss_fn(labels, logits)")
    print("```")

    print("\n📝 TensorFlow 다중 분류 예제 코드:")
    print("```python")
    print("# SparseCategoricalCrossentropy 사용 (더 편리)")
    print("loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)")
    print("logits = tf.constant([[2.0, 1.0, 0.1], [0.5, 2.5, 1.0]])")
    print("labels = tf.constant([0, 1])  # 정수 레이블")
    print("loss = loss_fn(labels, logits)")
    print("```")


def main():
    """메인 함수: 손실 함수 시각화 실행"""
    print("📈 Binary Cross-Entropy Loss Function 가이드 📈")
    print("=" * 60)
    print()

    visualize_binary_cross_entropy()

    print("=" * 60)
    print("🎉 손실 함수 시각화가 완료되었습니다! 🎉")
    print("=" * 60)

    print("\n🔥 PyTorch 손실 함수 예제\n")

    # PyTorch 예제 (설명만)
    pytorch_explanation()

    # 프레임워크 비교
    compare_frameworks()

    print("\n🔥 TensorFlow 손실 함수 예제\n")

    # TensorFlow 예제 (설명만)
    tensorflow_explanation()

    print("\n" + "=" * 50)
    print("✅ 모든 예제 완료!")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()

import matplotlib.pyplot as plt
import numpy as np

# 한글 폰트 설정 (MacOS 기준)
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False


# 1. 기본 Softmax 구현
def softmax(x):
    """
    Softmax 함수
    x: 입력 배열 (1D 또는 2D)
    """
    # 수치 안정성을 위해 최댓값을 빼줌 (결과는 동일)
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def softmax_with_temperature(x, temperature=1.0):
    """
    Temperature 파라미터가 있는 Softmax
    temperature > 1: 확률 분포가 더 평평해짐 (불확실성 증가)
    temperature < 1: 확률 분포가 더 뾰족해짐 (확신 증가)
    """
    x = x / temperature
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)


def example_1_basic_usage():
    """예시 1: 기본 사용법"""
    print("=" * 50)
    print("예시 1: 기본 사용법")
    print("=" * 50)

    scores = np.array([2.0, 1.0, 0.1])
    print(f"입력 점수: {scores}")

    probabilities = softmax(scores)
    print(f"Softmax 출력: {probabilities}")
    print(f"합계: {np.sum(probabilities):.6f}")
    print()


def example_2_image_classification():
    """예시 2: 3-클래스 이미지 분류"""
    print("=" * 50)
    print("예시 2: 3-클래스 이미지 분류")
    print("=" * 50)

    classes = ["고양이", "강아지", "새"]
    logits = np.array([3.2, 1.5, 0.8])  # 신경망 출력

    probs = softmax(logits)

    print("신경망 원본 점수 (logits):")
    for cls, score in zip(classes, logits):
        print(f"  {cls}: {score:.2f}")

    print("\nSoftmax 확률:")
    for cls, prob in zip(classes, probs):
        print(f"  {cls}: {prob:.4f} ({prob * 100:.2f}%)")

    predicted_class = classes[np.argmax(probs)]
    print(f"\n예측 결과: {predicted_class}")
    print()


def example_3_batch_processing():
    """예시 3: 배치 처리 (여러 샘플 동시 처리)"""
    print("=" * 50)
    print("예시 3: 배치 처리")
    print("=" * 50)

    batch_logits = np.array(
        [
            [2.0, 1.0, 0.1],  # 샘플 1
            [0.5, 2.5, 1.0],  # 샘플 2
            [1.0, 1.0, 1.0],  # 샘플 3
        ]
    )

    batch_probs = softmax(batch_logits)

    print("입력 (3개 샘플, 3개 클래스):")
    print(batch_logits)
    print("\nSoftmax 출력:")
    print(batch_probs)
    print("\n각 샘플의 확률 합:")
    print(np.sum(batch_probs, axis=1))
    print()


def example_4_temperature_scaling():
    """예시 4: Temperature Scaling"""
    print("=" * 50)
    print("예시 4: Temperature Scaling")
    print("=" * 50)

    scores = np.array([3.0, 1.0, 0.5])
    temperatures = [0.5, 1.0, 2.0]

    print(f"입력 점수: {scores}\n")
    for temp in temperatures:
        probs = softmax_with_temperature(scores, temp)
        print(f"Temperature = {temp}:")
        print(f"  확률: {probs}")
        print(f"  최대 확률: {np.max(probs):.4f}\n")


def example_5_visualization():
    """예시 5: Softmax 시각화"""
    print("=" * 50)
    print("예시 5: Softmax 시각화")
    print("=" * 50)

    # 입력 범위에 따른 softmax 출력 변화
    x_range = np.linspace(-5, 5, 100)
    y1 = softmax(np.array([x_range, np.zeros_like(x_range), np.zeros_like(x_range)]).T)[
        :, 0
    ]

    plt.figure(figsize=(10, 6))

    plt.subplot(1, 2, 1)
    plt.plot(x_range, y1, linewidth=2)
    plt.xlabel("Input Score", fontsize=12)
    plt.ylabel("Softmax Probability", fontsize=12)
    plt.title("Softmax: 3개 클래스 중 첫 번째 클래스\n(나머지는 0)", fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    scores_viz = np.array([3.0, 1.5, 0.5])
    probs_viz = softmax(scores_viz)
    bars = plt.bar(
        ["클래스 1", "클래스 2", "클래스 3"],
        probs_viz,
        color=["#FF6B6B", "#4ECDC4", "#45B7D1"],
    )
    plt.ylabel("Probability", fontsize=12)
    plt.title("Softmax 출력 예시", fontsize=12)
    plt.ylim(0, 1)

    # 막대 위에 확률 값 표시
    for bar, prob in zip(bars, probs_viz):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{prob:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    plt.tight_layout()
    plt.savefig("softmax_visualization.png", dpi=150, bbox_inches="tight")
    print("시각화가 'softmax_visualization.png'로 저장되었습니다!")

    plt.show()


def example_6_framework_comparison():
    """예시 6: 딥러닝 프레임워크와 비교"""
    print("\n" + "=" * 50)
    print("예시 6: 딥러닝 프레임워크와 함께 사용")
    print("=" * 50)

    print("PyTorch 예시:")
    print("```python")
    print("import torch")
    print("import torch.nn.functional as F")
    print()
    print("logits = torch.tensor([2.0, 1.0, 0.1])")
    print("probs = F.softmax(logits, dim=0)")
    print("print(probs)")
    print("```")
    print()

    print("TensorFlow/Keras 예시:")
    print("```python")
    print("import tensorflow as tf")
    print()
    print("logits = tf.constant([2.0, 1.0, 0.1])")
    print("probs = tf.nn.softmax(logits)")
    print("print(probs)")
    print("```")


def main():
    """메인 함수: 모든 예시를 순차적으로 실행"""
    print("🧠 Softmax 함수 완전 가이드 🧠")
    print("=" * 60)
    print()

    # 모든 예시 실행
    example_1_basic_usage()
    example_2_image_classification()
    example_3_batch_processing()
    example_4_temperature_scaling()
    example_5_visualization()
    example_6_framework_comparison()

    print("\n" + "=" * 60)
    print("🎉 모든 예시가 완료되었습니다! 🎉")
    print("=" * 60)


if __name__ == "__main__":
    main()

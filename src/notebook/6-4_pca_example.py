# flake8: noqa E501
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_olivetti_faces, load_digits, load_iris
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 한글 폰트 설정
plt.rcParams["font.family"] = "AppleGothic"  # Mac
plt.rcParams["axes.unicode_minus"] = False


def example1_basic_pca():
    """예제 1: 기본 PCA 구현"""
    print("\n" + "=" * 60)
    print("예제 1: 기본 PCA 구현")
    print("=" * 60)

    # 데이터 로드
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names

    print("원본 데이터 형태:", X.shape)
    print("특성 이름:", feature_names)
    print("\n첫 5개 샘플:")
    print(X[:5])

    # 데이터 표준화
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("\n표준화 후 평균:", X_scaled.mean(axis=0))
    print("표준화 후 표준편차:", X_scaled.std(axis=0))

    # PCA 적용 (4D → 2D)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    print("\n" + "=" * 50)
    print("PCA 결과")
    print("=" * 50)
    print("변환 후 데이터 형태:", X_pca.shape)
    print("\n각 주성분의 설명 분산 비율:")
    for i, var in enumerate(pca.explained_variance_ratio_):
        print(f"  PC{i + 1}: {var:.4f} ({var * 100:.2f}%)")

    print(
        f"\n누적 설명 분산: {pca.explained_variance_ratio_.sum():.4f} "
        f"({pca.explained_variance_ratio_.sum() * 100:.2f}%)"
    )

    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 원본 데이터 (2개 특성만 선택)
    ax1 = axes[0]
    colors = ["red", "green", "blue"]
    for i, color in enumerate(colors):
        mask = y == i
        ax1.scatter(
            X[mask, 0], X[mask, 1], c=color, label=target_names[i], alpha=0.6, s=50
        )
    ax1.set_xlabel(feature_names[0], fontsize=11)
    ax1.set_ylabel(feature_names[1], fontsize=11)
    ax1.set_title("원본 데이터 (2개 특성만 시각화)", fontsize=13, fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # PCA 변환 데이터
    ax2 = axes[1]
    for i, color in enumerate(colors):
        mask = y == i
        ax2.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            c=color,
            label=target_names[i],
            alpha=0.6,
            s=50,
        )
    ax2.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)", fontsize=11)
    ax2.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)", fontsize=11)
    ax2.set_title("PCA 변환 데이터 (4D → 2D)", fontsize=13, fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 주성분 해석
    print("\n" + "=" * 50)
    print("주성분 구성 요소 (Component Loadings)")
    print("=" * 50)
    print("각 주성분이 원본 특성들을 얼마나 사용하는지:\n")

    for i in range(2):
        print(f"PC{i + 1}:")
        for j, feature in enumerate(feature_names):
            print(f"  {feature}: {pca.components_[i, j]:+.3f}")
        print()

    return X_scaled, pca


def example2_variance_analysis(X_scaled):
    """예제 2: 설명 분산 비율로 최적 주성분 개수 찾기"""
    print("\n" + "=" * 60)
    print("예제 2: 설명 분산 비율 분석")
    print("=" * 60)

    # 모든 주성분 계산
    pca_full = PCA()
    X_pca_full = pca_full.fit_transform(X_scaled)
    print(X_pca_full)

    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 개별 설명 분산
    ax1 = axes[0]
    ax1.bar(
        range(1, len(pca_full.explained_variance_ratio_) + 1),
        pca_full.explained_variance_ratio_,
        color="#667eea",
        alpha=0.8,
    )
    ax1.set_xlabel("주성분", fontsize=12)
    ax1.set_ylabel("설명 분산 비율", fontsize=12)
    ax1.set_title("각 주성분의 설명 분산", fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3, axis="y")

    # 누적 설명 분산
    ax2 = axes[1]
    cumsum = np.cumsum(pca_full.explained_variance_ratio_)
    ax2.plot(
        range(1, len(cumsum) + 1),
        cumsum,
        "o-",
        linewidth=2,
        markersize=8,
        color="#667eea",
    )
    ax2.axhline(y=0.95, color="red", linestyle="--", alpha=0.7, label="95% 분산")
    ax2.set_xlabel("주성분 개수", fontsize=12)
    ax2.set_ylabel("누적 설명 분산 비율", fontsize=12)
    ax2.set_title("누적 설명 분산", fontsize=13, fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("누적 설명 분산:")
    for i, var in enumerate(cumsum):
        print(f"  PC1~PC{i + 1}: {var:.4f} ({var * 100:.2f}%)")


def example3_custom_pca(X_scaled):
    """예제 3: 직접 PCA 구현 (수학 이해용)"""
    print("\n" + "=" * 60)
    print("예제 3: 직접 PCA 구현")
    print("=" * 60)

    class MyPCA:
        """PCA를 직접 구현한 클래스"""

        def __init__(self, n_components):
            self.n_components = n_components
            self.components_ = None
            self.mean_ = None
            self.explained_variance_ratio_ = None

        def fit(self, X):
            # 1. 평균 중심화
            self.mean_ = np.mean(X, axis=0)
            X_centered = X - self.mean_

            # 2. 공분산 행렬 계산
            n_samples = X.shape[0]
            cov_matrix = (X_centered.T @ X_centered) / (n_samples - 1)

            # 3. 고유값, 고유벡터 계산
            eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

            # 4. 고유값 기준 내림차순 정렬
            idx = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            # 5. 상위 n_components개 선택
            self.components_ = eigenvectors[:, : self.n_components].T

            # 6. 설명 분산 비율 계산
            total_variance = np.sum(eigenvalues)
            self.explained_variance_ratio_ = (
                eigenvalues[: self.n_components] / total_variance
            )

            return self

        def transform(self, X):
            # 평균 중심화 후 주성분으로 투영
            X_centered = X - self.mean_
            return X_centered @ self.components_.T

        def fit_transform(self, X):
            self.fit(X)
            return self.transform(X)

    # 사용 예시
    my_pca = MyPCA(n_components=2)
    X_my_pca = my_pca.fit_transform(X_scaled)

    print("직접 구현한 PCA 결과:")
    print("변환 후 형태:", X_my_pca.shape)
    print("설명 분산 비율:", my_pca.explained_variance_ratio_)

    # sklearn과 비교
    pca_sklearn = PCA(n_components=2)
    X_pca = pca_sklearn.fit_transform(X_scaled)

    print("\nsklearn PCA와 비교:")
    print("차이:", np.abs(np.abs(X_my_pca) - np.abs(X_pca)).max())


def example4_image_compression():
    """예제 4: 실전 예제 - 이미지 압축"""
    print("\n" + "=" * 60)
    print("예제 4: 이미지 압축")
    print("=" * 60)

    # 얼굴 이미지 데이터 로드 (64x64 픽셀 = 4096 차원)
    faces = fetch_olivetti_faces()
    X_faces = faces.data  # (400, 4096)

    print("원본 이미지 형태:", X_faces.shape)
    print("픽셀 개수:", X_faces.shape[1])

    # 다양한 주성분 개수로 PCA 적용
    n_components_list = [10, 50, 100, 200]

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))

    # 원본 이미지
    axes[0, 0].imshow(X_faces[0].reshape(64, 64), cmap="gray")
    axes[0, 0].set_title("원본\n(4096 차원)", fontsize=10)
    axes[0, 0].axis("off")

    axes[1, 0].imshow(X_faces[1].reshape(64, 64), cmap="gray")
    axes[1, 0].set_title("원본\n(4096 차원)", fontsize=10)
    axes[1, 0].axis("off")

    # 다양한 주성분 개수로 복원
    for idx, n_comp in enumerate(n_components_list, 1):
        pca = PCA(n_components=n_comp)
        X_compressed = pca.fit_transform(X_faces)
        X_reconstructed = pca.inverse_transform(X_compressed)

        compression_ratio = (1 - n_comp / X_faces.shape[1]) * 100

        # 첫 번째 이미지
        axes[0, idx].imshow(X_reconstructed[0].reshape(64, 64), cmap="gray")
        axes[0, idx].set_title(
            f"{n_comp}개 주성분\n({compression_ratio:.1f}% 압축)", fontsize=10
        )
        axes[0, idx].axis("off")

        # 두 번째 이미지
        axes[1, idx].imshow(X_reconstructed[1].reshape(64, 64), cmap="gray")
        axes[1, idx].set_title(
            f"{n_comp}개 주성분\n({compression_ratio:.1f}% 압축)", fontsize=10
        )
        axes[1, idx].axis("off")

    plt.tight_layout()
    plt.show()

    # 압축률과 품질
    print("\n압축 결과:")
    for n_comp in n_components_list:
        pca = PCA(n_components=n_comp)
        pca.fit(X_faces)
        variance_retained = pca.explained_variance_ratio_.sum()
        compression_ratio = (1 - n_comp / X_faces.shape[1]) * 100
        print(
            f"{n_comp}개 주성분: {compression_ratio:.1f}% 압축, {variance_retained * 100:.1f}% 정보 보존"
        )


def example5_ml_with_pca():
    """예제 5: PCA 활용 - 차원 축소 후 머신러닝"""
    print("\n" + "=" * 60)
    print("예제 5: 차원 축소 후 머신러닝")
    print("=" * 60)

    # 손글씨 숫자 데이터 (64 차원)
    digits = load_digits()
    X = digits.data  # (1797, 64)
    y = digits.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("원본 데이터 형태:", X.shape)
    print()

    # 다양한 주성분 개수로 실험
    results = []

    for n_comp in [5, 10, 20, 30, 40, 64]:
        if n_comp < 64:
            # PCA 적용
            pca = PCA(n_components=n_comp)
            X_train_pca = pca.fit_transform(X_train)
            X_test_pca = pca.transform(X_test)
            variance = pca.explained_variance_ratio_.sum()
        else:
            # PCA 없이 원본 사용
            X_train_pca = X_train
            X_test_pca = X_test
            variance = 1.0

        # 모델 학습 및 평가
        start_time = time.time()
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train_pca, y_train)
        train_time = time.time() - start_time

        y_pred = clf.predict(X_test_pca)
        accuracy = accuracy_score(y_test, y_pred)

        results.append(
            {
                "n_components": n_comp,
                "variance": variance,
                "accuracy": accuracy,
                "train_time": train_time,
            }
        )

        print(
            f"주성분 {n_comp:2d}개: "
            f"정확도 {accuracy:.4f}, "
            f"분산 {variance * 100:5.1f}%, "
            f"학습시간 {train_time:.3f}초"
        )

    # 결과 시각화
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    n_comps = [r["n_components"] for r in results]
    accuracies = [r["accuracy"] for r in results]
    times = [r["train_time"] for r in results]

    ax1 = axes[0]
    ax1.plot(n_comps, accuracies, "o-", linewidth=2, markersize=8, color="#667eea")
    ax1.set_xlabel("주성분 개수", fontsize=12)
    ax1.set_ylabel("정확도", fontsize=12)
    ax1.set_title("주성분 개수 vs 정확도", fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(n_comps, times, "o-", linewidth=2, markersize=8, color="#e74c3c")
    ax2.set_xlabel("주성분 개수", fontsize=12)
    ax2.set_ylabel("학습 시간 (초)", fontsize=12)
    ax2.set_title("주성분 개수 vs 학습 시간", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def main():
    """메인 함수 - 모든 예제 실행"""
    print("=" * 60)
    print("PCA (Principal Component Analysis) 예제 모음")
    print("=" * 60)

    # 예제 1: 기본 PCA
    X_scaled, pca = example1_basic_pca()

    # 예제 2: 분산 분석
    example2_variance_analysis(X_scaled)

    # 예제 3: 직접 구현
    example3_custom_pca(X_scaled)

    # 예제 4: 이미지 압축
    example4_image_compression()

    # 예제 5: 머신러닝 활용
    example5_ml_with_pca()

    print("\n" + "=" * 60)
    print("모든 예제 실행 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()

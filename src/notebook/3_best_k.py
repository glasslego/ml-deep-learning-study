import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

# 한글 폰트 설정
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)


def example1_basic_selectkbest():
    """예제 1: SelectKBest 기본 사용법"""
    print("\n" + "=" * 60)
    print("예제 1: SelectKBest 기본 사용법")
    print("=" * 60)

    # 합성 데이터 생성 (10개 특성)
    X, y = make_regression(
        n_samples=200,
        n_features=10,
        n_informative=5,  # 실제로 유용한 특성 5개
        noise=10,
        random_state=42,
    )

    # DataFrame으로 변환
    feature_names = [f"Feature_{i + 1}" for i in range(10)]
    X_df = pd.DataFrame(X, columns=feature_names)

    print(f"원본 데이터 형태: {X_df.shape}")
    print("\n처음 5개 샘플:")
    print(X_df.head())

    # SelectKBest로 상위 3개 특성 선택
    selector = SelectKBest(score_func=f_regression, k=3)
    X_selected = selector.fit_transform(X, y)

    # 선택된 특성 확인
    selected_features = X_df.columns[selector.get_support()]
    print(f"\n선택된 특성 ({len(selected_features)}개):")
    print(selected_features.tolist())

    # 각 특성의 점수 확인
    scores = pd.DataFrame(
        {
            "Feature": feature_names,
            "Score": selector.scores_,
            "Selected": selector.get_support(),
        }
    ).sort_values("Score", ascending=False)

    print("\n특성별 F-통계량 점수:")
    print(scores)

    # 시각화
    plt.figure(figsize=(12, 5))

    # 특성별 점수
    plt.subplot(1, 2, 1)
    colors = ["green" if s else "gray" for s in selector.get_support()]
    plt.barh(feature_names, selector.scores_, color=colors)
    plt.xlabel("F-통계량 점수", fontsize=12)
    plt.title("특성별 중요도 (초록색 = 선택됨)", fontsize=13, fontweight="bold")
    plt.grid(True, alpha=0.3, axis="x")

    # 선택 전후 비교
    plt.subplot(1, 2, 2)
    categories = ["원본", "선택 후"]
    feature_counts = [X.shape[1], X_selected.shape[1]]
    plt.bar(categories, feature_counts, color=["#667eea", "#2ecc71"])
    plt.ylabel("특성 개수", fontsize=12)
    plt.title("특성 개수 비교", fontsize=13, fontweight="bold")
    for i, v in enumerate(feature_counts):
        plt.text(i, v + 0.2, str(v), ha="center", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.show()

    return X, y, selector


def example2_model_comparison(X, y):
    """예제 2: 특성 선택 전후 모델 성능 비교"""
    print("\n" + "=" * 60)
    print("예제 2: 특성 선택 전후 모델 성능 비교")
    print("=" * 60)

    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 표준화
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = []

    # 1. 원본 데이터로 학습
    print("\n1. 원본 데이터 (모든 특성 사용)")
    knn_full = KNeighborsRegressor(n_neighbors=5)
    knn_full.fit(X_train_scaled, y_train)
    y_pred_full = knn_full.predict(X_test_scaled)

    mse_full = mean_squared_error(y_test, y_pred_full)
    r2_full = r2_score(y_test, y_pred_full)

    print(f"   MSE: {mse_full:.2f}")
    print(f"   R² Score: {r2_full:.4f}")

    results.append({"Method": "원본 (10개 특성)", "MSE": mse_full, "R2": r2_full})

    # 2. 다양한 k값으로 특성 선택
    for k in [3, 5, 7]:
        print(f"\n2. SelectKBest (k={k})")

        selector = SelectKBest(score_func=f_regression, k=k)
        X_train_selected = selector.fit_transform(X_train_scaled, y_train)
        X_test_selected = selector.transform(X_test_scaled)

        knn_selected = KNeighborsRegressor(n_neighbors=5)
        knn_selected.fit(X_train_selected, y_train)
        y_pred_selected = knn_selected.predict(X_test_selected)

        mse_selected = mean_squared_error(y_test, y_pred_selected)
        r2_selected = r2_score(y_test, y_pred_selected)

        print(f"   MSE: {mse_selected:.2f}")
        print(f"   R² Score: {r2_selected:.4f}")

        results.append(
            {"Method": f"SelectKBest (k={k})", "MSE": mse_selected, "R2": r2_selected}
        )

    # 결과 시각화
    results_df = pd.DataFrame(results)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # MSE 비교
    ax1 = axes[0]
    ax1.barh(results_df["Method"], results_df["MSE"], color="#e74c3c")
    ax1.set_xlabel("MSE (낮을수록 좋음)", fontsize=12)
    ax1.set_title("평균 제곱 오차 비교", fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3, axis="x")

    # R² 비교
    ax2 = axes[1]
    ax2.barh(results_df["Method"], results_df["R2"], color="#2ecc71")
    ax2.set_xlabel("R² Score (높을수록 좋음)", fontsize=12)
    ax2.set_title("결정 계수 비교", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.show()

    return results_df


def main():
    """메인 함수"""
    print("=" * 60)
    print("Feature Selection (특성 선택) 완전 가이드")
    print("=" * 60)

    # 예제 1: 기본 사용법
    X, y, selector = example1_basic_selectkbest()

    # # 예제 2: 성능 비교
    results = example2_model_comparison(X, y)
    print(results)


if __name__ == "__main__":
    main()

"""
실습 2: 정규화 기법 비교 (Ridge & Lasso)

정규화(Regularization)의 핵심 개념:
- 과적합(Overfitting) 방지를 위한 기법
- 손실 함수에 페널티(penalty) 항을 추가하여 모델 복잡도 제어

정규화 유형:
1. Ridge (L2): 가중치의 제곱합에 비례하는 페널티
   - 손실함수 = MSE + α * Σ(βᵢ²)
   - 가중치를 0에 가깝게 만들지만 완전히 0으로 만들지는 않음
   - 모든 피처를 사용하되 영향력을 줄임

2. Lasso (L1): 가중치의 절댓값 합에 비례하는 페널티  
   - 손실함수 = MSE + α * Σ|βᵢ|
   - 일부 가중치를 정확히 0으로 만들어 피처 선택 효과
   - 자동으로 중요하지 않은 피처를 제거

3. Elastic Net: Ridge + Lasso 조합
   - 손실함수 = MSE + α₁ * Σ|βᵢ| + α₂ * Σ(βᵢ²)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_regression, fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score, validation_curve
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

def create_synthetic_data():
    """
    정규화 효과를 명확히 보기 위한 합성 데이터 생성
    - 고차원 데이터에서 정규화의 효과가 더 명확하게 나타남
    - 일부 피처는 노이즈, 일부는 실제 유용한 피처
    """
    print("=== 합성 데이터 생성 ===")
    
    # 1. 기본 회귀 데이터 생성
    X, y = make_regression(
        n_samples=200,      # 샘플 수를 적게 설정하여 과적합 유도
        n_features=20,      # 피처 수
        n_informative=5,    # 실제 유용한 피처 수 (나머지는 노이즈)
        noise=0.1,          # 노이즈 레벨
        random_state=42
    )
    
    feature_names = [f'feature_{i:02d}' for i in range(X.shape[1])]
    
    print(f"데이터 형태: {X.shape}")
    print(f"유용한 피처: 5개, 노이즈 피처: 15개")
    
    return X, y, feature_names

def load_real_data():
    """
    실제 데이터(캘리포니아 주택 데이터)로도 비교 분석
    """
    housing = fetch_california_housing()
    return housing.data, housing.target, housing.feature_names

def compare_regularization_models(X, y, feature_names, alpha_range=None):
    """
    다양한 정규화 모델 비교
    - 일반 선형회귀, Ridge, Lasso, Elastic Net 성능 비교
    - 교차검증을 통한 robust한 성능 평가
    """
    if alpha_range is None:
        alpha_range = np.logspace(-3, 2, 20)  # 0.001 ~ 100
    
    print("\n=== 정규화 모델 비교 ===")
    
    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 스케일링 (정규화에서 필수)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 모델 정의
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=1.0, max_iter=1000),
        'Elastic Net': ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=1000)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n--- {name} ---")
        
        # 모델 학습
        model.fit(X_train_scaled, y_train)
        
        # 예측
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        
        # 성능 지표
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        
        # 교차검증 점수 (더 robust한 평가)
        cv_scores = cross_val_score(model, X_train_scaled, y_train, 
                                   cv=5, scoring='r2')
        
        results[name] = {
            'model': model,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        print(f"  학습 R²: {train_r2:.4f}")
        print(f"  테스트 R²: {test_r2:.4f}")
        print(f"  CV R² 평균: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        print(f"  과적합 정도: {train_r2 - test_r2:.4f}")
        
        # 과적합 해석: 학습 점수와 테스트 점수의 차이가 클수록 과적합
    
    return results, (X_train_scaled, X_test_scaled, y_train, y_test)

def find_optimal_alpha(X_train, y_train, model_class, alpha_range):
    """
    교차검증을 통한 최적 알파(정규화 강도) 찾기
    
    알파 값의 의미:
    - 알파 = 0: 정규화 없음 (일반 선형회귀와 동일)
    - 알파 ↑: 정규화 강도 증가 (underfitting 위험)
    - 알파 ↓: 정규화 약화 (overfitting 위험)
    """
    print(f"\n=== {model_class.__name__} 최적 알파 탐색 ===")
    
    train_scores, valid_scores = validation_curve(
        model_class(max_iter=1000), X_train, y_train,
        param_name='alpha', param_range=alpha_range,
        cv=5, scoring='r2', n_jobs=-1
    )
    
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    valid_mean = valid_scores.mean(axis=1)
    valid_std = valid_scores.std(axis=1)
    
    # 최적 알파: 검증 점수가 가장 높은 지점
    best_alpha_idx = np.argmax(valid_mean)
    best_alpha = alpha_range[best_alpha_idx]
    
    print(f"최적 알파: {best_alpha:.4f}")
    print(f"최적 알파에서 검증 R²: {valid_mean[best_alpha_idx]:.4f}")
    
    return {
        'alpha_range': alpha_range,
        'train_mean': train_mean,
        'train_std': train_std,
        'valid_mean': valid_mean,
        'valid_std': valid_std,
        'best_alpha': best_alpha,
        'best_alpha_idx': best_alpha_idx
    }

def plot_validation_curves(ridge_results, lasso_results):
    """
    알파 값에 따른 성능 변화 시각화
    - Bias-Variance Tradeoff 시각화
    - 과적합/언더피팅 구간 식별
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Ridge 검증 곡선
    ax1.plot(ridge_results['alpha_range'], ridge_results['train_mean'], 
             'o-', color='blue', label='학습 점수')
    ax1.fill_between(ridge_results['alpha_range'], 
                     ridge_results['train_mean'] - ridge_results['train_std'],
                     ridge_results['train_mean'] + ridge_results['train_std'],
                     alpha=0.1, color='blue')
    
    ax1.plot(ridge_results['alpha_range'], ridge_results['valid_mean'],
             'o-', color='red', label='검증 점수')
    ax1.fill_between(ridge_results['alpha_range'],
                     ridge_results['valid_mean'] - ridge_results['valid_std'],
                     ridge_results['valid_mean'] + ridge_results['valid_std'],
                     alpha=0.1, color='red')
    
    ax1.axvline(ridge_results['best_alpha'], color='green', linestyle='--', 
                label=f'최적 α = {ridge_results["best_alpha"]:.4f}')
    ax1.set_xscale('log')
    ax1.set_xlabel('Alpha (정규화 강도)')
    ax1.set_ylabel('R² Score')
    ax1.set_title('Ridge 회귀 - 검증 곡선')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Lasso 검증 곡선
    ax2.plot(lasso_results['alpha_range'], lasso_results['train_mean'],
             'o-', color='blue', label='학습 점수')
    ax2.fill_between(lasso_results['alpha_range'],
                     lasso_results['train_mean'] - lasso_results['train_std'],
                     lasso_results['train_mean'] + lasso_results['train_std'],
                     alpha=0.1, color='blue')
    
    ax2.plot(lasso_results['alpha_range'], lasso_results['valid_mean'],
             'o-', color='red', label='검증 점수')
    ax2.fill_between(lasso_results['alpha_range'],
                     lasso_results['valid_mean'] - lasso_results['valid_std'],
                     lasso_results['valid_mean'] + lasso_results['valid_std'],
                     alpha=0.1, color='red')
    
    ax2.axvline(lasso_results['best_alpha'], color='green', linestyle='--',
                label=f'최적 α = {lasso_results["best_alpha"]:.4f}')
    ax2.set_xscale('log')
    ax2.set_xlabel('Alpha (정규화 강도)')
    ax2.set_ylabel('R² Score')
    ax2.set_title('Lasso 회귀 - 검증 곡선')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('regularization_validation_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_feature_selection(X, y, feature_names, alpha_range):
    """
    Lasso의 피처 선택 효과 분석
    - 알파 값에 따라 어떤 피처가 제거되는지 추적
    - 피처 선택의 안정성 평가
    """
    print("\n=== Lasso 피처 선택 분석 ===")
    
    # 데이터 준비
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # 각 알파값에 대해 모델 학습하고 계수 저장
    coef_path = []
    alphas_used = []
    
    for alpha in alpha_range:
        lasso = Lasso(alpha=alpha, max_iter=1000)
        lasso.fit(X_train_scaled, y_train)
        coef_path.append(lasso.coef_)
        alphas_used.append(alpha)
    
    coef_path = np.array(coef_path)
    
    # 각 알파에서 0이 아닌 피처 개수
    non_zero_features = [np.sum(coef != 0) for coef in coef_path]
    
    print(f"알파 범위: {min(alphas_used):.4f} ~ {max(alphas_used):.4f}")
    print(f"선택된 피처 수 범위: {min(non_zero_features)} ~ {max(non_zero_features)}")
    
    # 시각화
    plt.figure(figsize=(15, 5))
    
    # 1. 계수 경로 (Coefficient Path)
    plt.subplot(1, 2, 1)
    for i in range(len(feature_names)):
        plt.plot(alphas_used, coef_path[:, i], label=feature_names[i] if len(feature_names) <= 10 else None)
    
    plt.xscale('log')
    plt.xlabel('Alpha (정규화 강도)')
    plt.ylabel('계수 값')
    plt.title('Lasso 계수 경로')
    plt.grid(True, alpha=0.3)
    if len(feature_names) <= 10:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 2. 선택된 피처 개수
    plt.subplot(1, 2, 2)
    plt.plot(alphas_used, non_zero_features, 'o-', color='red', linewidth=2)
    plt.xscale('log')
    plt.xlabel('Alpha (정규화 강도)')
    plt.ylabel('0이 아닌 피처 개수')
    plt.title('Lasso 피처 선택')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('lasso_feature_selection.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return coef_path, alphas_used, non_zero_features

def compare_coefficients(results, feature_names):
    """
    각 모델의 계수 비교 분석
    - 정규화가 계수에 미치는 영향 시각화
    - 피처 중요도 해석
    """
    print("\n=== 모델별 계수 비교 ===")
    
    # 계수 데이터 준비 (Linear Regression, Ridge, Lasso만 비교)
    models_to_compare = ['Linear Regression', 'Ridge', 'Lasso']
    coef_data = {}
    
    for name in models_to_compare:
        if hasattr(results[name]['model'], 'coef_'):
            coef_data[name] = results[name]['model'].coef_
    
    # DataFrame 생성
    coef_df = pd.DataFrame(coef_data, index=feature_names)
    print("\n계수 비교 테이블:")
    print(coef_df.round(4))
    
    # 시각화
    plt.figure(figsize=(12, 8))
    
    # 히트맵으로 계수 비교
    plt.subplot(2, 1, 1)
    sns.heatmap(coef_df.T, annot=True, cmap='RdBu_r', center=0, 
                fmt='.3f', cbar_kws={'label': '계수 값'})
    plt.title('모델별 계수 히트맵')
    
    # 막대 그래프로 계수 비교
    plt.subplot(2, 1, 2)
    x = np.arange(len(feature_names))
    width = 0.25
    
    for i, name in enumerate(models_to_compare):
        offset = (i - 1) * width
        plt.bar(x + offset, coef_data[name], width, label=name, alpha=0.8)
    
    plt.xlabel('피처')
    plt.ylabel('계수 값')
    plt.title('모델별 계수 비교')
    plt.xticks(x, feature_names, rotation=45, ha='right')
    plt.legend()
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('coefficient_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Lasso의 제로 계수 피처 분석
    if 'Lasso' in coef_data:
        zero_features = [name for name, coef in zip(feature_names, coef_data['Lasso']) if abs(coef) < 1e-10]
        if zero_features:
            print(f"\nLasso에서 제거된 피처 ({len(zero_features)}개):")
            for feature in zero_features:
                print(f"  - {feature}")
        else:
            print("\nLasso에서 제거된 피처 없음")

def performance_summary(results):
    """
    모델 성능 요약 및 시각화
    """
    print("\n=== 모델 성능 요약 ===")
    
    # 성능 데이터 정리
    summary_data = []
    for name, result in results.items():
        summary_data.append({
            'Model': name,
            'Test R²': result['test_r2'],
            'CV Mean': result['cv_mean'],
            'CV Std': result['cv_std'],
            'Overfitting': result['train_r2'] - result['test_r2']
        })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.round(4))
    
    # 시각화
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. R² 점수 비교
    axes[0, 0].bar(summary_df['Model'], summary_df['Test R²'], color='skyblue')
    axes[0, 0].set_title('테스트 R² 점수')
    axes[0, 0].set_ylabel('R² Score')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. 교차검증 점수 (에러바 포함)
    axes[0, 1].bar(summary_df['Model'], summary_df['CV Mean'], 
                   yerr=summary_df['CV Std'], capsize=5, color='lightgreen')
    axes[0, 1].set_title('교차검증 R² 점수')
    axes[0, 1].set_ylabel('R² Score')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. 과적합 정도
    colors = ['red' if x > 0.05 else 'green' for x in summary_df['Overfitting']]
    axes[1, 0].bar(summary_df['Model'], summary_df['Overfitting'], color=colors)
    axes[1, 0].axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='과적합 기준선')
    axes[1, 0].set_title('과적합 정도 (학습 R² - 테스트 R²)')
    axes[1, 0].set_ylabel('R² 차이')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].legend()
    
    # 4. 종합 점수 (테스트 R² - 과적합 페널티)
    composite_score = summary_df['Test R²'] - np.maximum(0, summary_df['Overfitting'] - 0.02)
    axes[1, 1].bar(summary_df['Model'], composite_score, color='orange')
    axes[1, 1].set_title('종합 점수 (테스트 R² - 과적합 페널티)')
    axes[1, 1].set_ylabel('점수')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('regularization_performance_summary.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 최고 성능 모델 식별
    best_model = summary_df.loc[summary_df['Test R²'].idxmax(), 'Model']
    most_stable = summary_df.loc[summary_df['Overfitting'].idxmin(), 'Model']
    
    print(f"\n🏆 최고 테스트 성능: {best_model}")
    print(f"🎯 가장 안정적인 모델: {most_stable}")

def main():
    """
    메인 실행 함수
    """
    print("📊 정규화 기법 비교 분석 (Ridge vs Lasso)")
    print("=" * 60)
    
    # 알파 범위 설정
    alpha_range = np.logspace(-4, 1, 30)
    
    print("\n🔬 합성 데이터로 분석")
    print("-" * 30)
    
    # 1. 합성 데이터 분석
    X_synthetic, y_synthetic, synthetic_features = create_synthetic_data()
    
    # 모델 비교
    results_synthetic, data_splits = compare_regularization_models(
        X_synthetic, y_synthetic, synthetic_features, alpha_range
    )
    
    X_train, X_test, y_train, y_test = data_splits
    
    # 최적 알파 탐색
    ridge_validation = find_optimal_alpha(X_train, y_train, Ridge, alpha_range)
    lasso_validation = find_optimal_alpha(X_train, y_train, Lasso, alpha_range)
    
    # 검증 곡선 시각화
    plot_validation_curves(ridge_validation, lasso_validation)
    
    # 피처 선택 분석
    analyze_feature_selection(X_synthetic, y_synthetic, synthetic_features, alpha_range)
    
    # 계수 비교
    compare_coefficients(results_synthetic, synthetic_features)
    
    # 성능 요약
    performance_summary(results_synthetic)
    
    print("\n🏠 실제 데이터로 분석")
    print("-" * 30)
    
    # 2. 실제 데이터 분석
    X_real, y_real, real_features = load_real_data()
    
    results_real, _ = compare_regularization_models(X_real, y_real, real_features)
    compare_coefficients(results_real, real_features)
    performance_summary(results_real)
    
    print("\n" + "=" * 60)
    print("✅ 분석 완료!")
    print("📊 생성된 시각화 파일:")
    print("   - regularization_validation_curves.png")
    print("   - lasso_feature_selection.png")
    print("   - coefficient_comparison.png")
    print("   - regularization_performance_summary.png")
    
    print("\n📈 주요 인사이트:")
    print("1. Ridge: 모든 피처 유지, 계수 크기 축소")
    print("2. Lasso: 불필요한 피처 자동 제거, 스파스 모델 생성")
    print("3. 정규화 강도(α) 조정으로 Bias-Variance 균형 제어")
    print("4. 교차검증을 통한 최적 하이퍼파라미터 선택 중요")

if __name__ == "__main__":
    main()
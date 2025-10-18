"""
데이터 전처리 예제
학부생을 위한 데이터 정제, 스케일링, 특성 선택 실습 코드
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing, make_classification
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정 (Mac의 경우)
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

def create_sample_data_with_missing():
    """결측값이 있는 샘플 데이터 생성"""
    np.random.seed(42)
    
    # 기본 데이터 생성
    n_samples = 200
    data = {
        'age': np.random.randint(18, 80, n_samples),
        'income': np.random.normal(50000, 15000, n_samples),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
        'experience': np.random.randint(0, 40, n_samples),
        'score': np.random.normal(75, 10, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # 인위적으로 결측값 생성
    missing_indices = np.random.choice(n_samples, size=int(0.1 * n_samples), replace=False)
    df.loc[missing_indices[:len(missing_indices)//2], 'income'] = np.nan
    df.loc[missing_indices[len(missing_indices)//2:], 'score'] = np.nan
    
    return df

def missing_value_handling():
    """결측값 처리 예제"""
    print("=== 결측값 처리 예제 ===")
    
    # 결측값이 있는 데이터 생성
    df = create_sample_data_with_missing()
    
    print("원본 데이터 정보:")
    print(f"데이터 크기: {df.shape}")
    print("\n결측값 현황:")
    print(df.isnull().sum())
    
    # 결측값 시각화
    plt.figure(figsize=(15, 10))
    
    # 결측값 패턴
    plt.subplot(2, 3, 1)
    sns.heatmap(df.isnull(), cbar=True, cmap='viridis')
    plt.title('Missing Value Pattern')
    plt.ylabel('Sample Index')
    
    # 원본 데이터 분포 (income)
    plt.subplot(2, 3, 2)
    plt.hist(df['income'].dropna(), bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('Income')
    plt.ylabel('Frequency')
    plt.title('Original Income Distribution')
    
    # 결측값 처리 방법 비교
    # 1. 평균값으로 대체
    df_mean = df.copy()
    imputer_mean = SimpleImputer(strategy='mean')
    df_mean[['income', 'score']] = imputer_mean.fit_transform(df_mean[['income', 'score']])
    
    # 2. 중앙값으로 대체
    df_median = df.copy()
    imputer_median = SimpleImputer(strategy='median')
    df_median[['income', 'score']] = imputer_median.fit_transform(df_median[['income', 'score']])
    
    # 3. 행 삭제
    df_dropna = df.dropna()
    
    # 처리 결과 비교
    plt.subplot(2, 3, 3)
    plt.hist(df_mean['income'], bins=30, alpha=0.5, label='Mean Imputation', edgecolor='black')
    plt.hist(df_median['income'], bins=30, alpha=0.5, label='Median Imputation', edgecolor='black')
    plt.xlabel('Income')
    plt.ylabel('Frequency')
    plt.title('Imputation Methods Comparison')
    plt.legend()
    
    # 결측값 처리 후 통계
    plt.subplot(2, 3, 4)
    methods = ['Original', 'Mean Imp.', 'Median Imp.', 'Drop NA']
    means = [
        df['income'].mean(),
        df_mean['income'].mean(),
        df_median['income'].mean(),
        df_dropna['income'].mean()
    ]
    
    bars = plt.bar(methods, means, color=['skyblue', 'lightgreen', 'orange', 'pink'])
    plt.ylabel('Mean Income')
    plt.title('Mean Income by Method')
    plt.xticks(rotation=45)
    
    for bar, mean_val in zip(bars, means):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
                f'{mean_val:.0f}', ha='center', va='bottom')
    
    # 데이터 크기 변화
    plt.subplot(2, 3, 5)
    sizes = [len(df), len(df_mean), len(df_median), len(df_dropna)]
    plt.bar(methods, sizes, color=['skyblue', 'lightgreen', 'orange', 'pink'])
    plt.ylabel('Number of Samples')
    plt.title('Sample Size by Method')
    plt.xticks(rotation=45)
    
    for i, (method, size) in enumerate(zip(methods, sizes)):
        plt.text(i, size + 5, f'{size}', ha='center', va='bottom')
    
    # 처리 방법별 분포 비교
    plt.subplot(2, 3, 6)
    plt.boxplot([df['income'].dropna(), df_mean['income'], df_median['income'], df_dropna['income']], 
               labels=['Original', 'Mean', 'Median', 'Drop NA'])
    plt.ylabel('Income')
    plt.title('Income Distribution by Method')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n처리 후 데이터 크기:")
    print(f"평균값 대체: {df_mean.shape}")
    print(f"중앙값 대체: {df_median.shape}")
    print(f"행 삭제: {df_dropna.shape}")
    
    return df_mean

def feature_scaling_example():
    """특성 스케일링 예제"""
    print("\n=== 특성 스케일링 예제 ===")
    
    # 스케일이 다른 데이터 생성
    np.random.seed(42)
    data = {
        'age': np.random.randint(18, 80, 200),
        'income': np.random.normal(50000, 15000, 200),
        'experience': np.random.randint(0, 40, 200),
        'test_score': np.random.normal(75, 10, 200)
    }
    
    df = pd.DataFrame(data)
    X = df.values
    
    # 다양한 스케일링 방법
    standard_scaler = StandardScaler()
    minmax_scaler = MinMaxScaler()
    
    X_standard = standard_scaler.fit_transform(X)
    X_minmax = minmax_scaler.fit_transform(X)
    
    # 시각화
    plt.figure(figsize=(15, 10))
    
    # 원본 데이터
    plt.subplot(2, 3, 1)
    plt.boxplot(X, labels=df.columns)
    plt.title('Original Data')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    
    # 표준화
    plt.subplot(2, 3, 2)
    plt.boxplot(X_standard, labels=df.columns)
    plt.title('Standardized Data (Z-score)')
    plt.ylabel('Standardized Value')
    plt.xticks(rotation=45)
    
    # 정규화
    plt.subplot(2, 3, 3)
    plt.boxplot(X_minmax, labels=df.columns)
    plt.title('Normalized Data (Min-Max)')
    plt.ylabel('Normalized Value')
    plt.xticks(rotation=45)
    
    # 분포 비교 (income 특성)
    plt.subplot(2, 3, 4)
    plt.hist(X[:, 1], bins=30, alpha=0.7, label='Original', edgecolor='black')
    plt.xlabel('Income')
    plt.ylabel('Frequency')
    plt.title('Income Distribution - Original')
    plt.legend()
    
    plt.subplot(2, 3, 5)
    plt.hist(X_standard[:, 1], bins=30, alpha=0.7, label='Standardized', color='green', edgecolor='black')
    plt.xlabel('Standardized Income')
    plt.ylabel('Frequency')
    plt.title('Income Distribution - Standardized')
    plt.legend()
    
    plt.subplot(2, 3, 6)
    plt.hist(X_minmax[:, 1], bins=30, alpha=0.7, label='Normalized', color='orange', edgecolor='black')
    plt.xlabel('Normalized Income')
    plt.ylabel('Frequency')
    plt.title('Income Distribution - Normalized')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # 통계 정보 출력
    print("스케일링 전후 통계 정보:")
    print("\n원본 데이터:")
    print(df.describe())
    
    print("\n표준화 데이터:")
    df_standard = pd.DataFrame(X_standard, columns=df.columns)
    print(df_standard.describe())
    
    print("\n정규화 데이터:")
    df_minmax = pd.DataFrame(X_minmax, columns=df.columns)
    print(df_minmax.describe())
    
    return X_standard, X_minmax

def categorical_encoding_example():
    """범주형 데이터 인코딩 예제"""
    print("\n=== 범주형 데이터 인코딩 예제 ===")
    
    # 범주형 데이터가 포함된 샘플 데이터 생성
    np.random.seed(42)
    data = {
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 200),
        'city': np.random.choice(['Seoul', 'Busan', 'Incheon', 'Daegu'], 200),
        'employment_type': np.random.choice(['Full-time', 'Part-time', 'Contract'], 200),
        'age': np.random.randint(22, 65, 200),
        'salary': np.random.normal(50000, 15000, 200)
    }
    
    df = pd.DataFrame(data)
    
    print("원본 데이터 정보:")
    print(f"데이터 크기: {df.shape}")
    print(f"범주형 변수: {df.select_dtypes(include=['object']).columns.tolist()}")
    print(f"수치형 변수: {df.select_dtypes(include=[np.number]).columns.tolist()}")
    
    # 1. Label Encoding
    df_label = df.copy()
    label_encoders = {}
    
    for column in ['education', 'city', 'employment_type']:
        le = LabelEncoder()
        df_label[column] = le.fit_transform(df[column])
        label_encoders[column] = le
    
    # 2. One-Hot Encoding
    df_onehot = pd.get_dummies(df, columns=['education', 'city', 'employment_type'], prefix=['edu', 'city', 'emp'])
    
    # 시각화
    plt.figure(figsize=(15, 10))
    
    # 원본 데이터 분포
    plt.subplot(2, 3, 1)
    education_counts = df['education'].value_counts()
    plt.bar(education_counts.index, education_counts.values)
    plt.title('Education Distribution')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    plt.subplot(2, 3, 2)
    city_counts = df['city'].value_counts()
    plt.bar(city_counts.index, city_counts.values)
    plt.title('City Distribution')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    plt.subplot(2, 3, 3)
    emp_counts = df['employment_type'].value_counts()
    plt.bar(emp_counts.index, emp_counts.values)
    plt.title('Employment Type Distribution')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    # Label Encoding 결과
    plt.subplot(2, 3, 4)
    plt.hist(df_label['education'], bins=4, alpha=0.7, edgecolor='black')
    plt.title('Education (Label Encoded)')
    plt.xlabel('Encoded Value')
    plt.ylabel('Count')
    
    # One-Hot Encoding 차원 비교
    plt.subplot(2, 3, 5)
    dimensions = [df.shape[1], df_label.shape[1], df_onehot.shape[1]]
    methods = ['Original', 'Label Encoded', 'One-Hot Encoded']
    
    bars = plt.bar(methods, dimensions, color=['skyblue', 'lightgreen', 'orange'])
    plt.ylabel('Number of Features')
    plt.title('Feature Dimensions by Encoding Method')
    plt.xticks(rotation=45)
    
    for bar, dim in zip(bars, dimensions):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{dim}', ha='center', va='bottom')
    
    # 상관관계 히트맵 (Label Encoded)
    plt.subplot(2, 3, 6)
    correlation_matrix = df_label.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, square=True)
    plt.title('Correlation Matrix (Label Encoded)')
    
    plt.tight_layout()
    plt.show()
    
    # 인코딩 매핑 정보 출력
    print("\nLabel Encoding 매핑:")
    for column, encoder in label_encoders.items():
        mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
        print(f"{column}: {mapping}")
    
    print(f"\nOne-Hot Encoding 결과:")
    print(f"원본 특성 수: {df.shape[1]}")
    print(f"One-Hot 인코딩 후 특성 수: {df_onehot.shape[1]}")
    print(f"새로 생성된 특성들: {[col for col in df_onehot.columns if col not in df.columns]}")
    
    return df_label, df_onehot

def feature_selection_example():
    """특성 선택 예제"""
    print("\n=== 특성 선택 예제 ===")
    
    # 분류용 데이터 생성 (일부 특성은 노이즈)
    X, y = make_classification(n_samples=500, n_features=20, n_informative=5, 
                              n_redundant=5, n_clusters_per_class=1, random_state=42)
    
    # 특성 이름 생성
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    
    # 훈련/테스트 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 1. 단변량 특성 선택 (SelectKBest)
    selector_univariate = SelectKBest(score_func=f_classif, k=10)
    X_train_uni = selector_univariate.fit_transform(X_train, y_train)
    X_test_uni = selector_univariate.transform(X_test)
    
    # 2. 재귀적 특성 제거 (RFE)
    estimator = LogisticRegression(random_state=42, max_iter=1000)
    selector_rfe = RFE(estimator, n_features_to_select=10)
    X_train_rfe = selector_rfe.fit_transform(X_train, y_train)
    X_test_rfe = selector_rfe.transform(X_test)
    
    # 3. 특성 중요도 기반 선택 (Random Forest)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    feature_importances = rf.feature_importances_
    
    # 성능 비교
    models = {}
    
    # 모든 특성 사용
    lr_all = LogisticRegression(random_state=42, max_iter=1000)
    lr_all.fit(X_train, y_train)
    models['All Features'] = lr_all.score(X_test, y_test)
    
    # 단변량 선택
    lr_uni = LogisticRegression(random_state=42, max_iter=1000)
    lr_uni.fit(X_train_uni, y_train)
    models['Univariate Selection'] = lr_uni.score(X_test_uni, y_test)
    
    # RFE 선택
    lr_rfe = LogisticRegression(random_state=42, max_iter=1000)
    lr_rfe.fit(X_train_rfe, y_train)
    models['RFE Selection'] = lr_rfe.score(X_test_rfe, y_test)
    
    # 시각화
    plt.figure(figsize=(15, 10))
    
    # 단변량 점수
    plt.subplot(2, 3, 1)
    scores = selector_univariate.scores_
    plt.bar(range(len(scores)), scores)
    plt.xlabel('Feature Index')
    plt.ylabel('F-score')
    plt.title('Univariate Feature Scores')
    
    # RFE 순위
    plt.subplot(2, 3, 2)
    rankings = selector_rfe.ranking_
    colors = ['green' if rank == 1 else 'red' for rank in rankings]
    plt.bar(range(len(rankings)), rankings, color=colors)
    plt.xlabel('Feature Index')
    plt.ylabel('Ranking')
    plt.title('RFE Feature Rankings')
    
    # Random Forest 특성 중요도
    plt.subplot(2, 3, 3)
    sorted_indices = np.argsort(feature_importances)[::-1]
    plt.bar(range(len(feature_importances)), feature_importances[sorted_indices])
    plt.xlabel('Feature Index (sorted)')
    plt.ylabel('Importance')
    plt.title('Random Forest Feature Importance')
    
    # 선택된 특성 비교
    plt.subplot(2, 3, 4)
    selected_uni = selector_univariate.get_support()
    selected_rfe = selector_rfe.support_
    
    methods = ['Univariate', 'RFE']
    selected_counts = [np.sum(selected_uni), np.sum(selected_rfe)]
    
    plt.bar(methods, selected_counts, color=['skyblue', 'lightgreen'])
    plt.ylabel('Number of Selected Features')
    plt.title('Number of Selected Features')
    
    for i, count in enumerate(selected_counts):
        plt.text(i, count + 0.1, f'{count}', ha='center', va='bottom')
    
    # 성능 비교
    plt.subplot(2, 3, 5)
    method_names = list(models.keys())
    accuracies = list(models.values())
    
    bars = plt.bar(method_names, accuracies, color=['orange', 'skyblue', 'lightgreen'])
    plt.ylabel('Accuracy')
    plt.title('Model Performance Comparison')
    plt.xticks(rotation=45)
    
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    # 특성 선택 오버랩
    plt.subplot(2, 3, 6)
    overlap = np.sum(selected_uni & selected_rfe)
    venn_data = [
        np.sum(selected_uni & ~selected_rfe),  # Univariate only
        np.sum(~selected_uni & selected_rfe),  # RFE only
        overlap  # Both
    ]
    
    labels = ['Uni only', 'RFE only', 'Both']
    plt.pie(venn_data, labels=labels, autopct='%1.1f%%')
    plt.title('Feature Selection Overlap')
    
    plt.tight_layout()
    plt.show()
    
    # 결과 출력
    print("특성 선택 결과:")
    print(f"단변량 선택: {np.sum(selected_uni)}개 특성 선택")
    print(f"RFE 선택: {np.sum(selected_rfe)}개 특성 선택")
    print(f"공통 선택: {overlap}개 특성")
    
    print("\n성능 비교:")
    for method, accuracy in models.items():
        print(f"{method:20}: {accuracy:.3f}")

def complete_preprocessing_pipeline():
    """완전한 데이터 전처리 파이프라인 예제"""
    print("\n=== 완전한 데이터 전처리 파이프라인 ===")
    
    # 복합적인 문제가 있는 데이터 생성
    np.random.seed(42)
    n_samples = 300
    
    data = {
        'age': np.random.randint(18, 80, n_samples),
        'income': np.random.normal(50000, 15000, n_samples),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
        'experience': np.random.randint(0, 40, n_samples),
        'city': np.random.choice(['Seoul', 'Busan', 'Incheon'], n_samples),
        'score': np.random.normal(75, 10, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # 결측값 추가
    missing_indices = np.random.choice(n_samples, size=30, replace=False)
    df.loc[missing_indices[:15], 'income'] = np.nan
    df.loc[missing_indices[15:], 'score'] = np.nan
    
    # 타겟 변수 생성 (이진 분류)
    df['target'] = (df['score'] > 75).astype(int)
    
    print("전처리 전 데이터 상태:")
    print(f"데이터 크기: {df.shape}")
    print(f"결측값: {df.isnull().sum().sum()}개")
    print(f"범주형 변수: {df.select_dtypes(include=['object']).columns.tolist()}")
    
    # 전처리 파이프라인
    df_processed = df.copy()
    
    # 1. 결측값 처리
    imputer = SimpleImputer(strategy='median')
    df_processed[['income', 'score']] = imputer.fit_transform(df_processed[['income', 'score']])
    
    # 2. 범주형 인코딩
    df_processed = pd.get_dummies(df_processed, columns=['education', 'city'], prefix=['edu', 'city'])
    
    # 3. 특성과 타겟 분리
    X = df_processed.drop(['target'], axis=1)
    y = df_processed['target']
    
    # 4. 훈련/테스트 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 5. 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 6. 특성 선택
    selector = SelectKBest(score_func=f_classif, k=8)
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = selector.transform(X_test_scaled)
    
    # 모델 성능 비교
    models_comparison = {}
    
    # 전처리 전
    try:
        # 수치형 데이터만 사용 (범주형 제외)
        X_original = df[['age', 'income', 'experience', 'score']].dropna()
        y_original = df.loc[X_original.index, 'target']
        X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(
            X_original, y_original, test_size=0.2, random_state=42)
        
        lr_orig = LogisticRegression(random_state=42, max_iter=1000)
        lr_orig.fit(X_train_orig, y_train_orig)
        models_comparison['Original (no preprocessing)'] = lr_orig.score(X_test_orig, y_test_orig)
    except:
        models_comparison['Original (no preprocessing)'] = 0.0
    
    # 전처리 후
    lr_processed = LogisticRegression(random_state=42, max_iter=1000)
    lr_processed.fit(X_train_selected, y_train)
    models_comparison['Full preprocessing pipeline'] = lr_processed.score(X_test_selected, y_test)
    
    # 시각화
    plt.figure(figsize=(15, 8))
    
    # 전처리 전후 데이터 크기
    plt.subplot(2, 3, 1)
    stages = ['Original', 'Missing\nHandled', 'Encoded', 'Scaled', 'Selected']
    sizes = [df.shape[1], df.shape[1], X.shape[1], X.shape[1], X_train_selected.shape[1]]
    
    plt.plot(stages, sizes, 'bo-', linewidth=2, markersize=8)
    plt.ylabel('Number of Features')
    plt.title('Feature Count Through Pipeline')
    plt.xticks(rotation=45)
    
    # 결측값 처리 전후
    plt.subplot(2, 3, 2)
    missing_before = df.isnull().sum().sum()
    missing_after = df_processed.isnull().sum().sum()
    
    plt.bar(['Before', 'After'], [missing_before, missing_after], 
           color=['red', 'green'], alpha=0.7)
    plt.ylabel('Number of Missing Values')
    plt.title('Missing Values Handling')
    
    # 성능 비교
    plt.subplot(2, 3, 3)
    methods = list(models_comparison.keys())
    scores = list(models_comparison.values())
    
    bars = plt.bar(methods, scores, color=['orange', 'skyblue'])
    plt.ylabel('Accuracy')
    plt.title('Model Performance Comparison')
    plt.xticks(rotation=45)
    
    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom')
    
    # 특성 중요도 (선택된 특성)
    plt.subplot(2, 3, 4)
    selected_features = selector.get_support()
    scores = selector.scores_[selected_features]
    feature_names_selected = X.columns[selected_features]
    
    plt.barh(range(len(scores)), scores)
    plt.yticks(range(len(scores)), feature_names_selected)
    plt.xlabel('F-score')
    plt.title('Selected Features Importance')
    
    # 스케일링 전후 비교 (income 특성)
    income_idx = X.columns.get_loc('income')
    
    plt.subplot(2, 3, 5)
    plt.hist(X_train.iloc[:, income_idx], bins=20, alpha=0.7, label='Before Scaling', edgecolor='black')
    plt.hist(X_train_scaled[:, income_idx], bins=20, alpha=0.7, label='After Scaling', edgecolor='black')
    plt.xlabel('Income')
    plt.ylabel('Frequency')
    plt.title('Scaling Effect on Income')
    plt.legend()
    
    # 전처리 파이프라인 요약
    plt.subplot(2, 3, 6)
    pipeline_steps = ['Missing\nValues', 'Categorical\nEncoding', 'Scaling', 'Feature\nSelection']
    step_status = ['✓', '✓', '✓', '✓']
    
    colors = ['green'] * len(pipeline_steps)
    bars = plt.bar(pipeline_steps, [1]*len(pipeline_steps), color=colors, alpha=0.7)
    
    for i, (bar, status) in enumerate(zip(bars, step_status)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                status, ha='center', va='center', fontsize=20, color='white')
    
    plt.ylim(0, 1.2)
    plt.ylabel('Completed')
    plt.title('Preprocessing Pipeline Status')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n전처리 파이프라인 완료:")
    print(f"최종 특성 수: {X_train_selected.shape[1]}")
    print(f"훈련 데이터 크기: {X_train_selected.shape}")
    print(f"테스트 데이터 크기: {X_test_selected.shape}")
    print(f"최종 모델 성능: {models_comparison['Full preprocessing pipeline']:.3f}")

if __name__ == "__main__":
    # 예제 실행
    missing_value_handling()
    feature_scaling_example()
    categorical_encoding_example()
    feature_selection_example()
    complete_preprocessing_pipeline()
    
    print("\n실습 완료! 다음 사항들을 확인해보세요:")
    print("1. 결측값 처리 방법들의 장단점")
    print("2. 표준화와 정규화의 차이점과 사용 시기")
    print("3. 범주형 데이터 인코딩 방법 선택 기준")
    print("4. 특성 선택 방법들의 특징과 성능 영향")
    print("5. 전체 전처리 파이프라인의 순서와 중요성")
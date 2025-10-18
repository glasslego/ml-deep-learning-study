"""
클러스터링 분석 예제
학부생을 위한 K-Means와 계층적 클러스터링 실습 코드
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, load_iris
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정 (Mac의 경우)
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False


def kmeans_basic_example():
    """K-Means 기본 예제"""
    print("=== K-Means 클러스터링 기본 예제 ===")
    
    # 클러스터링용 데이터 생성
    X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, 
                          random_state=0)
    
    # K-Means 클러스터링 (k=4)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    y_pred = kmeans.fit_predict(X)
    
    # 성능 평가
    silhouette_avg = silhouette_score(X, y_pred)
    ari_score = adjusted_rand_score(y_true, y_pred)
    
    print(f"실루엣 점수: {silhouette_avg:.3f}")
    print(f"조정된 랜드 지수: {ari_score:.3f}")
    
    # 시각화
    plt.figure(figsize=(15, 5))
    
    # 원본 데이터
    plt.subplot(1, 3, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', alpha=0.7)
    plt.title('True Clusters')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    # K-Means 결과
    plt.subplot(1, 3, 2)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', alpha=0.7)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
               c='red', marker='x', s=200, linewidths=3, label='Centroids')
    plt.title('K-Means Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    
    # 클러스터 내 거리 분포
    plt.subplot(1, 3, 3)
    distances = []
    for i in range(4):
        cluster_points = X[y_pred == i]
        center = kmeans.cluster_centers_[i]
        cluster_distances = np.sqrt(np.sum((cluster_points - center)**2, axis=1))
        distances.extend(cluster_distances)
    
    plt.hist(distances, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('Distance to Centroid')
    plt.ylabel('Frequency')
    plt.title('Distribution of Distances to Centroids')
    
    plt.tight_layout()
    plt.show()
    
    return kmeans

def elbow_method_example():
    """엘보우 방법으로 최적 k 찾기"""
    print("\n=== 엘보우 방법으로 최적 k 찾기 ===")
    
    # 데이터 생성
    X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
    
    # 다양한 k 값에 대해 클러스터링 수행
    k_range = range(1, 11)
    inertias = []
    silhouette_scores = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
        
        if k > 1:  # 실루엣 점수는 k > 1일 때만 계산 가능
            silhouette_avg = silhouette_score(X, kmeans.labels_)
            silhouette_scores.append(silhouette_avg)
        else:
            silhouette_scores.append(0)
    
    # 시각화
    plt.figure(figsize=(12, 5))
    
    # 엘보우 방법
    plt.subplot(1, 2, 1)
    plt.plot(k_range, inertias, 'bo-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia (Within-cluster sum of squares)')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True, alpha=0.3)
    
    # 실루엣 점수
    plt.subplot(1, 2, 2)
    plt.plot(range(2, 11), silhouette_scores[1:], 'ro-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score vs Number of Clusters')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 최적 k 추천
    best_k_silhouette = np.argmax(silhouette_scores[1:]) + 2
    print(f"실루엣 점수 기준 최적 k: {best_k_silhouette}")

def hierarchical_clustering_example():
    """계층적 클러스터링 예제"""
    print("\n=== 계층적 클러스터링 예제 ===")
    
    # 아이리스 데이터셋 사용
    iris = load_iris()
    X = iris.data
    y_true = iris.target
    feature_names = iris.feature_names
    
    # 데이터 표준화
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 계층적 클러스터링
    hierarchical = AgglomerativeClustering(n_clusters=3, linkage='ward')
    y_pred = hierarchical.fit_predict(X_scaled)
    
    # 성능 평가
    silhouette_avg = silhouette_score(X_scaled, y_pred)
    ari_score = adjusted_rand_score(y_true, y_pred)
    
    print(f"실루엣 점수: {silhouette_avg:.3f}")
    print(f"조정된 랜드 지수: {ari_score:.3f}")
    
    # 덴드로그램 생성
    plt.figure(figsize=(15, 10))
    
    # 덴드로그램
    plt.subplot(2, 2, 1)
    linkage_matrix = linkage(X_scaled, method='ward')
    dendrogram(linkage_matrix, truncate_mode='level', p=3)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Sample Index or (Cluster Size)')
    plt.ylabel('Distance')
    
    # 원본 클래스
    plt.subplot(2, 2, 2)
    plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', alpha=0.7)
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title('True Classes')
    
    # 계층적 클러스터링 결과
    plt.subplot(2, 2, 3)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', alpha=0.7)
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title('Hierarchical Clustering Result')
    
    # 클러스터별 특성 분포
    plt.subplot(2, 2, 4)
    df = pd.DataFrame(X, columns=feature_names)
    df['Cluster'] = y_pred
    
    cluster_means = df.groupby('Cluster')[feature_names].mean()
    cluster_means.plot(kind='bar', ax=plt.gca())
    plt.title('Feature Means by Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Feature Value')
    plt.xticks(rotation=0)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()
    
    return hierarchical

def clustering_comparison():
    """다양한 클러스터링 방법 비교"""
    print("\n=== 클러스터링 방법 비교 ===")
    
    # 다양한 형태의 데이터 생성
    np.random.seed(42)
    
    # 1. 구형 클러스터
    X1, y1 = make_blobs(n_samples=100, centers=3, cluster_std=1.0, random_state=42)
    
    # 2. 길쭉한 클러스터
    X2 = np.random.randn(100, 2)
    X2[:50, 0] *= 3  # 첫 번째 클러스터를 길쭉하게
    X2[50:, 0] += 5  # 두 번째 클러스터를 이동
    
    datasets = [
        (X1, "Spherical Clusters"),
        (X2, "Elongated Clusters")
    ]
    
    plt.figure(figsize=(15, 8))
    
    for i, (X, title) in enumerate(datasets):
        # K-Means
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        y_kmeans = kmeans.fit_predict(X)
        
        # 계층적 클러스터링
        hierarchical = AgglomerativeClustering(n_clusters=3, linkage='ward')
        y_hierarchical = hierarchical.fit_predict(X)
        
        # 원본 데이터
        plt.subplot(2, 3, i*3 + 1)
        plt.scatter(X[:, 0], X[:, 1], alpha=0.7)
        plt.title(f'{title} - Original')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        
        # K-Means 결과
        plt.subplot(2, 3, i*3 + 2)
        plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis', alpha=0.7)
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                   c='red', marker='x', s=200, linewidths=3)
        plt.title(f'{title} - K-Means')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        
        # 계층적 클러스터링 결과
        plt.subplot(2, 3, i*3 + 3)
        plt.scatter(X[:, 0], X[:, 1], c=y_hierarchical, cmap='viridis', alpha=0.7)
        plt.title(f'{title} - Hierarchical')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        
        # 성능 비교
        print(f"\n{title}:")
        print(f"K-Means 실루엣 점수: {silhouette_score(X, y_kmeans):.3f}")
        print(f"계층적 클러스터링 실루엣 점수: {silhouette_score(X, y_hierarchical):.3f}")
    
    plt.tight_layout()
    plt.show()

def clustering_evaluation_metrics():
    """클러스터링 평가 지표 설명"""
    print("\n=== 클러스터링 평가 지표 ===")
    
    # 샘플 데이터
    X, y_true = make_blobs(n_samples=200, centers=3, cluster_std=1.0, random_state=42)
    
    # 다양한 k 값으로 클러스터링
    k_values = [2, 3, 4, 5]
    metrics_data = []
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        y_pred = kmeans.fit_predict(X)
        
        # 다양한 평가 지표 계산
        inertia = kmeans.inertia_
        silhouette = silhouette_score(X, y_pred)
        
        # 실제 클래스가 있는 경우만 ARI 계산
        if k == 3:  # 실제 클러스터 수와 같을 때
            ari = adjusted_rand_score(y_true, y_pred)
        else:
            ari = adjusted_rand_score(y_true, y_pred)
        
        metrics_data.append({
            'k': k,
            'Inertia': inertia,
            'Silhouette Score': silhouette,
            'Adjusted Rand Index': ari
        })
    
    # 결과 출력
    df_metrics = pd.DataFrame(metrics_data)
    print("\n클러스터링 평가 지표:")
    print(df_metrics.round(3))
    
    # 시각화
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(df_metrics['k'], df_metrics['Inertia'], 'bo-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Inertia vs k')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(df_metrics['k'], df_metrics['Silhouette Score'], 'ro-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score vs k')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.plot(df_metrics['k'], df_metrics['Adjusted Rand Index'], 'go-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Adjusted Rand Index')
    plt.title('ARI vs k')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 예제 실행
    kmeans_basic_example()
    elbow_method_example()
    hierarchical_clustering_example()
    clustering_comparison()
    clustering_evaluation_metrics()
    
    print("\n실습 완료! 다음 사항들을 확인해보세요:")
    print("1. K-Means와 계층적 클러스터링의 차이점")
    print("2. 엘보우 방법과 실루엣 점수를 이용한 최적 k 선택")
    print("3. 덴드로그램의 해석 방법")
    print("4. 다양한 클러스터링 평가 지표의 의미")
"""
딥러닝 기본 예제
학부생을 위한 신경망, CNN, RNN 기초 실습 코드
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# TensorFlow/Keras 임포트 (설치되어 있다면)
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
    print("TensorFlow 사용 가능")
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow가 설치되어 있지 않습니다. 기본 신경망 구현을 사용합니다.")

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'

def simple_neural_network_numpy():
    """NumPy로 구현한 간단한 신경망"""
    print("=== NumPy로 구현한 간단한 신경망 ===")
    
    class SimpleNeuralNetwork:
        def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
            # 가중치 초기화 (Xavier 초기화)
            self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
            self.b1 = np.zeros((1, hidden_size))
            self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
            self.b2 = np.zeros((1, output_size))
            self.learning_rate = learning_rate
            
        def sigmoid(self, x):
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        
        def sigmoid_derivative(self, x):
            return x * (1 - x)
        
        def forward(self, X):
            # 순전파
            self.z1 = np.dot(X, self.W1) + self.b1
            self.a1 = self.sigmoid(self.z1)
            self.z2 = np.dot(self.a1, self.W2) + self.b2
            self.a2 = self.sigmoid(self.z2)
            return self.a2
        
        def backward(self, X, y, output):
            # 역전파
            m = X.shape[0]
            
            # 출력층 오차
            dZ2 = output - y
            dW2 = (1/m) * np.dot(self.a1.T, dZ2)
            db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)
            
            # 은닉층 오차
            dZ1 = np.dot(dZ2, self.W2.T) * self.sigmoid_derivative(self.a1)
            dW1 = (1/m) * np.dot(X.T, dZ1)
            db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)
            
            # 가중치 업데이트
            self.W2 -= self.learning_rate * dW2
            self.b2 -= self.learning_rate * db2
            self.W1 -= self.learning_rate * dW1
            self.b1 -= self.learning_rate * db1
        
        def train(self, X, y, epochs=1000):
            losses = []
            for epoch in range(epochs):
                # 순전파
                output = self.forward(X)
                
                # 손실 계산 (평균 제곱 오차)
                loss = np.mean((output - y) ** 2)
                losses.append(loss)
                
                # 역전파
                self.backward(X, y, output)
                
                if epoch % 100 == 0:
                    print(f"Epoch {epoch}, Loss: {loss:.4f}")
            
            return losses
        
        def predict(self, X):
            output = self.forward(X)
            return (output > 0.5).astype(int)
    
    # 이진 분류 데이터 생성
    X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0,
                             n_informative=2, n_clusters_per_class=1, random_state=42)
    
    # 데이터 정규화
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 훈련/테스트 분할
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # 타겟 변수 reshape
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    
    # 신경망 생성 및 훈련
    nn = SimpleNeuralNetwork(input_size=2, hidden_size=10, output_size=1, learning_rate=0.1)
    losses = nn.train(X_train, y_train, epochs=1000)
    
    # 예측 및 평가
    y_pred = nn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"테스트 정확도: {accuracy:.3f}")
    
    # 시각화
    plt.figure(figsize=(15, 5))
    
    # 손실 함수 그래프
    plt.subplot(1, 3, 1)
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True, alpha=0.3)
    
    # 결정 경계 시각화
    plt.subplot(1, 3, 2)
    h = 0.02
    x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
    y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    
    Z = nn.forward(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, levels=50, alpha=0.8, cmap='RdBu')
    scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap='viridis', edgecolors='black')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Neural Network Decision Boundary')
    plt.colorbar()
    
    # 활성화 함수 시각화
    plt.subplot(1, 3, 3)
    x = np.linspace(-10, 10, 100)
    sigmoid_y = 1 / (1 + np.exp(-x))
    
    plt.plot(x, sigmoid_y, label='Sigmoid', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Sigmoid Activation Function')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return nn

def keras_mlp_example():
    """Keras를 사용한 다층 퍼셉트론 예제"""
    if not TENSORFLOW_AVAILABLE:
        print("TensorFlow가 설치되어 있지 않아 이 예제를 건너뜁니다.")
        return None
    
    print("\n=== Keras 다층 퍼셉트론 예제 ===")
    
    # 다중 클래스 분류 데이터 생성
    X, y = make_classification(n_samples=2000, n_features=10, n_informative=5,
                             n_redundant=2, n_classes=3, n_clusters_per_class=1,
                             random_state=42)
    
    # 데이터 정규화
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 훈련/검증/테스트 분할
    X_temp, X_test, y_temp, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)
    
    # 모델 구성
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(10,)),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(16, activation='relu'),
        layers.Dense(3, activation='softmax')
    ])
    
    # 모델 컴파일
    model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    
    print("모델 구조:")
    model.summary()
    
    # 모델 훈련
    history = model.fit(X_train, y_train,
                       batch_size=32,
                       epochs=50,
                       validation_data=(X_val, y_val),
                       verbose=0)
    
    # 모델 평가
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"테스트 정확도: {test_accuracy:.3f}")
    
    # 시각화
    plt.figure(figsize=(15, 5))
    
    # 훈련 손실
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 훈련 정확도
    plt.subplot(1, 3, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 활성화 함수 비교
    plt.subplot(1, 3, 3)
    x = np.linspace(-5, 5, 100)
    relu = np.maximum(0, x)
    sigmoid = 1 / (1 + np.exp(-x))
    tanh = np.tanh(x)
    
    plt.plot(x, relu, label='ReLU', linewidth=2)
    plt.plot(x, sigmoid, label='Sigmoid', linewidth=2)
    plt.plot(x, tanh, label='Tanh', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Activation Functions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return model

def simple_cnn_example():
    """간단한 CNN 예제 (손글씨 숫자 인식)"""
    if not TENSORFLOW_AVAILABLE:
        print("TensorFlow가 설치되어 있지 않아 이 예제를 건너뜁니다.")
        return None
    
    print("\n=== 간단한 CNN 예제 (손글씨 숫자 인식) ===")
    
    # 손글씨 숫자 데이터셋 로드
    digits = load_digits()
    X, y = digits.data, digits.target
    
    # 이미지 형태로 reshape (8x8 이미지)
    X = X.reshape(-1, 8, 8, 1)
    X = X.astype('float32') / 16.0  # 정규화 (0-16 범위를 0-1로)
    
    # 훈련/테스트 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # CNN 모델 구성
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(8, 8, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    
    # 모델 컴파일
    model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    
    print("CNN 모델 구조:")
    model.summary()
    
    # 모델 훈련
    history = model.fit(X_train, y_train,
                       batch_size=32,
                       epochs=20,
                       validation_split=0.2,
                       verbose=0)
    
    # 모델 평가
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"테스트 정확도: {test_accuracy:.3f}")
    
    # 시각화
    plt.figure(figsize=(15, 10))
    
    # 훈련 과정
    plt.subplot(2, 3, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('CNN Training Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('CNN Training Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 샘플 이미지들
    plt.subplot(2, 3, 3)
    fig, axes = plt.subplots(2, 5, figsize=(8, 4))
    for i in range(10):
        ax = axes[i//5, i%5]
        ax.imshow(X_test[i].reshape(8, 8), cmap='gray')
        ax.set_title(f'True: {y_test[i]}')
        ax.axis('off')
    plt.suptitle('Sample Test Images')
    
    # 예측 결과
    y_pred = model.predict(X_test[:10], verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    plt.subplot(2, 3, 4)
    correct = (y_pred_classes == y_test[:10]).sum()
    plt.bar(['Correct', 'Incorrect'], [correct, 10-correct], color=['green', 'red'])
    plt.ylabel('Count')
    plt.title(f'Prediction Results (First 10 samples)')
    
    # 예측 확률 분포
    plt.subplot(2, 3, 5)
    plt.hist(np.max(y_pred, axis=1), bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Max Prediction Probability')
    plt.ylabel('Frequency')
    plt.title('Prediction Confidence Distribution')
    
    # 특성 맵 시각화 (첫 번째 Conv2D 층)
    plt.subplot(2, 3, 6)
    # 중간층 출력을 위한 모델
    intermediate_model = keras.Model(inputs=model.input,
                                   outputs=model.layers[0].output)
    feature_maps = intermediate_model.predict(X_test[:1], verbose=0)
    
    # 첫 번째 특성 맵 표시
    plt.imshow(feature_maps[0, :, :, 0], cmap='viridis')
    plt.title('First Feature Map')
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()
    
    return model

def simple_rnn_example():
    """간단한 RNN 예제 (시계열 예측)"""
    if not TENSORFLOW_AVAILABLE:
        print("TensorFlow가 설치되어 있지 않아 이 예제를 건너뜁니다.")
        return None
    
    print("\n=== 간단한 RNN 예제 (시계열 예측) ===")
    
    # 간단한 사인파 시계열 데이터 생성
    def create_sine_wave_data(seq_length=50, n_samples=1000):
        np.random.seed(42)
        time_steps = np.linspace(0, 100, n_samples + seq_length)
        data = np.sin(time_steps) + 0.1 * np.random.randn(len(time_steps))
        
        X, y = [], []
        for i in range(n_samples):
            X.append(data[i:i+seq_length])
            y.append(data[i+seq_length])
        
        return np.array(X), np.array(y)
    
    # 데이터 생성
    seq_length = 20
    X, y = create_sine_wave_data(seq_length=seq_length, n_samples=1000)
    
    # 데이터 reshape (RNN 입력 형태: [samples, time_steps, features])
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    # 훈련/테스트 분할
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # RNN 모델 구성
    model = keras.Sequential([
        layers.SimpleRNN(50, activation='tanh', input_shape=(seq_length, 1)),
        layers.Dense(25, activation='relu'),
        layers.Dense(1)
    ])
    
    # 모델 컴파일
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    print("RNN 모델 구조:")
    model.summary()
    
    # 모델 훈련
    history = model.fit(X_train, y_train,
                       batch_size=32,
                       epochs=50,
                       validation_split=0.2,
                       verbose=0)
    
    # 예측
    y_pred = model.predict(X_test, verbose=0).flatten()
    
    # 성능 평가
    mse = np.mean((y_test - y_pred) ** 2)
    mae = np.mean(np.abs(y_test - y_pred))
    print(f"테스트 MSE: {mse:.4f}")
    print(f"테스트 MAE: {mae:.4f}")
    
    # 시각화
    plt.figure(figsize=(15, 10))
    
    # 훈련 과정
    plt.subplot(2, 3, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('RNN Training Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 예측 결과
    plt.subplot(2, 3, 2)
    plt.plot(y_test[:100], label='True Values', alpha=0.7)
    plt.plot(y_pred[:100], label='Predictions', alpha=0.7)
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.title('RNN Predictions vs True Values')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 오차 분포
    plt.subplot(2, 3, 3)
    errors = y_test - y_pred
    plt.hist(errors, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Prediction Error Distribution')
    plt.grid(True, alpha=0.3)
    
    # 산점도 (실제 vs 예측)
    plt.subplot(2, 3, 4)
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('True vs Predicted Values')
    plt.grid(True, alpha=0.3)
    
    # 시계열 데이터 패턴
    plt.subplot(2, 3, 5)
    original_data = np.sin(np.linspace(0, 10, 200)) + 0.1 * np.random.randn(200)
    plt.plot(original_data)
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.title('Original Sine Wave Pattern')
    plt.grid(True, alpha=0.3)
    
    # RNN vs LSTM 개념 비교 (시각적 설명)
    plt.subplot(2, 3, 6)
    plt.text(0.1, 0.8, 'RNN 특징:', fontsize=12, fontweight='bold')
    plt.text(0.1, 0.7, '• 단순한 구조', fontsize=10)
    plt.text(0.1, 0.6, '• 기울기 소실 문제', fontsize=10)
    plt.text(0.1, 0.5, '• 짧은 시퀀스에 적합', fontsize=10)
    plt.text(0.1, 0.3, 'LSTM 특징:', fontsize=12, fontweight='bold')
    plt.text(0.1, 0.2, '• 복잡한 구조 (게이트)', fontsize=10)
    plt.text(0.1, 0.1, '• 장기 의존성 학습', fontsize=10)
    plt.text(0.1, 0.0, '• 긴 시퀀스에 적합', fontsize=10)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title('RNN vs LSTM 비교')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return model

def neural_network_comparison():
    """다양한 신경망 아키텍처 비교"""
    print("\n=== 신경망 아키텍처 비교 ===")
    
    # 네트워크 타입별 특징 비교
    network_types = {
        'MLP (Multi-Layer Perceptron)': {
            'structure': 'Dense layers only',
            'input': 'Tabular data',
            'use_case': 'Classification, Regression',
            'pros': 'Simple, Fast training',
            'cons': 'Limited feature extraction'
        },
        'CNN (Convolutional Neural Network)': {
            'structure': 'Conv + Pool + Dense',
            'input': 'Images, 2D data',
            'use_case': 'Image recognition, Computer vision',
            'pros': 'Spatial feature extraction',
            'cons': 'Requires large datasets'
        },
        'RNN (Recurrent Neural Network)': {
            'structure': 'Recurrent connections',
            'input': 'Sequential data',
            'use_case': 'Time series, NLP',
            'pros': 'Handles sequences',
            'cons': 'Vanishing gradient problem'
        },
        'LSTM (Long Short-Term Memory)': {
            'structure': 'RNN with gates',
            'input': 'Sequential data',
            'use_case': 'Long sequences, NLP',
            'pros': 'Long-term dependencies',
            'cons': 'Complex, Slow training'
        }
    }
    
    # 시각화
    plt.figure(figsize=(15, 10))
    
    # 네트워크 구조 개념도
    plt.subplot(2, 2, 1)
    plt.text(0.5, 0.9, 'MLP Structure', ha='center', fontsize=14, fontweight='bold')
    # MLP 노드들을 원으로 표현
    for layer in range(3):
        for node in range(3):
            circle = plt.Circle((layer*0.3 + 0.2, node*0.2 + 0.3), 0.05, color='lightblue')
            plt.gca().add_patch(circle)
        if layer < 2:
            # 연결선 그리기
            for i in range(3):
                for j in range(3):
                    plt.plot([layer*0.3 + 0.25, (layer+1)*0.3 + 0.15], 
                            [i*0.2 + 0.3, j*0.2 + 0.3], 'k-', alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    
    # CNN 구조
    plt.subplot(2, 2, 2)
    plt.text(0.5, 0.9, 'CNN Structure', ha='center', fontsize=14, fontweight='bold')
    # 간단한 CNN 구조 표현
    rectangles = [
        (0.1, 0.4, 0.15, 0.3),  # Input
        (0.3, 0.5, 0.1, 0.2),   # Conv
        (0.45, 0.55, 0.05, 0.1), # Pool
        (0.6, 0.5, 0.1, 0.2),   # Conv
        (0.75, 0.55, 0.05, 0.1), # Pool
        (0.85, 0.4, 0.1, 0.3)   # Dense
    ]
    colors = ['lightgreen', 'lightblue', 'orange', 'lightblue', 'orange', 'pink']
    for rect, color in zip(rectangles, colors):
        plt.gca().add_patch(plt.Rectangle((rect[0], rect[1]), rect[2], rect[3], 
                                        facecolor=color, alpha=0.7))
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    
    # RNN 구조
    plt.subplot(2, 2, 3)
    plt.text(0.5, 0.9, 'RNN Structure', ha='center', fontsize=14, fontweight='bold')
    # RNN 순환 구조 표현
    for t in range(3):
        # RNN 셀
        circle = plt.Circle((t*0.25 + 0.2, 0.5), 0.08, color='lightcoral')
        plt.gca().add_patch(circle)
        plt.text(t*0.25 + 0.2, 0.5, f't{t+1}', ha='center', va='center')
        
        # 순환 연결
        if t < 2:
            plt.arrow(t*0.25 + 0.28, 0.5, 0.15, 0, head_width=0.02, 
                     head_length=0.02, fc='black', ec='black')
        
        # 입력/출력
        plt.arrow(t*0.25 + 0.2, 0.3, 0, 0.12, head_width=0.02, 
                 head_length=0.02, fc='blue', ec='blue')
        plt.arrow(t*0.25 + 0.2, 0.58, 0, 0.12, head_width=0.02, 
                 head_length=0.02, fc='red', ec='red')
    
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    
    # 비교 표
    plt.subplot(2, 2, 4)
    table_data = []
    for name, features in network_types.items():
        table_data.append([name.split()[0], features['input'], features['use_case']])
    
    table = plt.table(cellText=table_data,
                     colLabels=['Type', 'Input', 'Use Case'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.3, 0.3, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    plt.axis('off')
    plt.title('Neural Network Comparison', pad=20)
    
    plt.tight_layout()
    plt.show()
    
    # 상세 비교 정보 출력
    print("\n신경망 아키텍처 상세 비교:")
    for name, features in network_types.items():
        print(f"\n{name}:")
        for key, value in features.items():
            print(f"  {key.capitalize()}: {value}")

def activation_functions_demo():
    """활성화 함수 비교 데모"""
    print("\n=== 활성화 함수 비교 ===")
    
    # 다양한 활성화 함수 정의
    def relu(x):
        return np.maximum(0, x)
    
    def leaky_relu(x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)
    
    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def tanh(x):
        return np.tanh(x)
    
    def softmax(x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    # x 값 범위
    x = np.linspace(-5, 5, 100)
    
    # 시각화
    plt.figure(figsize=(15, 10))
    
    # ReLU
    plt.subplot(2, 3, 1)
    plt.plot(x, relu(x), 'b-', linewidth=2)
    plt.title('ReLU')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True, alpha=0.3)
    plt.text(-4, 3, 'f(x) = max(0, x)', fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    
    # Leaky ReLU
    plt.subplot(2, 3, 2)
    plt.plot(x, leaky_relu(x), 'g-', linewidth=2)
    plt.title('Leaky ReLU')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True, alpha=0.3)
    plt.text(-4, 2, 'f(x) = max(0.01x, x)', fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
    
    # Sigmoid
    plt.subplot(2, 3, 3)
    plt.plot(x, sigmoid(x), 'r-', linewidth=2)
    plt.title('Sigmoid')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True, alpha=0.3)
    plt.text(-4, 0.8, 'f(x) = 1/(1+e^(-x))', fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
    
    # Tanh
    plt.subplot(2, 3, 4)
    plt.plot(x, tanh(x), 'm-', linewidth=2)
    plt.title('Tanh')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True, alpha=0.3)
    plt.text(-4, 0.5, 'f(x) = tanh(x)', fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="plum"))
    
    # 모든 함수 비교
    plt.subplot(2, 3, 5)
    plt.plot(x, relu(x), 'b-', linewidth=2, label='ReLU')
    plt.plot(x, leaky_relu(x), 'g-', linewidth=2, label='Leaky ReLU')
    plt.plot(x, sigmoid(x), 'r-', linewidth=2, label='Sigmoid')
    plt.plot(x, tanh(x), 'm-', linewidth=2, label='Tanh')
    plt.title('All Activation Functions')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 활성화 함수 특징 표
    plt.subplot(2, 3, 6)
    features = [
        ['ReLU', 'Fast', 'Dead neurons'],
        ['Leaky ReLU', 'No dead neurons', 'Slight negative slope'],
        ['Sigmoid', 'Smooth', 'Vanishing gradient'],
        ['Tanh', 'Zero-centered', 'Vanishing gradient'],
        ['Softmax', 'Probability output', 'Output layer only']
    ]
    
    table = plt.table(cellText=features,
                     colLabels=['Function', 'Advantage', 'Disadvantage'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.3, 0.35, 0.35])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)
    plt.axis('off')
    plt.title('Activation Function Characteristics', pad=20)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 예제 실행
    simple_neural_network_numpy()
    
    if TENSORFLOW_AVAILABLE:
        keras_mlp_example()
        simple_cnn_example()
        simple_rnn_example()
    else:
        print("\nTensorFlow 예제들을 실행하려면 다음 명령어로 설치하세요:")
        print("pip install tensorflow")
    
    neural_network_comparison()
    activation_functions_demo()
    
    print("\n실습 완료! 다음 사항들을 확인해보세요:")
    print("1. 신경망의 순전파와 역전파 과정")
    print("2. CNN의 컨볼루션과 풀링 연산의 역할")
    print("3. RNN의 순환 구조와 시계열 데이터 처리")
    print("4. 다양한 활성화 함수의 특징과 선택 기준")
    print("5. 각 신경망 아키텍처의 적합한 사용 사례")
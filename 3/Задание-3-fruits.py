import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import pathlib


# 1. Параметры и загрузка данных (только тестовая выборка для визуализации)

data_dir = pathlib.Path(r'C:\Users\godta\Desktop\Учеба\ОМО')
test_dir = data_dir / 'test'

img_height, img_width = 150, 150
batch_size = 32

# Загружаем тестовый датасет (можно и train, но test быстрее)
test_ds = keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode='int',
    shuffle=False
)

class_names = test_ds.class_names
num_classes = len(class_names)
print(f"Классы: {class_names}")

# Извлекаем все изображения и метки в numpy массивы
X_all = []
y_all = []
for images, labels in test_ds:
    X_all.append(images.numpy())
    y_all.append(labels.numpy())
X_all = np.concatenate(X_all, axis=0)  # форма (N, 150, 150, 3)
y_all = np.concatenate(y_all, axis=0)  # форма (N,)

print(f"Размер тестовой выборки: {X_all.shape}")


# 2. Извлечение признаков с помощью MobileNetV2 (без верхушки)

# Загружаем предобученную модель без классификатора
base_model = MobileNetV2(weights='imagenet', include_top=False,
                         input_shape=(img_height, img_width, 3))
base_model.trainable = False

# Добавляем глобальный средний пулинг, чтобы получить вектор признаков
feature_extractor = keras.Sequential([
    base_model,
    keras.layers.GlobalAveragePooling2D()
])

# Нормализуем входные данные под MobileNetV2 (от -1 до 1)
X_preprocessed = preprocess_input(X_all.astype(np.float32))

# Извлекаем признаки
features = feature_extractor.predict(X_preprocessed, batch_size=batch_size)
print(f"Размер признаков: {features.shape}")  # (N, 1280) для MobileNetV2


# 3. PCA – линейное снижение до 2 компонент

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(features)
print(f"PCA объяснённая дисперсия: {pca.explained_variance_ratio_.sum():.3f}")

# Визуализация PCA
plt.figure(figsize=(8, 6))
for i, name in enumerate(class_names):
    plt.scatter(X_pca[y_all == i, 0], X_pca[y_all == i, 1], label=name, alpha=0.7, s=20)
plt.xlabel('Первая главная компонента')
plt.ylabel('Вторая главная компонента')
plt.title('PCA: тестовые изображения фруктов')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# 4. t‑SNE – нелинейное вложение в 2D

tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
X_tsne = tsne.fit_transform(features)

plt.figure(figsize=(8, 6))
for i, name in enumerate(class_names):
    plt.scatter(X_tsne[y_all == i, 0], X_tsne[y_all == i, 1], label=name, alpha=0.7, s=20)
plt.xlabel('t‑SNE компонента 1')
plt.ylabel('t‑SNE компонента 2')
plt.title('t‑SNE: тестовые изображения фруктов')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# 5. Кластеризация K‑Means (на исходных признаках)

kmeans = KMeans(n_clusters=num_classes, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(features)

# Оценка качества
ari = adjusted_rand_score(y_all, cluster_labels)
silhouette = silhouette_score(features, cluster_labels)
print(f"Adjusted Rand Index (ARI): {ari:.3f}")
print(f"Silhouette Score: {silhouette:.3f}")

# Визуализация кластеров на PCA-проекции
plt.figure(figsize=(8, 6))
for i in range(num_classes):
    plt.scatter(X_pca[cluster_labels == i, 0], X_pca[cluster_labels == i, 1],
                label=f'Кластер {i}', alpha=0.7, s=20)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            s=200, c='black', marker='X', label='Центроиды')
plt.xlabel('Первая главная компонента')
plt.ylabel('Вторая главная компонента')
plt.title('K‑Means кластеры (PCA проекция)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Визуализация кластеров на t‑SNE-проекции (центроиды в исходном пространстве, здесь не отображаем)
plt.figure(figsize=(8, 6))
for i in range(num_classes):
    plt.scatter(X_tsne[cluster_labels == i, 0], X_tsne[cluster_labels == i, 1],
                label=f'Кластер {i}', alpha=0.7, s=20)
plt.xlabel('t‑SNE компонента 1')
plt.ylabel('t‑SNE компонента 2')
plt.title('K‑Means кластеры (t‑SNE проекция)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
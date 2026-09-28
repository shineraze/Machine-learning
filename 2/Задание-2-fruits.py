import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import pathlib


# 1. Загрузка данных

data_dir = pathlib.Path(r'C:\Users\godta\Desktop\Учеба\ОМО')
train_dir = data_dir / 'train'
test_dir = data_dir / 'test'

img_height = 150
img_width = 150
batch_size = 32

# Загружаем train и test датасеты
train_ds = keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode='int'
)

test_ds = keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode='int',
    shuffle=False  # не перемешиваем, чтобы удобнее было строить confusion matrix
)

# Количество классов
class_names = train_ds.class_names
num_classes = len(class_names)
print(f"Классы: {class_names}")
print(f"Всего классов: {num_classes}")

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)


# 2. Своя CNN (обучение с нуля)

model_custom = keras.Sequential([
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),  # нормализация
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

model_custom.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])

model_custom.summary()

epochs = 15
history = model_custom.fit(train_ds, epochs=epochs, validation_data=test_ds)


# 3. Оценка и визуализация (своя CNN)

test_loss, test_acc = model_custom.evaluate(test_ds)
print(f"\nТочность своей CNN на тестовых данных: {test_acc:.4f}")

# Графики обучения
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.title('Точность (своя CNN)')
plt.xlabel('Эпоха')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.title('Потери (своя CNN)')
plt.xlabel('Эпоха')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

# Предсказания и метрики
y_true = np.concatenate([y for x, y in test_ds], axis=0)
y_pred_probs = model_custom.predict(test_ds)
y_pred = np.argmax(y_pred_probs, axis=1)

print("\nClassification Report (своя CNN):")
print(classification_report(y_true, y_pred, target_names=class_names))

# Матрица ошибок
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix (своя CNN)')
plt.xlabel('Предсказано')
plt.ylabel('Истина')
plt.show()

# Визуализация нескольких примеров из тестового набора
plt.figure(figsize=(12, 12))
for images, labels in test_ds.take(1):
    for i in range(min(25, len(images))):
        ax = plt.subplot(5, 5, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        pred_label = class_names[np.argmax(y_pred_probs[i])]
        true_label = class_names[labels[i]]
        color = 'green' if pred_label == true_label else 'red'
        plt.title(f"P:{pred_label}\nT:{true_label}", color=color, fontsize=8)
        plt.axis('off')
plt.tight_layout()
plt.show()

# Сохраняем модель
model_custom.save('fruit_cnn_custom.h5')
print("Модель сохранена как fruit_cnn_custom.h5")


# 4. Transfer Learning (MobileNetV2) для сравнения

# Загружаем предобученную базу без верха
base_model = keras.applications.MobileNetV2(
    input_shape=(img_height, img_width, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # замораживаем базовые слои

# Добавляем новые слои
model_tl = keras.Sequential([
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(num_classes, activation='softmax')
])

model_tl.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])

model_tl.summary()

# Обучаем меньше эпох, т.к. база уже предобучена
history_tl = model_tl.fit(train_ds, epochs=10, validation_data=test_ds)

tl_test_loss, tl_test_acc = model_tl.evaluate(test_ds)
print(f"\nТочность Transfer Learning (MobileNetV2): {tl_test_acc:.4f}")

# Графики обучения TL
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history_tl.history['accuracy'], label='train')
plt.plot(history_tl.history['val_accuracy'], label='val')
plt.title('Точность Transfer Learning')
plt.xlabel('Эпоха')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_tl.history['loss'], label='train')
plt.plot(history_tl.history['val_loss'], label='val')
plt.title('Потери Transfer Learning')
plt.xlabel('Эпоха')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

# 5. Сравнение итоговых точностей

print("\n=== СРАВНЕНИЕ МОДЕЛЕЙ ===")
print(f"Своя CNN:            {test_acc:.4f}")
print(f"Transfer Learning:   {tl_test_acc:.4f}")

# Сохраняем модель transfer learning
model_tl.save('fruit_cnn_transfer.h5')
print("Модель Transfer Learning сохранена как fruit_cnn_transfer.h5")
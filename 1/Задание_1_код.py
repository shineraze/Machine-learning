import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# Загрузка датасета Wine
wine = load_wine()
X = wine.data
y = wine.target

# Пропускаем большой вывод описания датасета (wine.DESCR)
# Выводим только необходимую информацию
print("\nПризнаки:", wine.feature_names)
print("Классы:", wine.target_names)

# Разделение на train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Визуализация данных (scatter plot для двух признаков)
plt.figure(figsize=(8, 6))
for i, target_name in enumerate(wine.target_names):
    plt.scatter(X[y == i, 0], X[y == i, 1], label=target_name, alpha=0.7)
plt.xlabel(wine.feature_names[0])
plt.ylabel(wine.feature_names[1])
plt.title('Wine Dataset: Alcohol vs Malic Acid')
plt.legend()
plt.show()

# Модель KNN с k=1
knn1 = KNeighborsClassifier(n_neighbors=1)
knn1.fit(X_train, y_train)
y_pred_knn1 = knn1.predict(X_test)
acc_knn1 = accuracy_score(y_test, y_pred_knn1)
print(f"\nKNN (k=1) Accuracy: {acc_knn1:.4f}")
print("Classification Report KNN (k=1):")
print(classification_report(y_test, y_pred_knn1, target_names=wine.target_names))

# Модель KNN с k=5
knn5 = KNeighborsClassifier(n_neighbors=5)
knn5.fit(X_train, y_train)
y_pred_knn5 = knn5.predict(X_test)
acc_knn5 = accuracy_score(y_test, y_pred_knn5)
print(f"\nKNN (k=5) Accuracy: {acc_knn5:.4f}")
print("Classification Report KNN (k=5):")
print(classification_report(y_test, y_pred_knn5, target_names=wine.target_names))

# Модель Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
print(f"\nRandom Forest Accuracy: {acc_rf:.4f}")
print("Classification Report Random Forest:")
print(classification_report(y_test, y_pred_rf, target_names=wine.target_names))

# Сравнение точности
models = ['KNN (k=1)', 'KNN (k=5)', 'Random Forest']
accuracies = [acc_knn1, acc_knn5, acc_rf]
plt.figure(figsize=(6, 4))
plt.bar(models, accuracies, color=['blue', 'green', 'red'])
plt.ylabel('Accuracy')
plt.title('Сравнение точности моделей')
plt.ylim(0, 1)
for i, v in enumerate(accuracies):
    plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
plt.show()

print("\nСравнение: Random Forest обычно лучше KNN на этом датасете из-за ансамбля деревьев.")
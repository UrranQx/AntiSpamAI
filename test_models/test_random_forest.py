"""
Тестирование Random Forest классификатора для спам-фильтра

Random Forest - это классический ML алгоритм на основе деревьев решений.
Использует TF-IDF векторизацию текста.
"""

import os
import time
import numpy as np
from data_loader import load_emails
from models.random_forest import RandomForestSpamClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = "../data/extracted/body"


def print_dataset_info(X_train, y_train, X_test, y_test):
    """Выводит информацию о датасете"""
    print("\n" + "="*60)
    print("ИНФОРМАЦИЯ О ДАТАСЕТЕ")
    print("="*60)

    train_ham = sum(1 for label in y_train if label == 0)
    train_spam = sum(1 for label in y_train if label == 1)
    test_ham = sum(1 for label in y_test if label == 0)
    test_spam = sum(1 for label in y_test if label == 1)

    print(f"\nОбучающая выборка: {len(X_train)} писем")
    print(f"  - Ham (не спам):  {train_ham} ({train_ham/len(y_train)*100:.1f}%)")
    print(f"  - Spam:           {train_spam} ({train_spam/len(y_train)*100:.1f}%)")

    print(f"\nТестовая выборка: {len(X_test)} писем")
    print(f"  - Ham (не спам):  {test_ham} ({test_ham/len(y_test)*100:.1f}%)")
    print(f"  - Spam:           {test_spam} ({test_spam/len(y_test)*100:.1f}%)")

    # Статистика длин текстов
    train_lengths = [len(text.split()) for text in X_train]
    test_lengths = [len(text.split()) for text in X_test]

    print(f"\nСтатистика длин текстов (обучающая выборка):")
    print(f"  - Средняя длина:  {np.mean(train_lengths):.0f} слов")
    print(f"  - Медиана:        {np.median(train_lengths):.0f} слов")
    print(f"  - Минимум:        {np.min(train_lengths)} слов")
    print(f"  - Максимум:       {np.max(train_lengths)} слов")
    print("="*60 + "\n")


def plot_confusion_matrix(cm, title="Confusion Matrix"):
    """Визуализирует матрицу ошибок"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Ham', 'Spam'],
                yticklabels=['Ham', 'Spam'])
    plt.title(title)
    plt.ylabel('Истинная метка')
    plt.xlabel('Предсказанная метка')
    plt.tight_layout()
    plt.savefig('random_forest_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print(f"✓ График сохранен: random_forest_confusion_matrix.png")
    plt.show()


def evaluate_model(model, X_test, y_test):
    """Оценивает качество модели"""
    print("\n" + "="*60)
    print("ОЦЕНКА МОДЕЛИ")
    print("="*60)

    # Предсказания
    start_time = time.time()
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    prediction_time = time.time() - start_time

    # Метрики
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    cm = confusion_matrix(y_test, y_pred)

    print(f"\nВремя предсказания: {prediction_time:.4f} сек")
    print(f"Скорость: {len(X_test)/prediction_time:.0f} писем/сек")

    print(f"\n{'Метрика':<20} {'Значение':<10}")
    print("-"*30)
    print(f"{'Accuracy (Точность)':<20} {accuracy*100:>6.2f}%")
    print(f"{'Precision (Точность)':<20} {precision*100:>6.2f}%")
    print(f"{'Recall (Полнота)':<20} {recall*100:>6.2f}%")
    print(f"{'F1-Score':<20} {f1*100:>6.2f}%")

    print("\nМатрица ошибок:")
    print(f"                Предсказано Ham  Предсказано Spam")
    print(f"Истинный Ham         {cm[0][0]:<10}      {cm[0][1]:<10}")
    print(f"Истинный Spam        {cm[1][0]:<10}      {cm[1][1]:<10}")

    # Детальный отчет
    print("\nДетальный классификационный отчет:")
    print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

    print("="*60 + "\n")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'prediction_time': prediction_time
    }


def test_on_samples(model, X_test, y_test, n_samples=5):
    """Показывает примеры предсказаний"""
    print("\n" + "="*60)
    print("ПРИМЕРЫ ПРЕДСКАЗАНИЙ")
    print("="*60)

    # Берем случайные примеры
    indices = np.random.choice(len(X_test), min(n_samples, len(X_test)), replace=False)

    for i, idx in enumerate(indices, 1):
        text = X_test[idx]
        true_label = "SPAM" if y_test[idx] == 1 else "HAM"
        pred = model.predict([text])[0]
        pred_label = "SPAM" if pred == 1 else "HAM"
        proba = model.predict_proba([text])[0]

        TEXT_MAX_LEN_PREVIEW = 1000
        # Показываем первые TEXT_MAX_LEN_PREVIEW символов письма
        text_preview = text[:TEXT_MAX_LEN_PREVIEW] + "..." if len(text) > TEXT_MAX_LEN_PREVIEW else text

        print(f"\n--- Пример {i} ---")
        print(f"Текст: \n{text_preview}")
        print(f"Истинная метка: {true_label}")
        print(f"Предсказание:   {pred_label}")
        print(f"Уверенность:    Ham={proba[0]*100:.1f}%, Spam={proba[1]*100:.1f}%")
        print(f"Результат:      {'✓ ВЕРНО' if pred == y_test[idx] else '✗ ОШИБКА'}")

    print("\n" + "="*60 + "\n")


def main():
    print("\n" + "="*60)
    print("ТЕСТИРОВАНИЕ RANDOM FOREST КЛАССИФИКАТОРА")
    print("="*60)

    # 1. Загрузка данных
    print("\n[1/5] Загрузка данных...")
    texts, labels = load_emails(DATA_DIR)
    print(f"✓ Загружено {len(texts)} писем")

    # 2. Разделение на train/test
    print("\n[2/5] Разделение на обучающую и тестовую выборки...")
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.3, random_state=42, stratify=labels
    )
    print_dataset_info(X_train, y_train, X_test, y_test)

    # 3. Создание и обучение модели
    print("[3/5] Обучение Random Forest модели...")
    print("Параметры: n_estimators=100, max_features=5000 (TF-IDF)")

    start_time = time.time()
    rf_model = RandomForestSpamClassifier(n_estimators=100)
    rf_model.train(X_train, y_train)
    training_time = time.time() - start_time

    print(f"✓ Обучение завершено за {training_time:.2f} сек")

    # 4. Оценка модели
    print("\n[4/5] Оценка качества модели...")
    results = evaluate_model(rf_model, X_test, y_test)

    # 5. Примеры предсказаний
    print("[5/5] Демонстрация работы на примерах...")
    test_on_samples(rf_model, X_test, y_test, n_samples=5)

    # 6. Визуализация
    print("\nСоздание визуализации матрицы ошибок...")
    plot_confusion_matrix(results['confusion_matrix'],
                         title="Random Forest - Матрица ошибок")

    # Итоговая сводка
    print("\n" + "="*60)
    print("ИТОГОВАЯ СВОДКА")
    print("="*60)
    print(f"Модель:           Random Forest")
    print(f"Время обучения:   {training_time:.2f} сек")
    print(f"Время предсказания: {results['prediction_time']:.4f} сек")
    print(f"Accuracy:         {results['accuracy']*100:.2f}%")
    print(f"F1-Score:         {results['f1']*100:.2f}%")
    print(f"Precision:        {results['precision']*100:.2f}%")
    print(f"Recall:           {results['recall']*100:.2f}%")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()


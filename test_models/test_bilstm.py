import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

from data_loader import load_emails, EmailDataset
from models.bilstm import BiLSTMSpamClassifier
from sklearn.model_selection import train_test_split

# Настройки
DATA_DIR = "../data/extracted/body"
MAX_LEN = 1604  # 95-й перцентиль
EMBEDDING_DIM = 128
HIDDEN_DIM = 64  # Уменьшено с 128
NUM_LAYERS = 2
DROPOUT = 0.5
BATCH_SIZE = 128  # Увеличено с 32
EPOCHS = 10
LEARNING_RATE = 0.001


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Обучение на одной эпохе"""
    model.train()
    total_loss = 0

    for texts, labels in dataloader:
        texts, labels = texts.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    """Оценка модели"""
    model.eval()
    total_loss = 0
    predictions = []
    true_labels = []

    with torch.no_grad():
        for texts, labels in dataloader:
            texts, labels = texts.to(device), labels.to(device)

            outputs = model(texts)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)

    return avg_loss, accuracy, precision, recall, f1, predictions, true_labels


def plot_confusion_matrix(y_true, y_pred):
    """Визуализация матрицы ошибок"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
    plt.title('Confusion Matrix - BiLSTM')
    plt.ylabel('True Label (0=Ham, 1=Spam)')
    plt.xlabel('Predicted Label (0=Ham, 1=Spam)')
    plt.tight_layout()
    plt.savefig('bilstm_confusion_matrix.png')
    plt.close()
    print("Confusion matrix сохранена в bilstm_confusion_matrix.png")


def plot_training_history(train_losses, val_losses, accuracies):
    """Визуализация процесса обучения"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(train_losses, label='Train Loss', color='blue')
    ax1.plot(val_losses, label='Validation Loss', color='red')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss - BiLSTM')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(accuracies, label='Validation Accuracy', color='green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Validation Accuracy - BiLSTM')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('bilstm_training_history.png')
    plt.close()
    print("Training history сохранена в bilstm_training_history.png")


if __name__ == "__main__":
    print("=" * 50)
    print("ТЕСТИРОВАНИЕ BIDIRECTIONAL LSTM КЛАССИФИКАТОРА")
    print("=" * 50)

    # Загрузка данных
    print("\n[1/5] Загрузка данных...")
    texts, labels = load_emails(DATA_DIR)
    print(f"Всего загружено писем: {len(texts)}")
    print(f"Распределение: Ham={labels.count(0)}, Spam={labels.count(1)}")

    # Разделение на train/test
    print("\n[2/5] Разделение на train/test выборки...")
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.3, random_state=42, stratify=labels
    )
    print(f"Тренировочная выборка: {len(X_train)} писем")
    print(f"Тестовая выборка: {len(X_test)} писем")
    print(f"Распределение в train: Ham={y_train.count(0)}, Spam={y_train.count(1)}")
    print(f"Распределение в test: Ham={y_test.count(0)}, Spam={y_test.count(1)}")

    # Создание датасетов
    print("\n[3/5] Создание датасетов и построение словаря...")
    print(f"Максимальная длина последовательности: {MAX_LEN}")
    train_dataset = EmailDataset(X_train, y_train, max_len=MAX_LEN)
    vocab = train_dataset.vocab
    vocab_size = len(vocab)
    test_dataset = EmailDataset(X_test, y_test, vocab=vocab, max_len=MAX_LEN)
    print(f"Размер словаря: {vocab_size} уникальных слов")

    # Создание DataLoader
    print("\n[4/5] Создание DataLoader...")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Количество батчей в train: {len(train_loader)}")
    print(f"Количество батчей в test: {len(test_loader)}")

    # Инициализация модели
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[5/5] Инициализация модели...")
    print(f"Используется устройство: {device}")

    model = BiLSTMSpamClassifier(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)

    print(f"Архитектура модели:")
    print(f"  - Embedding размерность: {EMBEDDING_DIM}")
    print(f"  - Hidden размерность: {HIDDEN_DIM}")
    print(f"  - Количество слоев LSTM: {NUM_LAYERS}")
    print(f"  - Bidirectional: True (выход = {HIDDEN_DIM * 2})")
    print(f"  - Dropout: {DROPOUT}")
    print(f"  - Всего параметров: {sum(p.numel() for p in model.parameters())}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Обучение
    print(f"\n" + "=" * 50)
    print("НАЧАЛО ОБУЧЕНИЯ")
    print("=" * 50)
    print(f"Количество эпох: {EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print("-" * 50)

    train_losses = []
    val_losses = []
    accuracies = []

    best_accuracy = 0
    best_f1 = 0

    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, accuracy, precision, recall, f1, _, _ = evaluate(
            model, test_loader, criterion, device
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        accuracies.append(accuracy)

        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_f1 = f1
            torch.save(model.state_dict(), 'best_bilstm_model.pth')
            print(f"  ✓ Модель сохранена! (best accuracy: {accuracy:.4f})")

    # Финальная оценка
    print("\n" + "=" * 50)
    print("ФИНАЛЬНАЯ ОЦЕНКА НА ТЕСТОВОЙ ВЫБОРКЕ")
    print("=" * 50)
    model.load_state_dict(torch.load('best_bilstm_model.pth'))
    _, accuracy, precision, recall, f1, predictions, true_labels = evaluate(
        model, test_loader, criterion, device
    )

    print(f"\nИтоговые результаты BiLSTM:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")

    # Дополнительная статистика
    cm = confusion_matrix(true_labels, predictions)
    tn, fp, fn, tp = cm.ravel()
    print(f"\nМатрица ошибок:")
    print(f"  True Negatives (Ham как Ham):   {tn}")
    print(f"  False Positives (Ham как Spam): {fp}")
    print(f"  False Negatives (Spam как Ham): {fn}")
    print(f"  True Positives (Spam как Spam): {tp}")

    # Визуализация
    print("\n" + "=" * 50)
    print("СОЗДАНИЕ ВИЗУАЛИЗАЦИЙ")
    print("=" * 50)
    plot_confusion_matrix(true_labels, predictions)
    plot_training_history(train_losses, val_losses, accuracies)

    print("\n" + "=" * 50)
    print("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО!")
    print("=" * 50)
    print(f"Лучшая модель сохранена в: best_bilstm_model.pth")
    print(f"Лучший результат: Accuracy={best_accuracy:.4f}, F1={best_f1:.4f}")

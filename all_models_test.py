import os
import torch
import torch.nn as nn
from data_loader import get_data_loaders
from models.random_forest import RandomForestSpamClassifier
from models.cnn_1d import CNN1DSpamClassifier
from models.bilstm import BiLSTMSpamClassifier
from models.cnn_lstm import CNNLSTMSpamClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = "data/extracted/body"


def train_pytorch_model(model, train_loader, test_loader, epochs=10, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {'train_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_acc = 100 * train_correct / train_total

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for texts, labels in test_loader:
                texts, labels = texts.to(device), labels.to(device)
                outputs = model(texts)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total

        history['train_loss'].append(train_loss / len(train_loader))
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {train_loss / len(train_loader):.4f}, '
              f'Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')

    return model, history


def evaluate_model(model, test_loader, is_sklearn=False, X_test=None, y_test=None):
    if is_sklearn:
        y_pred = model.predict(X_test)
        y_true = y_test
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.eval()
        y_pred = []
        y_true = []

        with torch.no_grad():
            for texts, labels in test_loader:
                texts = texts.to(device)
                outputs = model(texts)
                _, predicted = torch.max(outputs.data, 1)
                y_pred.extend(predicted.cpu().numpy())
                y_true.extend(labels.numpy())

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    cm = confusion_matrix(y_true, y_pred)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }


def main():
    print("=== Загрузка данных ===")
    train_loader, test_loader, vocab, (X_train, y_train, X_test, y_test) = get_data_loaders(
        DATA_DIR, batch_size=32, test_size=0.2, max_len=500
    )

    print(f"Размер словаря: {len(vocab)}")
    print(f"Обучающая выборка: {len(X_train)}")
    print(f"Тестовая выборка: {len(X_test)}")

    models_results = {}

    print("\n=== Выберите модель для обучения ===")
    print("1. Random Forest")
    print("2. 1D CNN")
    print("3. Bidirectional LSTM")
    print("4. CNN + LSTM")
    print("5. Все модели")

    choice = input("Ваш выбор (1-5): ")

    if choice in ['1', '5']:
        print("\n--- Обучение Random Forest ---")
        rf_model = RandomForestSpamClassifier(n_estimators=100)
        rf_model.train(X_train, y_train)
        rf_results = evaluate_model(rf_model, None, is_sklearn=True, X_test=X_test, y_test=y_test)
        models_results['Random Forest'] = rf_results
        print(f"Random Forest - Accuracy: {rf_results['accuracy']:.4f}, F1: {rf_results['f1']:.4f}")

    if choice in ['2', '5']:
        print("\n--- Обучение 1D CNN ---")
        cnn_model = CNN1DSpamClassifier(len(vocab))
        cnn_model, _ = train_pytorch_model(cnn_model, train_loader, test_loader, epochs=10)
        cnn_results = evaluate_model(cnn_model, test_loader)
        models_results['1D CNN'] = cnn_results
        print(f"1D CNN - Accuracy: {cnn_results['accuracy']:.4f}, F1: {cnn_results['f1']:.4f}")

    if choice in ['3', '5']:
        print("\n--- Обучение BiLSTM ---")
        lstm_model = BiLSTMSpamClassifier(len(vocab))
        lstm_model, _ = train_pytorch_model(lstm_model, train_loader, test_loader, epochs=10)
        lstm_results = evaluate_model(lstm_model, test_loader)
        models_results['BiLSTM'] = lstm_results
        print(f"BiLSTM - Accuracy: {lstm_results['accuracy']:.4f}, F1: {lstm_results['f1']:.4f}")

    if choice in ['4', '5']:
        print("\n--- Обучение CNN + LSTM ---")
        cnn_lstm_model = CNNLSTMSpamClassifier(len(vocab))
        cnn_lstm_model, _ = train_pytorch_model(cnn_lstm_model, train_loader, test_loader, epochs=10)
        cnn_lstm_results = evaluate_model(cnn_lstm_model, test_loader)
        models_results['CNN + LSTM'] = cnn_lstm_results
        print(f"CNN + LSTM - Accuracy: {cnn_lstm_results['accuracy']:.4f}, F1: {cnn_lstm_results['f1']:.4f}")

    # Вывод сравнительной таблицы
    print("\n=== Сравнение моделей ===")
    print(f"{'Модель':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 68)
    for name, results in models_results.items():
        print(f"{name:<20} {results['accuracy']:<12.4f} {results['precision']:<12.4f} "
              f"{results['recall']:<12.4f} {results['f1']:<12.4f}")


if __name__ == "__main__":
    main()
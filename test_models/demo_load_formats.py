"""
Демонстрация загрузки модели в трех разных форматах

Показывает как загружать:
1. State dict (.pth) - только веса
2. Полную модель (_full.pth) - с гиперпараметрами и метаданными
3. TorchScript (.pt) - production-ready, не требует исходного кода
"""

import torch
import sys
import os

# Добавляем корневую папку в путь
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cnn_lstm import CNNLSTMSpamClassifier
from data_loader import load_emails, EmailDataset
from sklearn.model_selection import train_test_split


def demo_state_dict_loading():
    """Формат 1: State Dict - только веса"""
    print("\n" + "="*70)
    print("ФОРМАТ 1: STATE DICT (.pth) - Только веса")
    print("="*70)

    print("\n📝 Что нужно:")
    print("  ✓ Файл с весами (best_cnn_lstm_model.pth)")
    print("  ✓ Исходный код класса модели")
    print("  ✓ Вручную указать все гиперпараметры")

    # Загружаем данные для построения словаря
    print("\n🔧 Загрузка данных для словаря...")
    texts, labels = load_emails("../data/extracted/body")
    X_train, _, y_train, _ = train_test_split(
        texts, labels, test_size=0.3, random_state=42, stratify=labels
    )

    # Строим словарь
    train_dataset = EmailDataset(X_train, y_train, max_len=1604)
    vocab_size = len(train_dataset.vocab)
    print(f"  ✓ Размер словаря: {vocab_size}")

    # ВАЖНО: Нужно вручную указать все гиперпараметры!
    print("\n🏗️ Создание архитектуры модели (вручную указываем параметры)...")
    model = CNNLSTMSpamClassifier(
        vocab_size=vocab_size,
        embedding_dim=128,      # Вручную!
        num_filters=256,        # Вручную!
        filter_sizes=[3, 4, 5], # Вручную!
        lstm_hidden=256,        # Вручную!
        dropout=0.5             # Вручную!
    )

    # Загружаем веса
    print("\n📦 Загрузка весов из файла...")
    if os.path.exists('best_cnn_lstm_model.pth'):
        model.load_state_dict(torch.load('best_cnn_lstm_model.pth', map_location='cpu'))
        model.eval()
        print("  ✅ Модель загружена успешно!")
        print(f"  ✓ Параметров в модели: {sum(p.numel() for p in model.parameters()):,}")
    else:
        print("  ❌ Файл best_cnn_lstm_model.pth не найден!")
        print("     Сначала обучите модель: python test_cnn_lstm.py")

    print("\n💡 Плюсы:  Минимальный размер файла")
    print("💡 Минусы: Нужно помнить все гиперпараметры")


def demo_full_model_loading():
    """Формат 2: Полная модель - с гиперпараметрами"""
    print("\n" + "="*70)
    print("ФОРМАТ 2: ПОЛНАЯ МОДЕЛЬ (_full.pth) - Веса + гиперпараметры")
    print("="*70)

    print("\n📝 Что нужно:")
    print("  ✓ Файл с полной моделью (best_cnn_lstm_full.pth)")
    print("  ✓ Исходный код класса модели")
    print("  ✗ Гиперпараметры загружаются автоматически!")

    # Загружаем данные для построения словаря
    print("\n🔧 Загрузка данных для словаря...")
    texts, labels = load_emails("../data/extracted/body")
    X_train, _, y_train, _ = train_test_split(
        texts, labels, test_size=0.3, random_state=42, stratify=labels
    )

    # Строим словарь
    train_dataset = EmailDataset(X_train, y_train, max_len=1604)
    vocab_size = len(train_dataset.vocab)
    print(f"  ✓ Размер словаря: {vocab_size}")

    # Загружаем checkpoint
    print("\n📦 Загрузка checkpoint...")
    if os.path.exists('best_cnn_lstm_full.pth'):
        checkpoint = torch.load('best_cnn_lstm_full.pth', map_location='cpu')

        # Автоматически извлекаем гиперпараметры!
        hyperparams = checkpoint['hyperparameters']
        print("\n✨ Гиперпараметры загружены автоматически:")
        for key, value in hyperparams.items():
            print(f"  - {key}: {value}")

        # Создаем модель с загруженными параметрами
        print("\n🏗️ Создание архитектуры (с автоматическими параметрами)...")
        model = CNNLSTMSpamClassifier(
            vocab_size=vocab_size,
            **hyperparams  # Автоматически распаковываем параметры!
        )

        # Загружаем веса
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print("  ✅ Модель загружена успешно!")

        # Показываем информацию об обучении
        if 'training_info' in checkpoint:
            info = checkpoint['training_info']
            print("\n📊 Информация об обучении:")
            print(f"  - Эпоха: {info['epoch']}")
            print(f"  - Accuracy: {info['accuracy']*100:.2f}%")
            print(f"  - F1-Score: {info['f1']*100:.2f}%")
            print(f"  - Precision: {info['precision']*100:.2f}%")
            print(f"  - Recall: {info['recall']*100:.2f}%")

        print(f"\n  ✓ Параметров в модели: {sum(p.numel() for p in model.parameters()):,}")
    else:
        print("  ❌ Файл best_cnn_lstm_full.pth не найден!")
        print("     Сначала обучите модель: python test_cnn_lstm.py")

    print("\n💡 Плюсы:  Автоматическая загрузка параметров + метаданные")
    print("💡 Минусы: Требует класс модели при загрузке")


# def demo_torchscript_loading():
#     """Формат 3: TorchScript - production-ready"""
#     print("\n" + "="*70)
#     print("ФОРМАТ 3: TORCHSCRIPT (.pt) - Production-ready")
#     print("="*70)
#
#     print("\n📝 Что нужно:")
#     print("  ✓ Файл TorchScript (best_cnn_lstm_traced.pt)")
#     print("  ✗ НЕ требует исходный код класса модели!")
#     print("  ✗ НЕ требует гиперпараметры!")
#
#     # Загрузка модели
#     print("\n📦 Загрузка TorchScript модели...")
#     if os.path.exists('best_cnn_lstm_traced.pt'):
#         # Загружаем модель БЕЗ определения класса!
#         model = torch.jit.load('best_cnn_lstm_traced.pt', map_location='cpu')
#         model.eval()
#         print("  ✅ Модель загружена успешно!")
#
#         # Тестируем предсказание
#         print("\n🧪 Тестовое предсказание...")
#         # Создаем случайный тензор (в реальности это закодированный текст)
#         test_input = torch.randint(0, 30000, (1, 1604))
#
#         with torch.no_grad():
#             output = model(test_input)
#             probabilities = torch.softmax(output, dim=1)
#             prediction = torch.argmax(probabilities, dim=1).item()
#
#         print(f"  ✓ Предсказание работает!")
#         print(f"  - Выход модели: {output.shape}")
#         print(f"  - Вероятности: Ham={probabilities[0][0]:.4f}, Spam={probabilities[0][1]:.4f}")
#         print(f"  - Класс: {'SPAM' if prediction == 1 else 'HAM'}")
#
#     else:
#         print("  ❌ Файл best_cnn_lstm_traced.pt не найден!")
#         print("     Сначала обучите модель: python test_cnn_lstm.py")
#
#     print("\n💡 Плюсы:  Не требует исходного кода, оптимизировано, production-ready")
#     print("💡 Минусы: Нельзя изменять архитектуру, некоторые динамические операции не поддерживаются")
#     print("💡 Идеально для: Интеграция в другие системы, развертывание в production")


def compare_file_sizes():
    """Сравнение размеров файлов"""
    print("\n" + "="*70)
    print("СРАВНЕНИЕ РАЗМЕРОВ ФАЙЛОВ")
    print("="*70 + "\n")

    files = [
        ('best_cnn_lstm_model.pth', 'State Dict'),
        ('best_cnn_lstm_full.pth', 'Полная модель'),
        ('best_cnn_lstm_traced.pt', 'TorchScript')
    ]

    print(f"{'Файл':<30} {'Формат':<20} {'Размер':<15}")
    print("-" * 70)

    for filename, format_name in files:
        if os.path.exists(filename):
            size_bytes = os.path.getsize(filename)
            size_mb = size_bytes / (1024 * 1024)
            print(f"{filename:<30} {format_name:<20} {size_mb:>8.2f} MB")
        else:
            print(f"{filename:<30} {format_name:<20} {'не найден':>15}")


def main():
    print("\n" + "="*70)
    print("🎓 ДЕМОНСТРАЦИЯ ТРЕХ ФОРМАТОВ СОХРАНЕНИЯ МОДЕЛЕЙ PYTORCH")
    print("="*70)

    # Демонстрация каждого формата
    demo_state_dict_loading()
    demo_full_model_loading()
    # demo_torchscript_loading()

    # Сравнение размеров
    compare_file_sizes()

    # Итоговые рекомендации
    print("\n" + "="*70)
    print("📌 ИТОГОВЫЕ РЕКОМЕНДАЦИИ")
    print("="*70)
    print("""
┌─────────────────────────────┬───────────────────────────────────────┐
│ Сценарий                    │ Рекомендуемый формат                  │
├─────────────────────────────┼───────────────────────────────────────┤
│ Эксперименты, дообучение    │ State Dict (.pth)                     │
│ Разработка, прототипирование│ Полная модель (_full.pth)             │
│                             │                                       │
└─────────────────────────────┴───────────────────────────────────────┘


""")

    print("="*70 + "\n")


if __name__ == "__main__":
    main()


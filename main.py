"""
Этот проект - создание ИИ, которое на основе текста письма,
должен класифицировать его в не спам (ham) или в спам (spam).

В папке data/extracted/body:
    Ham files: 2801
        - Ham easy files: 2551
        - Ham hard files: 250
    Spam files: 1397

Лучшие результаты:
    CNN+LSTM: Accuracy=98.49%, F1=97.72%
"""

import torch
import numpy as np
from data_loader import load_emails, EmailDataset
from models.cnn_1d import CNN1DSpamClassifier
from models.bilstm import BiLSTMSpamClassifier
from models.cnn_lstm import CNNLSTMSpamClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

DATA_DIR = "data/extracted/body"
MODEL_PATH = "test_models/best_cnn_lstm_model.pth"


class SpamClassifierPredictor:
    """Класс для загрузки обученной модели и предсказания на новых текстах"""

    def __init__(self, model_path, model_type='cnn_lstm', max_len=1604):
        self.model_path = model_path
        self.model_type = model_type
        self.max_len = max_len  # ВАЖНО: должно совпадать с обучением
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.vocab = None
        self.vocab_size = None

    def load_model(self, texts_for_vocab):
        """Загрузка модели и построение словаря"""
        print(f"\n{'='*60}")
        print(f"ЗАГРУЗКА МОДЕЛИ: {self.model_type.upper()}")
        print(f"{'='*60}")

        # Строим словарь на основе обучающих данных
        print(f"Построение словаря из {len(texts_for_vocab)} текстов...")
        temp_dataset = EmailDataset(texts_for_vocab, [0]*len(texts_for_vocab), max_len=self.max_len)
        self.vocab = temp_dataset.vocab
        self.vocab_size = len(self.vocab)
        print(f"✓ Размер словаря: {self.vocab_size} слов")

        # Инициализация модели с ТОЧНЫМИ параметрами из обучения
        if self.model_type == 'cnn_lstm':
            self.model = CNNLSTMSpamClassifier(
                vocab_size=self.vocab_size,
                embedding_dim=128,      # Совпадает с test_cnn_lstm.py
                num_filters=256,        # Совпадает с test_cnn_lstm.py
                filter_sizes=[3, 4, 5],
                lstm_hidden=256,        # Совпадает с test_cnn_lstm.py
                dropout=0.5             # Совпадает с test_cnn_lstm.py
            )
        elif self.model_type == 'bilstm':
            self.model = BiLSTMSpamClassifier(
                vocab_size=self.vocab_size,
                embedding_dim=100,
                hidden_dim=128,
                num_layers=1,
                dropout=0.3
            )
        elif self.model_type == 'cnn_1d':
            self.model = CNN1DSpamClassifier(
                vocab_size=self.vocab_size,
                embedding_dim=100,
                num_filters=64,
                filter_sizes=[3, 4, 5],
                dropout=0.3
            )
        else:
            raise ValueError(f"Неизвестный тип модели: {self.model_type}")

        # Загрузка весов
        print(f"Загрузка весов из {self.model_path}...")
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        print(f"✓ Модель загружена на {self.device}")
        print(f"{'='*60}\n")

    def predict(self, text):
        """Предсказание для одного текста"""
        if self.model is None:
            raise ValueError("Модель не загружена! Вызовите load_model() сначала.")

        # Кодирование текста
        dataset = EmailDataset([text], [0], vocab=self.vocab, max_len=self.max_len)
        encoded_text = torch.tensor(dataset.encoded_texts[0]).unsqueeze(0).to(self.device)

        # Предсказание
        with torch.no_grad():
            output = self.model(encoded_text)
            probabilities = torch.softmax(output, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()

        ham_prob = probabilities[0][0].item()
        spam_prob = probabilities[0][1].item()

        return {
            'prediction': 'SPAM' if prediction == 1 else 'HAM',
            'prediction_label': prediction,
            'ham_probability': ham_prob,
            'spam_probability': spam_prob,
            'confidence': max(ham_prob, spam_prob)
        }

    def predict_batch(self, texts):
        """Предсказание для нескольких текстов"""
        results = []
        for text in texts:
            results.append(self.predict(text))
        return results


def test_on_random_samples(predictor, X_test, y_test, n_samples=5):
    """Тестирование на случайных образцах"""
    print(f"\n{'='*60}")
    print(f"ТЕСТИРОВАНИЕ НА {n_samples} СЛУЧАЙНЫХ ОБРАЗЦАХ")
    print(f"{'='*60}\n")

    indices = np.random.choice(len(X_test), min(n_samples, len(X_test)), replace=False)

    correct = 0
    for i, idx in enumerate(indices, 1):
        text = X_test[idx]
        true_label = 'SPAM' if y_test[idx] == 1 else 'HAM'

        result = predictor.predict(text)
        pred_label = result['prediction']
        is_correct = (pred_label == true_label)
        correct += int(is_correct)

        # Показываем первые 1000 символов
        text_preview = text[:1000] + "..." if len(text) > 1000 else text

        print(f"--- Образец {i} ---")
        print(f"Текст (первые 300 символов):")
        print(f"  {text_preview}")
        print(f"\nИстинная метка:    {true_label}")
        print(f"Предсказание:      {pred_label}")
        print(f"Уверенность:")
        print(f"  Ham:  {result['ham_probability']*100:>6.2f}%")
        print(f"  Spam: {result['spam_probability']*100:>6.2f}%")

        status_icon = "✓" if is_correct else "✗"
        status_text = "ВЕРНО" if is_correct else "ОШИБКА"
        print(f"\nРезультат: {status_icon} {status_text}")
        print(f"{'-'*60}\n")

    accuracy = correct / len(indices) * 100
    print(f"Точность на выборке: {correct}/{len(indices)} ({accuracy:.1f}%)\n")


def test_on_custom_texts(predictor):
    """Тестирование на пользовательских текстах"""
    print(f"\n{'='*60}")
    print("ТЕСТИРОВАНИЕ НА ПОЛЬЗОВАТЕЛЬСКИХ ТЕКСТАХ")
    print(f"{'='*60}\n")

    custom_texts = [
        {
            'text': "Dear friend, you have won $1,000,000! Click here to claim your prize now!",
            'description': "Типичный спам про выигрыш"
        },
        {
            'text': "Hi, this is a reminder about our meeting tomorrow at 3pm. Please bring the documents.",
            'description': "Обычное деловое письмо"
        },
        {
            'text': "URGENT: Your account will be suspended unless you verify your identity immediately!",
            'description': "Фишинговое письмо"
        },
        {
            'text': "Thanks for your email. I will review the proposal and get back to you next week.",
            'description': "Обычный ответ на письмо"
        },
        {
            'text': "Congratulations! You are selected for a special offer. Limited time only. Act now!",
            'description': "Рекламный спам"
        },
        {
            'text': """
                Quality, value, style, service, selection, convenience
                Economy, savings, performance, experience, hospitality
                Low rates, friendly service, name brands, easy terms
                Affordable prices, money-back guarantee
                
                Free installation, free admission, free appraisal, free alterations
                Free delivery, free estimates, free home trial, and free parking
                
                No cash? No problem! No kidding! No fuss, no muss
                No risk, no obligation, no red tape, no down payment
                No entry fee, no hidden charges, no purchase necessary
                No one will call on you, no payments or interest till September
                
                Limited time only, though, so act now, order today, send no money
                Offer good while supplies last, two to a customer, each item sold separately
                Batteries not included, mileage may vary, all sales are final
                Allow six weeks for delivery, some items not available
                Some assembly required, some restrictions may apply
                
                So come on in for a free demonstration and a free consultation
                With our friendly, professional staff. Our experienced and
                Knowledgeable sales representatives will help you make a
                Selection that's just right for you and just right for your budget
                
                And say, don't forget to pick up your free gift: a classic deluxe
                Custom designer luxury prestige high-quality premium select
                Gourmet pocket pencil sharpener. Yours for the asking
                No purchase necessary. It's our way of saying thank you
                And if you act now, we'll include an extra added free complimentary
                Bonus gift at no cost to you: a classic deluxe custom designer
                Luxury prestige high-quality premium select gourmet combination
                Key ring, magnifying glass, and garden hose, in a genuine
                Imitation leather-style carrying case with authentic vinyl trim
                Yours for the asking, no purchase necessary. It's our way of
                Saying thank you
                
                Actually, it's our way of saying 'Bend over just a little farther
                So we can stick this big advertising dick up your ass a little bit
                Deeper, a little bit deeper, a little bit DEEPER, you miserable
                No-good dumbass fucking consumer!
                """,
            'description': "Advertising Lullabye"
        }

    ]

    for i, item in enumerate(custom_texts, 1):
        text = item['text']
        description = item['description']

        result = predictor.predict(text)
        pred_label = result['prediction']

        print(f"--- Пример {i}: {description} ---")
        print(f"Текст: \"{text}\"")
        print(f"\nПредсказание: {pred_label}")
        print(f"Уверенность:")
        print(f"  Ham:  {result['ham_probability']*100:>6.2f}%")
        print(f"  Spam: {result['spam_probability']*100:>6.2f}%")
        print(f"{'-'*60}\n")


def evaluate_on_test_set(predictor, X_test, y_test):
    """Оценка на всей тестовой выборке"""
    print(f"\n{'='*60}")
    print("ОЦЕНКА НА ПОЛНОЙ ТЕСТОВОЙ ВЫБОРКЕ")
    print(f"{'='*60}\n")

    print(f"Обработка {len(X_test)} писем...")
    predictions = []

    for i, text in enumerate(X_test, 1):
        if i % 100 == 0:
            print(f"  Обработано: {i}/{len(X_test)}")
        result = predictor.predict(text)
        predictions.append(result['prediction_label'])

    predictions = np.array(predictions)
    y_test_array = np.array(y_test)

    # Метрики
    accuracy = accuracy_score(y_test_array, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test_array, predictions, average='binary'
    )
    cm = confusion_matrix(y_test_array, predictions)

    print(f"\n{'Метрика':<20} {'Значение':<10}")
    print(f"{'-'*30}")
    print(f"{'Accuracy':<20} {accuracy*100:>6.2f}%")
    print(f"{'Precision':<20} {precision*100:>6.2f}%")
    print(f"{'Recall':<20} {recall*100:>6.2f}%")
    print(f"{'F1-Score':<20} {f1*100:>6.2f}%")

    print(f"\nМатрица ошибок:")
    tn, fp, fn, tp = cm.ravel()
    print(f"  True Negatives (Ham как Ham):   {tn}")
    print(f"  False Positives (Ham как Spam): {fp}")
    print(f"  False Negatives (Spam как Ham): {fn}")
    print(f"  True Positives (Spam как Spam): {tp}")
    print(f"{'='*60}\n")


def main():
    """Главная функция"""
    print("\n" + "="*60)
    print("АНТИСПАМ AI - СИСТЕМА КЛАССИФИКАЦИИ ПИСЕМ")
    print("="*60)
    print("Лучшая модель: CNN+LSTM (Accuracy: 98.49%, F1: 97.72%)")
    print("="*60 + "\n")

    # Загрузка данных
    print("[1/4] Загрузка данных...")
    texts, labels = load_emails(DATA_DIR)
    print(f"✓ Загружено {len(texts)} писем")

    # Разделение на train/test (для построения словаря)
    print("\n[2/4] Разделение на train/test выборки...")
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.3, random_state=42, stratify=labels
    )
    print(f"✓ Train: {len(X_train)}, Test: {len(X_test)}")

    # Загрузка модели
    print("\n[3/4] Загрузка обученной модели...")
    predictor = SpamClassifierPredictor(
        model_path=MODEL_PATH,
        model_type='cnn_lstm',
        max_len=1604  # ВАЖНО: совпадает с test_cnn_lstm.py
    )
    predictor.load_model(X_train)

    # Тестирование
    print("[4/4] Тестирование модели...")

    # 1. Тестирование на случайных образцах из тестовой выборки
    test_on_random_samples(predictor, X_test, y_test, n_samples=5)

    # 2. Тестирование на пользовательских текстах
    test_on_custom_texts(predictor)

    # 3. Полная оценка на тестовой выборке (опционально)
    print("\nХотите выполнить полную оценку на всей тестовой выборке?")
    print("(Это займет некоторое время...)")
    response = input("Введите 'yes' для продолжения или нажмите Enter для пропуска: ")

    if response.lower() in ['yes', 'y', 'да']:
        evaluate_on_test_set(predictor, X_test, y_test)

    print("\n" + "="*60)
    print("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО!")
    print("="*60)
    print("\nВы можете использовать predictor.predict(text) для классификации")
    print("любых новых текстов писем.")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()


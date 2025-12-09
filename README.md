# AntiSpam AI - Классификация спама по содержимому писем

## Описание 

Модель нейронной сети для классификации email-сообщений на **HAM** (обычные письма) и **SPAM** на основе текстового содержимого письма.

**Лучший результат:** CNN+LSTM архитектура с точностью **98.65%** и F1-Score **97.98%**

---
## Все результаты
### RANDOM FOREST
![random_forest_confusion_matrix.png](test_models/random_forest_confusion_matrix.png)
### CNN1D
![cnn1d_confusion_matrix.png](test_models/cnn1d_confusion_matrix.png)
### Bidirectional LSTM
![bilstm_confusion_matrix.png](test_models/bilstm_confusion_matrix.png)
### CNN+LSTM
![cnn_lstm_confusion_matrix.png](test_models/cnn_lstm_confusion_matrix.png)
## Результаты модели

| Метрика | Значение |
|---------|----------|
| **Accuracy** | 98.49% |
| **Precision** | 98.08% |
| **Recall** | 97.37% |
| **F1-Score** | 97.72% |

### Матрица ошибок:
- True Negatives (Ham -> Ham): **833**
- False Positives (Ham -> Spam): **8**
- False Negatives (Spam -> Ham): **11**
- True Positives (Spam -> Spam): **408**

---


## Крч, как использовать 

### 1. Загрузка модели и предсказание

```python
from main import SpamClassifierPredictor
from data_loader import load_emails
from sklearn.model_selection import train_test_split

# Загрузка данных для построения словаря
texts, labels = load_emails("data/extracted/body")
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.3, random_state=42, stratify=labels
)

# Инициализация предиктора
predictor = SpamClassifierPredictor(
    model_path="test_models/best_cnn_lstm_model.pth",
    model_type='cnn_lstm',
    max_len=1604
)

# Загрузка модели
predictor.load_model(X_train)

# Предсказание
text = "Congratulations! You won $1,000,000! Click here now!"
result = predictor.predict(text)

print(f"Prediction: {result['prediction']}")
print(f"Ham probability: {result['ham_probability']:.2%}")
print(f"Spam probability: {result['spam_probability']:.2%}")
```

### 2. Запуск демонстрации

```bash
python main.py
```

Этот скрипт:
- Загружает обученную модель
- Тестирует на случайных образцах
- Показывает примеры классификации
- Опционально: полная оценка на тестовой выборке

---

## Если мы в дальнейшем хотим объединять с другими моделями

### Класс `SpamClassifierPredictor`

#### Инициализация
```python
predictor = SpamClassifierPredictor(
    model_path="test_models/best_cnn_lstm_model.pth",
    model_type='cnn_lstm',
    max_len=1604
)
predictor.load_model(training_texts)
```

#### Методы

**`predict(text: str) -> dict`**

Классифицирует один какой-то конкретный текст.

**Вход:**
- `text` (str): текст письма

**Выход:**
```python
{
    'prediction': 'SPAM' | 'HAM',
    'prediction_label': 0 | 1,  # 0=HAM, 1=SPAM
    'ham_probability': float,   # 0.0-1.0
    'spam_probability': float,  # 0.0-1.0
    'confidence': float         # max(ham_prob, spam_prob)
}
```

**Пример:**
```python
result = predictor.predict("URGENT: Click here to claim your prize!")
# {
#     'prediction': 'SPAM',
#     'prediction_label': 1,
#     'ham_probability': 0.02,
#     'spam_probability': 0.98,
#     'confidence': 0.98
# }
```

---

**`predict_batch(texts: list) -> list`**

Классифицирует несколько текстов.

**Вход:**
- `texts` (list[str]): список текстов писем

**Выход:**
- `list[dict]`: список результатов для каждого текста

**Пример:**
```python
texts = [
    "Meeting at 3pm tomorrow",
    "You won a lottery! Click now!"
]
results = predictor.predict_batch(texts)
# [
#     {'prediction': 'HAM', 'spam_probability': 0.05, ...},
#     {'prediction': 'SPAM', 'spam_probability': 0.97, ...}
# ]
```

---

## способы объеднэинения с другими нейронками 

### Вариант 1: Ансамбль (голосование)

```python
# Ваша модель (текст)
text_result = text_predictor.predict(email_body)
text_spam_prob = text_result['spam_probability']

# Модель коллег (метаданные)
metadata_spam_prob = metadata_predictor.predict(email_metadata)

# Взвешенное голосование
final_spam_prob = 0.6 * text_spam_prob + 0.4 * metadata_spam_prob
final_prediction = 'SPAM' if final_spam_prob > 0.5 else 'HAM'
```

### Вариант 2: Последовательное применение

```python
# Сначала проверка по метаданным (быстро)
if metadata_spam_prob > 0.9:
    return 'SPAM'
elif metadata_spam_prob < 0.1:
    return 'HAM'
else:
    # Неоднозначный случай - проверяем текст
    text_result = text_predictor.predict(email_body)
    return text_result['prediction']
```

### Вариант 3: Конкатенация признаков

```python
# Объединение вероятностей как признаков для финальной модели
features = [
    text_result['spam_probability'],
    metadata_spam_prob,
    text_result['confidence'],
    # другие признаки...
]

# Финальный классификатор (например, Logistic Regression)
final_prediction = final_classifier.predict([features])
```

---


## Важные уточнения

1. **Словарь (vocab)**: Модель требует построения словаря на тренировочных данных. При интеграции используйте те же данные для построения словаря, что и при обучении.

2. **Длина последовательности**: Модель обучена на последовательностях длиной **1604 токена**. Более длинные тексты обрезаются, короткие - дополняются padding.

3. **Гиперпараметры**: При загрузке модели **критически важно** использовать те же гиперпараметры, что и при обучении. См. `MODEL_PARAMS.md`.
---

## коротко о латасете
**Источник**: Ham/Spam Email Dataset

**Статистика:**
- Ham письма: 2801 (Easy: 2551, Hard: 250)
- Spam письма: 1397
- **Всего**: 4198 писем

**Разделение:**
- Train: 70% (2938 писем)
- Test: 30% (1260 писем)

**Характеристики текста:**
- Средняя длина: 386 слов
- Медиана: 166 слов
- 95-й перцентиль: 1604 слов
- Максимум: 14954 слов

---




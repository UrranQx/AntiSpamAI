# Быстрая интеграция AntiSpam AI

## Для коллег: 3 шага до работающей модели

### Шаг 1: Установка зависимостей
```bash
# создайте и активируете виртуальное окружение python 3.12 + (рекомендуется)
pip install -r requirements.txt
```

### Шаг 2: Минимальный код для использования

```python
from main import SpamClassifierPredictor
from data_loader import load_emails
from sklearn.model_selection import train_test_split

# Загрузка данных для словаря (один раз)
texts, labels = load_emails("data/extracted/body")
X_train, _, _, _ = train_test_split(texts, labels, test_size=0.3, random_state=42)

# Инициализация модели (один раз)
predictor = SpamClassifierPredictor(
    model_path="test_models/best_cnn_lstm_model.pth",
    model_type='cnn_lstm',
    max_len=1604
)
predictor.load_model(X_train)

# Использование (многократно)
def classify_email(email_text):
    result = predictor.predict(email_text)
    return {
        'is_spam': result['prediction'] == 'SPAM',
        'spam_probability': result['spam_probability'],
        'confidence': result['confidence']
    }

# Пример
email = "Congratulations! You won $1,000,000!"
result = classify_email(email)
print(result)  # {'is_spam': True, 'spam_probability': 0.98, 'confidence': 0.98}
```

## Проверка работоспособности

```bash
python main.py
```

Должно показать тестирование на примерах и вывести метрики.


# 🚀 Быстрый старт: Использование трех форматов моделей

## 📝 Что у вас теперь есть

После обучения каждой модели создаются **2 файла**:

```
test_models/
├── best_cnn_lstm_model.pth      ← State Dict (только веса)
├── best_cnn_lstm_full.pth       ← Полная модель (веса + параметры)

```

---

## ⚡ Быстрые команды

### Обучение модели с тремя форматами:
```bash
cd test_models
python test_cnn_lstm.py
```

### Демонстрация всех форматов:
```bash
cd test_models
python demo_load_formats.py
```

### Запуск main.py с автоматической загрузкой:
```bash
python main.py
```

---

## 💻 Примеры кода

### 1. Загрузка для разработки (Full Model)

```python
import torch
from models.cnn_lstm import CNNLSTMSpamClassifier

# Загрузка checkpoint
checkpoint = torch.load('best_cnn_lstm_full.pth')

# Автоматическое извлечение параметров
model = CNNLSTMSpamClassifier(
    vocab_size=30000,
    **checkpoint['hyperparameters']  # Автоматически!
)

# Загрузка весов
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Просмотр метаданных
print(f"Эпоха: {checkpoint['training_info']['epoch']}")
print(f"Accuracy: {checkpoint['training_info']['accuracy']:.4f}")
```


### 2. Использование в main.py

```python
from main import SpamClassifierPredictor

# Инициализация
predictor = SpamClassifierPredictor(
    model_path="test_models/best_cnn_lstm_model.pth",
    model_type='cnn_lstm',
    max_len=1604
)

# Автоматическая загрузка параметров из _full.pth
predictor.load_model(X_train, use_full_model=True)

# Предсказание
result = predictor.predict("Your email text here")
print(f"Prediction: {result['prediction']}")
print(f"Spam probability: {result['spam_probability']:.2%}")
```

---

## 🎯 Какой формат использовать?

### Вы разработчик модели:
```python
# Эксперименты
torch.load('best_cnn_lstm_model.pth')  # State Dict

# Разработка и настройка
torch.load('best_cnn_lstm_full.pth')   # Full Model


```


---

## 📚 Документация

- **[README.md](README.md)** - Общее описание проекта
- **[MODEL_FORMATS.md](MODEL_FORMATS.md)** - Детальное описание форматов
- **[INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md)** - Руководство для коллег
- **[demo_load_formats.py](test_models/demo_load_formats.py)** - Примеры загрузки

---

## 🔍 Проверка файлов

### После обучения проверьте:
```bash
cd test_models
dir best_cnn_lstm*
```

Должны быть 2 файла:
- `best_cnn_lstm_model.pth` (~5-10 МБ)
- `best_cnn_lstm_full.pth` (~10-15 МБ)


---

Подробнее: **[INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md)**

---

## ❓ FAQ


**Q: Можно ли конвертировать между форматами?**  
A: Да, см. `MODEL_FORMATS.md` → раздел "Конвертация".

**Q: Какой формат занимает меньше места?**  
A: State Dict (`.pth`), но требует исходный код модели.

**Q: Где хранятся гиперпараметры?**  
A: В `_full.pth` файле, в ключе `hyperparameters`.

---

## 🎉 Готово!

Теперь вы можете:
- ✅ Обучать модели с тремя форматами
- ✅ Загружать модели автоматически
- ✅ Интегрировать модели в другие системы
- ✅ Легко делиться моделями с коллегами

**Удачи!** 🚀


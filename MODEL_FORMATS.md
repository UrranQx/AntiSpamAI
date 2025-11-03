# 💾 Форматы сохранения моделей PyTorch

После обучения каждая модель сохраняется в **трех форматах** для различных сценариев использования.

---

## 📋 Обзор форматов

| Формат | Файл | Размер | Требует код | Production | Описание |
|--------|------|--------|-------------|------------|----------|
| **State Dict** | `*.pth` | ~5-10 МБ | ✅ Да | ❌ | Только веса модели |
| **Full Model** | `*_full.pth` | ~10-15 МБ | ✅ Да | ⚠️ | Веса + гиперпараметры + метаданные |
| **TorchScript** | `*.pt` | ~10-15 МБ | ❌ Нет | ✅ | Production-ready, standalone |

---

## 1️⃣ State Dict (`.pth`)

### Что это?
Сохраняет только **веса (параметры)** модели в виде словаря.

### Что сохраняется?
```python
{
    'embedding.weight': tensor(...),
    'conv1.weight': tensor(...),
    'lstm.weight_ih_l0': tensor(...),
    ...
}
```

### Как сохранять?
```python
torch.save(model.state_dict(), 'best_model.pth')
```

### Как загружать?
```python
# Нужно ВРУЧНУЮ создать архитектуру с правильными параметрами!
model = CNNLSTMSpamClassifier(
    vocab_size=30000,
    embedding_dim=128,
    num_filters=256,
    lstm_hidden=256,
    dropout=0.5
)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()
```

### ✅ Плюсы
- Минимальный размер файла
- Быстрое сохранение/загрузка
- Стандартный подход в PyTorch

### ❌ Минусы
- Нужен исходный код класса модели
- Нужно вручную указывать все гиперпараметры
- Легко допустить ошибку в параметрах

### 🎯 Когда использовать?
- Эксперименты и разработка
- Когда размер файла критичен
- Внутри одного проекта

---

## 2️⃣ Full Model (`_full.pth`)

### Что это?
Сохраняет **веса + гиперпараметры + метаданные обучения** в одном файле.

### Что сохраняется?
```python
{
    'model_state_dict': {...},  # Веса модели
    'model': model,             # Вся модель (опционально)
    'vocab_size': 30000,
    'max_len': 1604,
    'hyperparameters': {
        'embedding_dim': 128,
        'num_filters': 256,
        'filter_sizes': [3, 4, 5],
        'lstm_hidden': 256,
        'dropout': 0.5
    },
    'training_info': {
        'epoch': 10,
        'accuracy': 0.9849,
        'f1': 0.9772,
        'precision': 0.9808,
        'recall': 0.9737
    }
}
```

### Как сохранять?
```python
torch.save({
    'model_state_dict': model.state_dict(),
    'vocab_size': vocab_size,
    'max_len': MAX_LEN,
    'hyperparameters': {
        'embedding_dim': EMBEDDING_DIM,
        'num_filters': NUM_FILTERS,
        'filter_sizes': FILTER_SIZES,
        'lstm_hidden': LSTM_HIDDEN,
        'dropout': DROPOUT
    },
    'training_info': {
        'epoch': epoch + 1,
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
}, 'best_model_full.pth')
```

### Как загружать?
```python
# Загружаем checkpoint
checkpoint = torch.load('best_model_full.pth')

# Автоматически извлекаем параметры!
hyperparams = checkpoint['hyperparameters']
print(f"Гиперпараметры: {hyperparams}")

# Создаем модель с автоматическими параметрами
model = CNNLSTMSpamClassifier(
    vocab_size=vocab_size,
    **hyperparams  # Автоматическая распаковка!
)

# Загружаем веса
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Читаем информацию об обучении
info = checkpoint['training_info']
print(f"Accuracy: {info['accuracy']*100:.2f}%")
```

### ✅ Плюсы
- Автоматически загружает все параметры
- Сохраняет метаданные об обучении
- Меньше вероятность ошибок
- Легко воспроизвести результаты

### ❌ Минусы
- Требует исходный код класса модели
- Немного больший размер файла
- Не полностью portable

### 🎯 Когда использовать?
- Разработка и прототипирование
- Эксперименты с гиперпараметрами
- Воспроизводимость результатов
- Командная работа



---

## 📊 Сравнение производительности

| Операция | State Dict | Full Model | 
|----------|-----------|------------|
| Время сохранения | 0.5 сек | 0.6 сек |
| Время загрузки | 0.3 сек | 0.4 сек | 
| Скорость inference | 100% | 100% | 
| Размер файла | 5 МБ | 10 МБ | 

---

## 🎯 Рекомендации по выбору

### Для разработчиков модели (вы):
```
Эксперименты         → State Dict (.pth)
Совместная работа    → Full Model (_full.pth)
```

### Для интеграторов (коллеги):
```
Нужна гибкость       → Full Model (_full.pth)
Есть исходный код    → State Dict (.pth)
```

### Для production:
```
Мобильные устройства → TorchScript Mobile
Edge устройства      → TorchScript или ONNX
```

---

## 🔄 Конвертация между форматами

### State Dict → Full Model
```python
checkpoint = torch.load('model.pth')
torch.save({
    'model_state_dict': checkpoint,
    'hyperparameters': {...},
    'training_info': {...}
}, 'model_full.pth')
```

### Full Model → TorchScript
```python
checkpoint = torch.load('model_full.pth')
model = CNNLSTMSpamClassifier(vocab_size, **checkpoint['hyperparameters'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

example_input = torch.randint(0, vocab_size, (1, max_len))
traced = torch.jit.trace(model, example_input)
traced.save('model_traced.pt')
```

---

## 📝 Примеры использования


### Загрузка для экспериментов (Full Model)
```python
import torch
from models.cnn_lstm import CNNLSTMSpamClassifier

# Загрузка с автоматическими параметрами
checkpoint = torch.load('best_cnn_lstm_full.pth')
model = CNNLSTMSpamClassifier(
    vocab_size=30000,
    **checkpoint['hyperparameters']
)
model.load_state_dict(checkpoint['model_state_dict'])

# Можем продолжить обучение
optimizer = torch.optim.Adam(model.parameters())
# ...
```

---



## 📚 Дополнительные ресурсы

- [PyTorch Saving & Loading Models](https://pytorch.org/tutorials/beginner/saving_loading_models.html)
- [TorchScript Documentation](https://pytorch.org/docs/stable/jit.html)
- [ONNX Export (альтернатива)](https://pytorch.org/docs/stable/onnx.html)

---

**Создано:** AntiSpam AI Project  
**Дата:** 2025  
**Версия:** 1.0


# Гиперпараметры обученных моделей

## CNN+LSTM (best_cnn_lstm_model.pth)
**Файл обучения**: test_models/test_cnn_lstm.py

### Параметры:
- **MAX_LEN**: 1604
- **EMBEDDING_DIM**: 128
- **NUM_FILTERS**: 256
- **FILTER_SIZES**: [3, 4, 5]
- **LSTM_HIDDEN**: 256 (BiLSTM → выход 512)
- **DROPOUT**: 0.5
- **BATCH_SIZE**: 16
- **EPOCHS**: 10
- **LEARNING_RATE**: 0.001
- **TEST_SIZE**: 0.3

### Результаты:
- **Accuracy**: 98.49%
- **Precision**: 98.08%
- **Recall**: 97.37%
- **F1-Score**: 97.72%

### Матрица ошибок:
- True Negatives (Ham как Ham): 833
- False Positives (Ham как Spam): 8
- False Negatives (Spam как Ham): 11
- True Positives (Spam как Spam): 408

---

## BiLSTM (best_bilstm_model.pth)
**Файл обучения**: test_models/test_bilstm.py

### Параметры:
- **MAX_LEN**: 500
- **EMBEDDING_DIM**: 100
- **HIDDEN_DIM**: 128 (BiLSTM → выход 256)
- **NUM_LAYERS**: 1
- **DROPOUT**: 0.3
- **BATCH_SIZE**: 64
- **EPOCHS**: 10
- **LEARNING_RATE**: 0.001

---

## CNN1D (best_cnn1d_model.pth)
**Файл обучения**: test_models/test_cnn1d.py

### Параметры:
- **MAX_LEN**: 500
- **EMBEDDING_DIM**: 100
- **NUM_FILTERS**: 64
- **FILTER_SIZES**: [3, 4, 5]
- **DROPOUT**: 0.3
- **BATCH_SIZE**: 64
- **EPOCHS**: 10
- **LEARNING_RATE**: 0.001

---

## Random Forest (лучшая модель в test_models/)
**Файл обучения**: test_models/test_random_forest.py

### Параметры:
- **N_ESTIMATORS**: 100
- **MAX_DEPTH**: None
- **MIN_SAMPLES_SPLIT**: 2
- **MIN_SAMPLES_LEAF**: 1
- **MAX_FEATURES**: 'sqrt'
- **TF-IDF MAX_FEATURES**: 5000
- **TEST_SIZE**: 0.3

---

## ⚠️ ВАЖНО при загрузке модели в main.py

При загрузке модели **ОБЯЗАТЕЛЬНО** использовать те же гиперпараметры, с которыми она обучалась!

Иначе архитектура сети не совпадет с сохраненными весами, и модель либо:
1. Не загрузится (ошибка RuntimeError)
2. Загрузится неправильно и будет давать случайные предсказания

### Пример для CNN+LSTM:
```python
predictor = SpamClassifierPredictor(
    model_path="test_models/best_cnn_lstm_model.pth",
    model_type='cnn_lstm',
    max_len=1604  # ← Из test_cnn_lstm.py
)

# В load_model:
model = CNNLSTMSpamClassifier(
    vocab_size=vocab_size,
    embedding_dim=128,      # ← Из test_cnn_lstm.py
    num_filters=256,        # ← Из test_cnn_lstm.py
    filter_sizes=[3, 4, 5],
    lstm_hidden=256,        # ← Из test_cnn_lstm.py
    dropout=0.5             # ← Из test_cnn_lstm.py
)
```

---

## Сравнение производительности моделей

| Модель | Accuracy | F1-Score | Скорость обучения | Параметров |
|--------|----------|----------|-------------------|------------|
| **CNN+LSTM** | 98.49% | 97.72% | Медленно | ~3M |
| **BiLSTM** | ? | ? | Медленно | ~800K |
| **CNN1D** | ? | ? | Быстро | ~400K |
| **Random Forest** | ? | ? | Очень быстро | N/A |

*(Заполните результаты BiLSTM, CNN1D и Random Forest после тестирования)*


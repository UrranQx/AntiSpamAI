# data_loader.py
import os
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import re
from sklearn.model_selection import train_test_split


class EmailDataset(Dataset):
    def __init__(self, texts, labels, vocab=None, max_len=500):
        self.texts = texts
        self.labels = labels
        self.max_len = max_len

        if vocab is None:
            self.vocab = self.build_vocab(texts)
        else:
            self.vocab = vocab

        self.encoded_texts = [self.encode_text(text) for text in texts]

    def build_vocab(self, texts, min_freq=1):
        # Подсчитывает, сколько раз каждое слово встречается во всех текстах
        # Оставляет только слова, которые встречаются >= min_freq раз (фильтрация редких слов)
        # Создает словарь {слово: индекс}

        # Можно поставить min_freq=2
        # Редкие слова (встречаются 1 раз) обычно не важны и увеличивают размер словаря.
        counter = Counter()
        for text in texts:
            tokens = self.tokenize(text)
            counter.update(tokens)

        vocab = {'<PAD>': 0, '<UNK>': 1}
        # <PAD> (padding, индекс 0) — заполнитель для коротких текстов.
        # Все тексты должны быть одной длины (например, 500 слов).
        # Если текст короче, добавляем <PAD>.

        # <UNK> (unknown, индекс 1) — заменяет редкие/неизвестные слова,
        # которых нет в словаре.
        for word, freq in counter.items():
            if freq >= min_freq:
                vocab[word] = len(vocab)
        return vocab

    def tokenize(self, text):
        # Переводит текст в нижний регистр: "Hello World" → "hello world"
        # Удаляет знаки препинания: "hello, world!" → "hello world"
        # Разбивает на слова: "hello world" → ["hello", "world"]

        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        return text.split()

    def encode_text(self, text):

        # Токенизирует текст -> получает список слов
        # Заменяет каждое слово на его индекс из vocab (или <UNK>, если слова нет)
        # Обрезает или дополняет до max_len символами <PAD>

        # Хранит все тексты в числовом виде для быстрого доступа.
        # Кодирование делается 1 раз при создании датасета,
        # а не каждый раз при обучении.

        tokens = self.tokenize(text)
        encoded = [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]

        if len(encoded) > self.max_len:
            encoded = encoded[:self.max_len]
        else:
            encoded = encoded + [self.vocab['<PAD>']] * (self.max_len - len(encoded))

        return encoded

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.encoded_texts[idx]), torch.tensor(self.labels[idx])


def load_emails(data_dir):
    texts = []
    labels = []

    for filename in os.listdir(data_dir):
        if not filename.endswith('.txt'):
            continue

        filepath = os.path.join(data_dir, filename)

        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
            texts.append(text)

            # Определяем метку по префиксу файла
            if filename.startswith('spam_'):
                labels.append(1)  # spam
            elif filename.startswith('easy_ham_') or filename.startswith('hard_ham_'):
                labels.append(0)  # ham
            else:
                # На случай неожиданных файлов
                continue

    return texts, labels


def get_data_loaders(data_dir, batch_size=32, test_size=0.3, max_len=1604):
    texts, labels = load_emails(data_dir)

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=42, stratify=labels
    ) # stratify=labels гарантирует сохранение пропорции spam/ham

    train_dataset = EmailDataset(X_train, y_train, max_len=max_len)
    vocab = train_dataset.vocab
    test_dataset = EmailDataset(X_test, y_test, vocab=vocab, max_len=max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader, vocab, (X_train, y_train, X_test, y_test)

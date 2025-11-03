"""
Если надо посмотреть на статистику наших данных
"""
import os

DATA_DIR = "data/extracted/body"


# # lets' count total files:
#
#
# total_files = 0
# for root, dirs, files in os.walk(DATA_DIR):
#     total_files += len(files)
#
# print(f"Total files in '{DATA_DIR}': {total_files}")

# let's count amount of ham and spam files, and ham easy vs ham hard
def print_data_statistics():
    ham_easy_count = 0
    ham_hard_count = 0
    spam_count = 0
    for root, dirs, files in os.walk(DATA_DIR):
        for file in files:
            if file.startswith("easy_ham_"):
                ham_easy_count += 1
            elif file.startswith("hard_ham_"):
                ham_hard_count += 1
            elif file.startswith("spam_"):
                spam_count += 1

    print(f'Ham files: {ham_easy_count + ham_hard_count}')
    print(f"\t- Ham easy files: {ham_easy_count}")
    print(f"\t- Ham hard files: {ham_hard_count}")
    print(f"Spam files: {spam_count}")
    print(f'{'-' * 28}\nTotal files: {ham_easy_count + ham_hard_count + spam_count}')

print_data_statistics()
# Анализ длин текстов
from data_loader import load_emails

texts, labels = load_emails("data/extracted/body")
lengths = [len(text.split()) for text in texts]

import numpy as np
print(f"Средняя длина: {np.mean(lengths):.0f} слов")
print(f"Медиана: {np.median(lengths):.0f} слов")
print(f"95-й перцентиль: {np.percentile(lengths, 95):.0f} слов")
print(f"Максимум: {np.max(lengths)} слов")

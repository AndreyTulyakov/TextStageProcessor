#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Основная структура данных где хранятся промежуточные варианты по тексту
class TextData:
    def __init__(self, filename):
        # Имя файла с текстом
        self.filename = filename
        # Исходные предложения не разделенные на слова
        self.original_sentences = []
        # Разделенные по словам предложения
        self.tokenized_sentences = []
        # Предложения без стоп-слов
        self.no_stop_words_sentences = []
        # Нормализованные предложения
        self.normalized_sentences = []
        # Предложения с преобразованным регистром (заглавные буквы только для имён)
        self.register_pass_centences = []
        # Словарь [слово:частота] по всему тексту
        self.word_frequency = dict()
        # Сортированный список [слово:частота]
        self.sorted_word_frequency = []
        # Счетчик слов в тексте
        self.word_count = 0
        # Здесь можно хранить промежуточные версии предложений для вашего алгоритма
        self.updated_sentences = []
        # Рассчитанные веса TF*IDF для каждого слова в тексте (с учетом соседних документов)
        self.words_tf_idf = dict()
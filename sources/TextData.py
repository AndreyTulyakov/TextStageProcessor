#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Основная структура данных где хранятся промежуточные варианты по тексту
import codecs


class TextData:
    def __init__(self, filename):
        # Имя файла с текстом
        self.filename = filename
        # Имя файла с полным путем
        self.full_filename = filename
        # Относительное имя файла (для древовидных алгоритмов), (заполняется отдельно)
        self.short_filename = ''
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
        # Категория документа (статьи)
        self.category = None

    def readSentencesFromInputText(self, subdir = None):
        full_filename = self.filename
        if(subdir != None):
            full_filename = subdir + '/' + self.filename

        with codecs.open(full_filename, 'r', "utf-8") as text_file:
            data = text_file.read().replace('\n', ' ')
            sentences = data.split('.')
            for i in range(len(sentences)):
                sentences[i] = sentences[i].strip().replace(',', '')
            self.original_sentences = sentences

    def constainsWord(self, test_word):
        for sentense in self.register_pass_centences:
            for word in sentense:
                if(word == test_word):
                    return True
        return False
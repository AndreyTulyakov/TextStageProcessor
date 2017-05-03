#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Основная структура данных где хранятся промежуточные варианты по тексту
import codecs


def readFullTextInputText(filename):
    data = None
    encode_fail = False
    encodings = ['utf-8', 'windows-1251']
    for encoding in encodings:
        try:
            with codecs.open(filename, 'r', encoding) as text_file:
                data = text_file.read()
        except UnicodeDecodeError:
            print('File:%s - Got unicode error with %s , trying different encoding' % (filename, encoding))
            encode_fail = True
        else:
            if encode_fail:
                print('Opening the file [%s] with encoding:  %s ' % (filename, encoding))
            break
    return data


def readSentencesListFromInputText(filename):
    sentences = []
    encode_fail = False
    encodings = ['utf-8', 'windows-1251']
    for encoding in encodings:
        try:
            with codecs.open(filename, 'r', encoding) as text_file:
                data = text_file.read().replace('\n', ' ')
                sentences = data.split('.')
                for i in range(len(sentences)):
                    sentences[i] = sentences[i].strip().replace(',', '')
        except UnicodeDecodeError:
            print('File:%s - Got unicode error with %s , trying different encoding' % (filename, encoding))
            encode_fail = True
        else:
            if encode_fail:
                print('Opening the file [%s] with encoding:  %s ' % (filename, encoding))
            break
    return sentences




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
        self.original_sentences = readSentencesListFromInputText(full_filename)

    def constainsWord(self, test_word):
        for sentense in self.register_pass_centences:
            for word in sentense:
                if(word == test_word):
                    return True
        return False
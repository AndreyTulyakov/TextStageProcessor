#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import namedtuple
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialog, QMessageBox, QTextEdit
from PyQt5 import QtCore, QtGui, uic
from sources.TextPreprocessing import *
from numpy.linalg import svd as singular_value_decomposition
import matplotlib

matplotlib.use('Qt5Agg')


class DialogAnnotationMaker(QDialog):
    MIN_DIMENSIONS = 3
    REDUCTION_RATIO = 1 / 1

    def __init__(self, filename, morph, configurations, parent):
        super().__init__()
        uic.loadUi('sources/DialogAnnotationMaker.ui', self)

        self.filename = filename
        self.morph = morph
        self.configurations = configurations
        self.parent = parent

        flags = Qt.Window | Qt.WindowSystemMenuHint | Qt.WindowMinimizeButtonHint | Qt.WindowCloseButtonHint;
        self.setWindowFlags(flags)

        self.all_idf_word_keys = []
        self.texts = []

        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

        self.buttonProcess.clicked.connect(self.processIt)
        self.textEdit.setText("")

    def processIt(self):
        self.textEdit.setText("")
        result_sentence_count = self.spinBoxOutputSentenceCount.value()

        # Загрузка текста
        self.text = TextData(self.filename)
        self.text.original_sentences = readSentencesFromInputText(self.filename, None)
        original_sentences = tuple(self.text.original_sentences)

        # Разделение текста на слова
        self.configurations["minimal_words_in_sentence"] = 4
        self.text.tokenized_sentences = tokenizeSingleText(self.text, self.configurations)

        # Удаление стоп-слов
        self.configurations["minimal_word_size"] = 3
        self.text.no_stop_words_sentences = removeStopWordsFromSentences(self.text.tokenized_sentences, self.morph,
                                                                         self.configurations)
        np.set_printoptions(suppress=False)

        # Нормализация
        texts, log_string = normalizeTexts([self.text], self.morph)
        self.text = texts[0]

        # Приведение регистра
        texts, log_string = fixRegisterInTexts(texts, self.morph)
        self.text = texts[0]

        # Расчет частотной таблицы слов
        texts, log_string = calculateWordsFrequencyInTexts(texts)
        self.text = texts[0]

        matrix, all_word_keys = self.CreateLSAMatrixForSummarization(self.text)
        matrix = self._compute_term_frequency(matrix)
        u, sigma, v = singular_value_decomposition(matrix, full_matrices=False)
        u = u + np.abs(np.min(u))
        v = v + np.abs(np.min(v))
        u, sigma, v = self.cutSingularValue(u, sigma, v)

        #print('U-MATRIX:\n', printMatrixToString(u, None, all_word_keys))
        #print('V-MATRIX:\n', printMatrixToString(v, range(v.shape[1]), None))

        self.calculateBySentenceValues(v, result_sentence_count)
        self.calculateByWordsValues(all_word_keys, u, result_sentence_count)

        # Готовая библиотека суммаризации
        # from sumy.nlp.stemmers import Stemmer
        # from sumy.nlp.tokenizers import Tokenizer
        # from sumy.parsers.plaintext import PlaintextParser
        # from sumy.utils import get_stop_words
        # from sumy.summarizers.lsa import LsaSummarizer as Summarizer
        # LANGUAGE = "russian"
        #
        # parser = PlaintextParser.from_file(self.filename, Tokenizer(LANGUAGE))
        # print(type(parser))
        # stemmer = Stemmer(LANGUAGE)
        #
        # summarizer = Summarizer(stemmer)
        # summarizer.stop_words = get_stop_words(LANGUAGE)
        #
        # self.textEdit.append('\nРезультирующие предложения:')
        # i = 1
        # for sentence in summarizer(parser.document, result_sentence_count):
        #     self.textEdit.append(str(i) + ') ' + str(sentence) + '\n')
        #     i = i + 1

        self.textEdit.append('\nУспешно завершено.')
        QMessageBox.information(self, "Внимание", "Автоматическое аннотирование завершено!")

    def calculateBySentenceValues(self, v, sentence_count_need):
        sentences_score = tuple((i, np.mean(v[0, i])) for i in range(v.shape[1]))
        sorted_sentences_score = sorted(sentences_score, key=lambda x: x[1], reverse=True)
        result_sentences = []
        self.textEdit.append('\nРезультирующие предложения (По весу предложения):')
        for i in range(min(sentence_count_need, len(sorted_sentences_score))):
            item = sorted_sentences_score[i]
            result_sentences.append(self.text.original_sentences[item[0]])
        i = 1
        for sentence in result_sentences:
            self.textEdit.append(str(i) + ') ' + str(sentence) + '\n')
            i += 1

    def calculateByWordsValues(self, all_word_keys, u, sentence_count_need):
        words_score = dict()
        for index, word in enumerate(all_word_keys):
            summa = []
            for y in range(u.shape[1]):
                summa.append(u[index][y])
            words_score[word] = np.mean(summa)

        sentences_res2 = list()
        for index, sentence in enumerate(self.text.register_pass_centences):
            sentence_sum = 0
            for word in sentence:
                sentence_sum += words_score[word]
            sentences_res2.append((index, sentence_sum))
        sorted_sentences_score2 = sorted(sentences_res2, key=lambda x: x[1], reverse=True)

        result_sentences2 = []
        self.textEdit.append('\nРезультирующие предложения (По весам слов):')
        for i in range(min(sentence_count_need, len(sorted_sentences_score2))):
            item = sorted_sentences_score2[i]
            result_sentences2.append(self.text.original_sentences[item[0]])
        i = 1
        for sentence in result_sentences2:
            self.textEdit.append(str(i) + ') ' + str(sentence) + '\n')
            i += 1

    # Сингулярное разложение на a = u, s, v (S - восстановленный до диагональной матрицы вектор)
    def divideSingular(self, matrix):
        u, s, v = np.linalg.svd(matrix, full_matrices=False)
        S = np.zeros((u.shape[0], v.shape[0]), dtype=complex)
        S[:v.shape[0], :v.shape[1]] = np.diag(s)
        return u, S, v, s

    # Создание матрицы слова-[частота]-документы
    def CreateLSAMatrixForSummarization(self, text):

        all_sentences_count = len(text.register_pass_centences);
        all_word_list = dict()

        # Создаем список уникальных слов
        for sentenceIndex in range(all_sentences_count):
            sentence = text.register_pass_centences[sentenceIndex]
            for word in sentence:
                all_word_list[word] = all_word_list.get(word, 0) + 1

        all_word_list_keys = list(all_word_list.keys())
        all_words_count = len(all_word_list_keys)

        lsa_matrix = np.zeros(shape=(all_words_count, all_sentences_count))

        for s in range(all_sentences_count):
            for i in range(all_words_count):

                current_word = all_word_list_keys[i]
                word_frequency_in_current_sentence = 0

                for word in text.register_pass_centences[s]:
                    if (word == current_word):
                        word_frequency_in_current_sentence = word_frequency_in_current_sentence + 1

                lsa_matrix[i][s] = math.sqrt(word_frequency_in_current_sentence)

        return lsa_matrix, all_word_list_keys

    def _compute_term_frequency(self, matrix, smooth=0.4):
        """
        Computes TF metrics for each sentence (column) in the given matrix.
        You can read more about smoothing parameter at URL below:
        http://nlp.stanford.edu/IR-book/html/htmledition/maximum-tf-normalization-1.html
        """
        assert 0.0 <= smooth < 1.0

        max_word_frequencies = np.max(matrix, axis=0)
        rows, cols = matrix.shape
        for row in range(rows):
            for col in range(cols):
                max_word_frequency = max_word_frequencies[col]
                if max_word_frequency != 0:
                    frequency = matrix[row, col] / max_word_frequency
                    matrix[row, col] = smooth + (1.0 - smooth) * frequency

        return matrix

    def cutSingularValue(self, u, sigma, v):

        #print('SING SIZE:', sigma.shape[0], str(sigma))
        singular_minimal_transfer = 3
        m = np.median(sigma)
        for i in range(sigma.shape[0]):
            if (sigma[i] > m):
                singular_minimal_transfer = i

        nu = u[0:, 0:(singular_minimal_transfer)]
        ns = sigma[0:(singular_minimal_transfer)]
        nv = v[0:(singular_minimal_transfer), 0:]
        #print('SING SIZE AFTER:', singular_minimal_transfer, str(ns))

        return nu, ns, nv


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from PyQt5.QtCore import QObject
from PyQt5.QtCore import QThread
from PyQt5.QtCore import Qt
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QDialog, QMessageBox, QTextEdit
from PyQt5 import QtCore, QtGui, uic
from sources.TextPreprocessing import *
from numpy.linalg import svd as singular_value_decomposition
from sources.utils import Profiler


class AnnotationMakerCalculatorSignals(QObject):
    PrintInfo = pyqtSignal(str)
    Finished = pyqtSignal()
    UpdateProgressBar = pyqtSignal(int)

class AnnotationMakerCalculator(QThread):

    METHOD_BY_WORDS_SUM = 0
    METHOD_BY_SENTENCE_VALUE = 1

    def __init__(self, filename, morph, configurations):
        super().__init__()
        self.filename = filename
        self.morph = morph
        self.configurations = configurations
        self.texts = []
        self.categories = dict()
        self.signals = AnnotationMakerCalculatorSignals()
        self.result_sentence_count = 1
        self.calculation_method = AnnotationMakerCalculator.METHOD_BY_WORDS_SUM
        self.output_dir = self.configurations.get("output_files_directory", "output_files") + '/auto_annotation/'

    def setCalculationMethod(self, method_type):
        self.calculation_method = method_type

    def run(self):
        self.signals.UpdateProgressBar.emit(0)

        # Загрузка текста
        self.text = TextData(self.filename)
        self.text.original_sentences = readSentencesFromInputText(self.filename, None)
        original_sentences = tuple(self.text.original_sentences)

        self.signals.UpdateProgressBar.emit(5)

        # Разделение текста на слова
        self.configurations["minimal_words_in_sentence"] = 4
        self.configurations['need_agresive_filtration'] = True
        self.text.tokenized_sentences = tokenizeSingleText(self.text, self.configurations)

        # Удаление стоп-слов
        self.configurations["minimal_word_size"] = 3
        self.text.no_stop_words_sentences = removeStopWordsFromSentences(self.text.tokenized_sentences, self.morph, self.configurations)

        if len(self.text.no_stop_words_sentences) > 0:

            np.set_printoptions(suppress=False)

            self.signals.UpdateProgressBar.emit(20)

            # Нормализация
            texts, log_string = normalizeTexts([self.text], self.morph)
            self.text = texts[0]

            # Приведение регистра
            texts, log_string = fixRegisterInTexts(texts, self.morph)
            self.text = texts[0]

            self.signals.UpdateProgressBar.emit(30)

            # Расчет частотной таблицы слов
            texts, log_string = calculateWordsFrequencyInTexts(texts)
            self.text = texts[0]

            self.signals.UpdateProgressBar.emit(40)

            matrix, all_word_keys = self.CreateLSAMatrixForSummarization(self.text)
            matrix = self._compute_term_frequency(matrix)

            self.signals.UpdateProgressBar.emit(50)

            u, sigma, v = singular_value_decomposition(matrix, full_matrices=False)
            u = u + np.abs(np.min(u))
            v = v + np.abs(np.min(v))
            u, sigma, v = self.cutSingularValue(u, sigma, v)

            self.signals.UpdateProgressBar.emit(70)

            if(self.calculation_method == AnnotationMakerCalculator.METHOD_BY_SENTENCE_VALUE):
                self.calculateBySentenceValues(v, self.result_sentence_count)

            if (self.calculation_method == AnnotationMakerCalculator.METHOD_BY_WORDS_SUM):
                self.calculateByWordsValues(all_word_keys, u, self.result_sentence_count)
            self.signals.PrintInfo.emit('\nУспешно завершено.')
        else:
            self.signals.PrintInfo.emit('\nНедостаточно входных данных и/или много неликвидных данных.')

        self.signals.UpdateProgressBar.emit(100)

        self.signals.Finished.emit()


    def calculateBySentenceValues(self, v, sentence_count_need):
        if(v.shape[0] == 0):
            self.signals.PrintInfo.emit('Слишком мало предложений! Прерывание...')
            return
        sentences_score = tuple((i, np.mean(v[0, i])) for i in range(v.shape[1]))
        sorted_sentences_score = sorted(sentences_score, key=lambda x: x[1], reverse=True)
        result_sentences = []
        self.signals.PrintInfo.emit('\nРезультирующие предложения (По весу предложения):')
        result_string = 'Результирующие предложения (По весу предложения):\n'
        for i in range(min(sentence_count_need, len(sorted_sentences_score))):
            item = sorted_sentences_score[i]
            result_sentences.append(self.text.original_sentences[item[0]])
        i = 1
        for sentence in result_sentences:
            self.signals.PrintInfo.emit(str(i) + ') ' + str(sentence) + '\n')
            result_string = result_string + (str(i) + ') ' + str(sentence) + '\n')
            i += 1

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        filename = self.output_dir + 'BySentenceValues_' + self.filename[1+self.filename.rfind('/'):]
        self.signals.PrintInfo.emit('Сохранено в файл:' + filename)
        writeStringToFile(result_string, filename)

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
        self.signals.PrintInfo.emit('\nРезультирующие предложения (По весам слов):')
        result_string = 'Результирующие предложения (По весам слов):\n'
        for i in range(min(sentence_count_need, len(sorted_sentences_score2))):
            item = sorted_sentences_score2[i]
            result_sentences2.append(self.text.original_sentences[item[0]])
        i = 1
        for sentence in result_sentences2:
            self.signals.PrintInfo.emit(str(i) + ') ' + str(sentence) + '\n')
            result_string += str(i) + ') ' + str(sentence) + '\n'
            i += 1

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        filename = self.output_dir + 'ByWordsValues_' + self.filename[1+self.filename.rfind('/'):]
        self.signals.PrintInfo.emit('Сохранено в файл:' + filename)
        writeStringToFile(result_string, filename)


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
        singular_minimal_transfer = 3
        m = np.mean(sigma)

        q = min(sigma.shape[0], singular_minimal_transfer)
        for i in range(sigma.shape[0]):
            if (sigma[i] > m or i < q):
                singular_minimal_transfer = i
        nu = u[0:, 0:(singular_minimal_transfer)]
        ns = sigma[0:(singular_minimal_transfer)]
        nv = v[0:(singular_minimal_transfer), 0:]
        return nu, ns, nv


class DialogAnnotationMaker(QDialog):

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

        self.profiler = Profiler()

        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

        self.buttonProcess.clicked.connect(self.process_it)
        self.textEdit.setText("")
        self.progressBar.setValue(0)

        self.calculator = AnnotationMakerCalculator(filename, morph, self.configurations)
        self.calculator.signals.Finished.connect(self.on_calculation_finish)
        self.calculator.signals.UpdateProgressBar.connect(self.on_update_progress_bar)
        self.calculator.signals.PrintInfo.connect(self.on_text_log_add)

    def on_text_log_add(self, QString):
        self.textEdit.append(QString + '\n')
        self.repaint()

    def on_update_progress_bar(self, value):
        self.progressBar.setValue(value)
        self.repaint()

    def on_calculation_finish(self):
        self.textEdit.append('Выполнено за ' + self.profiler.stop() + ' с.')
        QApplication.restoreOverrideCursor()
        self.buttonProcess.setEnabled(True)
        QMessageBox.information(self, "Внимание", "Автоматическое аннотирование завершено!")

    def process_it(self):
        self.buttonProcess.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.progressBar.setValue(0)
        self.textEdit.setText("")
        self.calculator.result_sentence_count = self.spinBoxOutputSentenceCount.value()

        if(self.radioButtonMethodWordsSum.isChecked()):
            self.calculator.setCalculationMethod(AnnotationMakerCalculator.METHOD_BY_WORDS_SUM)

        if (self.radioButtonMethodSentenceValue.isChecked()):
            self.calculator.setCalculationMethod(AnnotationMakerCalculator.METHOD_BY_SENTENCE_VALUE )
        self.profiler.start()
        self.calculator.start()



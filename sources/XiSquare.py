#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from PyQt5.QtCore import QObject
from PyQt5.QtCore import Qt
from PyQt5.QtCore import pyqtSignal

from PyQt5.QtWidgets import QDialog, QMessageBox, QTextEdit, QApplication
from PyQt5 import QtCore, QtGui, uic

from PyQt5.QtCore import QThread

from sources.TextPreprocessing import *
from sources.apriori_maker import makeAprioriForTexts
from sources.utils import Profiler
from stage_text_processor import output_dir


def right_div(a, b):
    if a == 0 and b == 0:
        return float('nan')
    if a == 0 and b != 0:
        return 0
    if b == 0:
        return float('nan')
    return a / b


def slog2(x):
    if x <= 0 or float(x) == float('nan'):
        return float('nan')

    return math.log(x, 2)


def a_log_b_div_c(a, b, c):
    if a == 0 and b == 0:
        return 0.0
    return a * slog2(right_div(b, c))


class XiCalculatorSignals(QObject):
    PrintInfo = pyqtSignal(str)
    Finished = pyqtSignal()
    UpdateProgressBar = pyqtSignal(int)


class XiCalculator(QThread):
    def __init__(self, filename, output_dir, morph, configurations):
        super().__init__()
        self.filename = filename
        self.output_dir = output_dir
        self.morph = morph
        self.configurations = configurations
        self.texts = []
        self.categories = dict()
        self.signals = XiCalculatorSignals()
        self.need_preprocessing = True
        self.need_apriori = False
        self.min_support = 0.01
        self.min_conf = 0.01

    def printMatrixToCsv(self, filename, header1, header2, matrix):
        matrix_csv_str = ';'
        header1_size = len(header1)
        header2_size = len(header2)

        for category in header1:
            matrix_csv_str = matrix_csv_str + str(category) + ';'
        matrix_csv_str = matrix_csv_str + '\n'

        for header2_index in range(header2_size):
            header2_value = header2[header2_index]
            matrix_csv_str = matrix_csv_str + header2_value + ';'
            for header1_index in range(header1_size):
                matrix_csv_str = matrix_csv_str + str(matrix[header1_index][header2_index]) + ';'
            matrix_csv_str = matrix_csv_str + '\n'

        # matrix_csv_str = matrix_csv_str.replace('nan','')
        matrix_csv_str = matrix_csv_str.replace('.', ',')
        writeStringToFile(matrix_csv_str, filename)

    def printDictToCsv(self, header, data, filename):
        matrix_csv_str = ''

        for text in header:
            matrix_csv_str = matrix_csv_str + str(text) + ';'
        matrix_csv_str = matrix_csv_str + '\n'

        for key, value in data.items():
            matrix_csv_str = matrix_csv_str + str(key) + ';' + str(value) + '\n'

        matrix_csv_str = matrix_csv_str.replace('.', ',')
        writeStringToFile(matrix_csv_str, filename)

    def run(self):
        self.signals.UpdateProgressBar.emit(0)
        # Считываем файл с информацией о категории и файлах
        self.signals.PrintInfo.emit('Чтение файла с категориям...')
        learn_groups = pd.read_csv(self.filename, index_col=None, na_values=['nan'], keep_default_na=False)
        self.input_path = self.filename[0:self.filename.rfind('/')]

        # Заполняем словарь КАТЕГОРИЯ:СПИСОК_ФАЙЛОВ
        for category in list(learn_groups):
            self.categories[category] = []
            for value in learn_groups[category]:
                value = str(value).strip()
                if (len(value) > 0):
                    self.categories[category].append(value)

        self.signals.UpdateProgressBar.emit(10)

        self.signals.PrintInfo.emit('Загрузка текстовых файлов...')
        # Загрузка текстовых файлов
        for key in self.categories.keys():
            for filename in self.categories[key]:
                if (filename != None and filename != 'nan' and len(filename) != 0):
                    text = TextData(filename)
                    text.readSentencesFromInputText(self.input_path)
                    text.category = key
                    self.texts.append(text)

        self.signals.UpdateProgressBar.emit(20)

        if self.need_preprocessing:
            self.signals.PrintInfo.emit('Предварительная обработка текстов...')
            self.configurations['need_agresive_filtration'] = True

            output_dir = self.configurations.get("output_files_directory", "output_files") + "/preprocessing/"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            # Разделяем предложения на слова
            self.texts = tokenizeTextData(self.texts)
            self.signals.PrintInfo.emit('Этап препроцессинга:')
            self.signals.UpdateProgressBar.emit(10)
            # Удаление стоп-слов из предложения (частицы, прилагательные и тд)
            self.signals.PrintInfo.emit('1) Удаление стоп-слов.')

            self.texts, log_string = removeStopWordsInTexts(self.texts, self.morph, self.configurations)
            writeStringToFile(log_string.replace('\n ', '\n'), output_dir + 'output_stage_1.txt')
            self.signals.UpdateProgressBar.emit(20)

            # Переводим обычное предложение в нормализованное (каждое слово)
            self.signals.PrintInfo.emit('2) Нормализация.')
            self.texts, log_string = normalizeTexts(self.texts, self.morph)
            writeStringToFile(log_string.replace('\n ', '\n'), output_dir + 'output_stage_2.txt')
            self.signals.UpdateProgressBar.emit(30)

            # Приведение регистра (все слова с маленькой буквы за исключением ФИО)
            self.signals.PrintInfo.emit('3) Приведение регистра.')
            self.texts, log_string = fixRegisterInTexts(self.texts, self.morph)
            writeStringToFile(log_string.replace('\n ', '\n'), output_dir + 'output_stage_3.txt')
            self.signals.UpdateProgressBar.emit(40)

            # Подсчет частоты слов в тексте
            self.signals.PrintInfo.emit('4) Расчет частотной таблицы слов.')
            self.texts, log_string = calculateWordsFrequencyInTexts(self.texts)
            writeStringToFile(log_string.replace('\n ', '\n'), output_dir + 'output_stage_4.csv')

            self.signals.UpdateProgressBar.emit(45)
        else:
            self.signals.PrintInfo.emit('Предварительная обработка текстов: Пропускается')
            texts = tokenizeTextData(self.texts)
            for text in self.texts:
                text.no_stop_words_sentences = text.tokenized_sentences
                text.normalized_sentences = text.tokenized_sentences
                text.register_pass_centences = text.tokenized_sentences
                self.texts, log_string = calculateWordsFrequencyInTexts(texts)

        if self.need_apriori:
            self.signals.PrintInfo.emit('Рассчет Apriori...')
            makeAprioriForTexts(self.texts, output_dir, self.min_support, self.min_conf)

        self.signals.PrintInfo.emit('Рассчет списка уникальных слов для матриц...')

        # Рассчет списка уникальных слов для матриц
        all_unique_words = dict()
        for text in self.texts:
            for sentence in text.register_pass_centences:
                for word in sentence:
                    all_unique_words[word] = all_unique_words.get(word, 0) + 1
        all_unique_words_list = list(all_unique_words.keys())
        unique_words_count = len(all_unique_words_list)
        categories_list = list(learn_groups)
        categories_count = len(categories_list)

        self.signals.PrintInfo.emit('Рассчет DF...')
        df_dict = dict()
        for word in all_unique_words_list:
            df_dict[word] = 0
            for text in self.texts:
                pre_df = df_dict[word]
                for sentence in text.register_pass_centences:
                    for current_text_word in sentence:
                        if current_text_word == word:
                            df_dict[word] += 1
                            break
                    if df_dict[word] > pre_df:
                        break


        self.signals.UpdateProgressBar.emit(55)
        self.signals.PrintInfo.emit('Создание и заполнение MI, IG, CHI матриц...')

        # Создание матриц (Категория * Слова)
        a_matrix = np.zeros(shape=(categories_count, unique_words_count))
        b_matrix = np.zeros(shape=(categories_count, unique_words_count))
        c_matrix = np.zeros(shape=(categories_count, unique_words_count))
        d_matrix = np.zeros(shape=(categories_count, unique_words_count))
        mi_matrix = np.zeros(shape=(categories_count, unique_words_count))
        ig_matrix = np.zeros(shape=(categories_count, unique_words_count))
        chi_matrix = np.zeros(shape=(categories_count, unique_words_count))

        self.signals.UpdateProgressBar.emit(60)

        # Рассчет А, B, C, D показателей (стр 173)
        for category_index in range(categories_count):
            category = categories_list[category_index]
            for word_index in range(unique_words_count):
                word = all_unique_words_list[word_index]

                for text in self.texts:
                    target_category = text.category == category
                    contains_word = text.constainsWord(word)
                    if (target_category and contains_word):
                        a_matrix[category_index][word_index] = a_matrix[category_index][word_index] + 1
                    if (not target_category and contains_word):
                        b_matrix[category_index][word_index] = b_matrix[category_index][word_index] + 1
                    if (target_category and not contains_word):
                        c_matrix[category_index][word_index] = c_matrix[category_index][word_index] + 1
                    if (not target_category and not contains_word):
                        d_matrix[category_index][word_index] = d_matrix[category_index][word_index] + 1

        self.signals.UpdateProgressBar.emit(80)

        for category_index in range(categories_count):
            for word_index in range(unique_words_count):
                u = len(self.texts)
                a = a_matrix[category_index][word_index]
                b = b_matrix[category_index][word_index]
                c = c_matrix[category_index][word_index]
                d = d_matrix[category_index][word_index]

                word = all_unique_words_list[word_index]
                category = categories_list[category_index]

                # Формула-5 стр173
                mi_value = right_div(a * u, (a + c) * (a + b))
                mi_matrix[category_index][word_index] = slog2(mi_value)

                # Формула 7 стр174
                ig_value_1d = (a + b) * (a + c)
                ig_value_2d = (c + d) * (a + c)
                ig_value_3d = (a + b) * (b + d)
                ig_value_4d = (c + d) * (b + d)

                ig_value_1 = a_log_b_div_c(a / u, u * a, ig_value_1d)
                ig_value_2 = a_log_b_div_c(c / u, u * c, ig_value_2d)
                ig_value_3 = a_log_b_div_c(b / u, u * b, ig_value_3d)
                ig_value_4 = a_log_b_div_c(d / u, u * d, ig_value_4d)

                ig_matrix[category_index][word_index] = ig_value_1 + ig_value_2 + ig_value_3 + ig_value_4

                # Формула-9 стр174
                chi_value = u * (((a * d) - (c * b)) ** 2)
                chi_value = right_div(chi_value, ((a + c) * (b + d) * (a + b) * (c + d)))

                chi_matrix[category_index][word_index] = chi_value

        self.signals.UpdateProgressBar.emit(90)

        self.printDictToCsv(('Слово', 'DF (Кол-во документов в которых встречается слово)'), df_dict, self.output_dir + '/df_table.csv')
        self.signals.PrintInfo.emit('Таблица DF записана в файл:' + self.output_dir + '/df_table.csv')

        # Сохраняем рассчитанные матрицы в CSV файлы
        self.printMatrixToCsv(self.output_dir + '/mi_matrix.csv', categories_list, all_unique_words_list, mi_matrix)
        self.signals.PrintInfo.emit('Матрица MI записана в файл:' + self.output_dir + '/mi_matrix.csv')

        self.printMatrixToCsv(self.output_dir + '/ig_matrix.csv', categories_list, all_unique_words_list, ig_matrix)
        self.signals.PrintInfo.emit('Матрица IG записана в файл:' + self.output_dir + '/ig_matrix.csv')

        self.printMatrixToCsv(self.output_dir + '/chi_matrix.csv', categories_list, all_unique_words_list, chi_matrix)
        self.signals.PrintInfo.emit('Матрица CHI записана в файл:' + self.output_dir + '/chi_matrix.csv')

        self.signals.UpdateProgressBar.emit(100)
        self.signals.PrintInfo.emit('Рассчеты закончены!')
        self.signals.Finished.emit()


class DialogXiSquare(QDialog):
    def __init__(self, filename, morph, configurations, parent):
        super().__init__()
        uic.loadUi('sources/XiSquare.ui', self)
        self.configurations = configurations

        flags = Qt.Window | Qt.WindowSystemMenuHint | Qt.WindowMinimizeButtonHint | Qt.WindowCloseButtonHint;
        self.setWindowFlags(flags)

        self.parent = parent
        self.all_idf_word_keys = []

        self.input_path = ''
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.buttonProcess.clicked.connect(self.processIt)
        self.textEdit.setText("")

        self.profiler = Profiler()

        self.configurations["minimal_word_size"] = 4
        self.configurations["cut_ADJ"] = False
        output_dir = self.configurations.get("output_files_directory", "output_files")
        self.progressBar.setValue(0)

        self.checkBoxNeedApriori.toggled.connect(self.onActivateApriori)
        self.groupBoxApriori.setVisible(False)

        self.calculator = XiCalculator(filename, output_dir, morph, self.configurations)
        self.calculator.signals.Finished.connect(self.onCalculationFinish)
        self.calculator.signals.UpdateProgressBar.connect(self.onUpdateProgressBar)
        self.calculator.signals.PrintInfo.connect(self.onTextLogAdd)

    def onActivateApriori(self):
        if self.checkBoxNeedApriori.isChecked():
            self.groupBoxApriori.setVisible(True)
        else:
            self.groupBoxApriori.setVisible(False)

    def onTextLogAdd(self, QString):
        self.textEdit.append(QString + '\n')
        self.repaint()

    def onUpdateProgressBar(self, value):
        self.progressBar.setValue(value)
        self.repaint()

    def onCalculationFinish(self):
        self.textEdit.append('Выполнено за ' + self.profiler.stop() + ' с.')
        QApplication.restoreOverrideCursor()
        QMessageBox.information(self, "Внимание", "Расчет: MI, IG, Хи-Квадрат завершен!")

    def processIt(self):
        self.buttonProcess.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.textEdit.setText("")
        self.calculator.need_preprocessing = self.checkBoxEnablePreprocessing.isChecked()
        self.calculator.need_apriori = self.checkBoxNeedApriori.isChecked()
        self.calculator.min_support = self.spinBoxMinSupport.value()
        self.calculator.min_conf = self.spinBoxMinConf.value()
        self.profiler.start()
        self.calculator.start()

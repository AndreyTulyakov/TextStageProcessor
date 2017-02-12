#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from PyQt5.QtCore import Qt

from PyQt5.QtWidgets import QDialog, QMessageBox, QTextEdit
from PyQt5 import QtCore, QtGui, uic

from sources.TextData import TextData
from sources.TextPreprocessing import *


class DialogXiSquare(QDialog):

    def __init__(self, filename, morph, configurations, parent):
        super().__init__()
        uic.loadUi('sources/XiSquare.ui', self)

        flags = Qt.Window | Qt.WindowSystemMenuHint | Qt.WindowMinimizeButtonHint | Qt.WindowCloseButtonHint;
        self.setWindowFlags(flags)

        self.filename = filename
        self.morph = morph
        self.configurations = configurations
        self.parent = parent

        self.all_idf_word_keys = []
        self.texts = []
        self.input_path = ''

        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

        self.buttonProcess.clicked.connect(self.processIt)

        self.textEdit.setText("")
        self.categories = dict()


    def processIt(self):
        self.textEdit.setText("")
        self.configurations["minimal_word_size"] = 4
        self.configurations["cut_ADJ"] = False

        output_dir = self.configurations.get("output_files_directory", "output_files")

        print('Step',1)

        # Считываем обучающий файл содержащий информацию о том,
        # какой файл к какой категории относится
        learn_groups = pd.read_csv(self.filename, index_col=None,na_values=['nan'], keep_default_na=False)
        self.input_path = self.filename[0:self.filename.rfind('/')]

        # Заполняем словарь КАТЕГОРИЯ:СПИСОК(ИМЯ_ФАЙЛА)
        for category in list(learn_groups):
            self.categories[category] = []
            for value in learn_groups[category]:
                value = str(value).strip()
                if(len(value) > 0):
                    self.categories[category].append(value)

        print('Step',2)

        for key in self.categories.keys():
            print(key)
            for filename in self.categories[key]:
                if(filename != None and filename != 'nan' and len(filename) != 0):
                    text = TextData(filename)
                    text.readSentencesFromInputText(self.input_path)
                    text.category = key

                    self.texts.append(text)

        print('Step', 3)

        # Предварительная обработка
        self.texts = tokenizeTextData(self.texts)
        self.texts, log_string = removeStopWordsInTexts(self.texts, self.morph, self.configurations)
        self.texts, log_string = normalizeTexts(self.texts, self.morph)
        self.texts, log_string = fixRegisterInTexts(self.texts, self.morph)
        self.texts, log_string = calculateWordsFrequencyInTexts(self.texts)

        print('Step', 4)

        # Создание матриц [Термы * Категории]
        all_unique_words = dict()
        for text in self.texts:
            for sentence in text.register_pass_centences:
                for word in sentence:
                    all_unique_words[word] = all_unique_words.get(word, 0) + 1
        all_unique_words_list = list(all_unique_words.keys())
        unique_words_count = len(all_unique_words_list)
        categories_list = list(learn_groups)
        categories_count = len(categories_list)

        print('Step', 5)

        xi_a_matrix = np.zeros(shape=(categories_count, unique_words_count))
        xi_b_matrix = np.zeros(shape=(categories_count, unique_words_count))
        xi_c_matrix = np.zeros(shape=(categories_count, unique_words_count))
        xi_d_matrix = np.zeros(shape=(categories_count, unique_words_count))
        xi_mi_matrix = np.zeros(shape=(categories_count, unique_words_count))
        xi_chi_matrix = np.zeros(shape=(categories_count, unique_words_count))

        print('Step', 6)

        print('Total unique words: ', len(all_unique_words.keys()))

        # Рассчет частот
        # A - кол-во док. принадлежащих к кат С и содержащих термин t
        print(unique_words_count, categories_count)


        for category_index in range(categories_count):
            category = categories_list[category_index]
            for word_index in range(unique_words_count):
                word = all_unique_words_list[word_index]

                for text in self.texts:
                    target_category = text.category == category
                    contains_word = text.constainsWord(word)
                    if(target_category and contains_word):
                        xi_a_matrix[category_index][word_index] = xi_a_matrix[category_index][word_index] + 1
                    if (not target_category and contains_word):
                        xi_b_matrix[category_index][word_index] = xi_b_matrix[category_index][word_index] + 1
                    if (target_category and not contains_word):
                        xi_c_matrix[category_index][word_index] = xi_c_matrix[category_index][word_index] + 1
                    if (not target_category and not contains_word):
                        xi_d_matrix[category_index][word_index] = xi_d_matrix[category_index][word_index] + 1

        print('Step', 7)

        for category_index in range(categories_count):
            category = categories_list[category_index]
            for word_index in range(unique_words_count):
                word = all_unique_words_list[word_index]
                u = len(self.texts)
                a = xi_a_matrix[category_index][word_index]
                b = xi_b_matrix[category_index][word_index]
                c = xi_c_matrix[category_index][word_index]
                d = xi_d_matrix[category_index][word_index]

                # Формула-5 стр173
                mi_value = (a * u) / ((a+c)*(a+b))
                xi_mi_matrix[category_index][word_index] = np.log2(mi_value)

                # Формула-9 стр174
                chi_value = u * (((a*d)-(c*b))**2)
                chi_value = chi_value / ((a+c)*(b+d)*(a+b)*(c+d))
                xi_chi_matrix[category_index][word_index] = chi_value

        print(xi_mi_matrix)

        self.printMatrixToCsv(output_dir + '/mi_matrix.csv', categories_list, all_unique_words_list, xi_mi_matrix)
        self.printMatrixToCsv(output_dir + '/chi_matrix.csv', categories_list, all_unique_words_list, xi_chi_matrix)



        print('Step', 8)

        QMessageBox.information(self, "Внимание", "Расчет Хи-Квадрат завершен!")


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

        writeStringToFile(matrix_csv_str, filename)



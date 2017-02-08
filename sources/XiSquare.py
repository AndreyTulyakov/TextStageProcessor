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
        self.setWindowFlags(Qt.WindowMinMaxButtonsHint|self.windowFlags())

        self.filename = filename
        self.morph = morph
        self.configurations = configurations
        self.parent = parent

        self.all_idf_word_keys = []
        self.texts = []

        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

        self.buttonProcess.clicked.connect(self.processIt)

        self.textEdit.setText("")
        self.categories = dict()


    def processIt(self):
        self.textEdit.setText("")
        self.configurations["minimal_word_size"] = 4
        self.configurations["cut_ADJ"] = False
        
        #self.texts = makePreprocessing(self.filenames, self.morph, self.configurations, self.textEdit)

        output_dir = self.configurations.get("output_files_directory", "output_files")

        learn_groups = pd.read_csv(self.filename, index_col=None)

        for category in list(learn_groups):
            self.categories[category] = []
            for value in learn_groups[category]:
                self.categories[category].append(value)

        print(self.categories)

        #
        # idf_word_data = calculateWordsIDF(self.texts)
        # sorted_IDF = sorted(idf_word_data.items(), key=lambda x: x[1], reverse=False)
        # calculateTFIDF(self.texts, idf_word_data)
        #
        #
        # log_string = writeWordTFIDFToString(self.texts, idf_word_data)
        # writeStringToFile(log_string.replace('\n ', '\n'), output_dir + '/output_stage_6.csv')
        #
        # # Вырезаем из TF-IDF % худших слов
        # removeTFIDFWordsWithMiniamlMultiplier(self.texts , self.spinBoxCutPercent.value()/100.0)
        #
        # log_string = writeWordTFIDFToString(self.texts, idf_word_data)
        # writeStringToFile(log_string.replace('\n ', '\n'), output_dir + '/output_stage_7.csv')
        #
        # self.textEdit.append('Успешно завершено.')

        QMessageBox.information(self, "Внимание", "Расчет Хи-Квадрат завершен!")

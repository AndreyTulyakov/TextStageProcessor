#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re

from PyQt5.QtCore import Qt

from sources.TextPreprocessing import *
from sources.classification.Rocchio import *
from sources.classification.NaiveBayes import *

from PyQt5.QtWidgets import QDialog, QMessageBox, QTextEdit
from PyQt5 import QtCore, QtGui, uic


class DialogConfigClassification(QDialog):

    def __init__(self, morph, configurations, parent):
        super().__init__()
        uic.loadUi('sources/DialogConfigClassification.ui', self)

        flags = Qt.Window | Qt.WindowSystemMenuHint | Qt.WindowMinimizeButtonHint | Qt.WindowCloseButtonHint;
        self.setWindowFlags(flags)

        self.morph = morph
        self.configurations = configurations
        self.parent = parent

        self.checkBoxNeedPreprocessing.setEnabled(False)

        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.buttonClassify.clicked.connect(self.makeClassification)
        self.textEdit.setText("")

        self.output_dir = configurations.get("output_files_directory", "output_files/classification") + "/"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    # Вызов при нажатии на кнопку "Классифицировать"
    def makeClassification(self):
        self.textEdit.setText("")
        self.lineEditInputDir.setEnabled(False)
        #self.input_dir = configurations.get("input_classification_files_directory", "input_files/classification")
        self.input_dir = self.lineEditInputDir.text() + '/'

        needPreprocessing = self.checkBoxNeedPreprocessing.isChecked()

        if(self.radioButtonNaiveBayes.isChecked()):
            self.classification_naive_bayes(needPreprocessing)

        if(self.radioButtonRocchio.isChecked()):
            self.classification_rocchio(needPreprocessing)

        self.textEdit.append('Завершено.')
        QMessageBox.information(self, "Внимание", "Процесс классификации завершен!")
        self.lineEditInputDir.setEnabled(True)

    # Алгоритм наивного Байеса
    def classification_naive_bayes(self, needPreprocessing):

        # Обучение
        self.textEdit.append('Обучение...')
        traindir = self.input_dir + "train/"
        DClasses = os.listdir(traindir)

        p = Pool()
        for i in DClasses:
            p.learn(traindir + i, i)

        # Классификация
        self.textEdit.append('Классификация...')
        testdir = self.input_dir + "test/"

        log_string = "Вероятности документов:\n"
        log_string_rates = "Процент похожих слов:\n"

        for i in DClasses:
            dir = os.listdir(testdir + i)
            correct = 0
            for file in dir:
                res = p.Probability(testdir + i + "/" + file)
                str_out = i + ": " + file + ": " + str(res)
                if i == res[0][0]:
                    correct = correct + 1
                log_string = log_string + str_out + '\n'
                log_string_rates += str(p.DocumentIntersectionWithClasses(testdir + i + "/" + file)) + "\n"
            log_string = log_string + 'Верных классов:' + i + ': ' + str(correct) + '\n'

        log_string_learn = "Словарь классов обучения:\n"
        for c in DClasses:
            log_string_learn += c + "\n"
            log_string_learn += str(p.BagOfWords_in_class(c)) + "\n"

        self.textEdit.append("Выходные файлы:")
        out_dir = self.output_dir + 'nb_out/'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        self.textEdit.append(out_dir + 'output_naive_bayes.csv')
        writeStringToFile(log_string, out_dir + 'output_naive_bayes.csv')
        self.textEdit.append(out_dir + 'naive_bayes_learn.csv')
        writeStringToFile(log_string_learn, out_dir + 'naive_bayes_learn.csv')
        self.textEdit.append(out_dir + 'naive_bayes_rates.csv')
        writeStringToFile(log_string_rates, out_dir + 'naive_bayes_rates.csv')

    # Алгоритм Рочио
    def classification_rocchio(self, needPreprocessing):
        # Обучение
        self.textEdit.append('Обучение...')
        root_path = self.input_dir + 'train/'
        folders = [root_path + folder + '/' for folder in os.listdir(root_path)]
        class_titles = os.listdir(root_path)
        files = {}
        for folder, title in zip(folders, class_titles):
            files[title] = [folder + f for f in os.listdir(folder)]
        train = files

        # Классификация
        self.textEdit.append('Классификация...')
        root_path = self.input_dir + 'test/'
        folders = [root_path + folder + '/' for folder in os.listdir(root_path)]
        class_titles = os.listdir(root_path)
        files = {}
        for folder, title in zip(folders, class_titles):
            files[title] = [folder + f for f in os.listdir(folder)]
        test = files

        pool = createTokenPool(class_titles, train)
        tdict = createDictionary(class_titles, pool)
        rocchio = Rocchio(class_titles, tdict)
        rocchio.train(pool)

        test_pool = createTokenPool(class_titles, test)
        test_lbl_pool = rocchio.predictPool(test_pool)

        log_string = "Роккио:\n"

        for i in class_titles:
            correct = 0
            log_string = log_string + i + '\n'
            for j in range(len(test_lbl_pool[i])):
                str_out = test[i][j] + ' = ' + test_lbl_pool[i][j] + '\n'
                log_string = log_string + str_out
            for val in test_lbl_pool[i]:
                correct = correct + (val == i)
            log_string = log_string + 'Верных классов: ' + i + ': ' + str(correct) + '\n'

        log_string_centr = "Центроиды классов:\n"
        idx = []
        for itm in list(tdict.values()):
            idx.append(itm[idx_lbl])

        for i in range(len(class_titles)):
            log_string_centr += class_titles[i] + '\n'
            log_string_centr += ';'.join([x for (y, x) in sorted(zip(idx, tdict.keys()))]) + '\n'
            log_string_centr += ';'.join(str(x) for x in sum(rocchio.centroids[i].tolist(), [])) + '\n'

        self.textEdit.append("Выходные файлы:")
        out_dir = self.output_dir + 'roc_out/'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        self.textEdit.append(out_dir + 'output_Rocchio.csv')
        writeStringToFile(log_string, out_dir + 'output_Rocchio.csv')

        self.textEdit.append(out_dir + 'Rocchio_centroids.csv')
        writeStringToFile(log_string_centr, out_dir + 'Rocchio_centroids.csv')

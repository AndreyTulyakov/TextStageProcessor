#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re

from PyQt5.QtCore import Qt

from sources.TextPreprocessing import *
from sources.classification.Rocchio import *
from sources.classification.NaiveBayes import *
from sources.classification.KNN import *

from PyQt5.QtWidgets import QDialog, QMessageBox, QTextEdit
from PyQt5 import QtCore, QtGui, uic

import numpy as np
from sources.classification.clsf_util import *
import copy


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

        if(self.radioButtonKNN.isChecked()):
            self.classification_knn(needPreprocessing)


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

        def findCentroid(nparray):
            return (np.sum(nparray, axis=0) / len(nparray))

        ##############PARAMS###################
        output_dir = self.output_dir + 'roc_out/'
        input_dir = self.input_dir
        sep = ";"
        eol = "\n"
        ###############ALGO##################

        fdata, fclass, split = makeFileList(input_dir)
        tfidf, uniq_words = makeTFIDF(fdata[:split], fdata[split:])
        class_titles = set(fclass)

        combiSet = addClassToTFIDF(copy.deepcopy(tfidf), fclass)
        trainSet = combiSet[:split]
        testSet = combiSet[split:]

        centroids = []
        for cl in class_titles:
            cl_array = []
            for i in range(len(trainSet)):
                if fclass[i] == cl:
                    cl_array.append(trainSet[i][:-1])
            centroids.append(findCentroid(np.array(cl_array)).round(3).tolist())

        centroids = addClassToTFIDF(centroids, list(class_titles))
        log_centr = "центроиды" + eol + sep.join(uniq_words) + eol
        for row in centroids:
            log_centr += sep.join(map(str, row)) + eol

        self.textEdit.append("Алгоритм Роккио")
        log_main = "Расстояние до центроидов" + eol
        predictions = []
        for doc in testSet:
            neighbors, dist = getNeighbors(centroids, testSet[0], len(centroids))
            log_main += str(doc) + eol + sep.join([x[0][-1] for x in dist]) + eol + sep.join(
                map(str, [x[1] for x in dist])) + eol
            self.textEdit.append('> результат =' + repr(dist[0][0][-1]) + ', на самом деле=' + repr(doc[-1]))
            predictions.append(dist[0][0][-1])
        accuracy = getAccuracy(testSet, predictions)
        self.textEdit.append('Точность: ' + repr(accuracy) + '%')

        ###############LOGS##################
        log_tfidf = sep.join(uniq_words) + eol
        split_names = makeFileList(input_dir, fread=False)[0]
        for i in range(len(combiSet)):
            row = combiSet[i]
            log_tfidf += sep.join(map(str, row)) + sep + split_names[i] + eol

        self.textEdit.append('Выходные файлы:')

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.textEdit.append(output_dir + 'output_Rocchio.csv')
        writeStringToFile2(log_main, output_dir + 'output_Rocchio.csv')
        self.textEdit.append(output_dir + 'Rocchio_centroids.csv')
        writeStringToFile2(log_centr, output_dir + 'Rocchio_centroids.csv')
        self.textEdit.append(output_dir + 'tfidf_matrix.csv')
        writeStringToFile2(log_tfidf, output_dir + 'tfidf_matrix.csv')


    # Алгоритм KNN
    def classification_knn(self, needPreprocessing):

        ##############PARAMS###################
        output_dir = self.output_dir + 'knn_out/'
        input_dir = self.input_dir
        sep = ";"
        eol = "\n"
        k = 1
        ###############ALGO##################

        fdata, fclass, split = makeFileList(input_dir)
        tfidf, uniq_words = makeTFIDF(fdata[:split], fdata[split:])

        trainingSet = addClassToTFIDF(tfidf[:split], fclass[:split])
        testSet = addClassToTFIDF(tfidf[split:], fclass[split:])

        self.textEdit.append("Алгоритм KNN")
        predictions = []
        log_neighbors = "Соседи и расстояния до них:" + eol
        log_votes = "Голоса соседей:" + eol
        for x in range(len(testSet)):
            neighbors, dist = getNeighbors(trainingSet, testSet[x], k)
            result = getResponse(neighbors)
            log_neighbors += "Документ:;" + str(testSet[x]) + eol
            for p in dist:
                log_neighbors += sep.join(map(str, p)) + eol
            log_votes += "Документ:;" + str(testSet[x]) + eol + str(result).strip("[]") + eol
            predictions.append(result[0][0])
            self.textEdit.append('> результат =' + repr(result[0][0]) + ', на самом деле=' + repr(testSet[x][-1]))
        accuracy = getAccuracy(testSet, predictions)
        self.textEdit.append('Точность: ' + repr(accuracy) + '%')

        ###############LOGS##################
        log_tfidf = sep.join(uniq_words) + eol
        combiSet = trainingSet + testSet
        split_names = makeFileList(input_dir, fread=False)[0]
        for i in range(len(combiSet)):
            row = combiSet[i]
            log_tfidf += sep.join(map(str, row)) + sep + split_names[i] + eol

        self.textEdit.append("Выходные файлы:")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.textEdit.append(output_dir + 'tfidf_matrix.csv')
        writeStringToFile(log_tfidf, output_dir + 'tfidf_matrix.csv')

        self.textEdit.append(output_dir + 'Соседи.csv')
        writeStringToFile(log_neighbors, output_dir + 'Соседи.csv')

        self.textEdit.append(output_dir + 'Голоса.csv')
        writeStringToFile(log_votes, output_dir + 'Голоса.csv')


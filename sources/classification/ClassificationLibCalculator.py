#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import math
import copy
import numpy as np
import shutil
import os

from PyQt5.QtCore import QObject
from PyQt5.QtCore import QThread
from PyQt5.QtCore import pyqtSignal
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.neighbors import KNeighborsClassifier

from sources.TextPreprocessing import writeStringToFile
from sources.classification.clsf_util import makeFileListLib
from sources.utils import makePreprocessingForAllFilesInFolder, clear_dir

class ClassificationLibCalculatorSignals(QObject):
    PrintInfo = pyqtSignal(str)
    Finished = pyqtSignal()
    UpdateProgressBar = pyqtSignal(int)


class ClassificationLibCalculator(QThread):

    def __init__(self, input_dir, output_dir, morph, configurations):
        super().__init__()
        self.input_dir = input_dir + '/'
        self.output_dir = output_dir + '/classification/'

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        clear_dir(self.output_dir)

        self.output_preprocessing_dir = self.output_dir + 'preprocessing/'
        self.first_call = True
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.morph = morph
        self.configurations = configurations
        self.texts = []
        self.categories = dict()
        self.signals = ClassificationLibCalculatorSignals()
        self.need_preprocessing = False

        if len(self.input_dir) > 0 and self.input_dir[-1] == '/':
            self.input_dir = self.input_dir[:-1]
        last_slash_index = self.input_dir.rfind('/')
        self.input_dir_short = ''
        if last_slash_index != -1:
            self.input_dir_short = self.input_dir[last_slash_index + 1:]

    def setMethod(self, method_name, arg_need_preprocessing):
        self.method = method_name
        self.need_preprocessing = arg_need_preprocessing

    def run(self):
        self.signals.UpdateProgressBar.emit(0)

        if not self.need_preprocessing:
            self.signals.PrintInfo.emit("Препроцессинг - пропускается")

        if (not self.first_call) and self.need_preprocessing:
            self.signals.PrintInfo.emit("Препроцессинг - используются предыдущие результаты.")

        # Делаем препроцессинг 1 раз
        if self.first_call and self.need_preprocessing:
            self.signals.PrintInfo.emit("Препроцессинг...")
            makePreprocessingForAllFilesInFolder(self.configurations,
                                                 self.input_dir,
                                                 self.output_preprocessing_dir,
                                                 self.output_dir,
                                                 self.morph)

        self.signals.UpdateProgressBar.emit(30)



        if self.need_preprocessing:
            self.method_input_dir = self.output_preprocessing_dir + self.input_dir_short + '/'
        else:
            self.method_input_dir = self.input_dir

        self.signals.UpdateProgressBar.emit(40)

        if self.method_index == 0:
            self.classification_knn()

        if self.first_call and self.need_preprocessing:
            self.first_call = False
        self.signals.UpdateProgressBar.emit(100)
        self.signals.PrintInfo.emit('Рассчеты закончены!')
        self.signals.Finished.emit()


    # Алгоритм KNN
    def classification_knn(self):

        output_dir = self.output_dir + 'knn_out/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        k = self.configurations.get('classification_knn_k')

        self.signals.PrintInfo.emit("Алгоритм KNN")

        fdata, fclass, split, filenames = makeFileListLib(self.method_input_dir)


        for index in range(len(filenames)):
            if '/' in filenames[index]:
                filenames[index] = filenames[index].split('/')[-1]

        trainingSet = fdata[:split]
        trainingClass = fclass[:split]
        testSet = fdata[split:]
        training_filenames = filenames[:split]
        test_filenames = filenames[split:]

        print('trainingSet:', trainingSet)
        print('trainingClass:', trainingClass)
        print('testSet:', testSet)
        print('filenames:', filenames)
        print('test_filenames:', test_filenames)

        # Переводим документы в векторы
        vectorizer = HashingVectorizer()
        fdata = vectorizer.fit_transform(fdata)
        trainingSet = fdata[:split]
        testSet = fdata[split:]

        # Создаем и тренируем классификатор а затем классифицируем
        classificator = KNeighborsClassifier(n_neighbors=self.knn_n_neighbors)
        classificator.fit(trainingSet, trainingClass)
        results = classificator.predict(testSet)

        self.signals.PrintInfo.emit("Результаты классификации:")
        self.signals.PrintInfo.emit("----------------------------------------------------------------------------")
        for index, result in enumerate(results):
            self.signals.PrintInfo.emit("Файл: " + test_filenames[index])
            self.signals.PrintInfo.emit("Класс: " + str(result))
            self.signals.PrintInfo.emit("----------------------------------------------------------------------------")

        # self.signals.PrintInfo.emit(output_dir + 'tfidf_matrix.csv')
        # writeStringToFile(log_tfidf, output_dir + 'tfidf_matrix.csv')
        #
        # self.signals.PrintInfo.emit(output_dir + 'Соседи.csv')
        # writeStringToFile(log_neighbors, output_dir + 'Соседи.csv')
        #
        # self.signals.PrintInfo.emit(output_dir + 'Голоса.csv')
        # writeStringToFile(log_votes, output_dir + 'Голоса.csv')

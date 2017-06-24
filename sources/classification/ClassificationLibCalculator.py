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
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

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
        #clear_dir(self.output_dir)

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

        self.fdata, self.fclass, self.split, self.filenames = makeFileListLib(self.method_input_dir)

        for index in range(len(self.filenames)):
            if '/' in self.filenames[index]:
                self.filenames[index] = self.filenames[index].split('/')[-1]

        self.trainingSet = self.fdata[:self.split]
        self.trainingClass = self.fclass[:self.split]
        self.testSet = self.fdata[self.split:]
        self.test_filenames = self.filenames[self.split:]

        self.signals.UpdateProgressBar.emit(40)

        if self.method_index == 0:
            self.classification_knn()

        if self.method_index == 1:
            self.classification_linear_svm()

        if self.method_index == 2:
            self.classification_rbf_svm()

        if self.method_index == 3:
            self.classification_gaussian_nb()

        if self.first_call and self.need_preprocessing:
            self.first_call = False
        self.signals.UpdateProgressBar.emit(100)
        self.signals.PrintInfo.emit('Рассчеты закончены!')
        self.signals.Finished.emit()


    def compile_result_string(self, results, proba, classes, filenames):
        result_s = ''
        result_s += "Результаты классификации:\n"
        result_s += "----------------------------------------------------------------------------\n"
        for index, result in enumerate(results):
            result_s += "Файл: " + self.test_filenames[index] + '\n'
            result_s += "Класс: " + str(result) + '\n'
            result_s += "----------------------------------------------------------------------------\n"
        return result_s

    def write_results_to_file(self, output_filename, results, proba, classes, filenames):
        result_s = ''
        result_s += "Результаты классификации:\n"
        result_s += "\n"
        for index, result in enumerate(results):
            result_s += "Файл;" + self.test_filenames[index] + '\n'
            result_s += "Класс;" + str(result) + '\n'
            result_s += "Вероятность;\n"
            for cl_index, cl in enumerate(classes):
                result_s += ";" + str(cl) + ';' + str(proba[index][cl_index]) + "\n"
            result_s += "\n"
        writeStringToFile(result_s, output_filename)


    def classification_knn(self):
        self.signals.PrintInfo.emit("Алгоритм KNN")
        output_dir = self.output_dir + 'knn_out/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Переводим документы в векторы
        # Особенно стоит обратить внимание на тип векторизатора, ведь существуют и другие.
        vectorizer = HashingVectorizer()
        fdata = vectorizer.fit_transform(self.fdata)
        trainingSet = fdata[:self.split]
        testSet = fdata[self.split:]

        # Создаем и тренируем классификатор а затем классифицируем
        classificator = KNeighborsClassifier(n_neighbors=self.knn_n_neighbors)
        classificator.fit(trainingSet, self.trainingClass)
        results = classificator.predict(testSet)
        proba = classificator.predict_proba(testSet)

        self.write_results_to_file(output_dir + 'results.csv', results, proba, classificator.classes_, self.test_filenames)
        out_text = self.compile_result_string(results, proba, classificator.classes_, self.test_filenames)
        self.signals.PrintInfo.emit(out_text)



    def classification_linear_svm(self):
        self.signals.PrintInfo.emit("Алгоритм Linear SVM")
        output_dir = self.output_dir + 'linear_svm_out/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        vectorizer = HashingVectorizer()
        fdata = vectorizer.fit_transform(self.fdata)
        trainingSet = fdata[:self.split]
        testSet = fdata[self.split:]

        classificator = SVC(kernel="linear", probability=True, C=self.linear_svm_c)
        classificator.fit(trainingSet, self.trainingClass)
        results = classificator.predict(testSet)
        proba = classificator.predict_proba(testSet)

        self.write_results_to_file(output_dir + 'results.csv', results, proba, classificator.classes_, self.test_filenames)
        out_text = self.compile_result_string(results, proba, classificator.classes_, self.test_filenames)
        self.signals.PrintInfo.emit(out_text)


    def classification_rbf_svm(self):
        self.signals.PrintInfo.emit("RBF SVM")
        output_dir = self.output_dir + 'rbf_svm_out/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        vectorizer = HashingVectorizer()
        fdata = vectorizer.fit_transform(self.fdata)
        trainingSet = fdata[:self.split]
        testSet = fdata[self.split:]

        classificator = SVC(gamma=2, probability=True, C=self.rbf_svm_c)
        classificator.fit(trainingSet, self.trainingClass)
        results = classificator.predict(testSet)
        proba = classificator.predict_proba(testSet)

        self.write_results_to_file(output_dir + 'results.csv', results, proba, classificator.classes_,self.test_filenames)
        out_text = self.compile_result_string(results, proba, classificator.classes_, self.test_filenames)
        self.signals.PrintInfo.emit(out_text)


    def classification_gaussian_nb(self):
        self.signals.PrintInfo.emit("Gaussian NB")
        output_dir = self.output_dir + 'gaussian_nb_out/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        vectorizer = HashingVectorizer()
        fdata = vectorizer.fit_transform(self.fdata)
        trainingSet = fdata[:self.split]
        testSet = fdata[self.split:]

        classificator = GaussianNB()
        classificator.fit(trainingSet.toarray(), self.trainingClass)
        results = classificator.predict(testSet.toarray())
        proba = classificator.predict_proba(testSet.toarray())

        self.write_results_to_file(output_dir + 'results.csv', results, proba, classificator.classes_,self.test_filenames)
        out_text = self.compile_result_string(results, proba, classificator.classes_, self.test_filenames)
        self.signals.PrintInfo.emit(out_text)
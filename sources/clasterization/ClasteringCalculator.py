#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import math
import copy
import numpy as np
import shutil
import os
import random

from PyQt5.QtCore import QObject
from PyQt5.QtCore import QThread
from PyQt5.QtCore import pyqtSignal

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

from sources.TextPreprocessing import writeStringToFile, makePreprocessing, makeFakePreprocessing, \
    getCompiledFromSentencesText
from sources.utils import makePreprocessingForAllFilesInFolder



# Сигналы для потока вычисления

class ClasterizationCalculatorSignals(QObject):
    PrintInfo = pyqtSignal(str)
    Finished = pyqtSignal()
    UpdateProgressBar = pyqtSignal(int)


# Класс-поток вычисления
class ClasteringCalculator(QThread):

    def __init__(self, filenames, output_dir, morph, configurations, textEdit):
        super().__init__()
        self.filenames = filenames
        self.output_dir = output_dir
        self.morph = morph
        self.configurations = configurations
        self.textEdit = textEdit
        self.texts = []
        self.categories = dict()
        self.signals = ClasterizationCalculatorSignals()
        self.method = '1'
        self.minimalWordsLen = 3
        self.clusterCount = 2
        self.eps = 0.01
        self.m = 2
        self.minPts = 0.3
        self.need_preprocessing = False
        self.first_call = True
        self.texts = []

    def setMethod(self, method_name):
        self.method = method_name

    def setMinimalWordsLen(self, value):
        self.minimalWordsLen = value

    def setEps(self, value):
        self.eps = value

    def setM(self,value):
        self.m = value

    def setMinPts(self,value):
        self.minPts = value

    def setClusterCount(self,value):
        self.clusterCount = value

    def run(self):
        self.signals.UpdateProgressBar.emit(1)

        if self.first_call:
            if self.need_preprocessing:
                self.signals.PrintInfo.emit("Препроцессинг...")
                self.texts = makePreprocessing(self.filenames, self.morph, self.configurations, self.textEdit)
            else:
                self.signals.PrintInfo.emit("Препроцессинг - пропускается")
                self.texts = makeFakePreprocessing(self.filenames)
        else:
            if self.need_preprocessing:
                self.signals.PrintInfo.emit("Препроцессинг - использование предыдущих результатов.")
            else:
                self.signals.PrintInfo.emit("Препроцессинг - пропускается")


        if(True or self.method == '2'):

            input_texts = list()
            for text in self.texts:
                input_texts.append(getCompiledFromSentencesText(text.register_pass_centences))
            short_filenames = [text.filename[text.filename.rfind('/') + 1:] for text in self.texts]

            vectorizer = CountVectorizer()
            X = vectorizer.fit_transform(input_texts)

            svd = TruncatedSVD(2)
            normalizer = Normalizer(copy=False)
            lsa = make_pipeline(svd, normalizer)
            X = lsa.fit_transform(X)

            km = KMeans(n_clusters=self.clusterCount, init='k-means++', max_iter=100, n_init=10)
            km.fit(X)


            predict_result = km.predict(X)

            self.signals.PrintInfo.emit('Прогноз по документам:')
            self.signals.PrintInfo.emit(str(predict_result))

            for i in range(self.clusterCount):
                self.signals.PrintInfo.emit('  Кластер ' + str(i) + ':')
                for predicted_cluster, filename in zip(predict_result, short_filenames):
                    if predicted_cluster == i:
                        self.signals.PrintInfo.emit("    " + str(filename))

            self.signals.PrintInfo.emit('')
            self.signals.PrintInfo.emit('Центры кластеров:')
            for index, cluster_center in enumerate(km.cluster_centers_):
                self.signals.PrintInfo.emit('  ' + str(index)+':' + str(cluster_center))

        if self.first_call and self.need_preprocessing:
            self.first_call = False

        self.signals.PrintInfo.emit('Рассчеты закончены!')
        self.signals.UpdateProgressBar.emit(100)
        self.signals.Finished.emit()


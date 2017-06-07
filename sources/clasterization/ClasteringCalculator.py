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
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt

from sources.TextPreprocessing import writeStringToFile, makePreprocessing, makeFakePreprocessing
from sources.utils import makePreprocessingForAllFilesInFolder



# Сигналы для потока вычисления
from stage_text_processor import stop_words_filename


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
            #self.makeClasterizationKMiddle(self.clusterCount)
            input_texts = list()
            for text in self.texts:
                input_texts.append(self.getCompiledFromSentencesText(text.register_pass_centences))

            russian_stop_words = []
            with open(stop_words_filename) as f:
                russian_stop_words = f.readlines()
            russian_stop_words = [x.strip() for x in russian_stop_words]

            vectorizer = CountVectorizer(min_df=1, stop_words=russian_stop_words)
            X = vectorizer.fit_transform(input_texts)

            km = KMeans(n_clusters=self.clusterCount, init='k-means++', max_iter=100, n_init=1)
            km.fit(X)

            self.signals.PrintInfo.emit("Top terms per cluster:")
            order_centroids = km.cluster_centers_.argsort()[:, ::-1]

            terms = vectorizer.get_feature_names()
            for i in range(self.clusterCount):
                self.signals.PrintInfo.emit(str("Cluster " + str(i) +':'))
                for ind in order_centroids[i, :10]:
                    self.signals.PrintInfo.emit(' ' + str(terms[ind]))

            y_pred = km.predict(X)
            print("y_pred:", y_pred)
            #
            # plt.subplot(111)
            # colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
            # colors = np.hstack([colors] * 20)
            #
            # plt.scatter(X[:, 0], X[:, 1], color=colors[y_pred].tolist(), s=100)
            #
            # if hasattr(km, 'cluster_centers_'):
            #     centers = km.cluster_centers_
            #     center_colors = colors[:len(centers)]
            #     plt.scatter(centers[:, 0], centers[:, 1], s=100, c=center_colors)
            # plt.xlim(-2, 2)
            # plt.ylim(-2, 2)
            # plt.xticks(())
            # plt.yticks(())
            # plt.text(.99, .01, '',
            #          transform=plt.gca().transAxes, size=15,
            #          horizontalalignment='right')
            # plt.show()

        if self.first_call and self.need_preprocessing:
            self.first_call = False

        self.signals.PrintInfo.emit('Рассчеты закончены!')
        self.signals.UpdateProgressBar.emit(100)
        self.signals.Finished.emit()


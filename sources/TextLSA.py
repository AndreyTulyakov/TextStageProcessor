#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import numpy
import math

import sys
import os

import sklearn
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn.cluster import KMeans, MiniBatchKMeans
import pandas as pd
import warnings

from PyQt5.QtCore import Qt
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QDialog, QMessageBox, QTextEdit, QProgressBar, QApplication
from PyQt5 import QtCore, QtGui, uic
from PyQt5.QtCore import QObject
from PyQt5.QtWidgets import QVBoxLayout
from PyQt5.QtWidgets import QWidget
from PyQt5.uic import loadUiType

import matplotlib
import matplotlib.pyplot as plt

from sources.utils import Profiler
from stage_text_processor import stop_words_filename, output_dir

matplotlib.use('Qt5Agg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar)

from sources.apriori_maker import *

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from PyQt5.QtCore import QThread

from sources.TextData import TextData, readFullTextInputText
from sources.TextPreprocessing import *

Ui_DialogPlotter, QDialog = loadUiType('sources/DialogLSAPlot.ui')

class DialogPlotter(QDialog, Ui_DialogPlotter):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        flags = Qt.Window | Qt.WindowSystemMenuHint | Qt.WindowMinMaxButtonsHint | Qt.WindowCloseButtonHint;
        self.setWindowFlags(flags)
        fig = Figure()
        self.addmpl(fig)

    def addfig(self, fig):
        self.rmmpl()
        self.addmpl(fig)

    def addmpl(self, fig):
        self.canvas = FigureCanvas(fig)
        self.mplvl.addWidget(self.canvas)
        self.canvas.draw()
        self.toolbar = NavigationToolbar(self.canvas, self.mplwindow, coordinates=True)
        self.verticalLayout.addWidget(self.toolbar)

    def rmmpl(self, ):
        self.mplvl.removeWidget(self.canvas)
        self.canvas.close()
        self.mplvl.removeWidget(self.toolbar)
        self.toolbar.close()

    def viewLSAGraphics2D(self, plt, xs, ys, filenames):

        plt.figure(1)
        plt.clf()
        plt.subplot(1, 2, 1)
        plt.plot(xs, ys, 'go')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Documents weights')
        plt.grid(True)

        for i in range(len(filenames)):
            plt.annotate(filenames[i], xy=(xs[i], ys[i]), textcoords='data')

        plt.subplot(1, 2, 2)
        ax = plt.gca()
        ax.quiver(0, 0, xs, ys, angles='xy', scale_units='xy', scale=1, linewidth=.01)
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        for i in range(len(filenames)):
            plt.annotate(filenames[i], xy=(xs[i], ys[i]), textcoords='data')
        plt.xlabel('X-component')
        plt.ylabel('Y-component')
        plt.title('Documents vectors')

        self.addfig(plt.gcf())


    def viewDocumentRelationsTable(self, plt, similarity, filenames):

        plt.figure()
        plt.clf()
        plt.title('Documents relation table')
        ax = plt.gca()
        img = plt.imshow(similarity, interpolation='none', cmap="gray")
        plt.xticks(np.arange(len(filenames)), filenames, rotation='vertical')
        plt.yticks(np.arange(len(filenames)), filenames)
        plt.colorbar()
        self.addfig(plt.gcf())


class LsaCalculatorSignals(QObject):
    PrintInfo = pyqtSignal(str)
    UpdateProgressBar = pyqtSignal(int)
    Finished = pyqtSignal(list, list, numpy.ndarray, list)


class LsaCalculator(QThread):

    def __init__(self, filenames, configurations, output_dir, morph, textEdit):
        super().__init__()
        self.filenames = filenames
        self.short_filenames = []
        self.output_dir = output_dir
        self.morph = morph
        self.configurations = configurations
        self.texts = []
        self.categories = dict()
        self.signals = LsaCalculatorSignals()
        self.textEdit = textEdit
        self.input_texts = list()

        for filename in self.filenames:
            self.short_filenames.append(filename[filename.rfind('/')+1:])

        self.output_dir = "output_files/lsa/"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.first_start = True

    def setConfiguration(self, configurations):
        self.configurations = configurations

    def run(self):
        self.texts = []
        self.signals.UpdateProgressBar.emit(0)
        xs = []
        ys = []
        similarity = []


        output_dir = self.configurations.get("output_files_directory", "output_files") + "/preprocessing/"

        need_full_preprocessing = self.configurations.get("need_full_preprocessing", True)
        if self.first_start == True:
            if need_full_preprocessing:
                for filename in self.filenames:
                        text = TextData(filename)
                        text.readSentencesFromInputText()
                        self.texts.append(text)

                self.signals.PrintInfo.emit('Токенизация...')
                self.texts = tokenizeTextData(self.texts, self.configurations)

                self.signals.UpdateProgressBar.emit(10)

                self.signals.PrintInfo.emit('Удаление стоп-слов...')
                self.texts, log_string = removeStopWordsInTexts(self.texts, self.morph, self.configurations)
                writeStringToFile(log_string.replace('\n ', '\n'), output_dir + 'output_stage_1.txt')
                self.signals.UpdateProgressBar.emit(15)

                self.signals.PrintInfo.emit('Приведение к нормальной форме...')
                self.texts, log_string = normalizeTexts(self.texts, self.morph)
                writeStringToFile(log_string.replace('\n ', '\n'), output_dir + 'output_stage_2.txt')
                self.signals.UpdateProgressBar.emit(25)

                self.signals.PrintInfo.emit('Приведение регистра...')
                self.texts, log_string = fixRegisterInTexts(self.texts, self.morph)
                writeStringToFile(log_string.replace('\n ', '\n'), output_dir + 'output_stage_3.txt')
                self.signals.UpdateProgressBar.emit(30)

                if self.configurations.get("need_apriori", False):
                    self.signals.PrintInfo.emit('Рассчет Apriori...')
                    makeAprioriForTexts(self.texts, output_dir)

                self.signals.PrintInfo.emit('...')
                for text in self.texts:
                    self.input_texts.append(getCompiledFromSentencesText(text.register_pass_centences))
            else:
                for filename in self.filenames:
                    self.input_texts.append(readFullTextInputText(filename))
        else:
            self.signals.PrintInfo.emit('Использование предыдущих результатов предварительной обработки')

        self.signals.UpdateProgressBar.emit(40)

        self.first_start = False

        if len(self.input_texts) < 3:
            self.signals.PrintInfo.emit('Недостаточно документов для корректного анализа!')
        else:
            # Добавим русские стоп-слова
            russian_stop_words = []
            with open(stop_words_filename) as f:
                russian_stop_words = f.readlines()
            russian_stop_words = [x.strip() for x in russian_stop_words]

            self.signals.UpdateProgressBar.emit(45)

            vectorizer = CountVectorizer(min_df=1, stop_words=russian_stop_words)
            dtm = vectorizer.fit_transform(self.input_texts)

            pre_svd_matrix = pd.DataFrame(dtm.toarray(), index=self.short_filenames,
                                          columns=vectorizer.get_feature_names()).head(10)
            pre_svd_matrix_filename = self.output_dir + 'pre_svd_matrix.csv'
            pre_svd_matrix.to_csv(pre_svd_matrix_filename, sep=";", decimal=',')
            self.signals.PrintInfo.emit('Файл с матрицей [слова * документы] для ЛСА:' + pre_svd_matrix_filename)
            features_count = len(vectorizer.get_feature_names())
            self.signals.PrintInfo.emit('Уникальных слов:' + str(features_count))

            self.signals.UpdateProgressBar.emit(50)

            max_component = min(len(self.input_texts), features_count)

            # Производим ЛСА и сжимаем пространство до 2-мерного
            if max_component <= self.lsa_components_count:
                self.signals.PrintInfo.emit('Внимание! Число компонент уменьшено с ' + str(self.lsa_components_count) + ' до ' + str(max_component - 1))
                self.lsa_components_count = max_component - 1


            lsa = TruncatedSVD(self.lsa_components_count, algorithm='arpack')
            dtm_lsa = lsa.fit_transform(dtm)
            dtm_lsa = Normalizer(copy=False).fit_transform(dtm_lsa)
            self.signals.UpdateProgressBar.emit(70)

            xs = [w[0] for w in dtm_lsa]
            ys = [w[1] for w in dtm_lsa]

            columns = ['Filename']
            if len(dtm_lsa) > 0:
                for column_index in range(len(dtm_lsa[0])):
                    columns.append('Component_' + str(column_index+1))

            docs_weight_df = pd.DataFrame(columns=columns, index=None)
            docs_weight_df[columns[0]] = self.short_filenames
            for column_index in range(1, len(columns)):
                docs_weight_df[columns[column_index]] = [w[column_index-1] for w in dtm_lsa]
            documents_weight_filename = self.output_dir + 'documents_weight.csv'
            docs_weight_df.to_csv(documents_weight_filename, sep=";", decimal=',')
            self.signals.PrintInfo.emit('Файл с весами документов:' + documents_weight_filename)

            self.signals.UpdateProgressBar.emit(90)

            # Вычислим таблицу соответствия докуметов
            similarity = np.asarray(numpy.asmatrix(dtm_lsa) * numpy.asmatrix(dtm_lsa).T)
            relationsTable = pd.DataFrame(similarity, index=self.short_filenames, columns=self.short_filenames).head(len(self.short_filenames))

            relation_table_filename = self.output_dir + 'document_relation_table.csv'
            relationsTable.to_csv(relation_table_filename, sep=";", decimal=',')
            self.signals.PrintInfo.emit('Файл с таблицей отношений документов:' + relation_table_filename)

        self.signals.PrintInfo.emit('Рассчеты закончены!')
        self.signals.UpdateProgressBar.emit(100)
        self.signals.Finished.emit(xs, ys, similarity, self.short_filenames)


class DialogConfigLSA(QDialog):

    def __init__(self, filenames, morph, configurations, parent):
        super().__init__()
        uic.loadUi('sources/DialogConfigLSA.ui', self)

        self.morph = morph
        self.configurations = configurations
        self.parent = parent

        self.xs = []
        self.ys = []
        self.short_filenames = []
        self.similarity = None
        self.profiler = Profiler()

        flags = Qt.Window | Qt.WindowSystemMenuHint | Qt.WindowMinimizeButtonHint | Qt.WindowCloseButtonHint;
        self.setWindowFlags(flags)

        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

        self.buttonMakeLSA.clicked.connect(self.makeLSA)
        self.button2DView.clicked.connect(self.make2DView)
        self.buttonRelationTable.clicked.connect(self.showRelationTable)

        self.radio_preprocessing_full.toggled.connect(self.onChangePreprocMethod)
        self.radio_preprocessing_stopwords.toggled.connect(self.onChangePreprocMethod)

        output_dir = self.configurations.get("output_files_directory", "output_files")

        self.calculator = LsaCalculator(filenames,  self.configurations, output_dir, morph, self.textEdit)
        self.calculator.signals.Finished.connect(self.onCalculationFinish)
        self.calculator.signals.UpdateProgressBar.connect(self.onUpdateProgressBar)
        self.calculator.signals.PrintInfo.connect(self.onTextLogAdd)

        self.textEdit.setText("")

    def onChangePreprocMethod(self):
        if self.radio_preprocessing_full.isChecked():
            self.groupBoxFullPreprocessingPanel.setVisible(True)
        else:
            self.groupBoxFullPreprocessingPanel.setVisible(False)

    def onTextLogAdd(self, QString):
        self.textEdit.append(QString + '\n')
        self.repaint()

    def onUpdateProgressBar(self, value):
        self.progressBar.setValue(value)
        self.repaint()

    def onCalculationFinish(self, xs, ys, similarity, short_filenames):
        self.textEdit.append('Выполнено за ' + self.profiler.stop() + ' с.')
        self.xs = xs
        self.ys = ys
        self.similarity = similarity
        self.short_filenames = short_filenames
        QApplication.restoreOverrideCursor()
        self.button2DView.setEnabled(True)
        self.buttonRelationTable.setEnabled(True)
        self.buttonMakeLSA.setEnabled(True)
        QMessageBox.information(self, "Внимание", "Латентно-семантический анализ завершен!")

    def makeLSA(self):
        self.calculator.setConfiguration(self.configurations)
        self.calculator.lsa_components_count = self.lsa_components_count.value()
        self.buttonMakeLSA.setEnabled(False)
        self.button2DView.setEnabled(False)
        self.buttonRelationTable.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.textEdit.setText("")
        self.configurations['need_full_preprocessing'] = self.radio_preprocessing_full.isChecked()
        self.configurations["minimal_word_size"] = self.spinBoxMinimalWordsLen.value()
        self.configurations["cut_ADJ"] = self.checkBoxPrilag.isChecked()
        self.configurations["need_apriori"] = self.checkBoxNeedApriori.isChecked()
        self.radio_preprocessing_full.setEnabled(False)
        self.radio_preprocessing_stopwords.setEnabled(False)
        self.groupBoxFullPreprocessingPanel.setEnabled(False)
        self.profiler.start()
        self.calculator.start()

    def make2DView(self):
        plot_dialog = DialogPlotter()
        plot_dialog.viewLSAGraphics2D(plt, self.xs, self.ys, self.short_filenames)
        plot_dialog.exec_()

    def showRelationTable(self):
        plot_dialog = DialogPlotter()
        plot_dialog.viewDocumentRelationsTable(plt, self.similarity, self.short_filenames)
        plot_dialog.exec_()

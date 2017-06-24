#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pymorphy2
import math

from PyQt5.QtCore import QObject
from PyQt5.QtCore import QThread
from PyQt5.QtCore import Qt
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QApplication
from matplotlib.figure import Figure
from pymorphy2 import tokenizers
import os
import random

from sklearn.cluster import DBSCAN
from PyQt5.QtWidgets import QDialog, QMessageBox, QTextEdit
from PyQt5 import QtCore, QtGui, uic
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
matplotlib.use('Qt5Agg')


from sources.TextData import TextData
from sources.TextPreprocessing import *
from sources.clasterization.ClasteringCalculator import ClasteringCalculator
from sources.clasterization.ClasterizationCalculator import ClasterizationCalculator
from sources.utils import Profiler


class DialogClastering(QDialog):

    def __init__(self, filenames, morph, configurations, parent):
        super().__init__()
        uic.loadUi('sources/DialogClastering.ui', self)

        flags = Qt.Window | Qt.WindowSystemMenuHint | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint
        self.setWindowFlags(flags)

        fig = Figure()
        self.addmpl(fig)

        self.filenames = filenames
        self.morph = morph
        self.configurations = configurations
        self.parent = parent

        output_dir = self.configurations.get("output_files_directory", "output_files")

        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.startMethod.clicked.connect(self.OnStartMethod)

        self.textEdit.setText("")
        self.progressBar.setValue(0)

        self.profiler = Profiler()

        self.calculator = ClasteringCalculator(filenames, output_dir, morph, self.configurations, self.textEdit)
        self.calculator.signals.Finished.connect(self.onCalculationFinish)
        self.calculator.signals.UpdateProgressBar.connect(self.onUpdateProgressBar)
        self.calculator.signals.PrintInfo.connect(self.onTextLogAdd)


    def onTextLogAdd(self, QString):
        self.textEdit.append(QString)
        self.repaint()

    def onUpdateProgressBar(self, value):
        self.progressBar.setValue(value)
        self.repaint()

    def onCalculationFinish(self):
        self.tabWidget.setEnabled(True)
        self.textEdit.append('Выполнено за ' + self.profiler.stop() + ' с.')
        QApplication.restoreOverrideCursor()
        self.addfig(plt.gcf())
        self.startMethod.setEnabled(True)
        self.checkBoxNeedCalculateTFIDF.setEnabled(True)
        QMessageBox.information(self, "Внимание", "Кластеризация завершена!")

    def OnStartMethod(self):
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.startMethod.setEnabled(False)
        self.tabWidget.setEnabled(False)

        self.textEdit.setText("")
        plt.cla()
        plt.clf()

        self.calculator.set_method_index(self.tabWidget.currentIndex())

        self.calculator.need_preprocessing = self.checkBoxNeedPreprocessing.isChecked()
        self.checkBoxNeedPreprocessing.setEnabled(False)

        self.calculator.need_tf_idf = self.checkBoxNeedCalculateTFIDF.isChecked()
        self.checkBoxNeedCalculateTFIDF.setEnabled(False)

        # Передает параметры с формы в процесс
        self.calculator.kmeans_cluster_count = self.kmeans_cluster_count.value()

        self.calculator.dbscan_min_pts = self.dbscan_min_pts.value()
        self.calculator.dbscan_eps = self.dbscan_eps.value()

        self.calculator.ward_clusters_count = self.ward_clusters_count.value()

        self.calculator.spectral_clusters_count = self.spectral_clusters_count.value()

        self.calculator.aa_damping = self.aa_damping.value()
        self.calculator.aa_max_iter = self.aa_max_iter.value()
        self.calculator.aa_no_change_stop = self.aa_no_change_stop.value()

        self.calculator.mean_shift_quantile = self.mean_shift_quantile.value()

        self.calculator.birch_threshold = self.birch_threshold.value()
        self.calculator.birch_branching_factor = self.birch_branching_factor.value()
        self.calculator.birch_clusters_count = self.birch_clusters_count.value()

        self.profiler.start()
        self.calculator.start()


    # Команды для построения графиков
    def addfig(self, fig):
        self.rmmpl()
        self.addmpl(fig)

    def addmpl(self, fig):
        self.canvas = FigureCanvas(fig)
        self.mplvl.addWidget(self.canvas)
        self.canvas.draw()
        self.toolbar = NavigationToolbar(self.canvas, self.mplwindow, coordinates=True)
        self.GraphicsLayout.addWidget(self.toolbar)

    def rmmpl(self, ):
        self.mplvl.removeWidget(self.canvas)
        self.canvas.close()
        self.mplvl.removeWidget(self.toolbar)
        self.toolbar.close()

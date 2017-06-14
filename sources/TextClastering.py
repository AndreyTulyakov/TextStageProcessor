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

        self.onChangeMethod()

        self.radioButton_KMiddle.toggled.connect(self.onChangeMethod)
        self.radioButton_DBSCAN.toggled.connect(self.onChangeMethod)
        self.radioButton_Ward.toggled.connect(self.onChangeMethod)

    def onChangeMethod(self):
        if (self.radioButton_KMiddle.isChecked()):
            self.calculator.setMethod('1')
            self.parameters_KMEANS.setVisible(True)
        else:
            self.parameters_KMEANS.setVisible(False)

        if(self.radioButton_DBSCAN.isChecked()):
            self.calculator.setMethod('2')
            self.parameters_DBSCAN.setVisible(True)
        else:
            self.parameters_DBSCAN.setVisible(False)

        if(self.radioButton_Ward.isChecked()):
            self.calculator.setMethod('3')
            self.parameters_Ward.setVisible(True)
        else:
            self.parameters_Ward.setVisible(False)


    def onTextLogAdd(self, QString):
        self.textEdit.append(QString)
        self.repaint()

    def onUpdateProgressBar(self, value):
        self.progressBar.setValue(value)
        self.repaint()

    def onCalculationFinish(self):
        self.methods.setEnabled(True)
        self.textEdit.append('Выполнено за ' + self.profiler.stop() + ' с.')
        QApplication.restoreOverrideCursor()
        self.addfig(plt.gcf())
        self.startMethod.setEnabled(True)
        QMessageBox.information(self, "Внимание", "Кластеризация завершена!")

    def OnStartMethod(self):
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.startMethod.setEnabled(False)

        self.textEdit.setText("")
        plt.cla()
        plt.clf()

        self.calculator.need_preprocessing = self.checkBoxNeedPreprocessing.isChecked()
        self.checkBoxNeedPreprocessing.setEnabled(False)

        self.calculator.setClusterCount(self.spinBox.value())


        self.calculator.setEps(self.doubleSpinBox_dbscan_eps.value())
        self.calculator.setMinPts(self.spinBox_dbscan_min_pts.value())

        self.calculator.ward_parameter_eps = self.spinbox_ward_eps.value()
        self.calculator.ward_parameter_clusters_count = self.spinbox_ward_clusters_count.value()

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

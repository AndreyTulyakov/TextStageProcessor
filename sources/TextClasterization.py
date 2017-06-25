#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pymorphy2
import math

from PyQt5.QtCore import QObject
from PyQt5.QtCore import QThread
from PyQt5.QtCore import Qt
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QApplication
from PyQt5.uic import loadUiType
from pymorphy2 import tokenizers
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
import os
import random

from PyQt5.QtWidgets import QDialog, QMessageBox, QTextEdit
from PyQt5 import QtCore, QtGui, uic


from sources.TextData import TextData
from sources.TextPreprocessing import *
from sources.clasterization.ClasterizationCalculator import ClasterizationCalculator
from sources.utils import Profiler

from .TextLSA import DialogPlotter

Ui_DialogPlotter, QDialog = loadUiType('sources/DialogLSAPlot.ui')

class DialogPlotterSOM(DialogPlotter):
    def viewSOMDiagram(self, plt, somMap, somDLocations):
        fig, ax = plt.subplots()
        ax.imshow(somMap, cmap=plt.cm.gray, interpolation='nearest')
        ax.set_title('SOM Map')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        for index, (x,y) in enumerate(somDLocations):
            ax.text(x, y, 'd{0}'.format(index + 1),
            verticalalignment='center',
            horizontalalignment='center')

        ax.spines['left'].set_position(('outward', len(somMap)))
        ax.spines['bottom'].set_position(('outward', len(somMap)))
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

        self.addfig(plt.gcf())

class DialogConfigClasterization(QDialog):

    def __init__(self, filenames, morph, configurations, parent):
        super().__init__()
        uic.loadUi('sources/DialogConfigClasterization.ui', self)

        flags = Qt.Window | Qt.WindowSystemMenuHint | Qt.WindowMinimizeButtonHint | Qt.WindowCloseButtonHint
        self.setWindowFlags(flags)

        self.filenames = filenames
        self.morph = morph
        self.configurations = configurations
        self.parent = parent

        self.somMap = []
        self.somDLocations = []

        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.startMethod.clicked.connect(self.OnStartMethod)
        self.textEdit.setText("")
        self.parameters.setVisible(False)
        self.parameters_DBSCAN.setVisible(False)
        self.parameters_SOM.setVisible(False)
        output_dir = self.configurations.get("output_files_directory", "output_files")
        self.progressBar.setValue(0)
        self.profiler = Profiler()

        self.calculator = ClasterizationCalculator(filenames, output_dir, morph, self.configurations, self.textEdit)
        self.calculator.setMethod('1')
        self.calculator.signals.Finished.connect(self.onCalculationFinish)
        self.calculator.signals.UpdateProgressBar.connect(self.onUpdateProgressBar)
        self.calculator.signals.PrintInfo.connect(self.onTextLogAdd)

        self.radioButton_Hierarhy.toggled.connect(self.onChangeMethod)
        self.radioButton_KMiddle.toggled.connect(self.onChangeMethod)
        self.radioButton_SMiddle.toggled.connect(self.onChangeMethod)
        self.radioButton_DBSCAN.toggled.connect(self.onChangeMethod)
        self.radioButton_C3M.toggled.connect(self.onChangeMethod)
        self.radioButton_SOM.toggled.connect(self.onChangeMethod)
        self.drawSOMDiagram.clicked.connect(self.onDrawSOMDiagram)

    def onChangeMethod(self):
        self.parameters.setVisible(False)
        self.parameters_DBSCAN.setVisible(False)
        self.parameters_SOM.setVisible(False)

        if(self.radioButton_DBSCAN.isChecked()):
            self.parameters_DBSCAN.setVisible(True)

        if (self.radioButton_KMiddle.isChecked() or self.radioButton_SMiddle.isChecked()):
            self.parameters.setVisible(True)

        if (self.radioButton_SOM.isChecked()):
            self.parameters_SOM.setVisible(True)




        if (self.radioButton_Hierarhy.isChecked()):
            self.calculator.setMethod('1')
        if(self.radioButton_KMiddle.isChecked()):
            self.calculator.setMethod('2')
        if (self.radioButton_SMiddle.isChecked()):
            self.calculator.setMethod('3')
        if (self.radioButton_DBSCAN.isChecked()):
            self.calculator.setMethod('4')
        if (self.radioButton_C3M.isChecked()):
            self.calculator.setMethod('5')
        if (self.radioButton_SOM.isChecked()):
            self.calculator.setMethod('6')

    def onTextLogAdd(self, QString):
        self.textEdit.append(QString + '\n')
        self.repaint()

    def onUpdateProgressBar(self, value):
        self.progressBar.setValue(value)
        self.repaint()

    def onCalculationFinish(self, somMap=None, somDLocations=None):
        self.methods.setEnabled(True)
        self.parameters.setEnabled(True)
        self.textEdit.append('Выполнено за ' + self.profiler.stop() + ' с.')
        QApplication.restoreOverrideCursor()
        QMessageBox.information(self, "Внимание", "Кластеризация завершена!")
        if somMap and somDLocations:
            self.somMap = somMap
            self.somDLocations = somDLocations
            self.drawSOMDiagram.setEnabled(True)

    def OnStartMethod(self):
        QApplication.setOverrideCursor(Qt.WaitCursor)

        self.textEdit.setText("")

        self.calculator.need_preprocessing = self.checkBoxNeedPreprocessing.isChecked()
        self.checkBoxNeedPreprocessing.setEnabled(False)
        self.calculator.setClusterCount(self.spinBox.value())
        if(self.radioButton_DBSCAN.isChecked()):
            self.calculator.setEps(self.lineEdit_4.text())
        else:
            self.calculator.setEps(self.lineEdit.text())
        self.calculator.setM(self.lineEdit_2.text())
        self.calculator.setMinPts(self.lineEdit_3.text())
        self.calculator.som_length = self.spinBox_SOM_length.value()
        self.profiler.start()
        self.calculator.start()

    def onDrawSOMDiagram(self):
        plotter = DialogPlotterSOM()
        plotter.viewSOMDiagram(plt, self.somMap, self.somDLocations)
        plotter.exec_()

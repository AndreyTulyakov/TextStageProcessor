#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pymorphy2
import math

from PyQt5.QtCore import QObject
from PyQt5.QtCore import QThread
from PyQt5.QtCore import Qt
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QApplication
from pymorphy2 import tokenizers
import os
import random

from PyQt5.QtWidgets import QDialog, QMessageBox, QTextEdit
from PyQt5 import QtCore, QtGui, uic


from sources.TextData import TextData
from sources.TextPreprocessing import *
from sources.clasterization.ClasteringCalculator import ClasteringCalculator
from sources.clasterization.ClasterizationCalculator import ClasterizationCalculator
from sources.utils import Profiler


class DialogClastering(QDialog):

    def __init__(self, filenames, morph, configurations, parent):
        super().__init__()
        uic.loadUi('sources/DialogConfigClasterization.ui', self)

        flags = Qt.Window | Qt.WindowSystemMenuHint | Qt.WindowMinimizeButtonHint | Qt.WindowCloseButtonHint
        self.setWindowFlags(flags)

        self.filenames = filenames
        self.morph = morph
        self.configurations = configurations
        self.parent = parent

        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.startMethod.clicked.connect(self.OnStartMethod)
        self.textEdit.setText("")
        self.parameters.setVisible(False)
        self.parameters_DBSCAN.setVisible(False)
        output_dir = self.configurations.get("output_files_directory", "output_files")
        self.progressBar.setValue(0)
        self.profiler = Profiler()

        self.calculator = ClasteringCalculator(filenames, output_dir, morph, self.configurations, self.textEdit)
        self.calculator.setMethod('1')
        self.calculator.signals.Finished.connect(self.onCalculationFinish)
        self.calculator.signals.UpdateProgressBar.connect(self.onUpdateProgressBar)
        self.calculator.signals.PrintInfo.connect(self.onTextLogAdd)

        self.radioButton_Hierarhy.toggled.connect(self.onChangeMethod)
        self.radioButton_KMiddle.toggled.connect(self.onChangeMethod)
        self.radioButton_SMiddle.toggled.connect(self.onChangeMethod)
        self.radioButton_DBSCAN.toggled.connect(self.onChangeMethod)

    def onChangeMethod(self):
        if (self.radioButton_Hierarhy.isChecked()):
            self.parameters.setVisible(False)
            self.parameters_DBSCAN.setVisible(False)
        else:
            if(self.radioButton_DBSCAN.isChecked()):
                self.parameters.setVisible(False)
                self.parameters_DBSCAN.setVisible(True)
            else:
                self.parameters_DBSCAN.setVisible(False)
                self.parameters.setVisible(True)

        if (self.radioButton_Hierarhy.isChecked()):
            self.calculator.setMethod('1')
        if(self.radioButton_KMiddle.isChecked()):
            self.calculator.setMethod('2')
        if (self.radioButton_SMiddle.isChecked()):
            self.calculator.setMethod('3')
        if (self.radioButton_DBSCAN.isChecked()):
            self.calculator.setMethod('4')

    def onTextLogAdd(self, QString):
        self.textEdit.append(QString + '\n')
        self.repaint()

    def onUpdateProgressBar(self, value):
        self.progressBar.setValue(value)
        self.repaint()

    def onCalculationFinish(self):
        self.methods.setEnabled(True)
        self.parameters.setEnabled(True)
        self.textEdit.append('Выполнено за ' + self.profiler.stop() + ' с.')
        QApplication.restoreOverrideCursor()
        QMessageBox.information(self, "Внимание", "Кластеризация завершена!")

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
        self.profiler.start()
        self.calculator.start()

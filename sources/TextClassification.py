#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

from sources.classification.ClassificationCalculator import *

from PyQt5.QtWidgets import QDialog, QMessageBox
from PyQt5 import QtCore, uic
from sources.classification.clsf_util import *
from sources.utils import Profiler


class DialogConfigClassification(QDialog):

    def __init__(self, dirname, morph, configurations, parent):
        super().__init__()
        uic.loadUi('sources/DialogConfigClassification.ui', self)

        flags = Qt.Window | Qt.WindowSystemMenuHint | Qt.WindowMinimizeButtonHint | Qt.WindowCloseButtonHint
        self.setWindowFlags(flags)

        self.morph = morph
        self.configurations = configurations
        self.parent = parent
        self.input_dir = dirname
        self.lineEditInputDir.setText(dirname)

        self.profiler = Profiler()

        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.buttonClassify.clicked.connect(self.makeClassification)
        self.textEdit.setText("")
        self.groupBox_KNN.setVisible(False)

        self.radioButtonNaiveBayes.toggled.connect(self.onChangeMethod)
        self.radioButtonRocchio.toggled.connect(self.onChangeMethod)
        self.radioButtonKNN.toggled.connect(self.onChangeMethod)
        self.radioButtonLLSF.toggled.connect(self.onChangeMethod)
        self.radioButtonID3.toggled.connect(self.onChangeMethod)

        output_dir = self.configurations.get("output_files_directory", "output_files")

        self.calculator = ClassificationCalculator(self.input_dir, output_dir, morph, self.configurations)
        self.calculator.signals.Finished.connect(self.onCalculationFinish)
        self.calculator.signals.UpdateProgressBar.connect(self.onUpdateProgressBar)
        self.calculator.signals.PrintInfo.connect(self.onTextLogAdd)

        self.output_dir = configurations.get("output_files_directory", "output_files/classification") + "/"



    def onChangeMethod(self):
        if(self.radioButtonKNN.isChecked()):
            self.groupBox_KNN.setVisible(True)
        else:
            self.groupBox_KNN.setVisible(False)

    def onTextLogAdd(self, QString):
        self.textEdit.append(QString + '\n')
        self.repaint()

    def onUpdateProgressBar(self, value):
        self.progressBar.setValue(value)
        self.repaint()

    def onCalculationFinish(self):
        # self.groupButtonsBox.setEnabled(True)
        self.textEdit.append('Выполнено за ' + self.profiler.stop() + ' с.')
        QApplication.restoreOverrideCursor()
        QMessageBox.information(self, "Внимание", "Процесс классификации завершен!")

    # Вызов при нажатии на кнопку "Классифицировать"
    def makeClassification(self):
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.textEdit.setText("")
        self.checkBoxNeedPreprocessing.setEnabled(False)
        need_preprocessing = self.checkBoxNeedPreprocessing.isChecked()
        self.configurations['classification_knn_k'] = self.spinBox_KNN_K.value()

        if self.radioButtonNaiveBayes.isChecked():
            self.calculator.setMethod(ClassificationCalculator.METHOD_NAIVE_BAYES, need_preprocessing)

        if self.radioButtonRocchio.isChecked():
            self.calculator.setMethod(ClassificationCalculator.METHOD_ROCCHIO, need_preprocessing)

        if self.radioButtonKNN.isChecked():
            self.calculator.setMethod(ClassificationCalculator.METHOD_KNN, need_preprocessing)

        if self.radioButtonLLSF.isChecked():
            self.calculator.setMethod(ClassificationCalculator.METHOD_LLSF, need_preprocessing)

        if self.radioButtonID3.isChecked():
            self.calculator.setMethod(ClassificationCalculator.METHOD_ID3, need_preprocessing)

        self.profiler.start()
        self.calculator.start()

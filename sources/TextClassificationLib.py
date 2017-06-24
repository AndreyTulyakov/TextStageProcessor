#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QDialog, QMessageBox
from PyQt5 import QtCore, uic

from sources.classification.ClassificationLibCalculator import ClassificationLibCalculator
from sources.utils import Profiler


class DialogClassificationLib(QDialog):

    def __init__(self, dirname, morph, configurations, parent):
        super().__init__()
        uic.loadUi('sources/DialogClassificationLib.ui', self)

        flags = Qt.Window | Qt.WindowSystemMenuHint | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint
        self.setWindowFlags(flags)

        self.morph = morph
        self.configurations = configurations
        self.parent = parent
        self.input_dir = dirname
        self.lineEditInputDir.setText(dirname)

        self.profiler = Profiler()

        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.buttonClassify.clicked.connect(self.makeClassification)

        output_dir = self.configurations.get("output_files_directory", "output_files")

        self.calculator = ClassificationLibCalculator(self.input_dir, output_dir, morph, self.configurations)
        self.calculator.signals.Finished.connect(self.onCalculationFinish)
        self.calculator.signals.UpdateProgressBar.connect(self.onUpdateProgressBar)
        self.calculator.signals.PrintInfo.connect(self.onTextLogAdd)
        self.output_dir = configurations.get("output_files_directory", "output_files/classification") + "/"


    def onTextLogAdd(self, QString):
        self.textEdit.append(QString)
        self.repaint()

    def onUpdateProgressBar(self, value):
        self.progressBar.setValue(value)
        self.repaint()

    def onCalculationFinish(self):
        self.textEdit.append('Выполнено за ' + self.profiler.stop() + ' с.')
        QApplication.restoreOverrideCursor()
        QMessageBox.information(self, "Внимание", "Процесс классификации завершен!")


    # Вызов при нажатии на кнопку "Классифицировать"
    def makeClassification(self):
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.textEdit.setText("")
        self.checkBoxNeedPreprocessing.setEnabled(False)

        self.calculator.knn_n_neighbors = self.knn_n_neighbors.value()
        self.calculator.linear_svm_c = self.linear_svm_c.value()
        self.calculator.rbf_svm_c = self.rbf_svm_c.value()

        self.calculator.need_preprocessing = self.checkBoxNeedPreprocessing.isChecked()
        self.calculator.method_index = self.tabWidget.currentIndex()
        self.profiler.start()
        self.calculator.start()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pymorphy2

from matplotlib import rc

import sys
from PyQt5.QtWidgets import QMainWindow, QMessageBox, QSizePolicy, QAction, qApp, QSpacerItem, QApplication, QWidget, QFileDialog, QDialog, QPushButton, QHBoxLayout, QVBoxLayout
from PyQt5.QtGui import QIcon
from PyQt5 import QtCore


from sources.TextData import TextData
from sources.TextPreprocessing import *
from sources.TextClasterization import *
from sources.TextClassification import *
from sources.TextLSA import *


# Для корректного отображение шрифтов на графиках в Windows
if(os.name != 'posix'):
    font = {'family': 'Verdana',
            'weight': 'normal'}
    rc('font', **font)


configurations = readConfigurationFile("configuration.cfg")
output_dir = configurations.get("output_files_directory", "output_files") + "/"

# Получаем экземпляр анализатора (10-20мб)
morph = pymorphy2.MorphAnalyzer()

# Класс главного окна
class MainWindow(QMainWindow):
    
    def __init__(self):
        super().__init__()
        self.initUI()
        self.texts = []


    def initUI(self):
        button_clasterization = QPushButton("Кластеризация")
        button_clasterization.setMinimumHeight(32)
        button_clasterization.clicked.connect(self.clasterization)

        button_classification = QPushButton("Классификация")
        button_classification.setMinimumHeight(32)
        button_classification.clicked.connect(self.classification)

        button_lsa = QPushButton("Латентно-семантический анализ")
        button_lsa.setMinimumHeight(32)
        button_lsa.clicked.connect(self.makeLSA)

        spacer = QSpacerItem(20,40,QSizePolicy.Minimum,QSizePolicy.Expanding)

        vbox = QVBoxLayout()
        vbox.addWidget(button_clasterization)
        vbox.addWidget(button_classification)
        vbox.addWidget(button_lsa)
        vbox.addItem(spacer)

        widget = QWidget();
        widget.setLayout(vbox);
        self.setCentralWidget(widget);
        self.setGeometry(300, 300, 480, 320)
        self.setWindowTitle('Этапный текстовый процессор')    
        self.show()


    def getFilenamesFromUserSelection(self):
        filenames, _ = QFileDialog.getOpenFileNames(self, "Открыть файлы для анализа", "", "Text Files (*.txt)", None)
        if(len(filenames) > 0):
            return filenames
        else:
            return None

    def clasterization(self):
        print("Кластеризация")
        filenames = self.getFilenamesFromUserSelection()
        if(filenames != None):
            dialogConfigClasterization = DialogConfigClasterization(filenames, morph, configurations, self)
            self.hide()
            dialogConfigClasterization.destroyed.connect(self.show)
            dialogConfigClasterization.exec_()

    def classification(self):
        print("Классификация")
        filenames = self.getFilenamesFromUserSelection()
        if(filenames != None):
            dialogConfigClassification = DialogConfigClassification(filenames, morph, configurations, self)
            self.hide()
            dialogConfigClassification.destroyed.connect(self.show)
            dialogConfigClassification.exec_()

    def makeLSA(self):
        print("LSA")
        filenames = self.getFilenamesFromUserSelection()
        if(filenames != None):
            dialogConfigLSA = DialogConfigLSA(filenames, morph, configurations, self)
            self.hide()
            dialogConfigLSA.destroyed.connect(self.show)
            dialogConfigLSA.exec_()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MainWindow()
    sys.exit(app.exec_())

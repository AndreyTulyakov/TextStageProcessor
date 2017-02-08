#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pymorphy2

from matplotlib import rc

import sys
from PyQt5.QtWidgets import QMainWindow, QMessageBox, QSizePolicy, QAction, qApp, QSpacerItem, QApplication, QWidget, QFileDialog, QDialog, QPushButton, QHBoxLayout, QVBoxLayout
from PyQt5.QtGui import QIcon
from PyQt5 import QtCore


from sources.TextData import TextData
from sources.XiSquare import *
from sources.TextClasterization import *
from sources.TextClassification import *
from sources.TextLSA import *
from sources.TextDecomposeAndRuleApply import *


# Для корректного отображение шрифтов на графиках в Windows
if(os.name != 'posix'):
    font = {'family': 'Verdana',
            'weight': 'normal'}
    rc('font', **font)


configurations = readConfigurationFile("configuration.cfg")
output_dir = configurations.get("output_files_directory", "output_files") + "/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

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

        button_analyze_and_rule_apply = QPushButton("Анализ и правила вывода предложений")
        button_analyze_and_rule_apply.setMinimumHeight(32)
        button_analyze_and_rule_apply.clicked.connect(self.analyze_and_rule_apply)

        button_xi_square = QPushButton("Критерий Хи-Квадрат")
        button_xi_square.setMinimumHeight(32)
        button_xi_square.clicked.connect(self.makeXiSquare)
        button_xi_square.setEnabled(False)


        spacer = QSpacerItem(20,40,QSizePolicy.Minimum,QSizePolicy.Expanding)

        vbox = QVBoxLayout()
        vbox.addWidget(button_clasterization)
        vbox.addWidget(button_classification)
        vbox.addWidget(button_lsa)
        vbox.addWidget(button_analyze_and_rule_apply)
        vbox.addWidget(button_xi_square
                       )
        vbox.addItem(spacer)

        widget = QWidget();
        widget.setLayout(vbox);
        self.setCentralWidget(widget);
        self.setGeometry(300, 300, 480, 320)
        self.setWindowTitle('Этапный текстовый процессор')    
        self.show()

    def getFilenameFromUserSelection(self):
        filenames, _ = QFileDialog.getOpenFileName(self, "Открыть файлы для анализа", "", "Any Files (*.*)", None)
        if(len(filenames) > 0):
            return filenames
        else:
            return None

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
        filenames = []
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

    def analyze_and_rule_apply(self):
        print("Анализ и применение правил вывода")
        filenames = self.getFilenamesFromUserSelection()
        if(filenames != None):
            dialogConfigDRA = DialogConfigDRA(filenames, morph, configurations, self)
            self.hide()
            dialogConfigDRA.destroyed.connect(self.show)
            dialogConfigDRA.exec_()

    def makeXiSquare(self):
        print("Применение критерия Хи-Квадрат")
        filename = self.getFilenameFromUserSelection()
        if(filename != None):
            dialogXiSquare = DialogXiSquare(filename, morph, configurations, self)
            self.hide()
            dialogXiSquare.destroyed.connect(self.show)
            dialogXiSquare.exec_()




if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MainWindow()
    sys.exit(app.exec_())

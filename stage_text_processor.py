#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import pymorphy2
from matplotlib import rc

from sources.TextClassificationLib import DialogClassificationLib
from sources.utils import *


# Для корректного отображение шрифтов на графиках в Windows
if(os.name != 'posix'):
    font = {'family': 'Verdana',
            'weight': 'normal'}
    rc('font', **font)

configurations = readConfigurationFile("configuration.cfg")
input_dir = configurations.get("input_files_directory", "input_files") + "/"
output_dir = configurations.get("output_files_directory", "output_files") + "/"
stop_words_filename = configurations.get("stop_words_file", "sources/russian_stop_words.txt")


from PyQt5.QtWidgets import QMainWindow, QSizePolicy, QSpacerItem, QFileDialog, QPushButton
from sources.XiSquare import *
from sources.Word2VecNew.DialogWord2VecMaker import *
from sources.TextClasterization import *
from sources.TextClassification import *
from sources.TextClastering import *
from sources.TextLSA import *
from sources.TextDecomposeAndRuleApply import *
from sources.AnnotationMaker import *
from sources.FastText.DialogFastTextMaker import *


if not os.path.exists(input_dir):
    os.makedirs(input_dir)

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

        button_clastering = QPushButton("Кластеризация (LIB)")
        button_clastering.setMinimumHeight(32)
        button_clastering.clicked.connect(self.clastering)

        button_classification = QPushButton("Классификация")
        button_classification.setMinimumHeight(32)
        button_classification.clicked.connect(self.classification)

        button_classification_lib = QPushButton("Классификация (LIB)")
        button_classification_lib.setMinimumHeight(32)
        button_classification_lib.clicked.connect(self.classification_lib)

        button_lsa = QPushButton("Латентно-семантический анализ")
        button_lsa.setMinimumHeight(32)
        button_lsa.clicked.connect(self.makeLSA)

        button_analyze_and_rule_apply = QPushButton("Анализ и правила вывода предложений")
        button_analyze_and_rule_apply.setMinimumHeight(32)
        button_analyze_and_rule_apply.clicked.connect(self.analyze_and_rule_apply)

        button_xi_square = QPushButton("Рассчет критериев для выделения термов")
        button_xi_square.setMinimumHeight(32)
        button_xi_square.clicked.connect(self.makeXiSquare)

        button_annotation = QPushButton("Создание аннотации документа")
        button_annotation.setMinimumHeight(32)
        button_annotation.clicked.connect(self.makeTextAnnotation)
        #button_annotation.setEnabled(False)

        button_w2v = QPushButton("Word2Vec")
        button_w2v.setMinimumHeight(32)
        button_w2v.clicked.connect(self.makeWord2Vec)

        button_fastText = QPushButton("FastText")
        button_fastText.setMinimumHeight(32)
        button_fastText.clicked.connect(self.makeFastText)

        spacer = QSpacerItem(20,40,QSizePolicy.Minimum,QSizePolicy.Expanding)

        vbox = QVBoxLayout()
        vbox.addWidget(button_clasterization)
        vbox.addWidget(button_clastering)
        vbox.addWidget(button_classification)
        vbox.addWidget(button_classification_lib)
        vbox.addWidget(button_lsa)
        vbox.addWidget(button_analyze_and_rule_apply)
        vbox.addWidget(button_xi_square)
        vbox.addWidget(button_annotation)
        vbox.addWidget(button_w2v)
        vbox.addWidget(button_fastText)
        vbox.addItem(spacer)

        widget = QWidget();
        widget.setLayout(vbox);
        self.setCentralWidget(widget);
        self.setGeometry(300, 300, 480, 320)
        self.setWindowTitle('Этапный текстовый процессор')    
        self.show()

    def clasterization(self):
        print("Кластеризация")
        filenames = getFilenamesFromUserSelection(input_dir + 'clasterization')
        if(filenames != None):
            dialogConfigClasterization = DialogConfigClasterization(filenames, morph, configurations, self)
            self.hide()
            dialogConfigClasterization.destroyed.connect(self.show)
            dialogConfigClasterization.exec_()

    def clastering(self):
        print("Кластеризация (LIB)")
        filenames = getFilenamesFromUserSelection(input_dir + 'clasterization')
        if(filenames != None):
            dialogConfigClastering = DialogClastering(filenames, morph, configurations, self)
            self.hide()
            dialogConfigClastering.destroyed.connect(self.show)
            dialogConfigClastering.exec_()

    def classification(self):
        print("Классификация")
        dirname = getDirFromUserSelection(input_dir + 'classification')
        if(dirname != None):
            dialogConfigClassification = DialogConfigClassification(dirname, morph, configurations, self)
            self.hide()
            dialogConfigClassification.destroyed.connect(self.show)
            dialogConfigClassification.exec_()

    def classification_lib(self):
        print("Классификация (LIB)")
        dirname = getDirFromUserSelection(input_dir + 'classification')
        if(dirname != None):
            dialogClassificationLib = DialogClassificationLib(dirname, morph, configurations, self)
            self.hide()
            dialogClassificationLib.destroyed.connect(self.show)
            dialogClassificationLib.exec_()

    def makeLSA(self):
        print("LSA")
        filenames = getFilenamesFromUserSelection(input_dir)
        if(filenames != None):
            dialogConfigLSA = DialogConfigLSA(filenames, morph, configurations, self)
            self.hide()
            dialogConfigLSA.destroyed.connect(self.show)
            dialogConfigLSA.exec_()

    def analyze_and_rule_apply(self):
        print("Анализ и применение правил вывода")
        filenames = getFilenamesFromUserSelection()
        if(filenames != None):
            dialogConfigDRA = DialogConfigDRA(filenames, morph, configurations, self)
            self.hide()
            dialogConfigDRA.destroyed.connect(self.show)
            dialogConfigDRA.exec_()

    def makeXiSquare(self):
        print("Рассчет критериев для выделения термов")
        filename = getFilenameFromUserSelection("CSV Files (*.csv)", input_dir)
        if(filename != None):
            dialogXiSquare = DialogXiSquare(filename, morph, configurations, self)
            self.hide()
            dialogXiSquare.destroyed.connect(self.show)
            dialogXiSquare.exec_()

    def makeTextAnnotation(self):
        print("Аннотирование текста")
        filename = getFilenameFromUserSelection("Text file (*.txt)", input_dir)
        if (filename != None):
            dialogAnnotationMaker = DialogAnnotationMaker(filename, morph, configurations, self)
            self.hide()
            dialogAnnotationMaker.destroyed.connect(self.show)
            dialogAnnotationMaker.exec_()

    def makeWord2Vec(self):
        print("Word2Vec")
        filename = getFilenameFromUserSelection("Text file (*.txt);;Model file (*.model)", input_dir)
        if (filename != None):
            dialogWord2VecMaker = DialogWord2VecMaker(input_dir, filename, morph, configurations, self)
            self.hide()
            dialogWord2VecMaker.destroyed.connect(self.show)
            dialogWord2VecMaker.exec_()

    def makeFastText(self):
        print("FastText")
        dialogFastTextMaker = DialogFastTextMaker(input_dir, morph, configurations, self)
        self.hide()
        dialogFastTextMaker.destroyed.connect(self.show)
        dialogFastTextMaker.exec_()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MainWindow()
    sys.exit(app.exec_())

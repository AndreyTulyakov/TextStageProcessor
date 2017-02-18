#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import math

from PyQt5.QtCore import Qt
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QDialog, QMessageBox, QTextEdit
from PyQt5 import QtCore, QtGui, uic
from PyQt5.QtCore import QObject
from PyQt5.QtWidgets import QVBoxLayout
from PyQt5.QtWidgets import QWidget
from PyQt5.uic import loadUiType

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar)

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from PyQt5.QtCore import QThread

from sources.TextData import TextData
from sources.TextPreprocessing import *


class DialogAnnotationMaker(QDialog):

    def __init__(self, filename, morph, configurations, parent):
        super().__init__()
        uic.loadUi('sources/DialogAnnotationMaker.ui', self)

        self.filename = filename
        self.morph = morph
        self.configurations = configurations
        self.parent = parent

        flags = Qt.Window | Qt.WindowSystemMenuHint | Qt.WindowMinimizeButtonHint | Qt.WindowCloseButtonHint;
        self.setWindowFlags(flags)

        self.all_idf_word_keys = []
        self.texts = []

        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

        self.buttonProcess.clicked.connect(self.processIt)
        self.textEdit.setText("")


    def processIt(self):
        self.textEdit.append('Успешно завершено.')
        QMessageBox.information(self, "Внимание", "Латентно-семантический анализ завершен!")

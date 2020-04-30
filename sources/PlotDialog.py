from PyQt5.QtCore import QObject
from PyQt5.QtCore import QThread
from PyQt5.QtCore import Qt
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QDialog, QMessageBox, QTextEdit
from PyQt5 import QtCore, QtGui, uic
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
matplotlib.use('Qt5Agg')
from matplotlib.figure import Figure


class PlotDialog(QDialog):
    def __init__(self):
        super().__init__()

        uic.loadUi('sources/PlotDialog.ui', self)

        flags = Qt.Window | Qt.WindowSystemMenuHint | Qt.WindowMinimizeButtonHint | Qt.WindowCloseButtonHint
        self.setWindowFlags(flags)

        self.fig = Figure()
        self.addmpl(self.fig)

    def doplot(self):
        self.addfig(plt.gcf())

    def addfig(self, fig):
        self.rmmpl()
        self.addmpl(fig)

    def addmpl(self, fig):
        self.canvas = FigureCanvas(fig)
        self.mplvl.addWidget(self.canvas)
        self.canvas.draw()
        self.toolbar = NavigationToolbar(
            self.canvas, self.mplwindow, coordinates=True)
        self.GraphicsLayout.addWidget(self.toolbar)

    def rmmpl(self):
        self.mplvl.removeWidget(self.canvas)
        self.canvas.close()
        self.mplvl.removeWidget(self.toolbar)
        self.toolbar.close()

from PyQt5 import QtWidgets

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
matplotlib.use('Qt5Agg')
from matplotlib.figure import Figure

from sources.common.plot.TsneMplForWidget import TsneMplForWidget

class PlotMaker(FigureCanvas):
    def __init__(self, plotVLayout, parent=None):
        self.fig = Figure() # подаем на вход рисунок
        self.ax = self.fig.add_subplot(111)
        self.plotVLayout = plotVLayout # подаем на вход слой элементов виджета
        self.visualizeData(self.fig)

    def drawPlot(self, fig):
        self.removePlot()
        self.visualizeData(fig)

    def visualizeData(self, fig):
        self.geomForMpl = self.plotVLayout
        self.canvas = TsneMplForWidget(self.fig)
        self.geomForMpl.addWidget(self.canvas)

        # self.toolbar = NavigationToolbar(self.canvas, self, coordinates=True)
        # self.geomForMpl.addWidget(self.toolbar)

    def removePlot(self):
        self.mplvl.removeWidget(self.canvas)
        self.canvas.close()
        self.mplvl.removeWidget(self.toolbar)
        self.toolbar.close()

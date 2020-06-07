from PyQt5 import QtWidgets

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
matplotlib.use('Qt5Agg')
from matplotlib.figure import Figure

from sources.common.plot.TsneMplForWidget import TsneMplForWidget

class PlotMaker(FigureCanvas):
    def __init__(self, plotVLayout, parent=None, title: str=None):
        self.toolbar = None
        self.fig = Figure() # подаем на вход рисунок
        matplotlib.pyplot.close(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title(title)
        self.plotVLayout = plotVLayout # подаем на вход слой элементов виджета
        self.visualizeData(self.fig)

    def drawPlot(self, fig):
        self.removePlot()
        self.visualizeData(fig)

    def visualizeData(self, fig):
        self.geomForMpl = self.plotVLayout
        self.canvas = TsneMplForWidget(self.fig)
        # self.checkWidget(self.geomForMpl) 
        self.geomForMpl.addWidget(self.canvas)

        # self.toolbar = NavigationToolbar(self.canvas, self, coordinates=True)
        # self.geomForMpl.addWidget(self.toolbar)
    
    def add_toolbar(self, parent):
        self.toolbar = NavigationToolbar(self.canvas, parent, coordinates=True)
        self.plotVLayout.addWidget(self.toolbar)

    # Попытка очистки графика одновременно с рисованием
    def checkWidget(self, plotVLayout):
        lcount = plotVLayout.count()
        if(lcount > 1):
            for i in reversed(range(1, lcount)):
                plotVLayout.itemAt(i).widget().deleteLater()
            # plotVLayout.removeItem(plotVLayout.itemAt(1))

    def removePlot(self):
        self.geomForMpl.removeWidget(self.canvas)
        self.canvas.close()
        self.geomForMpl.removeWidget(self.toolbar)
        self.toolbar.close()

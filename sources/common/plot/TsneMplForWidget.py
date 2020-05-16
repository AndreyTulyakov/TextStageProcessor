from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QSizePolicy

import matplotlib
from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg as FigureCanvas)
matplotlib.use('Qt5Agg')

from sources.Word2VecNew.DialogWord2Vec import Ui_Word2VecDialog as DialogWord2Vec # Импорт UI конвертированного в .py

class TsneMplForWidget(FigureCanvas):
    def __init__(self, fig, parent=None):
        self.fig = fig # подаем на вход рисунок
        FigureCanvas.__init__(self, self.fig) # инициализация холста
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding) # холст занимает всё место (поведение размеров)
        FigureCanvas.updateGeometry(self) # сообщаем системе, что геометрия изменилась
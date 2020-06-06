import csv
import math
import copy
import numpy as np
import shutil
import os

from PyQt5.QtCore import QObject, QThread, pyqtSignal
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

import sources.TextPreprocessing as textproc
from sources.classification.clsf_util import makeFileListLib
from sources.utils import makePreprocessingForAllFilesInFolder, clear_dir
from sources.common.helpers.TextHelpers import get_processed_word_lists_from_lines, get_processed_lines
from sources.common.helpers.PathHelpers import get_filename_from_path

from gensim.models import FastText
import gensim
import fasttext

FAST_TEXT_FILENAME_PREFIX = 'trainfile'

class FastTextCalculatorSignals(QObject):
    PrintInfo = pyqtSignal(str)
    Finished = pyqtSignal()
    Progress = pyqtSignal(FastText, int)
    ProgressBar = pyqtSignal(int)

# Класс, инкапсулирующий логику создания модели fasttext при помощи выбранной на форме библиотеки.
# Gensim - библиотека позволяет создавать многофункциональную модель, наиболее полно описывающую текст. Поддерживает поиск близких слов (синонимов).
# fasttext (от PyPI: https://pypi.org/project/fasttext/) - оригинальная реализация fasttext для Python от Facebook. Позволяет производить классификацию текста.
class FastTextCalculator(QThread):
    def __init__(self, train_path: str, morph, configurations, use_gensim: bool = False, only_nouns: bool = False):
        super().__init__()
        self.use_gensim = use_gensim
        self.filename = train_path
        self.train_filename = None
        with open('sources/russian_stop_words.txt', 'r', encoding='UTF-8') as file:
            self.stopwords = set(map(str.strip, file.readlines()))
        self.morph = morph
        self.configurations = configurations
        self.signals = FastTextCalculatorSignals()
        self.only_nouns = only_nouns
        self.gensim_fasttext_model = None
        self.fasttext_model = None
        self.output_dir = self.configurations.get(
            "output_files_directory", "output_files") + '/FastText/'
        if not use_gensim:
            self.set_train_file()

    # Основной метод класса, создание (или тренировка) модели.
    def run(self):
        if self.use_gensim:
            self.create_model_gensim()
            print('Модель создана (Gensim).')
        else:
            self.create_model()
            print('Модель создана.')
        self.signals.PrintInfo.emit('\n****************\nРассчеты закончены!')
        self.signals.Finished.emit()
        self.signals.ProgressBar.emit(100)

    def create_model_gensim(self):
        handler = EpochCallbackHandler(
            self.iter, self.signals.Progress, self.signals.ProgressBar)
        data = self._load_file_data(self.filename)
        self.gensim_fasttext_model = FastText(data, size=self.size, alpha=self.learn_rate,
                        min_count=self.min_count, iter=self.iter, window=self.window, callbacks=[handler])
        self.gensim_fasttext_model.callbacks = ()

    def create_model(self):
        self.signals.PrintInfo.emit('Препроцессинг файлов:')
        self.signals.PrintInfo.emit('Удаление стоп слов...')
        self.signals.PrintInfo.emit('Приведение к нормальной форме...')
        self.set_train_file()
        self.fasttext_model = fasttext.train_unsupervised(input=self.train_filename, dim=self.size, lr=self.learn_rate,
                        minCount=self.min_count, epoch=self.iter, ws=self.window)

    def set_train_file(self):
        train_text = []
        train_dir = os.path.dirname(self.filename)

        processed_lines = get_processed_lines(
                input_file=self.filename,
                morph=self.morph,
                stop_words=self.stopwords,
                only_nouns=self.only_nouns
            )
        self.train_filename = '{0}/{1}'.format(train_dir, '{0}_cleared.txt'.format(get_filename_from_path(self.filename).split('.')[0]))
        train_file = open(self.train_filename, 'w', encoding='utf-8')

        for line in processed_lines:
            train_file.write(line + '\n')
        train_file.close()

    # Поиск близких по значению слов (Gensim).
    def search_word(self, word: str):
        return self.gensim_fasttext_model.wv.most_similar(positive=[word], negative=None, topn=20) if self.use_gensim else None

    # Загрузка данных из файла (включена предварительная обработка).
    def _load_file_data(self, input_file: str):
        with open(input_file, 'r', encoding='UTF-8') as file:
            file_data = file.readlines()
            self.signals.ProgressBar.emit(5)
            lines = get_processed_word_lists_from_lines(lines=file_data, morph=self.morph, stop_words=self.stopwords, only_nouns=self.only_nouns)
        self.signals.ProgressBar.emit(10)
        return lines

class EpochCallbackHandler(gensim.models.callbacks.CallbackAny2Vec):
    def __init__(self, totalEpoches: int, epochEndSignal: pyqtSignal, updateProgressSignal: pyqtSignal):
        self.epochEndSignal = epochEndSignal
        self.updateProgressSignal = updateProgressSignal
        self.totalEpoches = totalEpoches
        self.epoch = 0

    def on_epoch_end(self, model):
        print(
            "Model alpha: {}".format(model.alpha),
            "Model min_alpha: {}".format(model.min_alpha),
            "Epoch saved: {}".format(self.epoch + 1),
            "Start next epoch"
        )
        self.epoch += 1
        self.epochEndSignal.emit(model, self.epoch)
        self.updateProgressSignal.emit(10 + round((self.epoch / self.totalEpoches) * 90))
import csv
import math
import copy
import numpy as np
import shutil
import os
import re

from PyQt5.QtCore import QObject, QThread, pyqtSignal
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

import sources.TextPreprocessing as textproc
from sources.classification.clsf_util import makeFileListLib
from sources.utils import makePreprocessingForAllFilesInFolder, clear_dir

from gensim.models import Word2Vec
import gensim


class Word2VecCalculatorSignals(QObject):
    PrintInfo = pyqtSignal(str)
    Finished = pyqtSignal()
    Progress = pyqtSignal(Word2Vec, int)
    ProgressBar = pyqtSignal(int)


class Word2VecCalculator(QThread):
    def __init__(self, filename: str, morph, configurations):
        super().__init__()
        self.filename = filename
        self.morph = morph
        self.configurations = configurations
        self.texts = []
        self.categories = dict()
        self.signals = Word2VecCalculatorSignals()
        self.result_sentence_count = 1
        self.only_nouns = False
        self.output_dir = self.configurations.get(
            "output_files_directory", "output_files") + '/Word2Vec/'
        if self.filename.endswith('.model'):
            self.model = Word2Vec.load(filename)

    def run(self):
        data = self._load_file_data(self.filename)
        handler = EpochCallbackHandler(
            self.iter, self.signals.Progress, self.signals.ProgressBar)
        # sg тип алгоритма для тренировки 0 - CBOW, 1 - skip-gram 
        self.model = Word2Vec(data, size=self.size, alpha=self.learn_rate, sg=self.sg,
                              min_count=self.min_count, iter=self.iter, window=self.window,
                              ns_exponent=self.ns_exponent, negative=self.negative, workers=4,
                              callbacks=[handler])
        self.model.callbacks = ()
        self.signals.PrintInfo.emit('Рассчеты закончены!')
        self.signals.Finished.emit()
        self.signals.ProgressBar.emit(100)

    def search_word(self, word: str):
        return self.model.wv.most_similar(positive=[word], negative=None, topn=20)

    def _load_file_data(self, input_file: str):
        with open(input_file, 'r', encoding='UTF-8') as file:
            print(input_file)
            file_data = file.readlines()
            lines = self._process(file_data)
        self.signals.ProgressBar.emit(10)
        return lines

    def _process(self, lines):
        def apply_filter(filter, lines):
            for i in range(len(lines)):
                lines[i] = filter(lines[i])

        def gensim_preprocess(line):
            return gensim.utils.simple_preprocess(line, min_len=3, max_len=20)

        def normalize_text(lines):
            lst = []
            pattern = re.compile(r'[?!.]+')
            for line in filter(lambda x: len(x) > 0, map(str.strip, lines)):
                try:
                    lst.extend(re.split(pattern, line))
                except Exception as e:
                    print(line)
                    print(e)
            lines = lst
            return lines

        def remove_stops(line):
            for word in line:
                if word not in stopwords:
                    yield word

        def morph_words(line):
            for word in line:
                if self.morph.word_is_known(word):
                    word = self.morph.parse(word)[0].normal_form
                yield word

        def morph_nouns(line):
            for word in line:
                if self.morph.word_is_known(word):
                    data = self.morph.parse(word)[0]
                    if 'NOUN' in data.tag:
                        yield data.normal_form

        with open('sources/russian_stop_words.txt', 'r', encoding='UTF-8') as file:
            stopwords = set(map(str.strip, file.readlines()))

        lines = normalize_text(lines)
        self.signals.ProgressBar.emit(2)
        apply_filter(gensim_preprocess, lines)
        apply_filter(remove_stops, lines)
        apply_filter(morph_nouns if self.only_nouns else morph_words, lines)
        apply_filter(remove_stops, lines)
        self.signals.ProgressBar.emit(5)
        return [line for line in map(list, lines) if len(line) > 1]


class EpochCallbackHandler(gensim.models.callbacks.CallbackAny2Vec):
    def __init__(self, totalEpoches: int, epochEndSignal: pyqtSignal, updateProgressSignal: pyqtSignal):
        self.epochEndSignal = epochEndSignal
        self.updateProgressSignal = updateProgressSignal
        self.totalEpoches = totalEpoches
        self.epoch = 0
    
    # def on_epoch_begin(self, model):
    #     print(
    #         "Model alpha: {}".format(model.alpha),
    #         "Model min_alpha: {}".format(model.min_alpha),
    #         "Epoch saved: {}".format(self.epoch + 1),
    #         "Start next epoch"
    #     )

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
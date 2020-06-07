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
from sources.common.helpers.TextHelpers import get_processed_lines

import nltk
import gensim
import re
import pymorphy2
from nltk.corpus import stopwords

from sources.common.helpers.PathHelpers import get_filename_from_path

import fasttext

class FastTextClassifierSignals(QObject):
    PrintInfo = pyqtSignal(str)
    Finished = pyqtSignal()
    ClassificationFinished = pyqtSignal()
    Progress = pyqtSignal(int)
    ProgressBar = pyqtSignal(int)

FAST_TEXT_TRAIN_FILENAME_PREFIX = 'trainfile'

# Для треннировки классификатора подаётся список файлов. Название каждого файла (только латиница) должно отражать категорию, к которой относится список текстов, содержащийся в этом файле. Например, файл, содержащий тексты категории "комедия", должен иметь название "comedy.txt"
# Каждая строка файла содержит категорию (__label__<category>) и следующий за категорией текст.
# Класс, реализующий классификацию текстового файла при помощи метода predict библиотеки fasttext(PyPI).
class FastTextClassifier(QThread):
    def __init__(self, morph, only_nouns, configurations, filenames=None, model_file=None):
        super().__init__()
        self.train_files = filenames
        self.train_filename = None
        self.model = None
        self.model_filename = model_file
        self.configurations = configurations
        self.classification_filename = None
        self.classification_result_filename = None
        self.predictedLabels = []
        self.morph = morph
        self.only_nouns = only_nouns
        self.cachedStopWords = stopwords.words("russian")
        self.signals = FastTextClassifierSignals()
        self.train_dir = None
        self.output_dir = self.configurations.get(
            "output_files_directory", "output_files") + '/FastText/classification/'
        if not model_file == None:
            self.model = fasttext.load_model(self.model_filename)
            self.signals.PrintInfo.emit('Загружена supervised модель {0}'.format(self.model_filename))

    # Основной метод класса.
    # Тренировка классификатора.
    def run(self):
        if not self.train_files == None:
            self.set_train_file()
            self.model = fasttext.train_supervised(input=self.train_filename, dim=self.size, lr=self.learn_rate,
                        minCount=self.min_count, epoch=self.iter, ws=self.window)
        self.signals.PrintInfo.emit('\n****************\nРассчеты закончены!')
        self.signals.Finished.emit()

    # Классификация строк файла.
    # Каждая строка файла - отдельный текст для классификации.
    # Получает массив строк нормализованного текста, вызывает метод predict, сохраняет результат в файл.
    def classify(self):
        textLines = get_processed_lines(
            input_file=self.classification_filename,
            morph=self.morph,
            stop_words=self.cachedStopWords,
            only_nouns=self.only_nouns
        )
        result = self.model.predict(textLines)
        self.predictedLabels = result[0]
        self.signals.PrintInfo.emit('\n****************\nКлассификация завершена!')
        self.save_result_to_file(result)
        self.signals.ClassificationFinished.emit()

    # Устанавливает рабочую модель для классификации.
    def set_model(self):
        if not self.model_filename == None:
            self.model = fasttext.load_model(self.model_filename)
            self.signals.PrintInfo.emit('Загружена supervised модель {0}'.format(self.model_filename))

    # Создаёт тренировочный файл с помеченными категориями строками.
    # TODO: Добавить разбиение (80:20) на тренировочный файл и файл для проверки модели - X.valid.
    # Подобная проверка осуществляется при помощи метода <fasttext_model>.test(valid_file), которая возвращает критерии точности: precision и recall.
    def set_train_file(self):
        train_file_name = ''
        train_text = []
        train_dir = os.path.dirname(self.train_files[0])

        for file in self.train_files:
            if not get_filename_from_path(file).split('_')[0] == FAST_TEXT_TRAIN_FILENAME_PREFIX:
                processed_lines = get_processed_lines(
                        input_file=file,
                        morph=self.morph,
                        stop_words=self.cachedStopWords,
                        only_nouns=self.only_nouns
                    )
                train_file_name = train_file_name.split('.')[0] + '_' + get_filename_from_path(file).split('.')[0]
                for line in processed_lines:
                    train_text.append('__label__{0} {1}'.format(get_filename_from_path(file).split('.')[0], line))
            else:
                self.train_filename = file
                self.model_filename = 'SUPERVISED_{0}.model'.format(get_filename_from_path(file).split('.')[0])
                break
        if self.train_filename == None:
            self.model_filename = 'SUPERVISED_{0}.model'.format(train_file_name)
            self.train_filename = '{0}/{1}_{2}.txt'.format(train_dir, FAST_TEXT_TRAIN_FILENAME_PREFIX, train_file_name)
            train_file = open(self.train_filename , 'w', encoding='utf-8')

            for line in train_text:
                train_file.write(line + '\n')
            train_file.close()

    # Классификация фразы.
    def classify_phrase(self, phrase):
        if not self.model == None:
            textLines = get_processed_lines(
                morph=self.morph,
                stop_words=self.cachedStopWords,
                lines=[phrase]
            )
            print(self.model)
            result = self.model.predict(textLines)
            print(result)
            self.signals.PrintInfo.emit('\n****************\nКлассификация фразы завершена: {0} (probability: {1})'.format(result[0][0], result[1][0]))


    # Сохраняет в файл результат классификации.
    # Для каждой строки сходного файла с текстами для классификации: __label__результат (вероятность: 0.0...).
    def save_result_to_file(self, classificationResult: list):
        result_filename = get_filename_from_path(self.classification_filename).split('.')[0] + '.txt'
        self.classification_result_filename = result_filename
        newFile = open(self.output_dir + result_filename, "w", encoding='utf-8')
        for i in range(len(classificationResult[0])):
            newFile.write(str(classificationResult[0][i][0]) + ' (probability: ' + str(classificationResult[1][i][0]) + ')\n')
        newFile.close()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import math
import copy
import numpy as np
import shutil

from PyQt5.QtCore import QObject
from PyQt5.QtCore import QThread
from PyQt5.QtCore import pyqtSignal

from sources.TextPreprocessing import writeStringToFile
from sources.classification.ID3 import Classification_Text_ID3
from sources.classification.KNN import getResponse
from sources.classification.NaiveBayes import *
from sources.classification.clsf_util import *
from sources.utils import makePreprocessingForAllFilesInFolder, clear_dir


class ClassificationCalculatorSignals(QObject):
    PrintInfo = pyqtSignal(str)
    Finished = pyqtSignal()
    UpdateProgressBar = pyqtSignal(int)


class ClassificationCalculator(QThread):

    METHOD_NAIVE_BAYES = 1
    METHOD_ROCCHIO = 2
    METHOD_KNN = 3
    METHOD_LLSF = 4
    METHOD_ID3 = 5

    def __init__(self, input_dir, output_dir, morph, configurations):
        super().__init__()
        self.input_dir = input_dir + '/'
        self.output_dir = output_dir + '/classification/'

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        clear_dir(self.output_dir)

        self.output_preprocessing_dir = self.output_dir + 'preprocessing/'
        self.first_call = True
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.morph = morph
        self.configurations = configurations
        self.texts = []
        self.categories = dict()
        self.signals = ClassificationCalculatorSignals()
        self.method = ClassificationCalculator.METHOD_NAIVE_BAYES
        self.need_preprocessing = False

        if len(self.input_dir) > 0 and self.input_dir[-1] == '/':
            self.input_dir = self.input_dir[:-1]
        last_slash_index = self.input_dir.rfind('/')
        self.input_dir_short = ''
        if last_slash_index != -1:
            self.input_dir_short = self.input_dir[last_slash_index+1:]

    def setMethod(self, method_name, arg_need_preprocessing):
        self.method = method_name
        self.need_preprocessing = arg_need_preprocessing

    def run(self):
        self.signals.UpdateProgressBar.emit(0)

        # Делаем препроцессинг 1 раз
        if self.first_call and self.need_preprocessing:
            self.signals.PrintInfo.emit("Препроцессинг...")
            makePreprocessingForAllFilesInFolder(self.configurations,
                                                 self.input_dir,
                                                 self.output_preprocessing_dir,
                                                 self.output_dir,
                                                 self.morph)
        else:
            self.signals.PrintInfo.emit("Препроцессинг - пропускается")
        self.signals.UpdateProgressBar.emit(10)

        if self.need_preprocessing:
            self.method_input_dir = self.output_preprocessing_dir + self.input_dir_short + '/'
        else:
            self.method_input_dir = self.input_dir

        if self.method == ClassificationCalculator.METHOD_NAIVE_BAYES:
            self.classification_naive_bayes(self.need_preprocessing)

        if self.method == ClassificationCalculator.METHOD_ROCCHIO:
            self.classification_rocchio(self.need_preprocessing)

        if self.method == ClassificationCalculator.METHOD_KNN:
            self.classification_knn(self.need_preprocessing)

        if self.method == ClassificationCalculator.METHOD_LLSF:
            self.classification_llsf(self.need_preprocessing)

        if self.method == ClassificationCalculator.METHOD_ID3:
            self.classification_id3(self.need_preprocessing);

        if self.first_call and self.need_preprocessing:
            self.first_call = False

        self.signals.UpdateProgressBar.emit(100)
        self.signals.PrintInfo.emit('Рассчеты закончены!')
        self.signals.Finished.emit()

    # Алгоритм наивного Байеса
    def classification_naive_bayes(self, needPreprocessing):

        output_dir = self.output_dir + 'nb_out/'
        input_dir = self.method_input_dir

        self.signals.PrintInfo.emit("Алгоритм наивного Байеса")
        # Классификация
        fdata, fclass, split = makeFileList(input_dir)
        self.signals.UpdateProgressBar.emit(15)

        trainingSet = fdata[:split]
        trainingClass = fclass[:split]
        testSet = fdata[split:]
        test_fnames = getBasePath(makeFileList(input_dir, fread=False)[0][split:])
        self.signals.UpdateProgressBar.emit(20)
        vocab = {}
        word_counts = defaultdict(dict)
        priors = dict.fromkeys(set(trainingClass), 0.)
        for cl in priors.keys():
            priors[cl] = trainingClass.count(cl)
        docs = []

        self.signals.UpdateProgressBar.emit(30)

        for i in range(len(trainingSet)):
            counts = count_words(trainingSet[i])
            cl = trainingClass[i]
            for word, count in counts.items():
                if word not in vocab:
                    vocab[word] = 0.0
                if word not in word_counts[cl]:
                    word_counts[cl][word] = 0.0
                vocab[word] += count
                word_counts[cl][word] += count
        self.signals.UpdateProgressBar.emit(40)
        scores = {}
        V = len(vocab)
        for i in range(len(testSet)):
            scores[test_fnames[i]] = []
            counts = count_words(testSet[i])
            for cl in priors.keys():
                Lc = sum(word_counts[cl].values())
                prior_cl = math.log10(priors[cl] / sum(priors.values()))
                log_prob = 0.0
                for w, cnt in counts.items():
                    Wic = word_counts[cl].get(w, 0.0)
                    log_prob += math.log10((Wic + 1)/(V + Lc)) * cnt
                scores[test_fnames[i]].append([cl, round((log_prob + prior_cl), 3)])
        self.signals.UpdateProgressBar.emit(60)
        self.signals.PrintInfo.emit("Выходные файлы:")
        out_dir = self.output_dir + 'nb_out/'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        self.signals.PrintInfo.emit(out_dir + 'Словарь всех.csv')
        dictToCsv(vocab, output_dir + 'Словарь всех.csv')
        self.signals.PrintInfo.emit(out_dir + 'Вероятности документов.csv')
        dictListToCsv(scores, output_dir + 'Вероятности документов.csv')
        self.signals.PrintInfo.emit(out_dir + 'Словарь по классам.csv')
        dictOfDictToCsv(word_counts, output_dir + 'Словарь по классам.csv')

    # Алгоритм Рочио
    def classification_rocchio(self, needPreprocessing):

        def findCentroid(nparray):
            return (np.sum(nparray, axis=0) / len(nparray))

        ##############PARAMS###################
        output_dir = self.output_dir + 'roc_out/'
        input_dir = self.method_input_dir
        sep = ";"
        eol = "\n"
        ###############ALGO##################

        fdata, fclass, split = makeFileList(input_dir)
        tfidf, uniq_words = makeTFIDF(fdata[:split], fdata[split:])
        class_titles = set(fclass)

        combiSet = addClassToTFIDF(copy.deepcopy(tfidf), fclass)
        trainSet = combiSet[:split]
        testSet = combiSet[split:]
        split_names = getBasePath(makeFileList(input_dir, fread=False)[0])

        self.signals.UpdateProgressBar.emit(20)
        centroids = []
        for cl in class_titles:
            cl_array = []
            for i in range(len(trainSet)):
                if fclass[i] == cl:
                    cl_array.append(trainSet[i][:-1])
            centroids.append(findCentroid(np.array(cl_array)).round(3).tolist())

        centroids = addClassToTFIDF(centroids, list(class_titles))
        log_centr = "центроиды" + eol + sep.join(uniq_words) + eol
        for row in centroids:
            log_centr += sep.join(map(str, row)).replace('.',',') + eol
        self.signals.UpdateProgressBar.emit(40)
        self.signals.PrintInfo.emit("Алгоритм Роккио")
        log_main = "Расстояние до центроидов" + eol
        predictions = []

        test_fnames = split_names[split:]
        for i in range(len(testSet)):
            neighbors, dist = getNeighbors(centroids, testSet[i], len(centroids))
            log_main += test_fnames[i] + sep + "Принадлежит классу:" + str(dist[0][0][-1]) + sep + eol 
            log_main += sep.join([x[0][-1] for x in dist]) + eol + sep.join(
                map(str, [x[1] for x in dist])).replace('.',',') + eol
            self.signals.PrintInfo.emit('> результат =' + repr(dist[0][0][-1]) + ', на самом деле=' + repr(testSet[i][-1]))
            predictions.append(dist[0][0][-1])
        accuracy = getAccuracy(testSet, predictions)
        self.signals.PrintInfo.emit('Точность: ' + repr(accuracy) + '%')
        self.signals.UpdateProgressBar.emit(60)
        ###############LOGS##################
        log_tfidf = sep.join(uniq_words) + eol
        for i in range(len(combiSet)):
            row = combiSet[i]
            log_tfidf += sep.join(map(str, row)).replace('.',',') + sep + split_names[i] + eol

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.signals.PrintInfo.emit('Выходные файлы:')

        self.signals.UpdateProgressBar.emit(80)
        self.signals.PrintInfo.emit(output_dir + 'output_Rocchio.csv')
        writeStringToFile2(log_main, output_dir + 'output_Rocchio.csv')
        self.signals.PrintInfo.emit(output_dir + 'Rocchio_centroids.csv')
        writeStringToFile2(log_centr, output_dir + 'Rocchio_centroids.csv')
        self.signals.PrintInfo.emit(output_dir + 'tfidf_matrix.csv')
        writeStringToFile2(log_tfidf, output_dir + 'tfidf_matrix.csv')

    # Алгоритм KNN
    def classification_knn(self, needPreprocessing):

        ##############PARAMS###################
        output_dir = self.output_dir + 'knn_out/'
        input_dir = self.method_input_dir
        sep = ";"
        eol = "\n"
        k = self.configurations.get('classification_knn_k')
        ###############ALGO##################

        fdata, fclass, split = makeFileList(input_dir)
        tfidf, uniq_words = makeTFIDF(fdata[:split], fdata[split:])
        self.signals.UpdateProgressBar.emit(20)
        trainingSet = addClassToTFIDF(tfidf[:split], fclass[:split])
        testSet = addClassToTFIDF(tfidf[split:], fclass[split:])
        self.signals.UpdateProgressBar.emit(30)
        split_names = getBasePath(makeFileList(input_dir, fread=False)[0])

        self.signals.PrintInfo.emit("Алгоритм KNN")
        predictions = []
        log_neighbors = "Соседи и расстояния до них:" + eol
        log_votes = "Голоса соседей:" + eol
        test_fnames = split_names[split:]
        for x in range(len(testSet)):
            neighbors, dist = getNeighbors(trainingSet, testSet[x], k)
            result = getResponse(neighbors)
            log_neighbors += "Документ:;" + str(test_fnames[x]) + eol + "Сосед" + sep + "Расстояние" + eol
            for j in range(len(dist)):
                log_neighbors += split_names[j] + sep + str(dist[j][1]).replace('.',',') + eol
            log_votes += "Документ:;" + str(test_fnames[x]) + eol + "Принадлежит классу" + sep + result[0][0] + eol
            log_votes += sep.join([str(x[0]) + sep + str(x[1]) for x in result]) + eol
            predictions.append(result[0][0])
            self.signals.PrintInfo.emit('> результат =' + repr(result[0][0]) + ', на самом деле=' + repr(testSet[x][-1]))
        accuracy = getAccuracy(testSet, predictions)
        self.signals.PrintInfo.emit('Точность: ' + repr(accuracy) + '%')
        self.signals.UpdateProgressBar.emit(50)
        ###############LOGS##################
        log_tfidf = sep.join(uniq_words) + eol
        combiSet = trainingSet + testSet

        for i in range(len(combiSet)):
            row = combiSet[i]
            log_tfidf += sep.join(map(str, row)).replace('.',',') + sep + split_names[i] + eol
        self.signals.UpdateProgressBar.emit(70)
        self.signals.PrintInfo.emit("Выходные файлы:")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.signals.PrintInfo.emit(output_dir + 'tfidf_matrix.csv')
        writeStringToFile(log_tfidf, output_dir + 'tfidf_matrix.csv')

        self.signals.PrintInfo.emit(output_dir + 'Соседи.csv')
        writeStringToFile(log_neighbors, output_dir + 'Соседи.csv')

        self.signals.PrintInfo.emit(output_dir + 'Голоса.csv')
        writeStringToFile(log_votes, output_dir + 'Голоса.csv')


    def classification_llsf(self, needPreprocessing):

        def listToCsv(lst, fpath):
            with open(fpath, 'w') as csv_file:
                writer = csv.writer(csv_file, delimiter = ';', lineterminator = '\n')
                for el in lst:
                    writer.writerow(localize_floats(el))

        ##############PARAMS###################
        output_dir = self.output_dir + 'llsf_out/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        input_dir = self.method_input_dir
        sep = ";"
        eol = "\n"
        ###############ALGO##################
        fdata, fclass, split = makeFileList(input_dir)
        tfidf, uniq_words = makeTFIDF(fdata[:split], fdata[split:])
        class_titles = list(set(fclass))
        self.signals.UpdateProgressBar.emit(20)
        A = np.array(tfidf[:split])
        B = []
        for cl in fclass[:len(A)]:
            row = [0] * len(class_titles)
            row[class_titles.index(cl)] = 1
            B.append(row)
        B = np.array(B)

        Fls = np.dot(np.transpose(B), np.transpose(np.linalg.pinv(A)))

        self.signals.UpdateProgressBar.emit(40)
        class_table = [class_titles + ["Принадлежит классу"]]
        for d in tfidf[split:]:
            d_class = np.round(np.dot(Fls, np.transpose(np.array(d))), 2).tolist()
            class_table.append(d_class + [class_table[0][d_class.index(max(d_class))]])
        
        self.signals.UpdateProgressBar.emit(60)

        self.signals.PrintInfo.emit("Алгоритм наименьших квадратов")

        ###############LOGS##################
        split_names = getBasePath(makeFileList(input_dir, fread=False)[0])

        A = A.tolist()
        B = B.tolist()
        B.insert(0, class_titles)
        Fls = addClassToTFIDF(Fls.tolist(), class_titles)
        Fls.insert(0, uniq_words)

        test_files= split_names[split:]
        test_files.insert(0, "Файл")
        class_table = addClassToTFIDF(class_table, test_files)

        log_tfidf = sep.join(uniq_words) + eol
        for i in range(len(tfidf)):
            row = tfidf[i]
            log_tfidf += sep.join(map(str, row)).replace('.',',') + sep + split_names[i] + eol

        self.signals.PrintInfo.emit('Выходные файлы:')

        self.signals.UpdateProgressBar.emit(80)
        self.signals.PrintInfo.emit(output_dir + 'A.csv')
        listToCsv(A, output_dir + 'A.csv')
        self.signals.PrintInfo.emit(output_dir + 'B.csv')
        listToCsv(B, output_dir + 'B.csv')
        self.signals.PrintInfo.emit(output_dir + 'Fls.csv')
        listToCsv(Fls, output_dir + 'Fls.csv')
        self.signals.PrintInfo.emit(output_dir + 'output_class.csv')
        listToCsv(class_table, output_dir + 'output_class.csv')
        self.signals.PrintInfo.emit(output_dir + 'tfidf_matrix.csv')
        writeStringToFile2(log_tfidf, output_dir + 'tfidf_matrix.csv')

    def classification_id3(self, needPreprocessing):
        output_dir = self.output_dir + 'id3_out/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        input_dir = self.method_input_dir
        fdata, fclass, split = makeFileList(input_dir)
        trainingSet = fdata[:split]
        trainingClass = fclass[:split]
        testSet = fdata[split:]

        Classification_Text_ID3(input_dir, output_dir, trainingSet, trainingClass, testSet)






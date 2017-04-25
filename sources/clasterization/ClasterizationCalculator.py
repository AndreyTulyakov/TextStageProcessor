#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import math
import copy
import numpy as np
import shutil
import os
import random

from PyQt5.QtCore import QObject
from PyQt5.QtCore import QThread
from PyQt5.QtCore import pyqtSignal

from sources.TextPreprocessing import writeStringToFile, makePreprocessing, makeFakePreprocessing
from sources.utils import makePreprocessingForAllFilesInFolder


def sim(D, i, j):
    summ = 0
    summ_sqrt_i = 0
    summ_sqrt_j = 0
    for k in range(len(D[0])):
        summ += D[i][k] * D[j][k]

    for k in range(len(D[0])):
        summ_sqrt_i += D[i][k] * D[i][k]
    summ_sqrt_i = math.sqrt(summ_sqrt_i)

    for k in range(len(D[0])):
        summ_sqrt_j += D[j][k] * D[j][k]
    summ_sqrt_j = math.sqrt(summ_sqrt_j)

    sim = summ / (summ_sqrt_i * summ_sqrt_j)
    return sim


def GetS(D):
    n = len(D)
    m = len(D[0])
    S = [[0] * m for i in range(n)]

    for row in range(n):
        for item in range(m):
            if (item < row):
                S[row][item] = sim(D, row, item)
            else:
                S[row][item] = 0

    return S


def indexes2DocNames(docIndexes,filenames):
    res =[]
    for doc in range(len(docIndexes)):
        res[doc] = filenames[docIndexes]
    return res


def agglomerative_hierarchical_clustering(D):
    S = GetS(D)
    F = []
    for k in range(len(D)):
        F[k] = 1
    A = []
    for k in range(len(D)):
        F[k] = 1

    return 0


def FindUnionDist(Dist, F):
    min_i = 0
    min_j = 0
    min = 1000
    n = len(Dist)
    m = len(Dist[0])
    for i in range(n):
        for j in range(i):
            if ((Dist[i][j] < min) & (F[i] == 1) & (j < i)):
                min = Dist[i][j]
                min_i = i
                min_j = j

    return [min_i, min_j], Dist[min_i][min_j]


def FindUnionSim(Sim, F):
    max_i = 0
    max_j = 0
    max = 0
    n = len(Sim)
    m = len(Sim[0])
    for i in range(n):
        for j in range(i):
            if ((Sim[i][j] > max) & (F[i] == 1) & (j < i)):
                max = Sim[i][j]
                max_i = i
                max_j = j

    return [max_i, max_j], Sim[max_i][max_j]


class Cluster:
    ap = 1 / 2
    aq = 1 / 2
    b = 0
    g = 0

    def __init__(self, doc, dist):
        self.docs = [{doc}]
        self.Dist = dist

    def AddDoc(self, doc, dist, n):
        i = 0
        self.docs.append(doc)
        newDist = []
        for k in range(len(dist)):
            newDist[k] = Cluster.ap * (self.Dist[k]) + Cluster.aq * (dist[k]) + Cluster.b * (
                self.Dist[n]) + Cluster.g * (math.fabs(self.Dist[k] - dist[k]))


def printDist(dist, texts, filenames):
    dist_string = ''

    for name in filenames:
        dist_string = dist_string + '; doc' + str(name)[str(name).find('/') + 1:str(name).find('.')]
    dist_string += '\n'
    for i in range(len(texts)):
        dist_string += 'doc' + str(filenames[i])[
                               str(filenames[i]).find('/') + 1:str(filenames[i]).find('.')]


def ClusterByDoc(x, clusters):
    for key, value in clusters.items():
        for i in range(len(clusters[key])):
            if (clusters[key][i] == x):
                return key


def UnionClusters(cluster1, cluster2, clusters):
    for i in range(len(clusters[cluster2])):
        clusters[cluster1].append(clusters[cluster2][i])
    clusters.pop(cluster2)


def Cluster2String(clusters, cluster):
    res = '{'
    for i in range(len(clusters[cluster])):
        res += str(clusters[cluster][i]+1)
        if(i<len(clusters[cluster])-1):
            res+=','
    res += '}'
    return res


def Cluster2StringNames(clusters, cluster, filenames):
    res = ''
    clusters[cluster].sort(key=int, reverse=False)
    for i in range(len(clusters[cluster])):
        res += str(clusters[cluster][i] + 1) + ' = ' + os.path.basename(filenames[clusters[cluster][i]]) + '\n'

    return res


# Сигналы для потока вычисления
class ClasterizationCalculatorSignals(QObject):
    PrintInfo = pyqtSignal(str)
    Finished = pyqtSignal()
    UpdateProgressBar = pyqtSignal(int)


# Класс-поток вычисления
class ClasterizationCalculator(QThread):

    def __init__(self, filenames, output_dir, morph, configurations, textEdit):
        super().__init__()
        self.filenames = filenames
        self.output_dir = output_dir
        self.morph = morph
        self.configurations = configurations
        self.textEdit = textEdit
        self.texts = []
        self.categories = dict()
        self.signals = ClasterizationCalculatorSignals()
        self.method = '1'
        self.minimalWordsLen = 3
        self.clusterCount = 2
        self.eps = 0.01
        self.m = 2
        self.minPts = 0.3
        self.need_preprocessing = False
        self.first_call = True
        self.texts = []

    def setMethod(self, method_name):
        self.method = method_name

    def setMinimalWordsLen(self, value):
        self.minimalWordsLen = value

    def setEps(self, value):
        self.eps = value

    def setM(self,value):
        self.m = value

    def setMinPts(self,value):
        self.minPts = value

    def setClusterCount(self,value):
        self.clusterCount = value

    def run(self):
        self.signals.UpdateProgressBar.emit(0)

        if self.first_call:
            if self.need_preprocessing:
                self.signals.PrintInfo.emit("Препроцессинг...")
                self.texts = makePreprocessing(self.filenames, self.morph, self.configurations, self.textEdit)
            else:
                self.signals.PrintInfo.emit("Препроцессинг - пропускается")
                self.texts = makeFakePreprocessing(self.filenames)
        else:
            if self.need_preprocessing:
                self.signals.PrintInfo.emit("Препроцессинг - использование предыдущих результатов.")
            else:
                self.signals.PrintInfo.emit("Препроцессинг - пропускается")


        if(self.method == '1'):
            self.makeHierarhyClasterization()

        if(self.method == '2'):
            self.makeClasterizationKMiddle(self.clusterCount)

        if(self.method == '3'):
            self.makeClasterizationSMiddle(self.clusterCount,self.eps,self.m)

        if (self.method == '4'):
            self.makeDBSCANClasterization(self.eps,self.minPts)

        if self.first_call and self.need_preprocessing:
            self.first_call = False

        self.signals.PrintInfo.emit('Рассчеты закончены!')
        self.signals.Finished.emit()

    def makeHierarhyClasterization(self):
        self.signals.UpdateProgressBar.emit(0)
        self.signals.PrintInfo.emit('Иерархическая кластеризация' + '\n')

        texts = self.texts

        output_dir = self.configurations.get("output_files_directory", "output_files") + "/clasterization/Hierarchical/"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Нахождение матрицы весов
        t_all = dict()

        for text in texts:
            for key, value in text.sorted_word_frequency:
                t_all[key] = t_all.get(key, 0) + 1

        # Найти df
        df_string = ''
        df_string = df_string + "Слово;Используется в документах\n"
        for key, value in t_all.items():
            df_string = df_string + key + ';' + str(value) + '\n'
        writeStringToFile(df_string.replace('\n ', '\n'), output_dir + 'df.csv')

        W = [[0 for x in range(len(t_all))] for y in range(len(texts))]
        print('len(texts)=' + str(len(texts)))
        print('len(t_all)=' + str(len(t_all)))
        W_norm = [0 for x in range(len(texts))]
        i = 0
        j = 0
        for row in range(len(texts)):
            j = 0
            for key, value in t_all.items():
                text = texts[row]
                if (key in text.word_frequency.keys()):
                    frequency_in_this_doc = text.word_frequency[key]
                else:
                    frequency_in_this_doc = 0
                W[i][j] = frequency_in_this_doc * math.log10(len(texts) / value)
                W_norm[i] += math.pow(W[i][j], 2)

                j += 1
            W_norm[i] = math.sqrt(W_norm[i])
            i += 1

        for i in range(len(texts)):
            for j in range(len(t_all)):
                W[i][j] /= W_norm[i]

        W_string = ''
        W_string = W_string + "Нормированные веса\n"
        for key, value in t_all.items():
            W_string = W_string + ';' + key
        W_string += '\n'
        i = 0
        for row in W:
            W_string += self.filenames[i]
            for item in row:
                W_string = W_string + ';' + str(round(item, 10))
            W_string += '\n'
            i += 1
        writeStringToFile(W_string.replace('\n ', '\n'), output_dir + 'W.csv')

        S = GetS(W)
        sim_string = ''
        for name in self.filenames:
            sim_string = sim_string + ';' + name
        sim_string += '\n'
        for i in range(len(texts)):
            sim_string += self.filenames[i]
            for j in range(len(t_all)):
                sim_string = sim_string + ';' + str(S[i][j])
            sim_string += '\n'
        writeStringToFile(sim_string.replace('\n ', '\n'), output_dir + 'sim.csv')

        n = len(texts)
        m = len(texts)
        S = [[0 for x in range(n)] for y in range(m)]

        #Находим таблицу Dist
        for i in range(n):
            for j in range(m):
                summ = 0
                for k in range(len(t_all)):
                    summ += math.pow(W[i][k] - W[j][k], 2)
                S[i][j] = math.sqrt(summ)

        dist_string = ''

        for name in self.filenames:
            dist_string = dist_string + ';doc' + str(name)[str(name).find('/') + 1:str(name).find('.')]
        dist_string += '\n'
        for i in range(len(texts)):
            dist_string += 'doc' + str(self.filenames[i])[
                                   str(self.filenames[i]).find('/') + 1:str(self.filenames[i]).find('.')]
            for j in range(len(texts)):
                dist_string = dist_string + ';' + str(round(S[i][j], 2))
            dist_string += '\n'
        writeStringToFile(dist_string.replace('\n ', '\n').replace('.', ','), output_dir + 'dist.csv')

        doc2cluster = [0 for x in range(len(texts))]
        for i in range(len(texts)):
            doc2cluster[i] = i

        clusters = dict()
        for i in range(len(texts)):
            clusters[i] = [i]

        F = [1 for x in range(len(texts))]

        result = ''
        clustersString = ''
        #Find unions with dist
        for k in range(len(texts) - 1):
            printDist(S, texts, self.filenames)
            union, currDist = FindUnionDist(S, F)
            firstCluster = ClusterByDoc(union[0], clusters)
            secondCluster = ClusterByDoc(union[1], clusters)
            print('found clusters = ' + str(firstCluster) + ' and ' + str(secondCluster))
            # print(str(union[0] + 1) + ' + ' + str(union[1] + 1) + ' with dist= ' + str(dist[union[0]][union[1]]))
            result += '\n\nStep' + str(k) + '\nUnion --->;' + Cluster2String(clusters, firstCluster) + ';+;' \
                      + Cluster2String(clusters, secondCluster) + ';=;'
            clustersString += '\n\nStep' + str(k) + '\nUnion --->;' + Cluster2String(clusters, firstCluster) + ';+;' \
                      + Cluster2String(clusters, secondCluster) + ';=;'
            UnionClusters(firstCluster, secondCluster, clusters)
            clustersString += Cluster2String(clusters, ClusterByDoc(union[0], clusters)) + ';dist = ' + str(currDist) + '\n'
            clustersString += Cluster2StringNames(clusters, ClusterByDoc(union[0], clusters), self.filenames) + '\n'
            result += Cluster2String(clusters, ClusterByDoc(union[0], clusters)) + ';dist = ' + str(currDist) + '\n'
            result += Cluster2StringNames(clusters, ClusterByDoc(union[0], clusters), self.filenames) + '\n'


            # doc2cluster[union[1]] = '{' + doc2cluster[union[1]] + ',' + doc2cluster[union[0]] + '}'
            # doc2cluster[union[0]] = ''

            F[union[0]] = 0
            # F[union[1]] = 0
            new_sim = [[0 for x in range(len(texts))] for y in range(len(texts))]
            for i in range(len(texts)):
                for j in range(len(texts)):
                    new_sim[i][j] = S[i][j]

            #Запишем новую таблицу расстояний
            result+="New Dist\n"
            for name in self.filenames:
                result += ';' + str(name)[str(name).rfind('/') + 1:str(name).find('.')]
            result += '\n'
            for i in range(len(texts)):
                result += '' + str(self.filenames[i])[
                                       str(self.filenames[i]).rfind('/') + 1:str(self.filenames[i]).find('.')]
                for j in range(len(texts)):
                    result += ';' + str(round(new_sim[i][j], 2))
                result += '\n'

            for j in range(len(texts)):
                for i in range(2):
                    new_sim[j][union[i]] = 0.5 * (S[j][union[0]]) + 0.5 * (S[j][union[1]])

            S = new_sim

        writeStringToFile(result.replace('\n ', '\n').replace('.',','), output_dir + 'stepsDist.csv')
        writeStringToFile(clustersString.replace('\n ', '\n').replace('.', ','), output_dir + 'clusters.csv')
        # Find unions with Sim
        F = [1 for x in range(len(texts))]
        result = ''
        clusters = dict()
        for i in range(len(texts)):
            clusters[i] = [i]

        for k in range(len(texts) - 1):
            #printDist(dist, texts, self.filenames)
            union, currSim = FindUnionSim(S, F)
            firstCluster = ClusterByDoc(union[0], clusters)
            secondCluster = ClusterByDoc(union[1], clusters)
            print('found clusters = ' + str(firstCluster) + ' and ' + str(secondCluster))
            # print(str(union[0] + 1) + ' + ' + str(union[1] + 1) + ' with dist= ' + str(sim[union[0]][union[1]]))
            result += '\n\nStep' + str(k) + '\nUnion --->;' + Cluster2String(clusters, firstCluster) + ';+;' \
                      + Cluster2String(clusters, secondCluster) + ';=;'
            UnionClusters(firstCluster, secondCluster, clusters)
            result += Cluster2String(clusters, ClusterByDoc(union[0], clusters)) + ';sim = ' + str(currSim) + '\n'
            result += Cluster2StringNames(clusters, ClusterByDoc(union[0], clusters), self.filenames) + '\n'

            # doc2cluster[union[1]] = '{' + doc2cluster[union[1]] + ',' + doc2cluster[union[0]] + '}'
            # doc2cluster[union[0]] = ''

            F[union[0]] = 0
            # F[union[1]] = 0
            new_sim = [[0 for x in range(len(texts))] for y in range(len(texts))]
            for i in range(len(texts)):
                for j in range(len(texts)):
                    new_sim[i][j] = S[i][j]

            for j in range(len(texts)):
                for i in range(2):
                    new_sim[j][union[i]] = 0.5 * (S[j][union[0]]) + 0.5 * (S[j][union[1]])

            S = new_sim

        writeStringToFile(result.replace('\n ', '\n'), output_dir + 'stepsSim.csv')

        # for i in range(len(texts)):
        #     self.signals.PrintInfo.emit(str(doc2cluster[i])+'\n')
        #     print(doc2cluster[i])
        self.signals.UpdateProgressBar.emit(100)

    def makeClasterizationKMiddle(self,ClusterCount):
        self.signals.UpdateProgressBar.emit(0)
        self.signals.PrintInfo.emit('Кластеризация к-средних' + '\n')

        texts = self.texts
        output_dir = self.configurations.get("output_files_directory", "output_files") + "/clasterization/KMiddle/"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.signals.UpdateProgressBar.emit(20)

        result = ''
        clusters =''
        #легенда в ответ
        for doc in range(len(self.filenames)):
            result += str(doc) + ' = ' + os.path.basename(self.filenames[doc]) + '\n'
            clusters += str(doc+1) + ' = ' + os.path.basename(self.filenames[doc]) + '\n'
        result += '\n'
        clusters += '\n'

        # Нахождение матрицы весов
        self.signals.PrintInfo.emit('Нахождение матрицы весов' + '\n')
        t_all = dict()

        for text in texts:
            for key, value in text.sorted_word_frequency:
                t_all[key] = t_all.get(key, 0) + 1

        # Найти df
        df_string = ''
        df_string = df_string + "Слово;Используется в документах\n"
        for key, value in t_all.items():
            df_string = df_string + key + ';' + str(value) + '\n'
        writeStringToFile(df_string.replace('\n ', '\n').replace('.', ','), output_dir + 'df.csv')
        self.signals.UpdateProgressBar.emit(25)

        W = [[0 for x in range(len(t_all))] for y in range(len(texts))]
        print('len(texts)=' + str(len(texts)))
        print('len(t_all)=' + str(len(t_all)))
        W_norm = [0 for x in range(len(texts))]
        i = 0
        j = 0
        for row in range(len(texts)):
            j = 0
            for key, value in t_all.items():
                text = texts[row]
                if (key in text.word_frequency):
                    frequency_in_this_doc = text.word_frequency[key]
                else:
                    frequency_in_this_doc = 0
                W[i][j] = frequency_in_this_doc * math.log10(len(texts) / value)
                W_norm[i] += math.pow(W[i][j], 2)
                # print('W[' + key + '][' + filenames[i] + '] = ' + str(frequency_in_this_doc) + '*Log(' + str(
                #    len(texts)) + '/' + str(value) + ') = ' + str(W[i][j]))

                j += 1
            W_norm[i] = math.sqrt(W_norm[i])
            print('wnorm = ' + str(W_norm[i]))
            i += 1

        for i in range(len(texts)):
            for j in range(len(t_all)):
                W[i][j] /= W_norm[i]

        W_string = ''
        W_string = W_string + "Нормированные веса\n"
        for key, value in t_all.items():
            W_string = W_string + ';' + key
        W_string += '\n'
        i = 0
        for row in W:
            W_string += self.filenames[i]
            for item in row:
                W_string = W_string + ';' + str(round(item, 10))
            W_string += '\n'
            i += 1
        writeStringToFile(W_string.replace('\n ', '\n').replace('.', ','), output_dir + 'W.csv')

        S = GetS(W)
        sim_string = ''
        for name in self.filenames:
            sim_string = sim_string + ';' + name
        sim_string += '\n'
        for i in range(len(texts)):
            sim_string += self.filenames[i]
            for j in range(len(t_all)):
                sim_string = sim_string + ';' + str(S[i][j])
            sim_string += '\n'
        writeStringToFile(sim_string.replace('\n ', '\n'), output_dir + 'sim.csv')

        n = len(texts)
        m = len(texts)
        S = [[0 for x in range(n)] for y in range(m)]

        self.signals.UpdateProgressBar.emit(50)

        for i in range(n):
            for j in range(m):
                summ = 0
                for k in range(len(t_all)):
                    summ += math.pow(W[i][k] - W[j][k], 2)
                S[i][j] = math.sqrt(summ)

        dist_string = ''

        for name in self.filenames:
            dist_string = dist_string + ';doc' + str(name)[str(name).find('/') + 1:str(name).find('.')]
        dist_string += '\n'
        for i in range(len(texts)):
            dist_string += 'doc' + str(self.filenames[i])[
                                   str(self.filenames[i]).find('/') + 1:str(self.filenames[i]).find('.')]
            for j in range(len(texts)):
                dist_string = dist_string + ';' + str(round(S[i][j], 2))
            dist_string += '\n'
        writeStringToFile(dist_string.replace('\n ', '\n').replace('.', ','), output_dir + 'dist.csv')
        self.signals.UpdateProgressBar.emit(75)
        # Проверим для каждого уровня
        centroidCount = ClusterCount
        # for centroidCount in range(len(texts), 0, -1):
        calc_string = ''

        if(centroidCount>0):
            result += 'Кол-во кластеров - ' + str(centroidCount) + '\n'
            calc_string += 'k=' + str(centroidCount) + '\n'
            clusterCenteroids = dict()
            # Шаг 1. Инициализация центров кластеров $, j = 1,k, например, случайными числами.
            #случайные числа
            clusterCenteroids = [[random.randrange(0, 100, 1)/10000 for x in range(len(t_all))] for y in range(centroidCount)]
            #китайцы - китайский пекин шанхай макао япония токио
            # index=0
            # for key, value in t_all.items():
            #     if(key == 'китайский'):
            #         clusterCenteroids[0][index] = 0.96
            #         clusterCenteroids[1][index] = 0.49
            #     if (key == 'пекин'):
            #         clusterCenteroids[0][index] = 0.8
            #         clusterCenteroids[1][index] = 0.14
            #     if (key == 'шанхай'):
            #         clusterCenteroids[0][index] = 0.42
            #         clusterCenteroids[1][index] = 0.91
            #     if (key == 'макао'):
            #         clusterCenteroids[0][index] = 0.79
            #         clusterCenteroids[1][index] = 0.96
            #     if (key == 'япония'):
            #         clusterCenteroids[0][index] = 0.66
            #         clusterCenteroids[1][index] = 0.04
            #     if (key == 'токио'):
            #         clusterCenteroids[0][index] = 0.85
            #         clusterCenteroids[1][index] = 0.93
            #     index = index + 1

            #запишем исходные кластеры
            calc_string += 'Изначальные кластеры\n;'
            for key, value in t_all.items():
                calc_string += key + ';'
            calc_string += '\n'
            for i in range(centroidCount):
                calc_string += 'C' + str(i+1)+';'
                for j in range(len(t_all)):
                    calc_string+= str(clusterCenteroids[i][j])+';'
                calc_string+='\n'

            #Шаг 2. Cj={} , j=1,k.
            doc2cluster = [0 for x in range(len(texts))]
            for i in range(len(texts)):
                doc2cluster[i] = -1

            # for i in range(len(texts)):
            clusterDist = [[0 for x in range(len(texts))] for y in range(centroidCount)]

            # Находим таблицу Dist для центроидов
            calc_string += '\n\nРасстояния между кластерами(row) и документами(col)\n'

            #Евклидово
            for i in range(centroidCount):
                for j in range(len(texts)):
                    summ = 0
                    for k in range(len(t_all)):
                        summ += math.pow(W[j][k] - clusterCenteroids[i][k], 2)
                    clusterDist[i][j] = math.sqrt(summ)
                    calc_string += str(clusterDist[i][j])+';'
                calc_string+='\n'

            #Манхеттен
            # for i in range(centroidCount):
            #     for j in range(len(texts)):
            #         summ = 0
            #         for k in range(len(t_all)):
            #             summ += math.fabs(W[j][k] - clusterCenteroids[i][k])
            #         clusterDist[i][j] = summ
            #         calc_string += str(clusterDist[i][j])+';'
            #     calc_string+='\n'

            while True:
                changes = False
                #Шаг 3. Для каждого  di∈ D:

                for doc in range(len(texts)):
                    minDistance= 9999
                    minCluster = -1
                    currentDistance = 0
                    for cluster in range(0, centroidCount):
                        #for dist in range(len(texts)-1):
                            currentDistance= abs(clusterDist[cluster][doc])
                            #currentDistance/=len(t_all)
                            if (currentDistance<minDistance):
                                minDistance = currentDistance
                                minCluster = cluster
                    if(doc2cluster[doc] != minCluster):
                        doc2cluster[doc] = minCluster
                        changes = True

                # запишем результаты
                calc_string += '\n\nРаспределение документов по кластерам\n'
                for cluster in range(centroidCount):
                    calc_string += ';C' + str(cluster+1)
                    for doc in range(len(texts)):
                        if (doc2cluster[doc] == cluster):
                            calc_string += '; ' + str(doc+1)
                    calc_string += '\n'
                calc_string += '\n'

                docCount=0
                newDistance = 0
                for cluster in range(centroidCount):
                    summ = 0
                    for doc in range(len(texts)):
                        if(doc2cluster[doc] == cluster):
                            if(docCount==0):
                                clusterCenteroids[cluster] = [0 for x in range(len(t_all))]
                            docCount+=1
                            for k in range(len(t_all)):
                                clusterCenteroids[cluster][k] += W[doc][k]
                    if(docCount!=0):
                        for k in range(len(t_all)):
                            clusterCenteroids[cluster][k] /= docCount
                    docCount=0

                #обновим центры кластеров
                calc_string += '\n\nНовые центроиды кластеров\n;'
                for key, value in t_all.items():
                    calc_string += key + ';'
                calc_string += '\n'
                for i in range(centroidCount):
                    calc_string += 'C' + str(i+1) + ';'
                    for j in range(len(t_all)):
                        calc_string += str(clusterCenteroids[i][j]) + ';'
                    calc_string += '\n'

                # Обновляем таблицу Dist для центроидов
                calc_string += '\n\nРасстояния между кластерами(row) и документами(col)\n'
                for i in range(centroidCount):
                    for j in range(len(texts)):
                        summ = 0
                        for k in range(len(t_all)):
                            summ += math.pow(W[j][k] - clusterCenteroids[i][k], 2)
                        clusterDist[i][j] = math.sqrt(summ)
                        calc_string += str(clusterDist[i][j]) + ';'
                    calc_string += '\n'

                if(changes==False):
                    print("Найдены кластеры в количестве " + str(centroidCount))
                    #запишем результаты
                    clusters += 'Кластеров -'+ str(centroidCount) + '\n'
                    for cluster in range (centroidCount):
                        clusters += ';Кластер'+ str(cluster+1)
                        for doc in range (len(texts)):
                            if (doc2cluster[doc] == cluster):
                                clusters += '; ' + str(doc+1)
                        clusters += '\n'
                    clusters += '\n'
                    break

        writeStringToFile(result.replace('\n ', '\n').replace('.',','), output_dir + 'steps.csv')
        writeStringToFile(calc_string.replace('\n ', '\n').replace('.',','), output_dir + 'calc.csv')
        writeStringToFile(clusters.replace('\n ', '\n').replace('.', ','), output_dir + 'clusters.csv')
        self.signals.UpdateProgressBar.emit(100)
        self.signals.PrintInfo.emit('Кластеризация к-средних завершена' + '\n')

    def makeClasterizationSMiddle(self, ClusterCount, m, eps):
        self.signals.UpdateProgressBar.emit(0)
        self.signals.PrintInfo.emit('Нечёткий алгоритм с-средних' + '\n')

        texts = self.texts
        output_dir = self.configurations.get("output_files_directory", "output_files") + "/clasterization/SMiddle/"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        m = int(m)
        eps = float(eps)

        # eps = 0.01
        result = ''
        clusters=''
        # легенда в ответ
        for doc in range(len(self.filenames)):
            result += str(doc) + ' = ' + os.path.basename(self.filenames[doc]) + '\n'
            clusters += str(doc) + ' = ' + os.path.basename(self.filenames[doc]) + '\n'
        result += '\n'
        clusters += '\n'

        # Нахождение матрицы весов
        self.signals.PrintInfo.emit('Нахождение матрицы весов' + '\n')
        t_all = dict()

        for text in texts:
            for key, value in text.sorted_word_frequency:
                t_all[key] = t_all.get(key, 0) + 1

        self.signals.UpdateProgressBar.emit(15)
        # Найти df
        df_string = ''
        df_string = df_string + "Слово;Используется в документах\n"
        for key, value in t_all.items():
            df_string = df_string + key + ';' + str(value) + '\n'
        writeStringToFile(df_string.replace('\n ', '\n'), output_dir + 'df.csv')

        W = [[0 for x in range(len(t_all))] for y in range(len(texts))]
        print('len(texts)=' + str(len(texts)))
        print('len(t_all)=' + str(len(t_all)))
        W_norm = [0 for x in range(len(texts))]
        i = 0
        j = 0
        for row in range(len(texts)):
            j = 0
            for key, value in t_all.items():
                text = texts[row]
                if (key in text.word_frequency):
                    frequency_in_this_doc = text.word_frequency[key]
                else:
                    frequency_in_this_doc = 0
                W[i][j] = frequency_in_this_doc * math.log10(len(texts) / value)
                W_norm[i] += math.pow(W[i][j], 2)
                # print('W[' + key + '][' + filenames[i] + '] = ' + str(frequency_in_this_doc) + '*Log(' + str(
                #    len(texts)) + '/' + str(value) + ') = ' + str(W[i][j]))

                j += 1
            W_norm[i] = math.sqrt(W_norm[i])
            print('wnorm = ' + str(W_norm[i]))
            i += 1

        for i in range(len(texts)):
            for j in range(len(t_all)):
                W[i][j] /= W_norm[i]

        W_string = ''
        W_string = W_string + "Нормированные веса\n"
        for key, value in t_all.items():
            W_string = W_string + ';' + key
        W_string += '\n'
        i = 0
        for row in W:
            W_string += self.filenames[i]
            for item in row:
                W_string = W_string + ';' + str(round(item, 10))
            W_string += '\n'
            i += 1
        writeStringToFile(W_string.replace('\n ', '\n'), output_dir + 'W.csv')
        self.signals.UpdateProgressBar.emit(25)

        # Проверим для каждого уровня
        # for centroidCount in range(len(texts), 0, -1):
        #     print('Finding clusters - ' + str(centroidCount))
        if(True):
            centroidCount = ClusterCount
            result += 'Кол-во кластеров - ' + str(centroidCount) + '\n'

            #степень нечеткости 1<m< infinity
            # m = 2

            #номер итерации
            t=0
            result += 'm = ' + str(m) + ';' + 'k = ' + str(centroidCount) + '\n'

            #заполним изначально случайными числами, в сумме по строке - 1
            U0 = [[0 for x in range(centroidCount)] for y in range(len(texts))]
            result += '\nU0\n'
            for i in range(len(texts)):
                remain = 1
                for j in range(centroidCount):
                    if (j != centroidCount - 1):
                        current = random.uniform(0, remain)
                        remain = remain - current
                        U0[i][j] = current
                    else:
                        U0[i][j] = remain
                    result += str(U0[i][j]) + ';'
                result += '\n'
            changes = False
            self.signals.UpdateProgressBar.emit(50)
            while True:
                t = t+1
                result += '\nИтерация' + str(t) + '\n'
                # print('iteration ' + str(t))
                centroids = [[0 for x in range(len(t_all))] for y in range(centroidCount)]
                # Находим таблицу центроидов
                result += '\nЦентроиды\n'
                for j in range(centroidCount):
                    for w_kl in range(len(t_all)):
                        summU = 0
                        summUD = 0
                        for i in range(len(texts)):
                           summU = summU + math.pow(U0[i][j],m)
                        for i in range(len(texts)):
                            #for k in range(len(t_all)):
                            summUD += math.pow(U0[i][j],m) * W[i][w_kl]
                        centroids[j][w_kl] = summUD/summU
                        result += str(centroids[j][w_kl]) + ';'
                    result += '\n'
                # print('centroids founded ')

                # Находим новую таблицу разбиения U1
                U1 = [[0 for x in range(centroidCount)] for y in range(len(texts))]
                for i in range(len(texts)):
                    for j in range(centroidCount):
                        summ = 0
                        for k in range(centroidCount):
                            diff_ij = 0
                            diff_ik = 0
                            for p in range(len(t_all)):
                                diff_ij += math.pow(W[i][p] - centroids[j][p],2)
                                diff_ik += math.pow(W[i][p] - centroids[k][p], 2)
                            diff_ik = math.sqrt(diff_ik)
                            diff_ij = math.sqrt(diff_ij)
                            summ+= math.pow(diff_ij/diff_ik,2/(m-1))
                        U1[i][j] = 1/summ
                # print('new u table founded')

                #запишем в файл новую таблицу
                result += 'U' + str(t) + '\n'
                for i in range(len(texts)):
                    for j in range(centroidCount):
                        result += str(U1[i][j]) + ';'
                    result += '\n'

                #проверим условие остановки
                Udiff = 0
                for i in range(len(texts)):
                    for j in range(centroidCount):
                        Udiff += math.pow(U0[i][j] - U1[i][j], 2)
                Udiff = math.sqrt(Udiff)
                if(Udiff<eps):
                    # #Выберем самые ближайшие к документам кластеры
                    doc2cluster = [-1 for x in range(len(texts))]
                    # Находим таблицу Dist для центроидов
                    clusterDist = [[0 for x in range(len(texts))] for y in range(centroidCount)]
                    for x in range(centroidCount):
                        for y in range(len(texts)):
                            summ = 0
                            for k in range(len(t_all)):
                                summ += math.pow(W[y][k] - centroids[x][k], 2)
                                clusterDist[x][y] = math.sqrt(summ)
                    for doc in range(len(texts)):
                        minDistance = 9999
                        minCluster = -1
                        currentDistance = 0
                        for cluster in range(0, centroidCount):
                            # for dist in range(len(texts)-1):
                            currentDistance = abs(clusterDist[cluster][doc])
                            # currentDistance/=len(t_all)
                            if (currentDistance < minDistance):
                                minDistance = currentDistance
                                minCluster = cluster
                        if (doc2cluster[doc] != minCluster):
                            doc2cluster[doc] = minCluster
                    clusters += '\nКластеры'
                    print("Найдены кластеры в количестве " + str(centroidCount))
                    # запишем результаты
                    # result += 'Кластеров -'+ str(centroidCount) + '\n'
                    for cluster in range(centroidCount):
                        clusters += '\nКластер' + str(cluster+1)
                        for doc in range(len(texts)):
                            if (doc2cluster[doc] == cluster):
                                clusters += '; ' + str(doc+1)
                    break
                U0 = U1
                if(t>1000):
                    self.signals.PrintInfo.emit('АЛГОРИТМ РАСХОДИТСЯ! ВЫБЕРИТЕ ДРУГИЕ ПАРАМЕТРЫ' + '\n')
                    return
                # print('continue iterations ')
        self.signals.UpdateProgressBar.emit(100)
        writeStringToFile(result.replace('\n ', '\n').replace('.', ','), output_dir + 'steps.csv')
        writeStringToFile(clusters.replace('\n ', '\n').replace('.', ','), output_dir + 'clusters.csv')
        self.signals.PrintInfo.emit('Кластеризация Нечёткий алгоритм с-средних завершена' + '\n')

    def makeDBSCANClasterization(self,eps, minPts):
        D = []

        self.signals.UpdateProgressBar.emit(0)
        self.signals.PrintInfo.emit('Алгоритм DBSCAN' + '\n')

        texts = self.texts
        output_dir = self.configurations.get("output_files_directory", "output_files") + "/clasterization/DBScan/"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        minPts = float(minPts)
        eps = float(eps)

        # eps = 0.01
        result = 'Алгоритм DBSCAN\n'
        clustersString = 'Кластеры\n'
        # легенда в ответ
        for doc in range(len(self.filenames)):
            result += str(doc+1) + ' = ' + os.path.basename(self.filenames[doc]) + '\n'
            clustersString += str(doc+1) + ' = ' + os.path.basename(self.filenames[doc]) + '\n'
        result += '\n'
        clustersString += '\n'

        # Нахождение матрицы весов
        self.signals.PrintInfo.emit('Нахождение матрицы весов' + '\n')
        t_all = dict()

        for text in texts:
            for key, value in text.sorted_word_frequency:
                t_all[key] = t_all.get(key, 0) + 1

        self.signals.UpdateProgressBar.emit(15)
        # Найти df
        df_string = ''
        df_string = df_string + "Слово;Используется в документах\n"
        for key, value in t_all.items():
            df_string = df_string + key + ';' + str(value) + '\n'
        writeStringToFile(df_string.replace('\n ', '\n'), output_dir + 'df.csv')

        W = [[0 for x in range(len(t_all))] for y in range(len(texts))]
        print('len(texts)=' + str(len(texts)))
        print('len(t_all)=' + str(len(t_all)))
        W_norm = [0 for x in range(len(texts))]
        i = 0
        j = 0
        for row in range(len(texts)):
            j = 0
            for key, value in t_all.items():
                text = texts[row]
                if (key in text.word_frequency):
                    frequency_in_this_doc = text.word_frequency[key]
                else:
                    frequency_in_this_doc = 0
                W[i][j] = frequency_in_this_doc * math.log10(len(texts) / value)
                W_norm[i] += math.pow(W[i][j], 2)
                # print('W[' + key + '][' + filenames[i] + '] = ' + str(frequency_in_this_doc) + '*Log(' + str(
                #    len(texts)) + '/' + str(value) + ') = ' + str(W[i][j]))

                j += 1
            W_norm[i] = math.sqrt(W_norm[i])
            # print('wnorm = ' + str(W_norm[i]))
            i += 1

        for i in range(len(texts)):
            for j in range(len(t_all)):
                W[i][j] /= W_norm[i]

        W_string = ''
        W_string = W_string + "Нормированные веса\n"
        for key, value in t_all.items():
            W_string = W_string + ';' + key
        W_string += '\n'
        i = 0
        for row in W:
            D.append([i+1,row])
            W_string += self.filenames[i]
            for item in row:
                W_string = W_string + ';' + str(round(item, 10))
            W_string += '\n'
            i += 1
        writeStringToFile(W_string.replace('\n ', '\n').replace('.', ','), output_dir + 'W.csv')
        self.signals.UpdateProgressBar.emit(25)

        pt = ()
        stepsString=''
        # lines = open(csv, 'r').read().splitlines()

        # Remember to set a value for eps and minPts. Here
        # they are set to 0.3 and 3.
        stepsString += 'Steps\n'
        stepsString+= 'eps =;' + str(eps) + '\nminPts=;' +str(minPts) + '\n\n\n\n'
        myDBSCAN = DBSCAN(D, eps, minPts)
        res = ''
        results= myDBSCAN.run()
        clusters = results[0]
        noise = results[1]
        stepsString+=results[2]

        noiseDocs =[]
        for doc in range(len(texts)):
            noiseDocs.append(doc+1)

        # Teting printClusters()
        #myDBSCAN.printClusters()
        id=0
        for doc in W:
            for cluster in clusters:
                for pts in cluster.pts[1]:
                    if(doc==pts):
                        cluster.addDoc(id+1)
                        if(noiseDocs.count(id+1)!=0):
                            noiseDocs.remove(id+1)
            id = id + 1
        # id = 0
        # for doc in W:
        #     for noisedoc in noise:
        #             if (doc == noisedoc):
        #                 noiseDocs.append(id + 1)

            # id=id+1
        # Manually printing
        # print('Clusters')
        clustersString += 'Clusters\n'
        for cluster in clusters:
            clustersString+= 'C' + str(cluster.cid+1) + ':;'
            for doc in cluster.docs:
                clustersString+=str(doc)+';'
            clustersString += '\n'

        clustersString +='\nNoise\n'
        for noiceDoc in noiseDocs:
            clustersString += str(noiceDoc) + ';'
        writeStringToFile(clustersString.replace('\n ', '\n').replace('.', ','), output_dir + 'Clusters.csv')
        stepsString+='\n\n' + clustersString
        writeStringToFile(stepsString.replace('\n ', '\n').replace('.', ','), output_dir + 'Steps.csv')

class Cluster(object):
    """ A Cluster is just a wrapper for a list of points.
    Each Cluster object has a unique id. DBSCAN.run()
    returns a list of Clusters.
    """

    cid = 0  # Keep track of cluster ids

    def __init__(self):
        self.cid = Cluster.cid
        self.docs = []
        self.pts = []

        Cluster.cid += 1  # Increment the global id

    def addPoint(self, p):
        self.pts.append(p)
    def addDoc(self, doc):
        self.docs.append(doc)

class DBSCAN(object):
    """ Parameters
        ----------
        D: list of tuples
            stores points as a list of tuples
            of the form (<string id>, <float x>, <float y>)
            E.g. D = [('001', 0.5, 2.1), ('002', 1.0, 2.4)]
            Point ids don't have to be unique.
        eps: float
            maximum distance for two points to be
            considered the same neighborhood
            E.g. 0.001
        minPts: int
            Minimum number of points in a neighborhood for
            a point to be considered a core point. This
            includes the point itself.
            E.g. 4
        Returns
        -------
        A tuple of a list of Cluster objects and a list of
        noise, e.i. ([<list clusters>, <list noise pts>])
        Methods
        -------
        printClusters() - handy method for printing results
        run()           - run DBSCAN
        Example Usage
        -------------
        import dbscan
        dbs = DBSCAN(D, 0.001, 4)
        clusters = dbs.scan()
        # Print with printClusters
        dbs.printClusters()
        # Print with iteration
        for cluster in clusters:
            print(cluster.cid, cluster.pts)
    """

    def __init__(self, D, eps, minPts):
        self.D = D
        self.minPts = minPts

        # This implementation uses Manhattan distance
        # so the eps is squared. This is because
        # calculating sqrt for Euclidean distance is much
        # slower than calculating squares.
        self.eps = eps

        self.Clusters = []  # Results stored here
        self.NOISE = []  # Noise points
        self.visited = []  # For keeping track of pts
        self.stepsString = ''

    def __regionQuery(self, pt):
        eps = self.eps
        D = self.D

        # This implementation uses Manhattan distance
        # to avoid doing an expensive sqrt calculation.
        NeighborhoodPts = []

        for p in D:
            if(p!=pt):
                res = 0
                for k in range(len(pt[1])):
                    res += math.pow(p[1][k] - pt[1][k], 2)
                res = math.sqrt(res)
                if (res<= eps):
                    NeighborhoodPts.append(p)
                    self.stepsString += 'Founded Neighborhood doc' + str(p[0]) + '\n'

        return NeighborhoodPts

    def __expandCluster(self, pt, NeighborhoodPts, C):
        C.addPoint(pt)
        self.stepsString+='Expand C'+str(C.cid+1)+'\n'
        # Localize for some performance boost.
        visited = self.visited
        appendVisited = visited.append
        regionQuery = self.__regionQuery
        minPts = self.minPts
        appendNeighborhoodPts = NeighborhoodPts.append
        Clusters = self.Clusters

        for p in NeighborhoodPts:
            if p not in visited:
                self.stepsString += 'Visit Neighborhood doc' + str(p[0]) + '\n'
                appendVisited(p)
                NewNeighborhoodPts = regionQuery(p)
                if len(NewNeighborhoodPts) >= minPts:
                    for n in NewNeighborhoodPts:
                        if n not in NeighborhoodPts:
                            self.stepsString += 'Add Neighborhood doc' +str(n[0]) + '\n'
                            appendNeighborhoodPts(n)

            # Check if p in any clusters
            for cluster in Clusters:
                if p not in cluster.pts:
                    if p not in C.pts:
                        C.addPoint(p)
                        self.stepsString += 'Added doc' + str(p[0]) + ' to cluster C' + str(C.cid+1) + '\n'
                        break

    def printClusters(self):
        for cluster in self.Clusters:
            cid = cluster.cid
            pts = cluster.pts
            print("Cluster %d" % cid)
            for p in pts:
                print(
                    "id = %s x = %f y = %f"
                    % (p[0], p[1], p[2])
                )

    def run(self):
        index=0
        for pt in self.D:
            self.stepsString+='See doc'+str(index+1)+'\n'
            if pt not in self.visited:
                self.stepsString += 'Visit doc' + str(index + 1) + '\n'
                self.visited.append(pt)
                NeighborhoodPts = self.__regionQuery(pt)
                if len(NeighborhoodPts) < self.minPts:
                    self.stepsString += 'Add to Noise doc' + str(index + 1) + '\n'
                    self.NOISE.append(pt)
                else:
                    C = Cluster()  # new cluster
                    self.Clusters.append(C)
                    self.stepsString += 'Create new cluster C' + str(C.cid + 1) + '\n'
                    self.__expandCluster(
                        pt, NeighborhoodPts, C
                    )
            index += 1
        return (self.Clusters, self.NOISE, self.stepsString)

# Test
def IrisTest(csv):
    D = []
    pt = ()

    lines = open(csv, 'r').read().splitlines()

    # If using the sample iris.data included in the
    # repository, each line is structured:
    #
    # <float sl>, <float sw>, <float pl>, <float pw>, <string id>
    #
    # For sepal length vs petal length comparison, use:
    #
    # pt = (cols[4], float(cols[2]), float(cols[0])
    #
    # Check for more information:
    # https://en.wikipedia.org/wiki/Iris_flower_data_set
    for line in lines:
        cols = line.split(',')
        pt = (cols[4], float(cols[2]), float(cols[0]))
        D.append(pt)

    # Remember to set a value for eps and minPts. Here
    # they are set to 0.3 and 3.
    myDBSCAN = DBSCAN(D, 0.3, 3)

    results = myDBSCAN.run()
    clusters = results[0]
    noise = results[1]

    # Teting printClusters()
    myDBSCAN.printClusters()

    # Manually printing
    for cluster in clusters:
        print(cluster.cid, cluster.pts)



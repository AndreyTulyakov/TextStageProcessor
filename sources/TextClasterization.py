#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pymorphy2
import math

from PyQt5.QtCore import Qt
from pymorphy2 import tokenizers
import os
import random

from PyQt5.QtWidgets import QDialog, QMessageBox, QTextEdit
from PyQt5 import QtCore, QtGui, uic


from sources.TextData import TextData
from sources.TextPreprocessing import *




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


def CombineAll():
    # Объединить все тексты
    for filename in input_filenames:
        with open(filename, 'r') as text_file:
            data = text_file.read()
            with open('combined.txt', 'a') as out_text_file:
                out_text_file.write(data)
    return 0


def dist(w1, w2):
    k = len(w1)
    dist = [0 for x in range(k)]
    summ = 0
    for k in range(len(t_all)):
        summ += math.pow(w1[k] - w2[k], 2)
    dist[k] = math.sqrt(summ)
    return dist


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

class DialogConfigClasterization(QDialog):

    def __init__(self, filenames, morph, configurations, parent):
        super().__init__()
        uic.loadUi('sources/DialogConfigClasterization.ui', self)

        flags = Qt.Window | Qt.WindowSystemMenuHint | Qt.WindowMinimizeButtonHint | Qt.WindowCloseButtonHint;
        self.setWindowFlags(flags)

        self.filenames = filenames
        self.morph = morph
        self.configurations = configurations
        self.parent = parent

        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

        self.buttonClasterization.clicked.connect(self.makeClasterization)
        self.pushButtonKMiddle.clicked.connect(self.makeClasterizationKMiddle)
        self.pushButtonDBSCAN.clicked.connect(self.makeClasterizationDBscan)
        self.pushButtonCMiddle.clicked.connect(self.makeClasterizationSMiddle)
        self.textEdit.setText("")


    def makeClasterization(self):

        self.textEdit.setText("")
        self.configurations["minimal_word_size"] = self.spinBoxMinimalWordsLen.value()
        self.textEdit.append('Иерархическая кластеризация' + '\n')
        texts = makePreprocessing(self.filenames, self.morph, self.configurations, self.textEdit)
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
                if (key in text.word_frequency):
                    frequency_in_this_doc = text.word_frequency[key]
                else:
                    frequency_in_this_doc = 0
                W[i][j] = frequency_in_this_doc * math.log10(len(texts) / value)
                W_norm[i] += math.pow(W[i][j], 2)
                #print('W[' + key + '][' + filenames[i] + '] = ' + str(frequency_in_this_doc) + '*Log(' + str(
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
        #Find unions with dist
        for k in range(len(texts) - 1):
            printDist(S, texts, self.filenames)
            union, currDist = FindUnionDist(S, F)
            firstCluster = ClusterByDoc(union[0], clusters)
            secondCluster = ClusterByDoc(union[1], clusters)
            print('found clusters = ' + str(firstCluster) + ' and ' + str(secondCluster))
            # print(str(union[0] + 1) + ' + ' + str(union[1] + 1) + ' with dist= ' + str(dist[union[0]][union[1]]))
            result += 'Step' + str(k) + '\nUnion --->;' + Cluster2String(clusters, firstCluster) + ';+;' \
                      + Cluster2String(clusters, secondCluster) + ';=;'
            UnionClusters(firstCluster, secondCluster, clusters)
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

            for j in range(len(texts)):
                for i in range(2):
                    new_sim[j][union[i]] = 0.5 * (S[j][union[0]]) + 0.5 * (S[j][union[1]])

            S = new_sim

        writeStringToFile(result.replace('\n ', '\n'), output_dir + 'stepsDist.csv')

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
            result += 'Step' + str(k) + '\nUnion --->;' + Cluster2String(clusters, firstCluster) + ';+;' \
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

        for i in range(len(texts)):
            self.textEdit.append(str(doc2cluster[i])+'\n')
            print(doc2cluster[i])

    def makeClasterizationKMiddle(self):
        self.textEdit.setText("")
        self.configurations["minimal_word_size"] = self.spinBoxMinimalWordsLen.value()
        self.textEdit.append('Кластеризация к-средних' + '\n')
        texts = makePreprocessing(self.filenames, self.morph, self.configurations, self.textEdit)
        output_dir = self.configurations.get("output_files_directory", "output_files") + "/clasterization/KMiddle/"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        result = ''
        #легенда в ответ
        for doc in range(len(self.filenames)):
            result += str(doc) + ' = ' + os.path.basename(self.filenames[doc]) + '\n'
        result += '\n'

        # Нахождение матрицы весов
        self.textEdit.append('Нахождение матрицы весов' + '\n')
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

        # Проверим для каждого уровня
        for centroidCount in range(len(texts), 0, -1):
            result += 'Кол-во кастеров - ' + str(i) + '\n'

            clusterCenteroids = dict()
            # Шаг 1. Инициализация центров кластеров $, j = 1,k, например, случайными числами.
            clusterCenteroids = [[random.randrange(0, 100, 1)/10000 for x in range(len(t_all))] for y in range(centroidCount)]

            #Шаг 2. Cj={} , j=1,k.
            doc2cluster = [0 for x in range(len(texts))]
            for i in range(len(texts)):
                doc2cluster[i] = -1

            for i in range(len(texts)):
                clusterDist = [[0 for x in range(len(texts))] for y in range(centroidCount)]

            # Находим таблицу Dist для центроидов
            for i in range(centroidCount):
                for j in range(len(texts)):
                    summ = 0
                    for k in range(len(t_all)):
                        summ += math.pow(W[j][k] - clusterCenteroids[i][k], 2)
                        clusterDist[i][j] = math.sqrt(summ)

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

                docCount=0
                newDistance = 0
                for cluster in range(centroidCount):
                    summ = 0

                    for doc in range(len(texts)-1):
                        if(doc2cluster[doc] == cluster):
                            if(docCount==0):
                                clusterCenteroids[cluster] = [0 for x in range(len(t_all))]
                            docCount+=1
                            for k in range(len(t_all)):
                                clusterCenteroids[cluster][k] += W[cluster][k]
                    if(docCount!=0):
                        for k in range(len(t_all)):
                            clusterCenteroids[cluster][k] /= docCount
                    docCount=0

                # Находим таблицу Dist для центроидов
                for i in range(centroidCount):
                    for j in range(len(texts)):
                        summ = 0
                        for k in range(len(t_all)):
                            summ += math.pow(W[j][k] - clusterCenteroids[i][k], 2)
                            clusterDist[i][j] = math.sqrt(summ)

                if(changes==False):
                    print("Найдены кластеры в количестве " + str(centroidCount))
                    #запишем результаты
                    #result += 'Кластеров -'+ str(centroidCount) + '\n'
                    for cluster in range (centroidCount):
                        result += ';Кластер'+ str(cluster)
                        for doc in range (len(texts)):
                            if (doc2cluster[doc] == cluster):
                                result += '; ' + str(doc)
                        result += '\n'
                    result += '\n'
                    break

        writeStringToFile(result.replace('\n ', '\n'), output_dir + 'steps.csv')
        self.textEdit.append('Кластеризация к-средних завершена' + '\n')

    def makeClasterizationDBscan(self):
        self.textEdit.setText("")
        self.configurations["minimal_word_size"] = self.spinBoxMinimalWordsLen.value()
        self.textEdit.append('Плотностный алгоритм DBScan' + '\n')
        texts = makePreprocessing(self.filenames, self.morph, self.configurations, self.textEdit)
        output_dir = self.configurations.get("output_files_directory", "output_files") + "/clasterization/DBSCAN/"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        eps = 0.001
        minPt = 0.001


        self.textEdit.append('Кластеризация DBscan завершена' + '\n')

    def makeClasterizationSMiddle(self):
        self.textEdit.setText("")
        self.configurations["minimal_word_size"] = self.spinBoxMinimalWordsLen.value()
        self.textEdit.append('Нечёткий алгоритм с-средних' + '\n')
        texts = makePreprocessing(self.filenames, self.morph, self.configurations, self.textEdit)
        output_dir = self.configurations.get("output_files_directory", "output_files") + "/clasterization/SMiddle/"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        eps = 0.01

        result = ''
        # легенда в ответ
        for doc in range(len(self.filenames)):
            result += str(doc) + ' = ' + os.path.basename(self.filenames[doc]) + '\n'
        result += '\n'

        # Нахождение матрицы весов
        self.textEdit.append('Нахождение матрицы весов' + '\n')
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

        for i in range(n):
            for j in range(m):
                summUD = 0
                for k in range(len(t_all)):
                    summUD += math.pow(W[i][k] - W[j][k], 2)
                S[i][j] = math.sqrt(summUD)

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

        # Проверим для каждого уровня
        for centroidCount in range(len(texts), 0, -1):
            result += 'Кол-во кластеров - ' + str(centroidCount) + '\n'

            #степень нечеткости 1<m< infinity
            m = 3

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
            while True:
                t = t+1
                result += '\nИтерация' + str(t) + '\n'
                centroids = [[0 for x in range(len(t_all))] for y in range(centroidCount)]
                # Находим таблицу центроидов
                result += '\nЦентроиды\n'
                for j in range(centroidCount):
                    summU = 0
                    for i in range(len(texts)):
                       summU = summU + math.pow(U0[i][j],m)
                    for i in range(len(texts)):
                        summUD = 0
                        for k in range(len(t_all)):
                            summUD += math.pow(U0[i][j],m) * W[j][k]
                        centroids[j][i] = summUD/summU
                        result += str(centroids[j][i]) + ';'
                    result += '\n'

                # Находим новую таблицу разбиения U1
                U1 = [[0 for x in range(centroidCount)] for y in range(len(texts))]
                for i in range(len(texts)):
                    for j in range(centroidCount):
                        summ = 0
                        for k in range(centroidCount):
                            diff_ij = 0
                            diff_ik = 0
                            for m in range(len(t_all)):
                                diff_ij += math.pow(W[i][m] - centroids[j][m],2)
                                diff_ik += math.pow(W[i][m] - centroids[k][m], 2)
                            diff_ik = math.sqrt(diff_ik)
                            diff_ij = math.sqrt(diff_ij)
                            summ+= math.pow(diff_ij/diff_ik,2/(m-1))
                        U1[i][j] = 1/summ

                #запишем в файл новую таблицу
                result += 'U' + str(t) + '\n'
                for i in range(len(texts)):
                    for j in range(centroidCount):
                        result += str(U1[i][j]) + ';'
                    result += '\n'



                #проверим условие остановки
                summ = 0
                for i in range(len(texts)):
                    for j in range(centroidCount):
                            summ += math.pow(U0[i][j] - U1[i][j], 2)
                summ = math.sqrt(summ)
                if(summ<eps):
                    break
                U0 = U1

        writeStringToFile(result.replace('\n ', '\n').replace('.', ','), output_dir + 'steps.csv')
        self.textEdit.append('Кластеризация Нечёткий алгоритм с-средних завершена' + '\n')
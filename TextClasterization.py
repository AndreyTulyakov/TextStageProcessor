#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pymorphy2
import math
from pymorphy2 import tokenizers
import os

from TextData import TextData

from TextPreprocessing import *


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


def FindUnion(Dist, F):
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

    return [min_i, min_j]


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


def makeClasterization(filenames, morph, configurations):

    texts = makePreprocessing(filenames, morph, configurations)
    output_dir = configurations.get("output_files_directory", "output_files") + "/clasterization/" 

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
        W_string += filenames[i]
        for item in row:
            W_string = W_string + ';' + str(round(item, 10))
        W_string += '\n'
        i += 1
    writeStringToFile(W_string.replace('\n ', '\n'), output_dir + 'W.csv')

    S = GetS(W)
    sim_string = ''
    for name in filenames:
        sim_string = sim_string + ';' + name
    sim_string += '\n'
    for i in range(len(texts)):
        sim_string += filenames[i]
        for j in range(len(t_all)):
            sim_string = sim_string + ';' + str(S[i][j])
        sim_string += '\n'
    writeStringToFile(sim_string.replace('\n ', '\n'), output_dir + 'sim.csv')

    n = len(texts)
    m = len(texts)
    dist = [[0 for x in range(n)] for y in range(m)]

    for i in range(n):
        for j in range(m):
            summ = 0
            for k in range(len(t_all)):
                summ += math.pow(W[i][k] - W[j][k], 2)
            dist[i][j] = math.sqrt(summ)





    dist_string = ''

    for name in filenames:
        dist_string = dist_string + ';doc' + str(name)[str(name).find('/') + 1:str(name).find('.')]
    dist_string += '\n'
    for i in range(len(texts)):
        dist_string += 'doc' + str(filenames[i])[
                               str(filenames[i]).find('/') + 1:str(filenames[i]).find('.')]
        for j in range(len(texts)):
            dist_string = dist_string + ';' + str(round(dist[i][j], 2))
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

    for k in range(len(texts) - 1):
        printDist(dist, texts, filenames)
        union = FindUnion(dist, F)
        firstCluster = ClusterByDoc(union[0], clusters)
        secondCluster = ClusterByDoc(union[1], clusters)
        print('found clusters = ' + str(firstCluster) + ' and ' + str(secondCluster))
        # print(str(union[0] + 1) + ' + ' + str(union[1] + 1) + ' with dist= ' + str(dist[union[0]][union[1]]))
        result += 'step' + str(k) + ';' + Cluster2String(clusters, firstCluster) + ';+;' \
                  + Cluster2String(clusters, secondCluster) + ';=;'
        UnionClusters(firstCluster, secondCluster, clusters)
        result += Cluster2String(clusters, ClusterByDoc(union[0], clusters)) + '\n'
        # doc2cluster[union[1]] = '{' + doc2cluster[union[1]] + ',' + doc2cluster[union[0]] + '}'
        # doc2cluster[union[0]] = ''

        F[union[0]] = 0
        # F[union[1]] = 0
        new_dist = [[0 for x in range(len(texts))] for y in range(len(texts))]
        for i in range(len(texts)):
            for j in range(len(texts)):
                new_dist[i][j] = dist[i][j]

        for j in range(len(texts)):
            for i in range(2):
                new_dist[j][union[i]] = 0.5 * (dist[j][union[0]]) + 0.5 * (dist[j][union[1]])

        dist = new_dist

    writeStringToFile(result.replace('\n ', '\n'), output_dir + 'steps.csv')

    for i in range(len(texts)):
        print(doc2cluster[i])

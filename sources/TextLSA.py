#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import math

from PyQt5.QtWidgets import QDialog, QMessageBox, QTextEdit
from PyQt5 import QtCore, QtGui, uic

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from sources.TextData import TextData
from sources.TextPreprocessing import *

# Реализация латентно семантического анализа. LSA

# Создание матрицы слова-[частота]-документы
def CreateLSAMatrix(texts, idf_word_data):

    all_documents_count = len(texts);
    all_idf_word_list = dict()
    for text in texts:
        for word in list(text.words_tf_idf.keys()):
            if(idf_word_data.get(word, None) != None):
                all_idf_word_list[word] = idf_word_data[word]


    all_idf_word_keys = list(all_idf_word_list.keys())
    print("TOTAL WORDS:" + str(len(all_idf_word_list)))
    words_count = len(all_idf_word_keys)

    lsa_matrix = np.zeros(shape=(words_count,all_documents_count))

    for t in range(len(texts)):
        for i in range(len(all_idf_word_keys)):

            current_word = all_idf_word_keys[i]
            word_frequency_in_current_text = texts[t].word_frequency.get(current_word, 0)

            lsa_matrix[i][t] = text.words_tf_idf.get(current_word, 0.0)*math.sqrt(word_frequency_in_current_text*10.0)
            #lsx_matrix[i][t] = text.words_tf_idf.get(current_word, 0.0)

    print("LSA:")
    print(lsa_matrix)

    return lsa_matrix, all_idf_word_keys


# Сингулярное разложение на a = u, s, v (S - восстановленный до диагональной матрицы вектор)
def divideSingular(matrix):
    u,s,v = np.linalg.svd(matrix, full_matrices = True)
    S = np.zeros((u.shape[0], v.shape[0]), dtype=complex)
    print("Сингулярный переход:" + str(s))
    S[:v.shape[0], :v.shape[1]] = np.diag(s)
    return u, S, v, s


def cutSingularValue(u, S, v, s):
    # Если сингулярный переход менее 2 измерений то не имеет смысла анализировать.
    if(s.shape[0] == 0):
        print("Сингулярный переход отсутствует. Добавьте документов или слов.")
        exit(0)

    if(s.shape[0] == 1):
        print("Сингулярный переход имеет размерность 1. Добавьте документов или слов.")
        exit(0)

    if(s.shape[0] == 2):
        print("Сингулярный переход имеет размерность 2. 3D Проекция будет отключена.")
        singular_minimal_transfer = 2

    if(s.shape[0] == 3):
        print("Сингулярный переход имеет размерность 3.")
        singular_minimal_transfer = 3

    if(s.shape[0] > 3):
        print("Сингулярный переход имеет размерность " + str(s.shape[0]) + ". Уменьшение до 3...")
        singular_minimal_transfer = 3

    nu = u[0:,0:(singular_minimal_transfer)]
    ns = S[0:(singular_minimal_transfer),0:(singular_minimal_transfer)]
    nv = v[0:(singular_minimal_transfer),0:]

    return nu, ns, nv


def viewLSAGraphics2D(plt, nu, nv, need_words, all_idf_word_keys, texts):
    plt.plot(nu[0],nu[1],'go')
    plt.plot(nv[0],nv[1],'go')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('LSA 2D')
    plt.grid(True)

    min_value = 0.1

    if(need_words):
        for i in range(int(nu.shape[0])):
            if(abs(nu[i][0])>min_value or abs(nu[i][1])>min_value or abs(nu[i][2])>min_value ):
                plt.annotate(str(all_idf_word_keys[i]), xy=(nu[i][0],nu[i][1]), textcoords='data')

    for i in range(len(texts)):
            plt.annotate(str(texts[i].filename), xy=(nv[0][i],nv[1][i]), textcoords='data')

    plt.show()


def viewLSAGraphics3D(plt, nu, nv, need_words, all_idf_word_keys, texts):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    nu = np.transpose(nu)

    min_value = 0.1

    nuxx = []
    nuxy = []
    nuxz = []

    for i in range(len(nu[0])):
        if(abs(nu[0][i])>min_value or abs(nu[1][i])>min_value or abs(nu[2][i])>min_value):
            nuxx.append(nu[0][i])
            nuxy.append(nu[1][i])
            nuxz.append(nu[2][i])

    if(need_words):
       ax.scatter(nuxx,nuxy,nuxz, c='r')#, marker='o')
    ax.scatter(nv[0],nv[1],nv[2], c='b', marker='^')

    for i in range(len(texts)):
           ax.text(nv[0][i], nv[1][i], nv[2][i], str(texts[i].filename), None)

    if(need_words):
       for i in range(len(nuxx)):
           if(abs(nu[0][i])>min_value or abs(nu[1][i])>min_value or abs(nu[2][i])>min_value):
               ax.text(nuxx[i],nuxy[i],nuxz[i], str(all_idf_word_keys[i]), None)

    plt.show()



class DialogConfigLSA(QDialog):

    def __init__(self, filenames, morph, configurations, parent):
        super().__init__()
        uic.loadUi('sources/DialogConfigLSA.ui', self)

        self.filenames = filenames
        self.morph = morph
        self.configurations = configurations
        self.parent = parent

        self.nu = []
        self.ns = []
        self.nv = []
        self.all_idf_word_keys = []
        self.texts = []

        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

        self.buttonMakeLSA.clicked.connect(self.makeLSA)
        self.button2DView.clicked.connect(self.make2DView)
        self.button3DView.clicked.connect(self.make3DView)

        self.textEdit.setText("")


    def makeLSA(self):
        self.textEdit.setText("")
        self.configurations["minimal_word_size"] = self.spinBoxMinimalWordsLen.value()
        self.configurations["cut_ADJ"] = self.checkBoxPrilag.isChecked()
        
        self.texts = makePreprocessing(self.filenames, self.morph, self.configurations, self.textEdit)
        output_dir = self.configurations.get("output_files_directory", "output_files")

        self.repaint()
        self.textEdit.append('Этап ЛСА:\n')
        self.textEdit.append('1) Вычисление показателей TF*IDF.\n')
        self.repaint()
        idf_word_data = calculateWordsIDF(self.texts)
        sorted_IDF = sorted(idf_word_data.items(), key=lambda x: x[1], reverse=False)
        calculateTFIDF(self.texts, idf_word_data)

        log_string = writeWordTFIDFToString(self.texts, idf_word_data)
        writeStringToFile(log_string.replace('\n ', '\n'), output_dir + '/output_stage_6.csv')

        # Вырезаем из TF-IDF % худших слов
        removeTFIDFWordsWithMiniamlMultiplier(self.texts , self.spinBoxCutPercent.value()/100.0)

        log_string = writeWordTFIDFToString(self.texts, idf_word_data)
        writeStringToFile(log_string.replace('\n ', '\n'), output_dir + '/output_stage_7.csv')

        self.textEdit.append('2) Латентно-семантический анализ.\n')
        self.repaint()
        lsa_matrix, self.all_idf_word_keys = CreateLSAMatrix(self.texts, idf_word_data)
        u, S, v, s = divideSingular(lsa_matrix)
        self.nu, self.ns, self.nv = cutSingularValue(u, S, v, s)

        self.button2DView.setEnabled(True)
        self.button3DView.setEnabled(True)

        self.textEdit.append('Успешно завершено.')

        QMessageBox.information(self, "Внимание", "Латентно-семантический анализ завершен!")

    def make2DView(self):
        need_words = self.checkBoxShowWords.isChecked();
        viewLSAGraphics2D(plt, self.nu, self.nv, need_words, self.all_idf_word_keys, self.texts)

    def make3DView(self):
        need_words = self.checkBoxShowWords.isChecked();
        viewLSAGraphics3D(plt, self.nu, self.nv, need_words, self.all_idf_word_keys, self.texts)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os

from PyQt5.QtCore import QObject
from PyQt5.QtCore import QThread
from PyQt5.QtCore import pyqtSignal

import matplotlib.pyplot as plt
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import Birch
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from sklearn.cluster import SpectralClustering
from sklearn.cluster import estimate_bandwidth
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

from sources.TextPreprocessing import writeStringToFile, makePreprocessing, makeFakePreprocessing, \
    getCompiledFromSentencesText
from sklearn.feature_extraction.text import TfidfVectorizer


# Сигналы для потока вычисления

class ClasterizationCalculatorSignals(QObject):
    PrintInfo = pyqtSignal(str)
    Finished = pyqtSignal()
    UpdateProgressBar = pyqtSignal(int)


# Класс-поток вычисления
class ClasteringCalculator(QThread):
    def __init__(self, filenames, output_dir, morph, configurations, textEdit):
        super().__init__()
        self.filenames = filenames
        self.output_dir = output_dir + '/clasterization/'
        self.morph = morph
        self.configurations = configurations
        self.textEdit = textEdit
        self.texts = []
        self.categories = dict()
        self.method = 0
        self.signals = ClasterizationCalculatorSignals()

        self.need_preprocessing = False
        self.need_tf_idf = True
        self.first_call = True
        self.texts = []

    def set_method_index(self, method_name):
        self.method = method_name

    def run(self):
        self.signals.UpdateProgressBar.emit(1)

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

        input_texts = list()
        for text in self.texts:
            input_texts.append(getCompiledFromSentencesText(text.register_pass_centences))
        self.short_filenames = [text.filename[text.filename.rfind('/') + 1:] for text in self.texts]

        if self.method == 0:
            self.make_k_means_clustering(self.short_filenames, input_texts)

        if self.method == 1:
            self.make_dbscan_clustering(self.short_filenames, input_texts)

        if self.method == 2:
            self.make_ward_clustering(self.short_filenames, input_texts)

        if self.method == 3:
            self.make_spectral_clustering(self.short_filenames, input_texts)

        if self.method == 4:
            self.make_aa_clustering(self.short_filenames, input_texts)

        if self.method == 5:
            self.make_mean_shift_clustering(self.short_filenames, input_texts)

        if self.method == 6:
            self.make_birch_clustering(self.short_filenames, input_texts)

        if self.first_call and self.need_preprocessing:
            self.first_call = False

        self.signals.PrintInfo.emit('Рассчеты закончены!')
        self.signals.UpdateProgressBar.emit(100)
        self.signals.Finished.emit()

    # Рассчет и запись матрицы TF-IDF
    def calculate_and_write_tf_idf(self, out_filename, input_texts):
        idf_vectorizer = TfidfVectorizer()
        tf_idf_matrix = idf_vectorizer.fit_transform(input_texts)
        feature_names = idf_vectorizer.get_feature_names()

        matrix_output_s = 'Слово'
        for filename in self.short_filenames:
            matrix_output_s += (';' + filename)
        matrix_output_s += ('; Сумма')
        matrix_output_s += '\n'

        tf_idf_matrix = tf_idf_matrix.toarray().transpose()
        total = []
        for row in range(tf_idf_matrix.shape[0]):
            current_total = np.sum(tf_idf_matrix[row])
            total.append((current_total, tf_idf_matrix[row], feature_names[row]))

        total.sort(key=lambda tup: tup[0], reverse=True)

        for row in range(len(total)):
            current_total = total[row]
            current_row = current_total[1]
            current_feature_name = current_total[2]
            matrix_output_s += current_feature_name
            for cell in range(current_row.shape[0]):
                matrix_output_s += ('; ' + str(current_row[cell]))
            matrix_output_s += ('; ' + str(current_total[0]))
            matrix_output_s += '\n'
        matrix_output_s = matrix_output_s.replace('.', ',')
        writeStringToFile(matrix_output_s, out_filename)
        result_msg = "Матрица TF-IDF записана: " + out_filename

        return result_msg

    def draw_clusters_plot(self, X, predict_result, short_filenames):
        plt.subplot(111)
        colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
        colors = np.hstack([colors] * 20)
        plt.scatter(X[:, 0], X[:, 1], color=colors[predict_result].tolist(), s=50)

        for label, x, y in zip(short_filenames, X[:, 0], X[:, 1]):
            plt.annotate(
                label,
                xy=(x, y), xytext=(-20, 20),
                textcoords='offset points', ha='right', va='bottom',
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        plt.xticks(())
        plt.yticks(())
        plt.grid()

    def make_k_means_clustering(self, short_filenames, input_texts):

        output_dir = self.output_dir + 'K_MEANS/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if self.need_tf_idf:
            self.signals.PrintInfo.emit("Расчет TF-IDF...")
            idf_filename = output_dir + 'tf_idf.csv'
            msg = self.calculate_and_write_tf_idf(idf_filename, input_texts)
            self.signals.PrintInfo.emit(msg)

        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(input_texts)

        svd = TruncatedSVD(2)
        normalizer = Normalizer(copy=False)
        lsa = make_pipeline(svd, normalizer)
        X = lsa.fit_transform(X)

        km = KMeans(n_clusters=self.kmeans_cluster_count, init='k-means++', max_iter=100, n_init=10)
        km.fit(X)

        predict_result = km.predict(X)

        self.signals.PrintInfo.emit('\nПрогноз по документам:\n')

        clasters_output = ''
        for claster_index in range(max(predict_result) + 1):
            clasters_output += ('Кластер ' + str(claster_index) + ':\n')
            for predict, document in zip(predict_result, short_filenames):
                if predict == claster_index:
                    clasters_output += ('  ' + str(document) + '\n')
            clasters_output += '\n'
        self.signals.PrintInfo.emit(clasters_output)

        self.signals.PrintInfo.emit('Сохранено в:' + str(output_dir + 'clusters.txt'))
        writeStringToFile(clasters_output, output_dir + 'clusters.txt')

        self.signals.PrintInfo.emit('')
        self.signals.PrintInfo.emit('Центры кластеров:')
        for index, cluster_center in enumerate(km.cluster_centers_):
            self.signals.PrintInfo.emit('  ' + str(index) + ':' + str(cluster_center))

        self.draw_clusters_plot(X, predict_result, short_filenames)

    def make_dbscan_clustering(self, short_filenames, input_texts):

        output_dir = self.output_dir + 'DBSCAN/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if self.need_tf_idf:
            self.signals.PrintInfo.emit("Расчет TF-IDF...")
            idf_filename = output_dir + 'tf_idf.csv'
            msg = self.calculate_and_write_tf_idf(idf_filename, input_texts)
            self.signals.PrintInfo.emit(msg)

        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(input_texts)

        svd = TruncatedSVD(2)
        normalizer = Normalizer(copy=False)
        lsa = make_pipeline(svd, normalizer)
        X = lsa.fit_transform(X)

        db = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_pts)
        predict_result = db.fit_predict(X)
        db.fit(X)

        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_

        self.signals.PrintInfo.emit('\nПрогноз по документам:\n')
        clasters_output = ''
        for claster_index in range(max(predict_result) + 1):
            clasters_output += ('Кластер ' + str(claster_index) + ':\n')
            for predict, document in zip(predict_result, short_filenames):
                if predict == claster_index:
                    clasters_output += ('  ' + str(document) + '\n')
            clasters_output += '\n'

        clasters_output += ('Шумовые элементы (-1):\n')
        for predict, document in zip(predict_result, short_filenames):
            if predict == -1:
                clasters_output += ('  ' + str(document) + '\n')
        clasters_output += '\n'
        self.signals.PrintInfo.emit(clasters_output)

        self.signals.PrintInfo.emit('Сохранено в:' + str(output_dir + 'clusters.txt'))
        writeStringToFile(clasters_output, output_dir + 'clusters.txt')

        self.draw_clusters_plot(X, predict_result, short_filenames)

    def make_ward_clustering(self, short_filenames, input_texts):

        output_dir = self.output_dir + 'WARD/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if self.need_tf_idf:
            self.signals.PrintInfo.emit("Расчет TF-IDF...")
            idf_filename = output_dir + 'tf_idf.csv'
            msg = self.calculate_and_write_tf_idf(idf_filename, input_texts)
            self.signals.PrintInfo.emit(msg)

        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(input_texts)

        svd = TruncatedSVD(2)
        normalizer = Normalizer(copy=False)
        lsa = make_pipeline(svd, normalizer)
        X = lsa.fit_transform(X)

        ward = AgglomerativeClustering(n_clusters=self.ward_clusters_count, linkage='ward')
        predict_result = ward.fit_predict(X)

        self.signals.PrintInfo.emit('\nПрогноз по документам:\n')

        clasters_output = ''
        for claster_index in range(max(predict_result) + 1):
            clasters_output += ('Кластер ' + str(claster_index) + ':\n')
            for predict, document in zip(predict_result, short_filenames):
                if predict == claster_index:
                    clasters_output += ('  ' + str(document) + '\n')
            clasters_output += '\n'
        self.signals.PrintInfo.emit(clasters_output)
        self.signals.PrintInfo.emit('Сохранено в:' + str(output_dir + 'clusters.txt'))
        writeStringToFile(clasters_output, output_dir + 'clusters.txt')

        self.draw_clusters_plot(X, predict_result, short_filenames)

    def make_spectral_clustering(self, short_filenames, input_texts):

        output_dir = self.output_dir + 'spectral/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if self.need_tf_idf:
            self.signals.PrintInfo.emit("Расчет TF-IDF...")
            idf_filename = output_dir + 'tf_idf.csv'
            msg = self.calculate_and_write_tf_idf(idf_filename, input_texts)
            self.signals.PrintInfo.emit(msg)

        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(input_texts)

        svd = TruncatedSVD(2)
        normalizer = Normalizer(copy=False)
        lsa = make_pipeline(svd, normalizer)
        X = lsa.fit_transform(X)

        spectral = SpectralClustering(n_clusters=self.spectral_clusters_count)
        predict_result = spectral.fit_predict(X)
        self.signals.PrintInfo.emit('\nПрогноз по документам:\n')

        clasters_output = ''
        for claster_index in range(max(predict_result) + 1):
            clasters_output += ('Кластер ' + str(claster_index) + ':\n')
            for predict, document in zip(predict_result, short_filenames):
                if predict == claster_index:
                    clasters_output += ('  ' + str(document) + '\n')
            clasters_output += '\n'
        self.signals.PrintInfo.emit(clasters_output)
        self.signals.PrintInfo.emit('Сохранено в:' + str(output_dir + 'clusters.txt'))
        writeStringToFile(clasters_output, output_dir + 'clusters.txt')

        self.draw_clusters_plot(X, predict_result, short_filenames)

    # aa = Affinity Propagation
    def make_aa_clustering(self, short_filenames, input_texts):

        output_dir = self.output_dir + 'affinity_propagation/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if self.need_tf_idf:
            self.signals.PrintInfo.emit("Расчет TF-IDF...")
            idf_filename = output_dir + 'tf_idf.csv'
            msg = self.calculate_and_write_tf_idf(idf_filename, input_texts)
            self.signals.PrintInfo.emit(msg)

        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(input_texts)

        svd = TruncatedSVD(2)
        normalizer = Normalizer(copy=False)
        lsa = make_pipeline(svd, normalizer)
        X = lsa.fit_transform(X)

        aa_clusterizator = AffinityPropagation(damping=self.aa_damping,
                                               max_iter=self.aa_max_iter,
                                               convergence_iter=self.aa_no_change_stop)

        predict_result = aa_clusterizator.fit_predict(X)
        self.signals.PrintInfo.emit('\nПрогноз по документам:\n')

        clasters_output = ''
        for claster_index in range(max(predict_result) + 1):
            clasters_output += ('Кластер ' + str(claster_index) + ':\n')
            for predict, document in zip(predict_result, short_filenames):
                if predict == claster_index:
                    clasters_output += ('  ' + str(document) + '\n')
            clasters_output += '\n'
        self.signals.PrintInfo.emit(clasters_output)
        self.signals.PrintInfo.emit('Сохранено в:' + str(output_dir + 'clusters.txt'))
        writeStringToFile(clasters_output, output_dir + 'clusters.txt')

        self.draw_clusters_plot(X, predict_result, short_filenames)

    def make_mean_shift_clustering(self, short_filenames, input_texts):

        output_dir = self.output_dir + 'mean_shift/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if self.need_tf_idf:
            self.signals.PrintInfo.emit("Расчет TF-IDF...")
            idf_filename = output_dir + 'tf_idf.csv'
            msg = self.calculate_and_write_tf_idf(idf_filename, input_texts)
            self.signals.PrintInfo.emit(msg)

        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(input_texts)

        svd = TruncatedSVD(2)
        normalizer = Normalizer(copy=False)
        lsa = make_pipeline(svd, normalizer)
        X = lsa.fit_transform(X)

        if (len(input_texts) * self.mean_shift_quantile) < 1.0:
            self.mean_shift_quantile = (1.0 / len(input_texts)) + 0.05

        bandwidth = estimate_bandwidth(X, quantile=self.mean_shift_quantile)
        if bandwidth == 0:
            bandwidth = 0.1

        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        predict_result = ms.fit_predict(X)
        self.signals.PrintInfo.emit('\nПрогноз по документам:\n')

        clasters_output = ''
        for claster_index in range(max(predict_result) + 1):
            clasters_output += ('Кластер ' + str(claster_index) + ':\n')
            for predict, document in zip(predict_result, short_filenames):
                if predict == claster_index:
                    clasters_output += ('  ' + str(document) + '\n')
            clasters_output += '\n'
        self.signals.PrintInfo.emit(clasters_output)
        self.signals.PrintInfo.emit('Сохранено в:' + str(output_dir + 'clusters.txt'))
        writeStringToFile(clasters_output, output_dir + 'clusters.txt')

        self.draw_clusters_plot(X, predict_result, short_filenames)

    def make_birch_clustering(self, short_filenames, input_texts):

        output_dir = self.output_dir + 'birch/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if self.need_tf_idf:
            self.signals.PrintInfo.emit("Расчет TF-IDF...")
            idf_filename = output_dir + 'tf_idf.csv'
            msg = self.calculate_and_write_tf_idf(idf_filename, input_texts)
            self.signals.PrintInfo.emit(msg)

        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(input_texts)

        svd = TruncatedSVD(2)
        normalizer = Normalizer(copy=False)
        lsa = make_pipeline(svd, normalizer)
        X = lsa.fit_transform(X)

        birch = Birch(threshold=self.birch_threshold,
                      branching_factor=self.birch_branching_factor,
                      n_clusters=self.birch_clusters_count)

        predict_result = birch.fit_predict(X)
        self.signals.PrintInfo.emit('\nПрогноз по документам:\n')

        clasters_output = ''
        for claster_index in range(max(predict_result) + 1):
            clasters_output += ('Кластер ' + str(claster_index) + ':\n')
            for predict, document in zip(predict_result, short_filenames):
                if predict == claster_index:
                    clasters_output += ('  ' + str(document) + '\n')
            clasters_output += '\n'
        self.signals.PrintInfo.emit(clasters_output)
        self.signals.PrintInfo.emit('Сохранено в:' + str(output_dir + 'clusters.txt'))
        writeStringToFile(clasters_output, output_dir + 'clusters.txt')

        self.draw_clusters_plot(X, predict_result, short_filenames)

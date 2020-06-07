from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import Qt, QObject, QThread, pyqtSignal
from PyQt5.QtWidgets import QApplication, QDialog, QLayout, QMessageBox

from sources.TextPreprocessing import writeStringToFile
from sources.utils import Profiler, getDirFromUserSelection, getFilenameFromUserSelection, getFilenamesFromUserSelection

from sources.FastText.FastTextCalculator import *
from sources.FastText.FastTextClassifier import *
from sources.FastText.ui.DialogFastText import Ui_FastTextDialog as DialogFastText # Импорт UI конвертированного в .py
from sources.common.plot.TsneMplForWidget import TsneMplForWidget
from sources.common.plot.PlotMaker import PlotMaker
import numpy

from sklearn.manifold import TSNE
import matplotlib
from matplotlib.backends.backend_qt5agg import (NavigationToolbar2QT as NavigationToolbar)
matplotlib.use('Qt5Agg')

from sources.common.helpers.PathHelpers import get_filename_from_path

# Импортируем файл, работающий с алгоритмом fastText
# Импортируем файл, работающий с моделью fastText (создание, загрузка, препроцессинг)
# Импортируем файл, работающий с визуализацией с помощью t-SNE

classifierTrainLabelMessage = "Для корректной классификации: название каждого файла (только латиница) должно отражать категорию,\nк которой относится список текстов, содержащийся в этом файле.\nНапример, файл, содержащий тексты категории 'комедия',\nдолжен иметь название 'comedy.txt\n'"

'''
Основной класс алгоритма fastText
Загрузка интерфейса
Перенаправление методов по созданию и работе с моделью
'''
class DialogFastTextMaker(QDialog, DialogFastText):
    def __init__(self, input_dir, morph, configurations, parent):
        QDialog.__init__(self)
        DialogFastText.__init__(self)
        self.setupUi(self)

        self.filename = None
        self.train_files = None # Файлы для тренировки модели fasttext PyPi
        self.trainFilename = None
        self.classifyFilename = None
        self.classifierModel = None
        self.morph = morph
        self.configurations = configurations
        self.parent = parent
        self.input_dir = input_dir
        self.calculator = None
        self.visualisation_model = None
        self.plot = None
        self.visualized_gensim_model = None
        self.visualized_model = None
        self.visualized_model_filename = None

        self.output_dir = self.configurations.get(
            "output_files_directory", "output_files") + '/FastText/'
        self.classification_output_dir = self.configurations.get(
            "output_files_directory", "output_files") + '/FastText/classification/'

        flags = Qt.Window | Qt.WindowSystemMenuHint | Qt.WindowMinimizeButtonHint | Qt.WindowCloseButtonHint | Qt.WindowMaximizeButtonHint
        self.setWindowFlags(flags)

        self.profiler = Profiler()
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        
        # Вкладка создание модели
        self.createModelBtn.setEnabled(self.filename != None)

        self.selectAnotherPathBtn.clicked.connect(self.select_text_file)
        self.createModelBtn.clicked.connect(self.create_model)
        self.retrainModelBtn.clicked.connect(self.retrain_model)
        self.retrainModelBtn.setVisible(False) # Скрыть кнопку повторной тренировки модели

        # Вкладка визуализация модели
        self.visualizeBtn.setEnabled(False) # Блокируем элементы до выбора модели
        self.searchQueryGBox.setEnabled(False) # Блокируем элементы до выбора модели
        self.plotVLayout.setEnabled(False) # Блокируем элементы до выбора модели
        self.clearPlotWidgetBtn.setEnabled(False)

        self.selectModelBtn.clicked.connect(self.select_model_file)
        self.searchQueryBtn.clicked.connect(self.search_word)
        self.visualizeBtn.clicked.connect(self.visualise_model)
        self.clearPlotWidgetBtn.clicked.connect(self.clear_plots_layout)
        self.filePathField.setText(self.filename)

        # Вкладка классификация модели
        # Блокируем элементы до тренировки модели
        self.selectClassifierFilesLbl.setToolTip(classifierTrainLabelMessage)
        self.selectClassifierFilesLbl.setToolTipDuration(20000)
        self.selectClassifierTrainFilesBtn.setToolTip(classifierTrainLabelMessage)
        self.selectAnotherPathBtn.setToolTipDuration(20000)
        self.trainClissifierBtn.setEnabled(False)
        self.classifyButton.setEnabled(False)
        self.phraseClassificationField.setEnabled(False)
        self.phraseClassificationBtn.setEnabled(False)

        self.selectClassifierTrainFilesBtn.clicked.connect(self.select_train_files)
        self.trainClissifierBtn.clicked.connect(self.train_classifier)
        self.selectAnotherClassificationFileBtn.clicked.connect(self.select_file_for_classification)
        self.classifyButton.clicked.connect(self.classify)
        self.phraseClassificationBtn.clicked.connect(self.classify_phrase)

    # Инициализация FastTextCalculator, установка сигналов для FastTextCalculator, установка выходного пути.
    def set_calc_and_signals(self):
        self.calculator = FastTextCalculator(
            train_path=self.filename,
            morph=self.morph,
            configurations=self.configurations,
            use_gensim=self.gensimCheckbox.isChecked(),
            only_nouns=self.nounOnlyCheck.isChecked()
        )
        os.makedirs(os.path.dirname(self.output_dir), exist_ok=True)
        self.calculator.signals.PrintInfo.connect(self.on_text_log_add)
        if self.gensimCheckbox.isChecked(): self.calculator.signals.Progress.connect(self.on_model_epoch_end)
        self.calculator.signals.ProgressBar.connect(self.on_progress)
        self.calculator.signals.Finished.connect(self.on_calculation_finish)
        print("Настройки калькулятора и сигналов заданы")

    # Инициализация классификатора FastTextClassifier, установка сигналов для FastTextClassifier, установка выходного пути.
    def set_classifier_and_signals(self):
        self.classifier = FastTextClassifier(
            model_file=self.classifierModel,
            filenames=self.train_files,
            morph=self.morph,
            only_nouns=self.classifyNounOnlyCheck.isChecked(),
            configurations=self.configurations
        )
        self.classify_output_dir = self.configurations.get(
            "output_files_directory", "output_files") + '/FastText/classification/'
        os.makedirs(os.path.dirname(self.classify_output_dir), exist_ok=True)
        self.classifier.signals.PrintInfo.connect(self.on_classification_text_log_add)
        self.classifier.signals.Finished.connect(self.on_classifier_train_finish)
        self.classifier.signals.ClassificationFinished.connect(self.on_classification_finish)

    # Заполнение progressBar.
    def on_progress(self, value):
        self.createModelBar.setValue(value)
        self.repaint()

    # Вывод текста в информационное поле раздела создания моделей.
    def on_text_log_add(self, QString):
        self.createLogTextEdit.append(QString + '\n')
        self.repaint()

    # Вывод текста в информационное поле раздела классификациию
    def on_classification_text_log_add(self, QString):
        self.classificationLogTextEdit.append(QString + '\n')
        self.repaint()

    # Вывод информации об эпохах обучения для модели gensim.
    def on_model_epoch_end(self, model, epoch):
        root_path = '{0}{1}'.format(self.output_dir, os.path.basename(os.path.splitext(self.filename)[0]))
        model.wv.save_word2vec_format(
            root_path + 'weight_matrix_epoch' + str(epoch) + '.txt')
        self.createLogTextEdit.append("Данные за эпоху " + str(epoch) + " сохранены по адресу output/FastText")

    # Обработчик завершения процесса построения модели.
    # Выводит информацию о сохранённых файлах и о завершённом процессе создания модели.
    def on_calculation_finish(self):
        self.setEnabled(True)
        self.createLogTextEdit.append(
            'Выполнено за ' + self.profiler.stop() + ' с.')
        QApplication.restoreOverrideCursor()
        # self.retrainModelBtn.setVisible(True) # TODO: Показать кнопку повтора расчетов
        self._log_output_data()

    # Обработчик завершения тренировки классификатора.
    def on_classifier_train_finish(self):
        self.setEnabled(True)
        self.trainClissifierBtn.setEnabled(True)
        QApplication.restoreOverrideCursor()
        self.classificationLogTextEdit.append('Затраченное время: {0}c.'.format(self.profiler.stop()))
        model_file = '{0}{1}'.format(self.classification_output_dir, self.classifier.model_filename)
        self.classifier.model.save_model(model_file)
        self.classifierModel = self.classifier.model
        self.classificationLogTextEdit.append('Обученая модель сохранена по адресу: {0}'.format(model_file))
        self.set_enable_classification()

    # Обработчик завершения классификации.
    def on_classification_finish(self):
        self.setEnabled(True)
        QApplication.restoreOverrideCursor()
        self.classificationLogTextEdit.append('Затраченное время: {0}c.'.format(self.profiler.stop()))
        self.classificationLogTextEdit.append('Результаты сохранены по адресу: {0}{1}'.format(self.output_dir, self.classifier.classification_result_filename))

    # Инициализация и настройка FastTextCalculator для создания моделей.
    # Настройка сигналов.
    # Создание модели.
    def create_model(self):
        self.createModelBtn.setEnabled(False)
        self.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.set_calc_and_signals()
        self.createLogTextEdit.append("Исходный файл {0}".format(self.filename))
        self.calculator.min_count = self.wordFrequencyField.value()
        self.calculator.size = self.vectorSizeField.value()
        self.calculator.learn_rate = self.trainingSpeedField.value()
        self.calculator.window = self.windowField.value()
        self.calculator.iter = self.epochNumberField.value()
        
        self.profiler.start()
        self.calculator.start()

    # Визуализация модели.
    def visualise_model(self):
        self.selectModelBtn.setEnabled(False)
        self.visualizeBtn.setEnabled(False)
        self.clearPlotWidgetBtn.setEnabled(False)
        model = None
        X = []
        words = []

        if not self.visualized_gensim_model == None:
            X = self.visualized_gensim_model.wv[self.visualized_gensim_model.wv.vocab]
            words = list(self.visualized_gensim_model.wv.vocab)

        if not self.visualized_model == None:
            try:
                X = []
                output_matrix_np_array = self.visualized_model.get_output_matrix()
                for item in output_matrix_np_array:
                    X.append(item.tolist())
                words = self.visualized_model.get_words()
            except:
                error = 'Для визуализации необходима модель supervised или созданная с помощью gensim'
                print(error)
                self.visualizeLogTextEdit.append('{0}\n'.format(error))

        tsne = TSNE(n_components=2)
        result = tsne.fit_transform(X)
        self.plot = PlotMaker(self.plotVLayout, parent=self, title=get_filename_from_path(self.visualized_model_filename))
        self.plot.add_toolbar(self)
        ax = self.plot.ax
        ax.scatter(result[:, 0], result[:, 1])
        ax.plot()
        try:
            for i, word in enumerate(words):
                ax.annotate(word, xy=(result[i, 0], result[i, 1]))
        except Exception as e:
            print(e)

        self.searchQueryGBox.setVisible(True)
        self.selectModelBtn.setEnabled(True)
        self.visualizeBtn.setEnabled(True)
        self.clearPlotWidgetBtn.setEnabled(True)

    # Тренировка классификатора.
    def train_classifier(self):
        self.trainClissifierBtn.setEnabled(False)
        self.classificationLogTextEdit.append("Исходные файлы:\n{0}".format('\n'.join(self.train_files)))
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.classifier.min_count = self.classifyWordFrequencyField.value()
        self.classifier.size = self.classifyVectorSizeField.value()
        self.classifier.learn_rate = self.classifyTrainingSpeedField.value()
        self.classifier.window = self.classifyWindowField.value()
        self.classifier.iter = self.classifyEpochNumberField.value()
        self.setEnabled(False)
        self.profiler.start()
        self.classifier.start()

    # Запускает метод классификации FastTextClassifier.
    def classify(self):
        self.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        if self.classifyFilename != None:
            self.classificationLogTextEdit.append("Файл для классификации:\n{0}".format(self.classifyFilename))
            self.classifier.classification_filename = self.classifyFilename
            self.profiler.start()
            self.classifier.classify()

    # Обработчик классификации фразы.
    def classify_phrase(self):
        if self.phraseClassificationField.text != None or self.phraseClassificationField.text != '':
            self.classifier.classify_phrase(self.phraseClassificationField.text())
        else:
            self.classificationLogTextEdit.append('Введите фразу для классификации!')

    # Перетренировка модели (фактически, создание заново).
    def retrain_model(self):
        print("Повторная тренировка модели")
        self.create_model() # TODO: перетренировать модель, а не создать заново

    # Выбрать текстовый файл.
    def select_text_file(self):
        input_dir = self.input_dir if self.gensimCheckbox.isChecked() else '{0}/train'.format(self.input_dir)
        self.filename = getFilenameFromUserSelection("Text file (*.txt)", input_dir)
        self.filePathField.setText(self.filename)
        self.createModelBtn.setEnabled(True)

    # Выбрать директорию для тренировки классификатора fasttext
    def select_train_files(self):
        files = getFilenamesFromUserSelection(self.input_dir, extensions="MODEL Files (*.model)")
        if not files == None:
            model_file = self.get_model_selected(files)
            if model_file:
                self.trainClissifierBtn.setEnabled(False)
                self.classifierModel = model_file
                self.set_classifier_and_signals()
                self.train_files = None
            else:
                if not files == None:
                    self.train_files = files
                    self.classifierModel = None
                    self.set_disable_classification()
                    self.set_classifier_and_signals()
                    self.trainClissifierBtn.setEnabled(True)
            self.set_enable_classification()

    # Возвращает файл .model, если такой был выбран.
    # Иначе - False
    def get_model_selected(self, files: list):
        for file in files:
            if get_filename_from_path(file).split('.')[1] == 'model':
                return file
        return False

    # Выбор существующей модели.
    def select_model_file(self):
        self.visualized_model_filename = getFilenameFromUserSelection("MODEL Files (*.model)", self.output_dir + 'FastText')
        if self.visualized_model_filename != None and len(self.visualized_model_filename.split('/')) > 0:
            self.visualized_gensim_model = None
            self.visualized_model = None
            try:
                self.visualized_model = fasttext.load_model(self.visualized_model_filename)
            except:
                self.visualized_model = None
            try:
                self.visualized_gensim_model = FastText.load(self.visualized_model_filename)
            except:
                self.visualized_gensim_model = None
            self.visualizeLogTextEdit.clear()
            self.set_enable_visualisation(self.visualized_model_filename)

    # Выбор файла для классификации.
    # Снятие блокировки с кнопки классификации.
    def select_file_for_classification(self):
        self.classifyFilename = getFilenameFromUserSelection("Text file (*.txt)", self.input_dir + 'FastText/classification')
        self.classificationFilePathField.setText(self.classifyFilename)
        self.set_enable_classification()

    # Поиск синонимов для слова.
    def search_word(self):
        word = self.searchQueryField.text().strip()
        result = None
        if word == '' or word is None:
            return
        self.visualizeLogTextEdit.append("Приведение слова к нормальной форме: " + word)
        word = self.morph.parse(word)[0].normal_form
        self.visualizeLogTextEdit.append("Поиск слова...")

        try:
            if not self.visualized_gensim_model == None:
                result = self.visualized_gensim_model.wv.most_similar([word.lower()])
            if not self.visualized_model == None:
                result = self.visualized_model.get_nearest_neighbors(word.lower())
            self._display_search_results(word, result)
        except:
            error_text = 'Слово {0} не содержится в словаре'.format(word)
            self.visualizeLogTextEdit.append(error_text)

    # Очистка области визуализации.
    def clear_plots_layout(self):
        self.plot.removePlot()

    # Отображения результатов поиска слова-синонима.
    def _display_search_results(self, word, results):
        self.visualizeLogTextEdit.append("Слово {0} употребляется со следующими словами:".format(word))
        self.visualizeLogTextEdit.append(';'.join(["({0} - {1})".format(tpl[0], tpl[1]) for tpl in results]))

    # Вывод данных по окончании процесса создания модели.
    # Информация о расположении выходных файлов.
    def _log_output_data(self):
        root_path = '{0}{1}'.format(self.output_dir, os.path.basename(os.path.splitext(self.filename)[0]))

        def log_data(name, data):
            outpath = '{0}_{1}.txt'.format(root_path, name)
            writeStringToFile(str(data), outpath)

        def yield_vocab(vocab):
            for key in vocab:
                w = vocab[key]
                # Генератор внутри цикла.Код вызывается и не хранит значения в памяти
                yield '{0} -> count: {1}, index: {2}, sample_int: {3}'.format(key, w.count, w.index, w.sample_int)
        if self.gensimCheckbox.isChecked():
            model_filename = 'GENSIM_{0}_output.model'.format(get_filename_from_path(root_path))
            model_file = '{0}/{1}'.format(os.path.dirname(root_path), model_filename)
            log_data('vocab', '\n'.join(
                yield_vocab(self.calculator.gensim_fasttext_model.wv.vocab)))
            self.createLogTextEdit.append("Файл со словарем для текста сохранен по адресу output/FastText")
            log_data('index2word', list(
                enumerate(self.calculator.gensim_fasttext_model.wv.index2word)))
            self.createLogTextEdit.append("Параметр index2word сохранен по адресу output/FastText")
            self.calculator.gensim_fasttext_model.save(model_file)
            self.createLogTextEdit.append("Модель gensim сохранена по адресу {0}".format(model_file))
            self.calculator.gensim_fasttext_model.wv.save_word2vec_format(
                root_path + 'weight_matrix.txt')
            self.createLogTextEdit.append("Матрица весов weight_matrix сохранена по адресу output/FastText \n\n")
        else:
            model_filename = 'UNSUPERVISED_{0}_output.model'.format(get_filename_from_path(root_path))
            model_file = '{0}/{1}'.format(os.path.dirname(root_path), model_filename)
            self.calculator.fasttext_model.save_model(model_file)
            self.createLogTextEdit.append("Модель сохранена по адресу {0}".format(model_file))

    # Снятие блокировки с элементов интерфейса для визуализации модели.
    def set_enable_visualisation(self, modelFile):
        nameStrArray = modelFile.split('/')
        self.visualisation_model = modelFile
        self.selectModelField.setText(nameStrArray[-3] + '/' + nameStrArray[-2] + '/' + nameStrArray[-1])
        self.visualizeLogTextEdit.append('Модель выбрана')
        self.visualizeBtn.setEnabled(True) # Делаем доступными элементы после выбора модели
        self.searchQueryGBox.setEnabled(True) # Делаем доступными элементы после выбора модели
        self.plotVLayout.setEnabled(True) # Делаем доступными элементы после выбора модели

    # Снятие блокировки с элементов интерфейса для классификации.
    def set_enable_classification(self):
        if not self.classifierModel == None and not self.classifyFilename == None:
            self.classifyButton.setEnabled(True)
        if not self.classifierModel == None:
            self.classifyButton.setEnabled(True)
            self.phraseClassificationBtn.setEnabled(True)
            self.phraseClassificationField.setEnabled(True)

    # Блокирует элементы интерфейса для классификации.
    def set_disable_classification(self):
        self.classifyButton.setEnabled(False)
        self.phraseClassificationBtn.setEnabled(False)
        self.phraseClassificationField.setEnabled(False)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import codecs
import os
import shutil
import time
from PyQt5.QtWidgets import QFileDialog

from sources.TextPreprocessing import loadInputFilesFromList, tokenizeTextData, removeStopWordsInTexts, \
    calculateWordsFrequencyInTexts, fixRegisterInTexts, normalizeTexts, writeStringToFile

def readConfigurationFile(filename):
    with codecs.open(filename, 'r', "utf-8") as text_file:
        data = text_file.read()
        lines = data.split("\n")
        result = dict()
        for line in lines:
            line = line.strip()
            if(line.startswith("#") == False):
                keyvalue = line.split("=")
                if(len(keyvalue) == 2):
                    result[keyvalue[0]]=keyvalue[1]
        return result

def getFilenameFromUserSelection(file_types="Any Files (*.*)", path = ''):
    filenames, _ = QFileDialog.getOpenFileName(None, "Выбрать файл", path, file_types, None)
    if (len(filenames) > 0):
        return filenames
    else:
        return None

def getFilenamesFromUserSelection(path = ''):
    filenames, _ = QFileDialog.getOpenFileNames(None, "Выбрать файлы", path, "Text Files (*.txt)", None)
    if (len(filenames) > 0):
        return filenames
    else:
        return None

def getDirFromUserSelection(path):
    dir_name = QFileDialog.getExistingDirectory(None, "Выбрать каталог", path)
    if (len(dir_name) > 0):
        return dir_name
    else:
        return None


# Преобразует адреса вида /home/user/files/file.txt /home/user/files
# в вид: files/file.txt
def make_relative_files_path(filename, root_folder):
    folder = root_folder
    if(len(root_folder)>0 and root_folder[-1] == '/'):
        folder = root_folder[:-1]
    start_position = folder.rfind('/')
    if(start_position == -1):
        start_position = 0
    if len(filename) > 0 and filename[0] == '/':
        start_position += 1
    return filename[start_position:]

def clear_dir(path):
    for name in os.listdir(path):
        full_name = path + name
        if (os.path.isfile(full_name)):
            os.remove(full_name)
        else:
            shutil.rmtree(full_name)


def makePreprocessingForAllFilesInFolder(configurations,
                                         input_dir_name,
                                         output_files_dir,
                                         output_log_dir, morph):

    input_filenames_with_dir = []

    for top, dirs, files in os.walk(input_dir_name):
        for nm in files:
            input_filenames_with_dir.append(os.path.join(top, nm))

    # Загружаем предложения из нескольких файлов
    texts = loadInputFilesFromList(input_filenames_with_dir)

    for text in texts:
        text.short_filename = make_relative_files_path(text.full_filename, input_dir_name)

    # Разделяем предложения на слова
    texts = tokenizeTextData(texts)

    print('Этап препроцессинга:')

    print('1) Удаление стоп-слов.')

    texts, log_string = removeStopWordsInTexts(texts, morph, configurations)
    writeStringToFile(log_string.replace('\n ', '\n'), output_log_dir + '/output_stage_1.txt')

    # Переводим обычное предложение в нормализованное (каждое слово)
    print('2) Нормализация.')
    texts, log_string = normalizeTexts(texts, morph)
    writeStringToFile(log_string.replace('\n ', '\n'), output_log_dir + '/output_stage_2.txt')

    # Приведение регистра (все слова с маленькой буквы за исключением ФИО)
    print('3) Приведение регистра.')
    texts, log_string = fixRegisterInTexts(texts, morph)
    writeStringToFile(log_string.replace('\n ', '\n'), output_log_dir + '/output_stage_3.txt')

    # Подсчет частоты слов в тексте
    print('4) Расчет частотной таблицы слов.')
    texts, log_string = calculateWordsFrequencyInTexts(texts)
    writeStringToFile(log_string.replace('\n ', '\n'), output_log_dir + '/output_stage_4.csv')

    for text in texts:
        text_filename = output_files_dir + text.short_filename
        os.makedirs(os.path.dirname(text_filename), exist_ok=True)
        with open(text_filename, 'w', encoding='utf-8') as out_text_file:
            for sentence in text.register_pass_centences:
                for word in sentence:
                    out_text_file.write(word)
                    out_text_file.write(' ')


# Измерение времени выполнения блока кода
class Profiler(object):

    def __init__(self):
        self._startTime = 0

    def start(self):
        self._startTime = time.time()

    def stop(self):
        return str("{:.3f}").format(time.time() - self._startTime)
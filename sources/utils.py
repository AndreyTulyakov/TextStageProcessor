#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import shutil

from PyQt5.QtWidgets import QFileDialog

from sources.TextPreprocessing import loadInputFilesFromList, tokenizeTextData, removeStopWordsInTexts, \
    calculateWordsFrequencyInTexts, fixRegisterInTexts, normalizeTexts, writeStringToFile


def getFilenameFromUserSelection(file_types="Any Files (*.*)"):
    filenames, _ = QFileDialog.getOpenFileName(None, "Выбрать файл", "", file_types, None)
    if (len(filenames) > 0):
        return filenames
    else:
        return None

def getFilenamesFromUserSelection():
    filenames, _ = QFileDialog.getOpenFileNames(None, "Выбрать файлы", "", "Text Files (*.txt)", None)
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
        with open(text_filename, 'w') as out_text_file:
            for sentence in text.register_pass_centences:
                for word in sentence:
                    out_text_file.write(word)
                    out_text_file.write(' ')

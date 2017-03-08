#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from PyQt5.QtWidgets import QFileDialog


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

def getDirFromUserSelection():
    dir_name = QFileDialog.getExistingDirectory(None, "Выбрать каталог", "")
    if (len(dir_name) > 0):
        return dir_name
    else:
        return None
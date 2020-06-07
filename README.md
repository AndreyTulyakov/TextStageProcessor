# Text Stage Processor
Open source project for text mining process exploration.

## Требования:
1. Операционная система: Windows, Linux, MacOS.
2. Язык программирования: Python 3 (>= версия 3.6)
3. Входные файлы: каталог или отдельные текстовые файлы с расширением .txt в кодировке UTF-8 содержащие текст на русском языке.
4. Выходные файлы: формат TXT и CSV (помещаются в специальный каталог для выходных файлов)
5. Библиотеки: фреймворк Anakonda 3, pymorphy2
6. Алгоритмы ТextМining должны быть реализованы кодом.
7. Программная реализация алгоритмов тестируется.

## Установка на Windows
### Установка программного обеспечения
1. Наиболее простой вариант:
 - Если в системе имеется Python3, то удалить его. (Если вы не собираетесь его использовать далее)
 - Установить пакет Anakonda 3. (https://www.continuum.io/downloads)
 - Установить следующие библиотеки на Anakonda 3:
   + gensim
   + pymorphy2
   + pymorphy2-dicts
   + pymorphy2-dicts-ru

Команды установки указаны в start_conda_prequirement.bat (можно просто запустить данные файл)

- Установить на python внутри Anaconda3 через pip библиотеку fasttext

  pip install fasttext

  При возникновении проблем с установкой пакета fasttext необходимо установить файл .whl командой, директория /wheel:  
  pip install fasttext-0.9.2-cp37-cp37m-win_amd64.whl  
  Если версия Python не 3.7.X или система не x64 - установить подходящий .whl-файл:  
  https://www.lfd.uci.edu/~gohlke/pythonlibs/#fasttext  
  cpXXX - ближайшая версия Python  
  win-amdXX - ОС и разрядность системы  

2. В случае использования чистого языка Python 3 необходимо установить библиотеки (точное их перечисление есть в файле requirements.txt):
 - scipy
 - matplotlib
 - gensim
 - numpy
 - pandas
 - pymorphy2
 - PyQt5==5.13.*
 - PyQt5-tools==5.13.*
 - scikit_learn
 - pymorphy2-dicts-ru>=2.4
 - pymorphy2-dicts>=2.4
 - sklearn
 - fasttext>=0.9.2
 - nltk<=3.5
 - stopwords<=0.1.3

При возникновении проблем с установкой пакета fasttext необходимо установить файл .whl командой, директория /wheel:  
pip install fasttext-0.9.2-cp37-cp37m-win_amd64.whl  
Если версия Python не 3.7.X или система не x64 - установить подходящий .whl-файл:  
https://www.lfd.uci.edu/~gohlke/pythonlibs/#fasttext  
cpXXX - ближайшая версия Python  
win-amdXX - ОС и разрядность системы  

Для установки зависимостей (при наличии Python>=3.7.*) можно набрать команду:  
pip install -r requirements.txt

### Установка программы
Склонировать с этой версии github:  
git clone https://github.com/ElleyKo/TextStageProcessor
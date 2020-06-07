import gensim
import re

trash_chars = ['', '\n', ' \n', ' ', '\xa0\n', '\xa0', 'xa0']

# Обработка входного текста. Возвращает list of word lists.
# Нормализация текста, удаление стоп-слов.
def get_processed_word_lists_from_lines(lines, morph, stop_words, use_gensim_preprocess = True, only_nouns = False):
    def apply_filter(filter, lines):
        for i in range(len(lines)):
            lines[i] = filter(lines[i])

    def gensim_preprocess(line):
        return gensim.utils.simple_preprocess(line, min_len=3, max_len=20)

    def normalize_text(lines):
        lst = []
        pattern = re.compile(r'[?!.]+')
        for line in filter(lambda x: len(x) > 0, map(str.strip, lines)):
            try:
                lst.extend(re.split(pattern, line))
            except Exception as e:
                print(e)
        lines = lst
        return lines

    def remove_stops(line):
        for word in line:
            if word not in stop_words:
                yield word

    def morph_words(line):
        for word in line:
            if morph.word_is_known(word):
                word = morph.parse(word)[0].normal_form
            yield word

    def morph_nouns(line):
        for word in line:
            if morph.word_is_known(word):
                data = morph.parse(word)[0]
                if 'NOUN' in data.tag:
                    yield data.normal_form

    lines = normalize_text(lines)
    if use_gensim_preprocess: apply_filter(gensim_preprocess, lines)
    apply_filter(remove_stops, lines)
    apply_filter(morph_nouns if only_nouns else morph_words, lines)
    apply_filter(remove_stops, lines)
    return [line for line in map(list, lines) if len(line) > 1]

# Получает массив строк с нормализированными словами из исходного текстового файла.
def get_processed_lines(morph, stop_words, input_file = None, lines = None, only_nouns = False):
    def _process(lines):
        def remove_stops(line):
            newLine = ''
            words = line.split(' ')
            for word in words:
                if word.lower() not in stop_words and word not in trash_chars:
                    newLine += word + ' '
            return newLine

        def morph_words(line):
            newLine = ''
            words = line.split(' ')
            for word in words:
                if morph.word_is_known(word):
                    word = morph.parse(word)[0].normal_form
                    newLine += word + ' '
            return newLine

        def morph_nouns(line):
            newLine = ''
            words = line.split(' ')
            for word in words:
                if morph.word_is_known(word):
                    data = morph.parse(word)[0]
                    if 'NOUN' in data.tag:
                        newLine += data.normal_form + ' '
            return newLine

        for line in lines:
            if line.strip() and line not in trash_chars:
                newLine = remove_stops(line)
                newLines.append(morph_nouns(newLine).rstrip() if only_nouns else morph_words(newLine).rstrip())
        return newLines

    newLines = []
    if input_file != None:
        with open(input_file, 'r', encoding='UTF-8') as file:
            newLines = _process(file.readlines())
    if lines != None:
        newLines = _process(lines)
    return newLines

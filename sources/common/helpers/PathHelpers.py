# Возвращает имя файла с расширением из строки, содержащей полный путь до файла.
def get_filename_from_path(path: str):
    return path.split('/').pop()
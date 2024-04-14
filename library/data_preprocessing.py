import os
import sys
from rich.progress import Progress
from PIL import Image
import numpy as np
from library.custom_logging import setup_logging
import streamlit as st
import time

# Задаем переменную логирования
log = setup_logging()


def data_list(data_dir):
    """
    Возвращает список путей к изображениям (.jpg и .png) в указанной директории.

    :param data_dir: путь к директории с изображениями.
    :return image_files: список путей к изображениям.
    """

    # Проверка на наличие изображений в директории
    try:
        check_directory_for_images(data_dir)

        # Проверка наличия директории
        if not os.path.exists(data_dir):
            log.error(f"Директория '{data_dir}' не найдена")

        # Список хранения путей к изображениям
        image_files = []

        # Перебор изображений
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                image_files.append(os.path.join(root, file))

    except FileNotFoundError:
        log.error(f"Директория '{data_dir}' пуста")
        st.warning(f"Директория {data_dir} пуста")
        return None

    return image_files


def is_image_file(filename):
    """
    Проверяет, является ли файл изображением по расширению.

    :param filename: имя файла.
    :return True: если файл является изображением, иначе False.
    """

    return filename.lower().endswith((".jpg", ".png"))


def check_directory_for_images(directory):
    """
    Проверяет директорию на наличие изображений.

    :param directory: Путь к директории.
    :return True: если в директории есть изображения, иначе False.
    """

    if not os.path.exists(directory):
        raise FileNotFoundError(f"Директория '{directory}' не найдена")

    files = os.listdir(directory)
    image_files = [file for file in files if is_image_file(file)]

    if not image_files:
        raise FileNotFoundError(f"В директории '{directory}' нет изображений")

    return True


def image_resize(image, x, y):
    """
    Изменяет размер изображения.

    :param image: исходное изображение (PIL.Image).
    :param x: новая ширина изображения.
    :param y: новая высота изображения.
    :return resized_image: измененное изображение (PIL.Image).
    """

    if not isinstance(x, int) or not isinstance(y, int) or x <= 0 or y <= 0:
        log.error("Размеры изображения должны быть положительными целыми числами")
    else:
        resized_image = image.resize((x, y))
    return resized_image


def image_resize_and_save(image_files, new_width, new_height, dir_changed):
    """
    Изменяет размер и сохраняет каждое изображение из списка.

    :param image_files: список путей к изображениям.
    :param new_width: новая ширина изображения.
    :param new_height: новая высота изображения.
    :return: None.
    """

    # Проверка наличия путей к файлам
    if image_files is not None:

        log.warning("Дождитесь завершения изменения размеров изображений")
        progress_text = "Операция в процессе. Пожалуйста подождите..."
        bar = st.progress(0, text=progress_text)

        with Progress() as progress:
            # Прогресс бар
            task = progress.add_task("[yellow] Изменение размера...", total=len(image_files))

            # Перебор изображений
            for i, file in enumerate(image_files):
                bar.progress((i + 1) / len(image_files), text=progress_text)
                try:
                    # Загрузка изображения
                    image = Image.open(file)

                    # Изменение размера
                    resized_image = image_resize(image, new_width, new_height)

                    # Сохранение измененного изображения
                    resized_image.save(os.path.join(dir_changed, os.path.basename(file)))

                    # Обновление прогресс-бара
                    progress.update(task, advance=1)

                except Exception as e:
                    log.error(f"Ошибка при изменении размера изображения {file}: {e}")
                    sys.exit()

        log.info("Изменение размера изображений завершено")
        st.success("Операция успешно завершена!")


def gray_to_rgb(image_files, dir_changed):
    """
    Трансформация изображения из одноканального в трехканальный.

    :param image_files: список путей к изображениям.

    :return: None.
    """

    # Проверка наличия путей к файлам
    if image_files is not None:

        log.warning("Дождитесь завершения L to RGB")
        progress_text = "Операция в процессе. Пожалуйста подождите..."
        bar = st.progress(0, text=progress_text)

        with Progress() as progress:
            # Прогресс бар
            task = progress.add_task("[yellow] Processing...", total=len(image_files))

            # Перебор изображений
            for i, file in enumerate(image_files):
                bar.progress((i + 1) / len(image_files), text=progress_text)
                try:
                    # Загружаем изображение
                    gray_image = Image.open(file)

                    # Создаем новое RGB изображение с такими же размерами
                    rgb_image = Image.new("RGB", gray_image.size)

                    # Заполняем каналы R, G и B одними и теми же значениями из оттенков серого
                    rgb_image.paste(gray_image, (0, 0))

                    # Cохранение обрезанного изображения
                    rgb_image.save(os.path.join(dir_changed, os.path.basename(file)))

                    # Обновление прогресс-бара
                    progress.update(task, advance=1)

                except Exception as e:
                    log.error(f"Ошибка при трансформации L to RGB {file}: {e}")
                    sys.exit()

        log.info("Операция L to RGB завершена")
        st.success("Операция успешно завершена!")

import streamlit as st
from library.custom_logging import setup_logging
from library.IndentationHelper import IndentationHelper
from config_file import load_config, is_value_changed, save_config
from PIL import Image, ImageDraw, ImageFont
import string
import os
import torchvision.transforms as transforms
import numpy as np
import math
import cv2
from library.components import add_text_to_image
from datetime import datetime
import time
import shutil
from stqdm import stqdm
import library.style_button as sb
import json
from mysql.connector import connect, Error
import pickle
from efficientnet_pytorch import EfficientNet
import torch
import pandas as pd
from joblib import Memory
# хранение в памяти database
memory = Memory(location="./cachedir", verbose=0)
from scipy.spatial.distance import cosine


class PageTools:

    def __init__(self):

        # Назначение переменной логирования
        self.log = setup_logging()
        # Загрузка конфигурационного файла
        self.config_data = load_config()
        # Класс функций для отступов
        self.helper = IndentationHelper()
        # Обновляем изменяемые переменные после предыдущего запуска
        self._init_params()
        # Выгружаем и инициализируем переменные стандартных путей
        self._init_default_path()
        # Создаем директории, если они не существуют
        self._make_os_dir()
        # Инициализируем листы
        self._init_list()
        # Инициализируем словари
        self._init_dir()

        self.features_list = []
        self.features = None
        self.similar_images = None
        self.find_similar_images(self.features, self.features_list, top_n=10)

    def _time(self):
        self.timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")

    def _init_default_path(self):
        paths = self.config_data["dir_paths"]
        for key, value in paths.items():
            setattr(self, key, self.config_data[f"{value}"])

    def _init_params(self):
        params = self.config_data["dir_params"]
        for key, value in params.items():
            setattr(self, key, self.config_data[f"{value}"])

    def _init_list(self):
        lists = self.config_data["dir_lists"]
        for key, value in lists.items():
            setattr(self, key, self.config_data[f"{value}"])

    def _init_dir(self):
        # Словарь для хранения информации
        dirs = self.config_data["dir_dirs"]
        for key, value in dirs.items():
            setattr(self, key, self.config_data[f"{value}"])

    def _make_os_dir(self):
        paths = self.config_data["dir_paths"]
        for key, value in paths.items():
            if key is not "script_path":
                print(self.config_data[f"{value}"])
                os.makedirs(self.config_data[f"{value}"], exist_ok=True)

    def _clean_temp(self):
        pass

    def _refresh_inference_params(self):
        pass

    def save_to_json(self, data, file_path):
        """
        Сохраняет данные в формате JSON в указанный файл.

        Параметры:
        - data: словарь или список данных для сохранения в JSON.
        - file_path: строка, путь к файлу JSON.

        Возвращает:
        - None
        """

        with open(file_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)

    def add_key_value(self, dictionary, key, value):
        """
        Добавляет новый ключ и значение в словарь.

        Параметры:
        - dictionary: словарь, в который добавляется новый ключ и значение.
        - key: ключ для добавления.
        - value: значение для добавления.

        Возвращает:
        - dictionary: словарь с добавленным ключом и значением.
        """
        dictionary[key] = value
        return dictionary

    def run(self):
        """
        Запуск приложения.
        """

        # Содержимое страницы
        self.title_page()

        # Загрузка dataframe
        self._load_csv()

        # Загрузка database
        self._create_database(self.df)

        # Контейнер ввода параметров
        self.input_param_container()

        # Контейнер для загрузки изображений
        self._load_images()

        # Загрузка словаря с метками
        self._load_class_labels()

        # Получение предсказания
        self._load_pretrained_model()

        if self.uploaded_images is not None and self.button_start:

            # Получение класса
            self._predict_class()

            # Визуализация
            self.container_visualise_image()

            self._calculate_time_work()

            self.get_features_by_class(self.predict_class)

            self.extract_features_from_image(self.uploaded_images)


    def title_page(self):
        """
        Содержимое страницы ввиде вступительного текста.
        """
        self.helper.create_indentations(1)
        self.progress = st.container()

    def input_param_container(self):

        # Контейнер для ввода параметров
        self.cont_param = st.container()

        self.button_start = st.button("Нажмите для обработки")

    def _calculate_time_work(self):
        if self.uploaded_images is not None:
            # Время окончания выполнения функции
            self.end_time = time.time()
            # Вычисление времени выполнения
            self.execution_time = self.end_time - self.start_time
            # Вывод времени выполнения на Streamlit
            self.progress.success(f"Время работы алгоритма: {self.execution_time:.2f} сек")

    def _load_images(self):
        """
        Загрузка изображений пользователя.
        """

        # Создаем контейнер с помощью st.container()
        self.cont_load_images = st.container()

        self.helper.create_indentations_in_container(1, self.cont_load_images)

        self.cont_load_images.divider()

        # Загрузка изображения через Streamlit
        self.uploaded_images = self.cont_load_images.file_uploader("Загрузите изображение",
                                                                   type=["jpg", "png", "jpeg", "tiff"],
                                                                   accept_multiple_files=False)

        if self.uploaded_images is not None:
            name_img = self.uploaded_images.name
            self.path_to_image = os.path.join(self.path_to_temp, f"{name_img}")
            image = Image.open(self.uploaded_images)
            image = image.convert('RGB')
            image = image.resize(size=(224, 224))
            image.save(self.path_to_image)

            # Определяем время загрузки изображения
            self._time()
            # Время начала выполнения
            self.start_time = time.time()

        self.cont_load_images.divider()

    def _load_pretrained_model(self):

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.model = EfficientNet.from_pretrained(model_name='efficientnet-b3',
                                                  weights_path=os.path.join(self.path_to_weights, "weights_class/2024_04_14_05_37_32_EfficientNetB3.pth"),
                                                  num_classes=self.n_class)
        self.model.eval().to(self.device)

    def _preprocess(self, img):
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        img = self.preprocess(img)

        return img

    def _load_csv(self):
        # Загрузка датафрейма
        self.df = pd.read_csv(os.path.join(self.path_to_database, "test_df.csv"))

    def _load_class_labels(self):
        # Загрузка словаря с метками
        with open(os.path.join(self.path_to_class_labels, "class_labels.json"), "r", encoding="utf-8") as json_file:
            self.class_to_labels = json.load(json_file)

    def _predict_class(self):

        img = Image.open(self.path_to_image)
        img = self._preprocess(img).unsqueeze(0)

        image_tensor = img.to(self.device)

        outputs = self.model(image_tensor)

        predicted_labels = torch.argmax(torch.nn.functional.softmax(outputs, dim=1), dim=1)

        predicted_label_index = predicted_labels.argmax().item() + 1

        predicted_label_text = next(
            (key for key, value in self.class_to_labels.items() if value == int(predicted_label_index)),
            None)

        self.predict_class = predicted_label_text

    @memory.cache
    def _create_database(self, df=None):

        try:
            with connect(
                    host="localhost",
                    user='user',
                    password="qwerty",
            ) as connection:
                with connection.cursor() as cursor:
                    create_database_query = "CREATE DATABASE IF NOT EXISTS image_database"
                    cursor.execute(create_database_query)
                    use_database_query = 'USE image_database'
                    cursor.execute(use_database_query)
                    create_table_query = '''
                    CREATE TABLE IF NOT EXISTS Images (
                        image_id INT AUTO_INCREMENT PRIMARY KEY,
                        image_path VARCHAR(255),
                        image_class VARCHAR(255),
                        vector MEDIUMBLOB
                    )
                    '''
                    cursor.execute(create_table_query)
                    self.progress.success(f"База данных готова к использованию!")
                    insert_data_query = """
                        INSERT INTO Images
                        (image_path, image_class, vector)
                        VALUES ( %s, %s, %s)
                        """
                    for i in range(len(df['category'])):
                        name = df['filename'][i]
                        category = df['category'][i]
                        path_to_img = os.path.join(path_to_dir_image, name)
                        img = Image.open(path_to_img)
                        image_tensor = preprocess(img).unsqueeze(0)
                        features = pickle.dumps(model.extract_features(image_tensor).view(-1).detach().numpy())

                        cursor.executemany(insert_data_query, [(path_to_img, category, features)])

        except Error as e:
            print(e)

    def get_features_by_class(self, target_class):
        """
        Получает признаки объектов заданного класса из базы данных.

        Параметры:
        - target_class: строка, класс объектов, признаки которых нужно получить.

        Возвращает:
        - features_list: список признаков объектов заданного класса.
        """

        try:
            with connect(
                    host="localhost",
                    user='user',
                    password="qwerty",
                    database="image_database"
            ) as connection:
                with connection.cursor() as cursor:
                    # Выбираем все записи с заданным классом
                    select_query = "SELECT vector FROM Images WHERE image_class = %s"
                    cursor.execute(select_query, (target_class,))
                    rows = cursor.fetchall()

                    # Извлекаем признаки из каждой записи
                    for row in rows:
                        features_list.append(pickle.loads(row[0]))

        except Error as e:
            print(e)

        return self.features_list

    def extract_features_from_image(self, image):
        """
        Извлекает вектор признаков из загруженного изображения.

        Параметры:
        - image_path: строка, путь к загруженному изображению.

        Возвращает:
        - features: вектор признаков изображения.
        """

        img = Image.open(image)
        img = self._preprocess(img).unsqueeze(0)

        # Перемещаем изображение на устройство (GPU или CPU)
        image_tensor = img.to(self.device)

        # Применяем модель к изображению для извлечения признаков
        with torch.no_grad():
            self.model.eval()
            outputs = self.model(image_tensor)

        # Получаем вектор признаков из выходов модели
        self.features = outputs.squeeze().cpu().numpy()

        return self.features

    def find_similar_images(self, query_features, all_features, top_n=10):
        """
        Находит топ N изображений, ближайших к вектору признаков загруженного изображения по косинусному расстоянию.

        Параметры:
        - query_features: numpy массив, вектор признаков загруженного изображения.
        - all_features: список numpy массивов, признаки всех изображений.
        - top_n: целое число, количество ближайших изображений для вывода.

        Возвращает:
        - similar_images: список кортежей (индекс изображения, расстояние) топ N ближайших изображений.
        """
        similar_images = []

        # Перебираем признаки всех изображений
        for idx, features in enumerate(all_features):
            # Вычисляем косинусное расстояние между признаками
            distance = cosine(query_features, features)
            # Добавляем индекс и расстояние в список
            similar_images.append((idx, distance))

        # Сортируем список по расстоянию
        similar_images.sort(key=lambda x: x[1])

        # Возвращаем топ N ближайших изображений
        self.similar_images = similar_images[:top_n]
        print(self.similar_images)
        return self.similar_images

    def container_visualise_image(self):
        """
        Контейнер, содержащий визуализацию аннотированного изображения.
        """

        self.visualise_cont = st.container()

        st.info(f"Класс найденного объекта: {self.predict_class}")
        st.image(self.uploaded_images, caption=f"{self.predict_class}", width=512)

    def container_visualise_top(self):

        self.progress.info(f"Топ 10:  {self.similar_images}")


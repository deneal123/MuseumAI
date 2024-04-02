import os
import json
from library.custom_logging import setup_logging

# Назначение переменной логирования
log = setup_logging()


# Создание стандартных настроек конфигурационного файла
def save_default_config():
    """
    Создает и сохраняет стандартные настройки конфигурационного файла.

    :param: None.
    :return: None.
    """

    config_data = {

        "script_path": None,
        "path_to_temp": None,
        "path_to_inference": None,
        "path_to_weights": None,


        "dir_paths": {

            "script_path": "script_path",
            "path_to_temp": "path_to_temp",
            "path_to_inference": "path_to_inference",
            "path_to_weights": "path_to_weights",

        },

        "uploaded_images": None,
        "start_time": None,
        "end_time": None,
        "execution_time": None,
        "progress": None,

        "dir_params": {

            "uploaded_images": "uploaded_images",
            "start_time": "start_time",
            "end_time": "end_time",
            "execution_time": "execution_time",
            "progress": "progress",

        },

        "name_uploaded_images": [],
        "name_uploaded_images_without_ex": [],

        "dir_lists": {

            "name_uploaded_images": "name_uploaded_images",
            "name_uploaded_images_without_ex": "name_uploaded_images_without_ex"

        },

        "dir_data": {},

        "dir_dirs": {

            "dir_data": "dir_data"

        }

    }

    with open('config.json', 'w') as config_file:
        json.dump(config_data, config_file, indent=4)

    log.info('Применение стандартных настроек конфигурационного файла')


# Глобальная переменная для хранения предыдущих параметров
previous_config_data = None


def parse_list(self, input_str):
    try:
        # Пробуем разобрать введенную строку как JSON
        list_values = json.loads(input_str)
        # Проверяем, что полученный объект - это список
        if not isinstance(list_values, list):
            raise ValueError("Введенные данные не являются списком.")
        return list_values
    except ValueError as e:
        st.error(f"Ошибка: {e}")
        return None


def load_config():
    """
    Загружает данные конфигурационного файла и сохраняет их в глобальную переменную.

    :param previous_config_file: Глобальная переменная хранящая список.
    :return config_data: Список данных выгруженных из конфигурационного файла.
    """

    global previous_config_data

    # Загрузка конфигурационных параметров из файла
    with open('config.json', 'r') as config_file:
        config_data = json.load(config_file)

    # Сохраняем текущие параметры в глобальной переменной
    previous_config_data = config_data

    return config_data


def is_value_changed(key, new_value):
    """
    Создает и сохраняет стандартные настройки конфигурационного файла.

    :param previous_config_file: Глобальная переменная хранящая список.
    :return previous_config_data[key] != new_value: Возвращает в глобальную переменную...
    Новое значение для соответствующего ключа, если это значение ключа...
    Было измененно.
    :return True: Возвращает true, если значение ключа не изменилось.
    """

    global previous_config_data

    # Проверяем, изменилось ли значение параметра
    if previous_config_data is not None and key in previous_config_data:
        return previous_config_data[key] != new_value
    else:
        # Если предыдущих данных нет, считаем, что значение изменилось
        return True


def update_config(self, **kwargs):
    """
    Обновляет конфигурационные данные на основе переданных параметров.
    """

    # Аналогичная функция _update_config, но для классов

    for key, value in kwargs.items():
        # Используем get для проверки, что ключ существует в конфигурационных данных
        if key in self.config_data and is_value_changed(key, value):
            self.config_data[key] = value

    save_config(self.config_data)


def _update_config(config_data, **kwargs):
    """
    Обновляет конфигурационные данные на основе переданных параметров.
    """

    for key, value in kwargs.items():
        # Используем get для проверки, что ключ существует в конфигурационных данных
        if key in config_data and is_value_changed(key, value):
            config_data[key] = value

    save_config(config_data)


def save_config(config_data):
    """
    Сохраняет параметры в конфигурационный файл.

    :param: None.
    :return: None.
    """

    # Сохранение конфигурационных параметров в файл
    with open('config.json', 'w') as config_file:
        json.dump(config_data, config_file, indent=4)


def default_path():
    """
    Получить путь к текущему исполняемому скрипту.

    :param: None.
    :return: None.
    """
    config_data = load_config()
    script_path = os.path.realpath(__file__)
    script_path, _ = os.path.split(script_path)

    paths = {

        "path_to_temp": "temp",
        "path_to_inference": "inference",
        "path_to_weights": "weights",

    }

    for key, value in paths.items():
        paths[key] = os.path.join(script_path, value)

    paths["script_path"] = script_path

    _update_config(config_data=config_data, **paths)


if __name__ == '__main__':
    save_default_config()
    default_path()

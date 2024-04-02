import streamlit as st
from library.custom_logging import setup_logging
from library.IndentationHelper import IndentationHelper
from config_file import load_config, is_value_changed, save_config


class PageInfo:

    def __init__(self):

        # Назначение переменной логирования
        self.log = setup_logging()
        # Загрузка конфигурационного файла
        self.config_data = load_config()
        # Класс функций для отступов
        self.helper = IndentationHelper()

    def run(self):
        """
        Запуск приложения.
        """

        # Содержимое страницы
        self.title_page()

    def title_page(self):
        """
        Содержимое страницы ввиде вступительного текста.
        """
        self.helper.create_indentations(1)

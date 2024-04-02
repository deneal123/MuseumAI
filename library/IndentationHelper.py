import streamlit as st


class IndentationHelper:
    def __init__(self):
        self.num_indentations = 3

    def create_indentations(self, multiply):
        """
        Создает отступы в интерфейсе streamlit.

        :param multiply: множитель отступов.
        :return: None.
        """
        for _ in range(self.num_indentations * multiply):
            st.write("\n")

    def create_indentations_in_container(self, multiply, container):
        """
        Создает отступы в интерфейсе streamlit внутри заданного контейнера.

        :param multiply: множитель отступов.
        :param container: созданный контейнер.
        :return: None.
        """
        for _ in range(self.num_indentations * multiply):
            container.write("\n")

    @staticmethod
    def css_indentation_write(indentation, container, text):
        # Создаем стиль для центрирования текста
        style = '<style>.text-center {text-align: center; font-family: Comic Sans MS;}</style>'

        # Вставляем стиль в Streamlit
        container.markdown(style, unsafe_allow_html=True)

        # Добавляем текст с классом "text-center" для применения стиля
        container.write(f'<p class="text-center">{text}</p>', unsafe_allow_html=True)

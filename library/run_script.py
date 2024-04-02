import subprocess


def run_script(name, tool):
    """
    Функция для запуска скрипта с использованием указанного инструмента (python или streamlit).

    :param name: название скрипта.
    :param tool: инструмент для запуска (python или streamlit).
    :return: None.
    """

    if tool not in ["python", "streamlit"]:
        print("Неподдерживаемый инструмент. Допустимые значения: 'python' или 'streamlit'.")
        return

    # Выполнение команды в консоли
    subprocess.run([tool, 'run', name])

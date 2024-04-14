import os
from PIL import Image


def delete_files(file_paths):
    for file_path in file_paths:
        try:
            os.remove(file_path)
            print(f"Файл {file_path} удален.")
        except OSError as e:
            print(f"Ошибка при удалении файла {file_path}: {e}")


def delete_broken_images(directory_images, directory_json):
    broken_images = []
    for filename in os.listdir(directory_images):
        image_path = os.path.join(directory_images, filename)
        json_path = os.path.join(directory_json, filename.replace('.png', '.json'))  # Предполагается, что формат изображений JPEG и соответствующих им JSON файлов
        try:
            with Image.open(image_path) as img:
                img.verify()
        except (IOError, SyntaxError) as e:
            broken_images.append(image_path)
            if os.path.exists(json_path):
                broken_images.append(json_path)

    # Удаление поврежденных изображений и соответствующих JSON файлов
    delete_files(broken_images)


# Пример использования функции
delete_broken_images('C:/Users/NightMare/PycharmProjects/MuseumAI/data_with_class/images/',
                     'C:/Users/NightMare/PycharmProjects/MuseumAI/data_with_class/jsons/')
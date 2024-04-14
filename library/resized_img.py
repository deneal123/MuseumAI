from PIL import Image
import os


def resize_images(folder_path, new_size=(512, 512)):
    try:
        # Проверяем, существует ли указанная папка
        if not os.path.exists(folder_path):
            print(f"Папка {folder_path} не существует.")
            return

        # Получаем список файлов в папке
        files = os.listdir(folder_path)

        # Проходим по каждому файлу
        for file in files:
            # Полный путь к файлу
            file_path = os.path.join(folder_path, file)

            # Проверяем, является ли файл изображением
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tif', '.tiff')):
                with Image.open(file_path) as img:

                    # Изменяем размер
                    resized_img = img.resize(new_size)

                    # Сохраняем измененное изображение
                    resized_file_path = os.path.join(folder_path, f"{file}")
                    resized_img.save(resized_file_path)

                    print(f"Изображение {file} изменено и сохранено в {resized_file_path}")

        print("Изменение размера завершено.")
    except Exception as e:
        print(f"Произошла ошибка: {e}")


list_ = ["C:/Users/NightMare/PycharmProjects/MuseumAI/data_with_description/test_df",
         "C:/Users/NightMare/PycharmProjects/MuseumAI/data_with_description/train_df"]

for index in list_:
    # Пример использования
    resize_images(index,
                  new_size=(224, 224))

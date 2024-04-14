from PIL import Image
import os


def convert_images_to_rgb(directory_path):
    # Проверяем, существует ли указанный путь
    if not os.path.exists(directory_path):
        print(f"Директория {directory_path} не существует.")
        return

    # Получаем список файлов в директории
    image_files = [f for f in os.listdir(directory_path) if f.endswith(('.jpg', '.jpeg', '.png', '.TIF'))]

    # Перебираем все изображения в директории
    for image_file in image_files:
        image_path = os.path.join(directory_path, image_file)

        try:
            # Открываем изображение
            img = Image.open(image_path)

            # Если изображение не в формате RGB, конвертируем
            if img.mode != 'RGB':
                rgb_img = img.convert('RGB')

                # Сохраняем обновленное изображение
                rgb_img.save(image_path)
                print(f"Изображение {image_file} успешно сконвертировано в RGB формат.")
            else:
                print(f"Изображение {image_file} уже в RGB формате.")

        except Exception as e:
            print(f"Ошибка при обработке изображения {image_file}: {e}")


# Пример использования функции
directory_path = ["C:/Users/NightMare/PycharmProjects/MuseumAI/data_with_class/train_dataset_mincult-train/train_df",
                  "C:/Users/NightMare/PycharmProjects/MuseumAI/data_with_class/train_dataset_mincult-train/test_df"]
for path in directory_path:
    convert_images_to_rgb(path)

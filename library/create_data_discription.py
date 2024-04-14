import pandas as pd
import os
from sklearn.model_selection import train_test_split
import shutil
import re


def clean_description(text):
    if isinstance(text, str):
        # Удаляем знаки пунктуации с помощью регулярных выражений
        cleaned_text = re.sub(r'[^\w\s]', '', text)
        return cleaned_text
    else:
        return None


def get_df(path_to_train_csv, path_to_dir_image):
    with open(path_to_train_csv, 'r', encoding='utf-8') as csv_file:
        df = pd.read_csv(csv_file, sep=';')

    filenames = []
    categories = []

    for index in range(len(df['object_id'])):
        object_id = df['object_id'][index]
        group = clean_description(df['description'][index])
        name = df['img_name'][index]
        path_to_img = os.path.join(f"{object_id}", name)
        filenames.append(path_to_img)
        categories.append(group)

    # Формируем дату.
    df = pd.DataFrame({
        'filename': filenames,
        'category': categories})

    # Удаляем строки с отсутствующим текстовым описанием (None)
    df.dropna(subset=['category'], inplace=True)

    return df


def split_data(df, test_size=0.2, random_state=None):
    # Разбиваем на train и test
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)

    return train_df, test_df


def split_dir(train_df, test_df, path_to_dir_image, path_to):
    # Создаем директории для train и test
    train_dir = os.path.join(path_to, "train_df")
    test_dir = os.path.join(path_to, "test_df")

    for dir_path in [train_dir, test_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # Сохраняем новые имена файлов в датафреймы train_df и test_df
    new_filenames_train = []
    new_filenames_test = []

    for df, data_split, new_filenames in [(train_df, train_dir, new_filenames_train),
                                          (test_df, test_dir, new_filenames_test)]:
        for index, row in df.iterrows():
            src = os.path.join(path_to_dir_image, row['filename'])
            new_filename = f"({index}).png"
            new_filenames.append(new_filename)
            dst = os.path.join(data_split, new_filename)
            shutil.copy(src, dst)

    # Обновляем столбец с именами файлов в датафреймах train_df и test_df
    train_df['filename'] = new_filenames_train
    test_df['filename'] = new_filenames_test

    # Сохраняем CSV файлы для train и test
    train_df.to_csv(os.path.join(path_to, "train_df.csv"), index=False)
    test_df.to_csv(os.path.join(path_to, "test_df.csv"), index=False)


# Загружаем данные
df = get_df(
    path_to_train_csv="C:/Users/NightMare/PycharmProjects/MuseumAI/data_with_class/train_dataset_mincult-train/train.csv",
    path_to_dir_image="C:/Users/NightMare/PycharmProjects/MuseumAI/data_with_class/train_dataset_mincult-train/train/")
print(df)
# Разделяем данные на train, valid, test
train_df, test_df = split_data(df, test_size=0.1, random_state=42)
print("Размер тренировочной выборки:", len(train_df))
print("Размер тестовой выборки:", len(test_df))
# Выгружаем из папки данные и разделяем по папкам
split_dir(train_df, test_df,
          path_to_dir_image="C:/Users/NightMare/PycharmProjects/MuseumAI/data_with_class/train_dataset_mincult-train/train",
          path_to="C:/Users/NightMare/PycharmProjects/MuseumAI/data_with_description")

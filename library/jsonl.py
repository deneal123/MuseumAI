import pandas as pd
import os
import json


def get_df_text(path_to_train_csv, path_to_dir_image):
    with open(path_to_train_csv, 'r', encoding='utf-8') as csv_file:
        df = pd.read_csv(csv_file, sep=';')

    filenames = []
    categories = []

    for index in range(len(df['object_id'])):
        object_id = df['object_id'][index]
        group = str(df['description'][index])
        name = str(df['img_name'][index])
        path_to_img = str(os.path.join(f"{object_id}", name))
        filenames.append(path_to_img)
        categories.append(group)

    # Формируем дату.
    df = pd.DataFrame({
        'filename': filenames,
        'category': categories})

    # Удаляем строки с отсутствующим текстовым описанием (None)
    df.dropna(subset=['category'], inplace=True)

    return df


def get_caption(df):

    # Инициализируем список для хранения словарей
    captions = []

    # Итерируем по строкам DataFrame
    for index, row in df.iterrows():
        # Создаем словарь для каждой строки
        caption = {
            "file_name": str(row["filename"]),
            "text": str(row["category"])
        }
        # Добавляем словарь в список
        captions.append(caption)

    return captions


def create_JSONL(root, captions, name_jsonl='metadata'):

    # add metadata.jsonl file to this folder
    with open(root + f"{name_jsonl}.jsonl", 'w') as f:
        for item in captions:
            f.write(json.dumps(item) + "\n")


# Загружаем данные
df = get_df_text(
    path_to_train_csv="C:/Users/NightMare/PycharmProjects/MuseumAI/data_with_class/train_dataset_mincult-train/train.csv",
    path_to_dir_image="C:/Users/NightMare/PycharmProjects/MuseumAI/data_with_class/train_dataset_mincult-train/train/")

captions = get_caption(df)

create_JSONL(
    root="C:/Users/NightMare/PycharmProjects/MuseumAI/data_with_class/train_dataset_mincult-train/",
    captions=captions
)

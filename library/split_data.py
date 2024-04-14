import pandas as pd
import os
from sklearn.model_selection import train_test_split
import shutil


def get_df(path_to_train_csv, path_to_dir_image):
    with open(path_to_train_csv, 'r', encoding='utf-8') as csv_file:
        df = pd.read_csv(csv_file, sep=';')

    filenames = []
    categories = []

    for index in range(len(df['object_id'])):
        object_id = df['object_id'][index]
        group = df['group'][index]
        name = df['img_name'][index]
        path_to_img = os.path.join(f"{object_id}", name)
        filenames.append(path_to_img)
        categories.append(group)

    # Формируем дату.
    df = pd.DataFrame({
        'filename': filenames,
        'category': categories})

    df['category'].unique()

    dir_class_name = {
        'Археология': 'предметы археалогии',
        'Оружие': 'оружие',
        'Прочие': 'прочие',
        'Нумизматика': 'предметы нумизматики',
        'Фото, негативы': 'фотографии и негативы',
        'Редкие книги': 'редкие книги',
        'Документы': 'документы',
        'Печатная продукция': 'предметы печатной продукции',
        'ДПИ': 'декоративно прикладное искусство',
        'Скульптура': 'скульптура',
        'Графика': 'графика',
        'Техника': 'предметы техники',
        'Живопись': 'живопись',
        'Естественнонауч.коллекция': 'предметы естественнонаучной коллекции',
        'Минералогия': 'предметы минералогической коллекции'
    }

    df['category'] = df['category'].map(dir_class_name)

    return df


def split_data(df, test_size=0.2, random_state=None):
    # Разбиваем на train и test
    train_df, test_df = train_test_split(df, test_size=test_size, stratify=df['category'], random_state=random_state)

    return train_df, test_df


def split_dir(train_df, test_df, path_to_dir_image, path_to):
    # Создаем директории для train, val и test
    train_dir = os.path.join(path_to, "train_df")
    test_dir = os.path.join(path_to, "test_df")

    for dir_path in [train_dir, test_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # Перемещаем изображения в соответствующие директории
    for df, data_split in [(train_df, train_dir), (test_df, test_dir)]:
        for index, row in df.iterrows():
            src = os.path.join(path_to_dir_image, row['filename'])
            dst = os.path.join(data_split, f"{row['category']} ({index}).png")
            shutil.copy(src, dst)


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
          path_to="C:/Users/NightMare/PycharmProjects/MuseumAI/data_with_class/train_dataset_mincult-train/")

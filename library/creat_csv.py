import pandas as pd
import json
import os
from jsonl import get_df_text


def _load_data(path, save):

    save_path_labels = os.path.join(save,
                                    f"test_df.csv")

    # Собираем данные
    filenames = os.listdir(path)
    categories = [filename.split(' (')[0] for filename in filenames]
    dataframe = pd.DataFrame({'filename': filenames, 'category': categories})

    df = dataframe

    # Сохраняем датафрейм в CSV файл
    df.to_csv(save_path_labels, index=False, encoding='utf-8')

    return df


df = _load_data(path="C:/Users/NightMare/PycharmProjects/MuseumAI/data_with_class/train_dataset_mincult-train/test_df",
                save="C:/Users/NightMare/PycharmProjects/MuseumAI/data_with_class/train_dataset_mincult-train")

print(df)


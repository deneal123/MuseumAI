import os
import math
import statistics
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from library.data_preprocessing import data_list
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import bitsandbytes as bnb
import threading
import random
import numpy as np
from datetime import datetime
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras import backend as K
from sklearn.metrics import accuracy_score, f1_score as f1, precision_score, recall_score
from efficientnet_pytorch import EfficientNet
import json
from torch.optim.lr_scheduler import LambdaLR


class CustomImageDataset(Dataset):
    def __init__(self, dataframe, root_dir, IMAGE_WIDTH, IMAGE_HEIGHT, num_classes, seed=None, is_test=False):
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.IMAGE_WIDTH = IMAGE_WIDTH
        self.IMAGE_HEIGHT = IMAGE_HEIGHT
        self.seed = seed
        self.is_test = is_test
        self.N_classes = num_classes

        # Трансформер для аугментации данных

        # Общий трансформер для изображений и масок
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=(self.IMAGE_WIDTH, self.IMAGE_HEIGHT), antialias=True),
            transforms.RandomRotation(degrees=180),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ElasticTransform(),
            transforms.RandomAdjustSharpness(sharpness_factor=0.5),
            transforms.GaussianBlur(kernel_size=3),
            transforms.RandomAutocontrast(),
        ])

        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=(self.IMAGE_WIDTH, self.IMAGE_HEIGHT), antialias=True)
        ])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        img_name = os.path.join(self.root_dir, self.dataframe.iloc[index, 0])
        image = Image.open(img_name)
        category = int(self.dataframe.iloc[index, 1])

        if self.transform:
            if self.seed is None:
                self.seed = np.random.randint(2147483647)

            if self.is_test:
                random.seed(self.seed)
                torch.manual_seed(self.seed)
                image = self.transform_test(image)
            else:
                random.seed(self.seed)
                torch.manual_seed(self.seed)
                image = self.transform(image)

        category_one_hot = torch.zeros(self.N_classes)
        category_one_hot[category - 1] = 1

        return image, category_one_hot


class LossBasedLRScheduler:
    def __init__(self, optimizer, factor=0.1, patience=3, threshold=0.01, min_lr=0):
        self.optimizer = optimizer
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.min_lr = min_lr
        self.prev_loss = float('inf')
        self.counter = 0

    def __call__(self, loss):
        if loss > self.prev_loss - self.threshold:
            self.counter += 1
            if self.counter >= self.patience:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = max(param_group['lr'] * self.factor, self.min_lr)
                self.counter = 0
        else:
            self.counter = 0
        self.prev_loss = loss


class ImageClassifierTrainer:
    def __init__(self,
                 img_w: int = 128,
                 img_h: int = 128,
                 n_class: int = 15,
                 img_channels: int = 3,
                 num_epoch: int = 1,
                 batch_size: int = 8,
                 model: str = 'ResNet',
                 test_dir: str = '',
                 train_dir: str = '', ):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Загрузка параметров обучения и параметров моделей
        self.IMAGE_HEIGHT = img_h
        self.IMAGE_WIDTH = img_w
        self.N_CLASS = n_class
        self.IMAGE_CHANNELS = img_channels
        self.NUM_EPOCH = num_epoch
        self.BATCH_SIZE = batch_size
        self.selected_model = model
        self.test_dir = test_dir
        self.train_dir = train_dir
        self.path_to_class_to_label = {}

        self.df = pd.DataFrame()
        self.temp_df = pd.DataFrame()

        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None

        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.timestamp = None

        self.train_losses = []
        self.train_accuracy = []
        self.train_f1_score = []
        self.train_precision = []
        self.train_recall = []

        self.valid_accuracy = []
        self.valid_f1_score = []
        self.valid_precision = []
        self.valid_recall = []

        self.test_accuracy = []
        self.test_f1_score = []
        self.test_precision = []
        self.test_recall = []

        self.scheduler = None

    def _load_data(self, path):
        # Получаем список уникальных классов из имен файлов
        class_names = set(filename.split(' (')[0] for filename in os.listdir(path))

        # Создаем словарь для отображения классов в числовые метки
        self.class_to_label = {class_name: i + 1 for i, class_name in enumerate(class_names)}

        labels_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   f"class_labels/{self.timestamp}_{self.selected_model}_class_labels/")
        os.makedirs(labels_path, exist_ok=True)
        save_path_labels = os.path.join(labels_path,
                                        f"{self.timestamp}_{self.selected_model}_class_labels.json")

        # Сохранение словаря в файл
        with open(save_path_labels, 'w', encoding='utf-8') as f:
            json.dump(self.class_to_label, f, ensure_ascii=False, indent=4)

        # Собираем данные
        filenames = os.listdir(path)
        categories = [self.class_to_label[filename.split(' (')[0]] for filename in filenames]
        dataframe = pd.DataFrame({'filename': filenames, 'category': categories})

        self.df = dataframe

    def _split_data(self, is_test=False):
        if is_test:
            datasets = {
                'test': CustomImageDataset(self.df,
                                           self.test_dir,
                                           self.IMAGE_WIDTH, self.IMAGE_HEIGHT, is_test=True, num_classes=self.N_CLASS)
            }

            loaders = {
                split: DataLoader(datasets[split], batch_size=self.BATCH_SIZE, shuffle=False,
                                  pin_memory=True)
                for split in ['test']}

            self.test_loader = loaders['test']

        else:
            train_df, valid_df = train_test_split(self.df, test_size=0.1)

            datasets = {
                'train': CustomImageDataset(train_df,
                                            self.train_dir,
                                            self.IMAGE_WIDTH, self.IMAGE_HEIGHT, num_classes=self.N_CLASS),
                'valid': CustomImageDataset(valid_df,
                                            self.train_dir,
                                            self.IMAGE_WIDTH, self.IMAGE_HEIGHT, num_classes=self.N_CLASS)
            }

            loaders = {
                split: DataLoader(datasets[split], batch_size=self.BATCH_SIZE, shuffle=(split == 'train'),
                                  pin_memory=True)
                for split in ['train', 'valid']}

            self.train_loader, self.valid_loader = loaders['train'], loaders['valid']

    def _load_model(self):

        if self.selected_model == "ResNet":
            self.model = models.resnet18(weights='IMAGENET1K_V1')
            self.model.fc = torch.nn.Linear(self.model.fc.in_features, self.N_CLASS)
        elif self.selected_model == "EfficientNetB3":
            self.model = EfficientNet.from_pretrained(model_name='efficientnet-b3',
                                                      weights_path="C:/Users/NightMare/PycharmProjects/MuseumAI/"
                                                                   "weights/weights_class/"
                                                                   "adv-efficientnet-b3-cdd7c0f4.pth",
                                                      num_classes=self.N_CLASS)

        self.model.to(self.device)

        self.optimizer = bnb.optim.AdamW8bit(self.model.parameters(), lr=0.0001, betas=(0.9, 0.999))
        # Создание объекта LossBasedLRScheduler
        self.scheduler = LossBasedLRScheduler(self.optimizer, factor=0.98, patience=3, threshold=0.01, min_lr=0)

        """# Веса для каждого класса
        class_weights = {
            '5': 0.6697942386831276, '3': 0.8908593322386426, '0': 0.8974910394265233,
            '8': 0.9064884433305486, '13': 0.9172161172161172, '14': 0.9321878579610539,
            '7': 0.943536231884058, '11': 0.975779376498801, '9': 1.0575698505523068,
            '4': 1.1049558723693143, '6': 1.1397759103641456, '2': 1.171778257739381,
            '10': 1.252963818321786, '12': 1.284102564102564, '1': 1.3136400322841
        }

        # Преобразование весов в тензор и перемещение на устройство (GPU или CPU)
        weights = torch.tensor(list(class_weights.values())).to(self.device)"""
        self.criterion = torch.nn.CrossEntropyLoss()

    def _compute_accuracy(self, predictions, labels):
        _, predicted = torch.max(predictions, 1)

        # Конвертируем one-hot encoding обратно в индексы классов
        _, labels = labels.max(1)

        # Используем функцию из библиотеки scikit-learn для вычисления точности
        accuracy = accuracy_score(labels.cpu().numpy(), predicted.cpu().numpy())

        return accuracy

    def _compute_f1_score(self, predictions, labels):
        _, predicted = torch.max(predictions, 1)

        # Конвертируем one-hot encoding обратно в индексы классов
        _, labels = labels.max(1)

        # Используем функцию из библиотеки scikit-learn для вычисления F1-score
        f1_score = f1(labels.cpu().numpy(), predicted.cpu().numpy(), average='macro', zero_division=0)

        return f1_score

    def _compute_precision(self, predictions, labels):
        _, predicted = torch.max(predictions, 1)

        # Конвертируем one-hot encoding обратно в индексы классов
        _, labels = labels.max(1)

        # Используем функцию из библиотеки scikit-learn для вычисления precision
        precision = precision_score(labels.cpu().numpy(), predicted.cpu().numpy(), average='macro', zero_division=0)

        return precision

    def _compute_recall(self, predictions, labels):
        _, predicted = torch.max(predictions, 1)

        # Конвертируем one-hot encoding обратно в индексы классов
        _, labels = labels.max(1)

        # Используем функцию из библиотеки scikit-learn для вычисления recall
        recall = recall_score(labels.cpu().numpy(), predicted.cpu().numpy(), average='macro', zero_division=0)

        return recall

    def _time(self):
        self.timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    def _save_json_data(self, data, file_path):
        with open(file_path, 'w') as json_file:
            json.dump(data, json_file)

    def _save_plot(self, metric_data, metric_name):
        fig, ax = plt.subplots()

        for metric_type, color, label in zip(metric_data, ['b', 'g', 'y'], ['Тренировка', 'Валидация', 'Тест']):
            ax.plot(range(1, self.NUM_EPOCH + 1), metric_type, marker='.', color=color, label=label)

        ax.set(xlabel='Номер эпохи', ylabel=f'Значение метрики f{metric_name}', title=f'{metric_name}')
        ax.legend()
        fig_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                f"plot/plot_class/{self.timestamp}_{self.selected_model}_plot/")
        os.makedirs(fig_path, exist_ok=True)
        save_path = os.path.join(fig_path,
                                 f"{self.timestamp}_{self.selected_model}_{metric_name.lower()}_plot.png")
        json_data = {'epochs': list(range(1, self.NUM_EPOCH + 1)), 'data': metric_data}
        self._save_json_data(json_data, save_path + ".json")
        fig.savefig(save_path, format='png', dpi=500)
        plt.close(fig)

    def _plot_final_plot(self):
        self._save_plot([self.train_losses], 'Функция потерь')
        self._save_plot([self.train_accuracy, self.valid_accuracy, self.test_accuracy], 'Истинность (Accuracy)')
        self._save_plot([self.train_f1_score, self.valid_f1_score, self.test_f1_score], 'F1-мера (F1-score)')
        self._save_plot([self.train_precision, self.valid_precision, self.test_precision], 'Точность (Precision)')
        self._save_plot([self.train_recall, self.valid_recall, self.test_recall], 'Полнота (Recall)')

    def _save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def _train_one_epoch(self):
        self.model.train()
        total_loss = 0.0
        total_accuracy = 0.0
        total_f1_score = 0.0
        total_precision = 0.0
        total_recall = 0.0

        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), leave=False)
        for batch_idx, (images, target) in pbar:
            images, target = images.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)

            # Вычисление кросс-энтропии
            loss = self.criterion(outputs, target)
            total_loss += loss.item()
            self.scheduler(loss.item())

            # Вычисление метрики accuracy
            accuracy = self._compute_accuracy(torch.nn.functional.softmax(outputs, dim=1), target)
            total_accuracy += accuracy

            # Вычисление метрики f1_score
            f1_score = self._compute_f1_score(torch.nn.functional.softmax(outputs, dim=1), target)
            total_f1_score += f1_score

            # Вычисление метрики precision
            precision = self._compute_precision(torch.nn.functional.softmax(outputs, dim=1), target)
            total_precision += precision

            # Вычисление метрики recall
            recall = self._compute_recall(torch.nn.functional.softmax(outputs, dim=1), target)
            total_recall += recall

            loss.backward()
            self.optimizer.step()

            pbar.set_description(f"Loss:{(total_loss / (batch_idx + 1)): .4f}"
                                 f" Accuracy:{(total_accuracy / (batch_idx + 1)):.4f}"
                                 f" F1-score: {(total_f1_score / (batch_idx + 1)):.4f}"
                                 f" Precision: {(total_precision / (batch_idx + 1)):.4f}"
                                 f" Recall: {(total_recall / (batch_idx + 1)):.4f}")

        avg_loss = total_loss / len(self.train_loader)
        avg_accuracy = total_accuracy / len(self.train_loader)
        avg_f1_score = total_f1_score / len(self.train_loader)
        avg_precision = total_precision / len(self.train_loader)
        avg_recall = total_recall / len(self.train_loader)
        return avg_loss, avg_accuracy, avg_f1_score, avg_precision, avg_recall

    def _validate_one_epoch(self):
        self.model.eval()
        total_accuracy = 0.0
        total_f1_score = 0.0
        total_precision = 0.0
        total_recall = 0.0

        with torch.no_grad():
            pbar = tqdm(enumerate(self.valid_loader), total=len(self.valid_loader), leave=False)
            for batch_idx, (images, target) in pbar:
                images, target = images.to(self.device), target.to(self.device)
                outputs = self.model(images)

                # Вычисление метрики accuracy
                accuracy = self._compute_accuracy(torch.nn.functional.softmax(outputs, dim=1), target)
                total_accuracy += accuracy

                # Вычисление метрики f1_score
                f1_score = self._compute_f1_score(torch.nn.functional.softmax(outputs, dim=1), target)
                total_f1_score += f1_score

                # Вычисление метрики precision
                precision = self._compute_precision(torch.nn.functional.softmax(outputs, dim=1), target)
                total_precision += precision

                # Вычисление метрики recall
                recall = self._compute_recall(torch.nn.functional.softmax(outputs, dim=1), target)
                total_recall += recall

                pbar.set_description(f" Accuracy:{(total_accuracy / (batch_idx + 1)): .4f}"
                                     f" F1-score: {(total_f1_score / (batch_idx + 1)):.4f}"
                                     f" Precision: {(total_precision / (batch_idx + 1)):.4f}"
                                     f" Recall: {(total_recall / (batch_idx + 1)):.4f}")

        avg_accuracy = total_accuracy / len(self.valid_loader)
        avg_f1_score = total_f1_score / len(self.valid_loader)
        avg_precision = total_precision / len(self.valid_loader)
        avg_recall = total_recall / len(self.valid_loader)
        return avg_accuracy, avg_f1_score, avg_precision, avg_recall

    def _predict_on_test_set(self):
        self.model.eval()
        total_accuracy = 0.0
        total_f1_score = 0.0
        total_precision = 0.0
        total_recall = 0.0

        # Сохранение изображений, предсказанных и истинных меток
        script_dir = os.path.dirname(os.path.abspath(__file__))

        with torch.no_grad():
            pbar = tqdm(enumerate(self.test_loader), total=len(self.test_loader), leave=False)
            for batch_idx, (images, target) in pbar:
                images, target = images.to(self.device), target.to(self.device)
                outputs = self.model(images)

                # Визуализация оригинального изображения, предсказанной и тестовой масок
                plt.figure(figsize=(6, 6))

                # Выбор первого изображения из батча для отображения
                single_image = np.transpose(images[0].cpu().numpy(), (1, 2, 0))

                predicted_labels = torch.argmax(torch.nn.functional.softmax(outputs, dim=1), dim=1)

                true_label_index = torch.where(target[0] == 1)[0].cpu().numpy()[0] + 1
                predicted_label_index = predicted_labels.argmax().item() + 1

                # Получаем текстовое представление меток и словаря
                true_label_text = next(
                    (key for key, value in self.class_to_label.items() if value == int(true_label_index)),
                    None)
                predicted_label_text = next(
                    (key for key, value in self.class_to_label.items() if value == int(predicted_label_index)),
                    None)
                print(true_label_text)
                plt.imshow(single_image)
                plt.title(f'Истинная метка = {true_label_text},\n'
                          f' Предсказанная метка = {predicted_label_text}')

                # Сохранение графика в отдельный файл
                fig_path = os.path.join(script_dir,
                                        f"test/test_class/{self.timestamp}_{self.selected_model}_test/")
                os.makedirs(fig_path, exist_ok=True)
                save_path = os.path.join(fig_path,
                                         f"{self.timestamp}_{self.selected_model}_{batch_idx + 1}_test.png")

                plt.savefig(save_path, format='png', dpi=500)
                plt.tight_layout()
                plt.close()

                # Вычисление метрики accuracy
                accuracy = self._compute_accuracy(torch.nn.functional.softmax(outputs, dim=1), target)
                total_accuracy += accuracy

                # Вычисление метрики f1_score
                f1_score = self._compute_f1_score(torch.nn.functional.softmax(outputs, dim=1), target)
                total_f1_score += f1_score

                # Вычисление метрики precision
                precision = self._compute_precision(torch.nn.functional.softmax(outputs, dim=1), target)
                total_precision += precision

                # Вычисление метрики recall
                recall = self._compute_recall(torch.nn.functional.softmax(outputs, dim=1), target)
                total_recall += recall

                pbar.set_description(f" Accuracy: {(total_accuracy / (batch_idx + 1)): .4f}"
                                     f" F1-score: {(total_f1_score / (batch_idx + 1)):.4f}"
                                     f" Precision: {(total_precision / (batch_idx + 1)):.4f}"
                                     f" Recall: {(total_recall / (batch_idx + 1)):.4f}")

        avg_accuracy = total_accuracy / len(self.test_loader)
        avg_f1_score = total_f1_score / len(self.test_loader)
        avg_precision = total_precision / len(self.test_loader)
        avg_recall = total_recall / len(self.test_loader)
        return avg_accuracy, avg_f1_score, avg_precision, avg_recall

    def train_classification(self):

        # Загрузка модели UNet
        self._load_model()
        self._time()

        for epoch in range(1, self.NUM_EPOCH + 1):
            print(f"\nНомер эпохи {epoch}/{self.NUM_EPOCH}")

            # Загрузка данных для обучения
            self._load_data(
                self.train_dir
            )

            self._split_data()

            # Обучение на тренировочных данных
            train_loss, train_accuracy, train_f1_score, train_precision, train_recall = self._train_one_epoch()
            self.train_losses.append(train_loss)
            self.train_accuracy.append(train_accuracy)
            self.train_f1_score.append(train_f1_score)
            self.train_precision.append(train_precision)
            self.train_recall.append(train_recall)

            # Валидация на валидационных данных
            valid_accuracy, valid_f1_score, valid_precision, valid_recall = self._validate_one_epoch()
            self.valid_accuracy.append(valid_accuracy)
            self.valid_f1_score.append(valid_f1_score)
            self.valid_precision.append(valid_precision)
            self.valid_recall.append(valid_recall)

            # Загрузка данных для теста
            self._load_data(
                self.test_dir
            )

            self._split_data(is_test=True)

            # Предсказание на тестовых данных
            test_accuracy, test_f1_score, test_precision, test_recall = self._predict_on_test_set()
            self.test_accuracy.append(test_accuracy)
            self.test_f1_score.append(test_f1_score)
            self.test_precision.append(test_precision)
            self.test_recall.append(test_recall)

            print(f"\n--------------------------------------------------------------------------------")
            print(f"\n Результаты по окончанию эпохи: \n"
                  f"\nОбучение\n"
                  f" - Loss: {train_loss: .4f}\n"
                  f" - Accuracy: {train_accuracy: .4f}\n"
                  f" - F1-score: {train_f1_score: .4f}\n"
                  f" - Precision: {train_precision: .4f}\n"
                  f" - Recall: {train_recall: .4f}\n")
            print(f"Валидация\n"
                  f" - Accuracy: {valid_accuracy: .4f}\n"
                  f" - F1-score: {valid_f1_score: .4f}\n"
                  f" - Precision: {valid_precision: .4f}\n"
                  f" - Recall: {valid_recall: .4f}\n")
            print(f"Тест\n"
                  f" - Accuracy: {test_accuracy: .4f}\n"
                  f" - F1-score: {test_f1_score: .4f}\n"
                  f" - Precision: {test_precision: .4f}\n"
                  f" - Recall: {test_recall: .4f}\n")
            print(f"\n--------------------------------------------------------------------------------")

        # Сохранение обученной модели
        self._save_model(
            f"weights/weights_class/{self.timestamp}_{self.selected_model}.pth")

        # Отображение графиков
        self._plot_final_plot()


test_dir = "C:/Users/NightMare/PycharmProjects/MuseumAI/data_with_class/train_dataset_mincult-train/test_df"
# test_dir = "C:/Users/NightMare/PycharmProjects/MuseumAI/data_with_class/train_dataset_mincult-train/test_df_check"
train_dir = "C:/Users/NightMare/PycharmProjects/MuseumAI/data_with_class/train_dataset_mincult-train/train_df"
# train_dir = "C:/Users/NightMare/PycharmProjects/MuseumAI/data_with_class/train_dataset_mincult-train/train_df_check"
train = ImageClassifierTrainer(img_w=224,
                               img_h=224,
                               n_class=15,
                               img_channels=3,
                               num_epoch=5,
                               batch_size=5,
                               model='EfficientNetB3',
                               test_dir=test_dir,
                               train_dir=train_dir)
train.train_classification()

from datasets import load_dataset
from joblib import Memory
from torch.utils.data import Dataset
from transformers import AutoProcessor
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
from transformers import AutoModelForCausalLM
from transformers import pipeline
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt
from PIL import Image
import os

memory = Memory(location="./cachedir", verbose=0)


@memory.cache
def load_cached_dataset():
    return load_dataset('csv', data_files={
        'train': "C:/Users/NightMare/PycharmProjects/MuseumAI/data_with_description/train_df.csv",
        'test': "C:/Users/NightMare/PycharmProjects/MuseumAI/data_with_description/test_df.csv"})


def show_image(image):
    image.show()


class ImageCaptioningDataset(Dataset):
    def __init__(self, dataset, processor, is_test=False):
        self.processor = processor
        if is_test:
            self.dataset = dataset['test']
            self.path_to_dir = "C:/Users/NightMare/PycharmProjects/MuseumAI/data_with_description/test_df"
        else:
            self.dataset = dataset['train']
            self.path_to_dir = "C:/Users/NightMare/PycharmProjects/MuseumAI/data_with_description/train_df"

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        image = Image.open(os.path.join(self.path_to_dir, self.dataset['filename'][idx]))
        text = self.dataset['category'][idx]

        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Замените height и width на требуемые значения
        ])

        # В конструкторе класса ImageCaptioningDataset примените этот transform к изображениям:
        image = transform(image)

        encoding_image = self.processor(images=image,
                                        return_tensors="pt")
        encoding_text = self.processor(text=text, padding='max_length', max_length=5000,
                                       return_tensors="pt")

        encoding = {**encoding_image, **encoding_text}
        encoding = {k: v.squeeze() for k, v in encoding.items()}

        return encoding


@memory.cache
def processor_cached():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return tokenizer

@memory.cache
def model_cached():
    model = BertModel.from_pretrained('bert-base-uncased')
    return model

def preprocess_image(image, mean=None, std=None):
    if mean is None:
        mean = np.array([123.675, 116.280, 103.530]) / 255
    if std is None:
        std = np.array([58.395, 57.120, 57.375]) / 255

    unnormalized_image = (image.numpy() * np.array(std)[:, None, None]) + np.array(mean)[:, None, None]
    unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
    unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)
    return Image.fromarray(unnormalized_image)


def train_model(model, train_dataloader, optimizer, device, num_epochs=1):
    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        print("Epoch:", epoch)
        for idx, batch in enumerate(train_dataloader):
            # print(batch)
            input_ids = batch.pop("input_ids").to(device)
            attention_mask = batch.pop("attention_mask").to(device)
            pixel_values = batch.pop("pixel_values").to(device)

            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            pixel_values=pixel_values,
                            labels=input_ids)

            loss = outputs.loss

            print("Loss:", loss.item())

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

        save_model_weights(model, optimizer,
                           save_dir='C:/Users/NightMare/PycharmProjects/MuseumAI/weights/weights_image_captioning')


def generate_caption(model, processor, image, device, max_length=1000):
    inputs = processor(images=image, return_tensors="pt").to(device)
    pixel_values = inputs.pixel_values

    generated_ids = model.generate(pixel_values=pixel_values, max_length=max_length)
    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_caption


# Загрузка набора данных
dataset = load_cached_dataset()

# Создание процессора для текста и модели
processor = processor_cached()
model = model_cached()

# Создание датасета для обучения
train_dataset = ImageCaptioningDataset(dataset, processor, is_test=False)
test_dataset = ImageCaptioningDataset(dataset, processor, is_test=True)

# Создание DataLoader для обучения
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=1)
test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=1)

# Обучение модели
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
device = "cuda" if torch.cuda.is_available() else "cpu"
train_model(model, train_dataloader, optimizer, device)


def display_image_mosaic(images, captions):
    """
    Выводит мозаику изображений с их предсказанными описаниями.

    Аргументы:
    - images: список изображений (PIL.Image)
    - captions: список предсказанных описаний для изображений (строки)

    Обратите внимание, что количество изображений в списке images и в списке captions
    должно быть одинаковым и равным 9.
    """
    # Проверка на соответствие количества изображений и описаний
    assert len(images) == len(captions) == 9, "Количество изображений и описаний должно быть равным 9"

    # Создание сетки из 3x3 для отображения мозаики
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))

    # Проход по каждому изображению и его описанию для отображения
    for i, (image, caption) in enumerate(zip(images, captions)):
        ax = axes[i // 3, i % 3]
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(caption)

    plt.tight_layout()
    plt.show()


def predict_caption(model, processor, image, device, max_length=50):
    """
    Предсказывает текст для изображения с использованием модели.

    Аргументы:
    - model: модель для генерации текста (AutoModelForCausalLM)
    - processor: предварительно обученный процессор (AutoProcessor)
    - image: изображение для предсказания подписи (PIL.Image)
    - device: устройство для выполнения (строка: "cuda" или "cpu")
    - max_length: максимальная длина сгенерированного текста

    Возвращает:
    - generated_caption: сгенерированный текст для изображения (строка)
    """
    # Подготовка изображения для модели
    inputs = processor(images=image, return_tensors="pt").to(device)
    pixel_values = inputs.pixel_values

    # Генерация подписи для изображения
    generated_ids = model.generate(pixel_values=pixel_values, max_length=max_length)
    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return generated_caption


def save_model_weights(model, optimizer, save_dir):
    """
    Сохраняет все веса модели и оптимизатора в указанную директорию.

    Аргументы:
    - model: модель PyTorch
    - optimizer: оптимизатор PyTorch
    - save_dir: путь к директории, в которую сохранить веса модели
    """
    # Создаем директорию, если её еще нет
    os.makedirs(save_dir, exist_ok=True)

    # Сохраняем веса модели
    model_save_path = os.path.join(save_dir, "model_weights.pth")
    torch.save(model.state_dict(), model_save_path)

    # Сохраняем веса оптимизатора
    optimizer_save_path = os.path.join(save_dir, "optimizer_weights.pth")
    torch.save(optimizer.state_dict(), optimizer_save_path)


# Подготовка изображений и их предсказанных описаний
example_images = [dataset['test'][i]["image"] for i in range(9)]  # Взять первые 9 изображений из набора данных
example_captions = [predict_caption(model, processor, image, device) for image in example_images]

# Отображение мозаики изображений и их описаний
display_image_mosaic(example_images, example_captions)


def save_image_mosaic(images, captions, save_dir, filename="image_mosaic.png"):
    """
    Сохраняет мозаику изображений с их предсказанными описаниями в указанную директорию.

    Аргументы:
    - images: список изображений (PIL.Image)
    - captions: список предсказанных описаний для изображений (строки)
    - save_dir: путь к директории, в которую сохранить мозаику
    - filename: имя файла для сохранения (по умолчанию: "image_mosaic.png")

    Обратите внимание, что количество изображений в списке images и в списке captions
    должно быть одинаковым и равным 9.
    """
    # Проверка на соответствие количества изображений и описаний
    assert len(images) == len(captions) == 9, "Количество изображений и описаний должно быть равным 9"

    # Создание сетки из 3x3 для отображения мозаики
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))

    # Проход по каждому изображению и его описанию для отображения
    for i, (image, caption) in enumerate(zip(images, captions)):
        ax = axes[i // 3, i % 3]
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(caption)

    plt.tight_layout()

    # Сохранение мозаики в указанную директорию
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path)

    print(f"Мозаика сохранена в {save_path}")


# Пример использования:
save_dir = "C:/Users/NightMare/PycharmProjects/MuseumAI/test/test_image_caption_mosaic"
save_image_mosaic(example_images, example_captions, save_dir)

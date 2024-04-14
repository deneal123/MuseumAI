import os
import requests
import json
import re


class GetData:
    def __init__(self,
                 key_api: str = "934e7eea2f89b4a78b7970df3aa10cff5554c556c275061292104bb3ccca658d",
                 prefix: str = "https://opendata.mkrf.ru/v2/museum-exhibits/$?l=1",
                 path_to_img: str = "",
                 path_to_json: str = "",
                 count_iter: int = 1):
        self.key_api = key_api
        self.header = {'X-API-KEY': self.key_api}
        self.prefix = prefix

        # Используем регулярное выражение для поиска строки '?l=' и последующих цифр после знака равно
        match = re.search(r'\?l=(\d+)', self.prefix)
        limit_value = int(match.group(1))
        self.count = limit_value

        self.url_to_img = None
        self.name = None
        self.description = None
        self.regNumber = None
        self.typologyDesc = None
        self.technologies = None
        self.periodStr = None
        self.name_list = []
        self.description_list = []
        self.regNumber_list = []
        self.typologyDesc_list = []
        self.technologies_list = []
        self.periodStr_list = []
        self.jsn_list = []
        self.url_to_img_part = "http://goskatalog.ru/muzfo-imaginator/rest/images/original/"
        self.next_page = None
        self.path_to_img = path_to_img
        self.path_to_json = path_to_json
        self.count_iter = count_iter

    def __str__(self):
        return (f"-------------------------------------------------------\n"
                f"Имя предмета: {self.name_list}\n"
                f"Описание: {self.description_list}\n"
                f"regNumber: {self.regNumber_list}\n"
                f"Класс объекта: {self.typologyDesc_list}\n"
                f"Технология изготовления: {self.technologies_list}\n"
                f"Эпоха создания предмета: {self.periodStr_list}\n"
                f"Количество предметов: {len(self.name_list)}\n"
                f"-------------------------------------------------------\n")

    # Функция для скачивания изображения по ссылке
    def download_image(self, url, filename):
        try:
            # Отправляем GET-запрос по указанному URL
            response = requests.get(url)
            # Проверяем успешность запроса
            if response.status_code == 200:
                # Открываем файл для записи в бинарном режиме
                with open(filename, 'wb') as f:
                    # Записываем содержимое ответа в файл
                    f.write(response.content)
                print(f"Изображение успешно сохранено как {filename}")
            else:
                # Выводим сообщение об ошибке, если запрос не успешен
                print(f"Ошибка {response.status_code}: Не удалось загрузить изображение")
        except Exception as e:
            # Выводим сообщение об ошибке, если что-то пошло не так
            print(f"Произошла ошибка при загрузке изображения: {str(e)}")

    def _get_img(self, regNumber):
        print(f"url_to_img: {self.url_to_img}")
        self.download_image(
            url=self.url_to_img,
            filename=os.path.join(self.path_to_img, f"{regNumber}.png"))

    def _write_json(self, data, regNumber):
        # Проверяем, было ли успешно загружено изображение
        if os.path.exists(os.path.join(self.path_to_img, f"{regNumber}.png")):
            with open(os.path.join(self.path_to_json, f"{regNumber}.json"), 'w', encoding='utf-8') as json_file:
                json.dump(data, json_file, ensure_ascii=False)
                print(f"JSON-файл успешно записан: {json_file.name}")
        else:
            print("Изображение не было загружено, JSON-файл не сохранен.")

    def _get_data(self, next_page=None):
        # Отправка запроса
        if next_page is None:
            resp = requests.get(self.prefix, headers={'X-API-KEY': self.key_api})
        else:
            resp = requests.get(next_page, headers={'X-API-KEY': self.key_api})

        # Проверка статуса ответа
        if resp.status_code == 200:
            try:
                for index in range(self.count):
                    jsn = resp.json()['data'][index].get('data')

                    # self.regNumber_list.append(self.regNumber)
                    self.typologyDesc = jsn.get('typologyDesc')

                    if self.typologyDesc is None:
                        continue  # Пропускаем итерацию, если значение typologyDesc равно None

                    # self.jsn_list.append(jsn)
                    self.name = jsn.get('name')
                    # self.name_list.append(self.name)
                    self.description = jsn.get('description')

                    """if self.description is None:
                        continue  # Пропускаем итерацию, если значение typologyDesc равно None"""

                    # self.description_list.append(self.description)
                    self.regNumber = jsn.get('regNumber')

                    # self.typologyDesc_list.append(self.typologyDesc)
                    self.technologies = jsn.get('technologies')
                    # self.technologies_list.append(self.technologies)
                    self.periodStr = jsn.get('periodStr')
                    # self.periodStr_list.append(self.periodStr)

                    self.url_to_img = self.url_to_img_part + f"{self.regNumber}"
                    self._get_img(self.regNumber)

                    data = {
                        "regNumber": self.regNumber,
                        "name": self.name,
                        "description": self.description,
                        "typologyDesc": self.typologyDesc,
                        "technologies": self.technologies,
                        "periodStr": self.periodStr
                    }

                    self._write_json(data, self.regNumber)

                return resp.json()['nextPage']

            except ValueError:
                print("Ошибка: Ответ не в формате JSON.")
        else:
            print("Ошибка: Не удалось получить данные из API.")

    def _next(self):
        for index in range(self.count_iter):
            if index == 0:
                self.next_page = self._get_data()
            else:
                self.next_page = self._get_data(self.next_page)

            print(f"last_page: {self.next_page}")


"""https://opendata.mkrf.ru/v2/museum-exhibits/$?l=9"""
Data = GetData(prefix="https://opendata.mkrf.ru/v2/museum-exhibits/$?l=9&cursor=AoE%2FAjVjM2RmYjI2MGIyY2IwNTc1YjcyNTcxMyExMDAyMjIyOQ%3D%3D",
               path_to_img="C:/Users/NightMare/PycharmProjects/MuseumAI/data_with_class/images/",
               path_to_json="C:/Users/NightMare/PycharmProjects/MuseumAI/data_with_class/jsons/",
               count_iter=10000)
Data._next()

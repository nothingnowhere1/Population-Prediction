import pandas as pd
import requests

# API ключ и адрес электронной почты
api_key = ""
email = ""

# Функция для загрузки данных о военных событиях с использованием ACLED API
def fetch_acled_data(api_key, email, page_num, limit):
    api_url = f"https://api.acleddata.com/acled/read?key={api_key}&email={email}&page={page_num}&limit={limit}"

    response = requests.get(api_url)

    if response.status_code == 200:
        data = response.json()
        if data['success']:
            df = pd.DataFrame(data['data'], columns=['event_id_cnty', 'country', 'year', 'event_type', 'fatalities'])  # Выбираем только нужные столбцы
            return df
        else:
            print("Ошибка при получении данных:", data['error']['message'])
            return None
    else:
        print("Ошибка при загрузке данных:", response.status_code)
        return None

# Параметры запроса
page_num = 1 # Начинаем с первой страницы
limit = 10000  # Увеличиваем лимит строк на страницу

# Загружаем данные путем последовательной загрузки каждой страницы
all_data = pd.DataFrame()  # Создаем пустой DataFrame для хранения всех данных

while True:
    page_data = fetch_acled_data(api_key, email, page_num, limit)
    if page_data is None or len(page_data) == 0:
        break  # Прекращаем загрузку, если данные закончились или произошла ошибка
    all_data = pd.concat([all_data, page_data], ignore_index=True)  # Объединяем данные текущей страницы с общими данными
    page_num += 1  # Переходим к следующей странице

# Отфильтровать строки, в которых fatalities равно нулю
all_data_filtered = all_data[all_data['fatalities'] != '0']

# Выбор необходимых столбцов из исходного датасета
subset = all_data_filtered[['country', 'year', 'event_type']]

# Сохранение данных в CSV файл
subset.to_csv('events.csv', index=False)

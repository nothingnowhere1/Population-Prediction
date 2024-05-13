import streamlit as st
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import wbdata
import itertools
import matplotlib.pyplot as plt

# Получение списка всех стран с их кодами
all_countries = wbdata.get_countries()

# Создание словаря из кода и названия стран
country_dict = {country['name']: country['id'] for country in all_countries}
df_country_dict = pd.DataFrame(country_dict.items(), columns=['Country', 'Country Code'])
df_country_dict.to_csv('country_dict.csv', index=False)

# Создание интерфейса
st.title('Прогнозирование численности населения')

# Выбор страны
country_name = st.selectbox('Выберите страну:', list(country_dict.keys()))
country_code = country_dict.get(country_name)

# Выбор года
year = st.number_input('Введите год прогноза:', min_value=2000, max_value=2050)
year2 = st.number_input('Введите начальный год для построения графика:', min_value=1980, max_value=2000)

# Определение параметров запроса данных
indicators = {"SP.POP.TOTL": "population"}  # Индикатор для численности населения
start_date = '1980-01-01'
end_date = '2022-01-01'

# Запрос данных из базы данных Всемирного банка
population_data = wbdata.get_dataframe(indicators, country=country_code, date=(start_date, end_date))

# Сортировка данных по возрастанию года
population_data.sort_index(inplace=True)

population_data.loc['2023'] = population_data.loc['2022'] + (population_data.loc['2022'] - population_data.loc['2021'])
population_data.loc['2024'] = population_data.loc['2022'] + 2 * (population_data.loc['2022'] - population_data.loc['2021'])

for i in range(1,year-2024):
    population_data.loc[str(2024+i)] = population_data.loc['2022'] + i * (population_data.loc['2022'] - population_data.loc['2021'])/10

# Запись данных в CSV файл
population_data.to_csv('population.csv')

# Загрузка данных о численности населения
population_data = pd.read_csv("population.csv")
population_data['date'] = pd.to_datetime(population_data['date'], format='%Y')  # Преобразование столбца с датой в формат datetime
population_data.set_index('date', inplace=True)

# Получение данных о событиях
events_data = pd.read_csv("../Population-Prediction/events.csv")
events_data['year'] = pd.to_datetime(events_data['year'], format='%Y')  # Преобразование столбца с годом в формат datetime

# Фильтрация данных о событиях по выбранной стране
events_data_country = events_data[events_data['country'] == country_name]

# Группировка данных о событиях по годам
events_grouped = events_data_country.groupby(events_data['year']).size().reset_index(name='events_count')
events_grouped.set_index('year', inplace=True)

# Объединение данных о численности населения и событиях
merged_data = pd.concat([population_data, events_grouped], axis=1, join='outer')

# Заполнение пропущенных значений нулями
merged_data.fillna(0, inplace=True)

# Подготовка данных для модели ARIMA
endog = merged_data['population']
exog = merged_data[['events_count']]

# Подбор параметров модели ARIMA с использованием Grid Search
p_range = range(0, 3)  # от 0 до 2
d_range = range(0, 2)  # от 0 до 1
q_range = range(0, 3)  # от 0 до 2
parameter_combinations = list(itertools.product(p_range, d_range, q_range))

best_aic = float("inf")
best_parameters = None

for params in parameter_combinations:
    try:
        # Строим модель ARIMA для текущих параметров
        model = ARIMA(endog, exog, order=params)
        model_fit = model.fit()

        # Вычисляем значение AIC для текущей модели
        aic = model_fit.aic

        # Сравниваем с текущим лучшим значением AIC
        if aic < best_aic:
            best_aic = aic
            best_parameters = params

    except:
        continue

# Разделение данных на обучающий и тестовый наборы
# Находим индекс 2024 года
index_2024 = pd.to_datetime('2024-01-01')

# Разделяем данные на обучающий и тестовый наборы
train_endog = endog.loc[:index_2024]
train_exog = exog.loc[:index_2024]
forecast_endog = endog.loc[index_2024:]
forecast_exog = exog.loc[index_2024:]

# Обучение модели на обучающем наборе
arima_model = ARIMA(train_endog, train_exog, order=best_parameters)
arima_model_fit = arima_model.fit()

if year > 2024:
    # Прогнозирование на модели
    forecast = model_fit.forecast(steps=len(forecast_endog), exog=forecast_exog)

    # Вывод результата прогноза
    st.write('Прогнозируемая численность населения %s в %d году составляет %d человек.' % (country_name, year, round(forecast[0])))

    # Визуализация прогноза и фактических значений
    fig, ax = plt.subplots()
    ax.plot(train_endog.index, train_endog, label='Training set')
    ax.plot(forecast_endog.index, forecast, label='Forecast')
    ax.set_title('Population forecast for {}'.format(country_name))
    ax.set_xlabel('Date')
    ax.set_ylabel('Population')
    ax.legend()

    # Отображение графика в Streamlit
    st.pyplot(fig)
else:
    # Вывод результата
    index = pd.to_datetime(str(year)+'-01-01')
    st.write('Численность населения %s в %d году составляла %d человек.' % (country_name, year, round(train_endog[index])))

    # Определение начального индекса для графика обучающего набора
    year = 2024 - year

    # Визуализация прогноза и фактических значений
    fig, ax = plt.subplots()
    ax.plot(train_endog.index[:-year], train_endog[:-year], label='Population set')
    ax.set_title('Population forecast for {}'.format(country_name))
    ax.set_xlabel('Date')
    ax.set_ylabel('Population')
    ax.legend()

    # Отображение графика в Streamlit
    st.pyplot(fig)

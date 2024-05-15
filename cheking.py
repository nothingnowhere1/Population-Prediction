import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import itertools
import wbdata
import numpy as np
import statsmodels.api as sm

# Получение списка всех стран с их кодами
all_countries = wbdata.get_countries()

# Создание словаря из кода и названия стран
country_dict = {country['name']: country['id'] for country in all_countries}
df_country_dict = pd.DataFrame(country_dict.items(), columns=['Country', 'Country Code'])
df_country_dict.to_csv('country_dict.csv', index=False)

# Выбор кода страны из словаря
country_name = 'United States'
country_code = country_dict.get(country_name)

# Определение параметров запроса данных
indicators = {"SP.POP.TOTL": "population"}  # Индикатор для численности населения
start_date = '1980-01-01'
end_date = '2022-01-01'

# Запрос данных из базы данных Всемирного банка
population_data = wbdata.get_dataframe(indicators, country=country_code, date=(start_date, end_date))

# Сортировка данных по возрастанию года
population_data.sort_index(inplace=True)

population_data.loc['2023'] = population_data.loc['2022']
population_data.loc['2024'] = population_data.loc['2022']

# Запись данных в CSV файл
population_data.to_csv('population.csv')

# Загрузка данных о численности населения
population_data = pd.read_csv("population.csv")
population_data['date'] = pd.to_datetime(population_data['date'], format='%Y')  # Преобразование столбца с датой в формат datetime
population_data.set_index('date', inplace=True)  # Установка индекса в качестве столбца с датой

# Загрузка данных о событиях
events_data = pd.read_csv("events.csv")
events_data['year'] = pd.to_datetime(events_data['year'], format='%Y')  # Преобразование столбца с годом в формат datetime

# Фильтрация данных о событиях по стране
events_data_country = events_data[events_data['country'] == country_name]

# Группировка данных о событиях по годам
events_grouped = events_data_country.groupby(events_data['year']).size().reset_index(name='events_count')
events_grouped.set_index('year', inplace=True)

# Объединение данных о численности населения и событиях
merged_data = pd.concat([population_data, events_grouped], axis=1, join='outer')

# Заменяем пропущенные значения нулями
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
train_size = int(len(endog) * 0.8)  # 80% данных для обучения
test_size = int(len(endog) * 0.8)
train_endog, test_endog = endog[:train_size-2], endog[test_size:-2]
train_exog, test_exog = exog[:train_size-2], exog[test_size:-2]

# Обучение модели на обучающем наборе
arima_model = ARIMA(train_endog, train_exog, order=best_parameters)
arima_model_fit = arima_model.fit()

# Визуализация прогноза модели на тестовом наборе
def plot_arima_forecast(model_fit, endog_train, exog_train, endog_test, exog_test):
    # Прогноз на будущие периоды
    forecast = model_fit.forecast(steps=len(endog_test), exog=exog_test)

    # Визуализация прогноза и фактических значений
    plt.plot(endog_train.index, endog_train, label='Training set')
    plt.plot(endog_test.index, endog_test, label='Test set')
    plt.plot(endog_test.index, forecast, label='Forecast')
    plt.title('Population forecast for {}'.format(country_name))
    plt.xlabel('Date')
    plt.ylabel('Population')
    plt.legend()
    plt.show()

# Вызов функции для визуализации прогноза на тестовом наборе
plot_arima_forecast(arima_model_fit, train_endog, train_exog, test_endog, test_exog)

# Оценка модели и диагностика
def evaluate_model(model_fit, endog_train, exog_train, endog_test, exog_test):
    # Анализ остатков модели
    residuals = model_fit.resid
    print(model_fit.summary())

    # Графики остатков
    fig, ax = plt.subplots(2, 2, figsize=(14, 8))

    # График остатков
    ax[0, 0].plot(residuals[1:])
    ax[0, 0].set_title('Residuals Plot')
    ax[0, 0].set_xlabel('Index')
    ax[0, 0].set_ylabel('Residuals')

    # График распределения остатков
    sm.graphics.qqplot(residuals[1:], line='45', fit=True, ax=ax[0, 1])
    ax[0, 1].set_title('QQ Plot')

    # График автокорреляции остатков
    sm.graphics.tsa.plot_acf(residuals[1:], lags=11, ax=ax[1, 0])
    ax[1, 0].set_title('Autocorrelation Function (ACF)')

    # График частичной автокорреляции остатков
    sm.graphics.tsa.plot_pacf(residuals[1:], lags=11, ax=ax[1, 1])
    ax[1, 1].set_title('Partial Autocorrelation Function (PACF)')

    plt.tight_layout()
    plt.show()

# Вызов функции для оценки модели и диагностики
evaluate_model(arima_model_fit, train_endog, train_exog, test_endog, test_exog)

# Прогноз на тестовом наборе
forecast = arima_model_fit.forecast(steps=len(test_endog), exog=test_exog)

# Mean Absolute Percentage Error (MAPE)
def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape = calculate_mape(test_endog, forecast)
print("MAPE:", mape)

# Root Mean Squared Error (RMSE)
from sklearn.metrics import mean_squared_error
def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

rmse = calculate_rmse(test_endog, forecast)
print("RMSE:", rmse)

# Mean Absolute Error (MAE)
from sklearn.metrics import mean_absolute_error
def calculate_mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

mae = calculate_mae(test_endog, forecast)
print("MAE:", mae)

# Mean Absolute Scaled Error (MASE)
def calculate_mase(y_true, y_pred, y_naive):
    numerator = np.mean(np.abs(y_true - y_pred))
    denominator = np.mean(np.abs(y_true - y_naive))
    return numerator / denominator

# Прогноз наивной модели (среднее значение)
y_naive = np.mean(train_endog)

mase = calculate_mase(test_endog, forecast, y_naive)
print("MASE:", mase)

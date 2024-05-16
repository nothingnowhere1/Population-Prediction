# Прогнозирование численности населения стран

Проект, предназначенный для прогнозирования численности населения с учетом исторических событий на основе метода ARIMA с использованием экзогенных переменных.

# Список используемых библиотек
```
import pandas as pd
import requests
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import itertools
import wbdata
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA
import wbdata
from clearml import Task, Logger
```

# Структура проекта

/population.csv - файл с данными о численности населения
/events.csv - файл с данными о событиях
/country_dict.csv - файл с данными о страннах и их аббревиатурами

/evets.py - скрипт для выгрузки событий
/cheking.py - скрипт для прогнозирования и проверки
/interface.py - скрипт итогового проекта

# Участники проекта
Бисеров василий, Кашафутдинова Карина, Ситдиков Рушан, Хайбуллина Светлана, Хисаметдинов Данир, Яруллина Эльвина


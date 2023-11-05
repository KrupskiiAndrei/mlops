from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import boxcox
from catboost import CatBoostRegressor
import psycopg2

app = Flask(__name__)
app.secret_key = 'some_secret_key'

# Глобальная переменная для хранения последнего введенного значения PPP
last_ppp_value = ""

def inv_boxcox(y, lambda_):
    if lambda_ == 0:
        return np.exp(y)
    else:
        return np.exp(np.log(lambda_ * y + 1) / lambda_)

@app.route('/', methods=['GET', 'POST'])
def index():
    global last_ppp_value
    if request.method == 'POST':
        ppp_value = request.form.get('ppp_value')
        method = request.form.get('method')
        
        if not ppp_value:
            flash('Введите число')
            return redirect(url_for('index'))
        
        try:
            ppp_value = float(ppp_value.replace(',', '.'))
            last_ppp_value = ppp_value
            
            if method == "sarima":
                return forecast_sarima(ppp_value)
            elif method == "catboost":
                return forecast_catboost(ppp_value)
            else:
                flash('Выбран неверный метод прогнозирования')
                return redirect(url_for('index'))
        except ValueError:
            flash('Введите корректное число')
            return redirect(url_for('index'))
    return render_template('index.html', last_ppp_value=last_ppp_value)

def forecast_sarima(ppp_value):
    # Подключение к базе данных
    conn = psycopg2.connect(host="192.168.1.121", database="infordb", user="infor", password="123")
    data = pd.read_sql("SELECT * FROM knal;", conn)  # Используйте нижний регистр для имени таблицы
    conn.close()

    # Прогнозирование с использованием SARIMA
    endog_transformed, lambda_ = boxcox(data['value'])  # Используйте нижний регистр для названия столбца
    data['ppp_lag1'] = data['ppp'].shift(1)            # Используйте нижний регистр для названия столбцов
    data['ppp_lag12'] = data['ppp'].shift(12)          # Используйте нижний регистр для названия столбцов
    data = data.dropna()
    exog = data[['ppp', 'ppp_lag1', 'ppp_lag12']]      # Используйте нижний регистр для названия столбцов
    endog_transformed = endog_transformed[data.index]

    model = sm.tsa.statespace.SARIMAX(endog_transformed, exog=exog, order=(1,1,1), seasonal_order=(1,1,0,12))
    results = model.fit(disp=-1, method='nm', maxiter=100)

    known_exog = np.array([[ppp_value, data['ppp'].iloc[-1], data['ppp'].iloc[-12]]])
    forecast = results.get_forecast(steps=1, exog=known_exog)
    mean_forecast = forecast.predicted_mean
    mean_forecast = round(inv_boxcox(mean_forecast, lambda_).iloc[0], 5)

    return render_template('forecast.html', predicted_value=mean_forecast)

def forecast_catboost(ppp_value):
    # Подключение к базе данных
    conn = psycopg2.connect(host="192.168.1.121", database="infordb", user="infor", password="123")
    data = pd.read_sql("SELECT * FROM knal;", conn)  # Используйте нижний регистр для имени таблицы
    conn.close()

    # Прогнозирование с использованием CatBoost
    X = data[['ppp']]  # Используйте нижний регистр для названия столбцов
    y = data['value']  # Используйте нижний регистр для названия столбца

    model = CatBoostRegressor(iterations=1000, learning_rate=0.1, depth=5, verbose=0)
    model.fit(X, y)

    predicted_value = model.predict([[ppp_value]])[0]
    predicted_value = round(predicted_value, 5)

    return render_template('forecast.html', predicted_value=predicted_value)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

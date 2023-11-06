from flask import Flask, render_template, request, redirect, url_for, flash
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import boxcox
from catboost import CatBoostRegressor

app = Flask(__name__)
app.secret_key = 'some_secret_key'

# Глобальные переменные для хранения последних введенных значений
last_ppp_value = ""
last_dpm_value = ""
last_aes_value = ""

# Создание движка SQLAlchemy для подключения к базе данных
engine = create_engine('postgresql://infor:123@192.168.1.121:5432/infordb')

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
            
            flash('Выбран неверный метод прогнозирования')
            return redirect(url_for('index'))
        except ValueError:
            flash('Введите корректное число')
            return redirect(url_for('index'))
    return render_template('index.html', last_ppp_value=last_ppp_value)

@app.route('/index_2', methods=['GET', 'POST'])
def index_2():
    global last_dpm_value, last_aes_value
    if request.method == 'POST':
        dpm_value = request.form.get('dpm_value')
        aes_value = request.form.get('aes_value')
        method = request.form.get('method')
        
        if not dpm_value or not aes_value:
            flash('Введите все числа')
            return redirect(url_for('index_2'))
        
        try:
            dpm_value = float(dpm_value.replace(',', '.'))
            aes_value = float(aes_value.replace(',', '.'))
            last_dpm_value = dpm_value
            last_aes_value = aes_value
            
            if method == "sarima":
                return forecast_sarima_with_additional_vars(dpm_value, aes_value)
            elif method == "catboost":
                return forecast_catboost_with_additional_vars(dpm_value, aes_value)
            
            flash('Выбран неверный метод прогнозирования')
            return redirect(url_for('index_2'))
        except ValueError:
            flash('Введите корректные числа')
            return redirect(url_for('index_2'))
    return render_template('index_2.html', last_dpm_value=last_dpm_value, last_aes_value=last_aes_value)

@app.route('/advanced', methods=['POST'])
def advanced():
    global last_dpm_value, last_aes_value
    dpm_value = request.form.get('dpm_value')
    aes_value = request.form.get('aes_value')
    method = request.form.get('method')
    
    if not dpm_value or not aes_value:
        flash('Введите все числа')
        return redirect(url_for('index_2'))
    
    try:
        dpm_value = float(dpm_value.replace(',', '.'))
        aes_value = float(aes_value.replace(',', '.'))
        last_dpm_value = dpm_value
        last_aes_value = aes_value
        
        if method == "sarima":
            return forecast_sarima_with_additional_vars(dpm_value, aes_value)
        elif method == "catboost":
            return forecast_catboost_with_additional_vars(dpm_value, aes_value)
        
        flash('Выбран неверный метод прогнозирования')
        return redirect(url_for('index_2'))
    except ValueError:
        flash('Введите корректные числа')
        return redirect(url_for('index_2'))

def forecast_sarima(ppp_value):
    try:
        data = pd.read_sql("SELECT * FROM knal;", engine)
        flash(f"Data loaded successfully with {len(data)} rows", "info")

        endog_transformed, lambda_ = boxcox(data['value'])
        if lambda_ == 0:
            flash("Lambda for boxcox transformation is zero, transformation would be log", "warning")

        data['ppp_lag1'] = data['ppp'].shift(1)
        data['ppp_lag12'] = data['ppp'].shift(12)
        data = data.dropna()
        if data.empty:
            flash("No data after dropna operation", "error")
            return redirect(url_for('index'))

        exog = data[['ppp', 'ppp_lag1', 'ppp_lag12']]
        endog_transformed = endog_transformed[data.index]

        model = sm.tsa.statespace.SARIMAX(endog_transformed, exog=exog, order=(1,1,1), seasonal_order=(1,1,0,12))
        results = model.fit(disp=-1, method='nm', maxiter=100)
        flash("Model fitted successfully", "info")

        known_exog = np.array([[ppp_value, data['ppp'].iloc[-1], data['ppp'].iloc[-12]]])
        forecast = results.get_forecast(steps=1, exog=known_exog)
        mean_forecast = forecast.predicted_mean
        mean_forecast = round(inv_boxcox(mean_forecast, lambda_).iloc[0], 5)

        return render_template('forecast.html', predicted_value=mean_forecast)
    except Exception as e:
        flash(f"An error occurred: {e}", "error")
        return redirect(url_for('index'))

def forecast_catboost(ppp_value):
    try:
        data = pd.read_sql("SELECT * FROM knal;", engine)
        flash(f"Data loaded successfully with {len(data)} rows", "info")

        X = data[['ppp']]
        y = data['value']

        model = CatBoostRegressor(iterations=1000, learning_rate=0.1, depth=5, verbose=0)
        model.fit(X, y)
        flash("Model fitted successfully", "info")

        predicted_value = model.predict([[ppp_value]])[0]
        predicted_value = round(predicted_value, 5)

        return render_template('forecast.html', predicted_value=predicted_value)
    except Exception as e:
        flash(f"An error occurred: {e}", "error")
        return redirect(url_for('index'))

def forecast_sarima_with_additional_vars(dpm_value, aes_value):
    try:
        data = pd.read_sql("SELECT * FROM knal;", engine)
        flash(f"Data loaded successfully with {len(data)} rows", "info")

        endog_transformed, lambda_ = boxcox(data['value'])
        if lambda_ == 0:
            flash("Lambda for boxcox transformation is zero, transformation would be log", "warning")

        data['ppp_lag1'] = data['ppp'].shift(1)
        data['ppp_lag12'] = data['ppp'].shift(12)
        data['dpm_lag1'] = data['dpm'].shift(1)
        data['aes_lag1'] = data['aes'].shift(1)
        data = data.dropna()
        if data.empty:
            flash("No data after dropna operation", "error")
            return redirect(url_for('index_2'))

        exog = data[['ppp', 'ppp_lag1', 'ppp_lag12', 'dpm_lag1', 'aes_lag1']]
        endog_transformed = endog_transformed[data.index]

        model = sm.tsa.statespace.SARIMAX(endog_transformed, exog=exog, order=(1,1,1), seasonal_order=(1,1,0,12))
        results = model.fit(disp=-1, method='nm', maxiter=100)
        flash("Model fitted successfully", "info")

        known_exog = np.array([[last_ppp_value, data['ppp'].iloc[-1], data['ppp'].iloc[-12], dpm_value, aes_value]])
        forecast = results.get_forecast(steps=1, exog=known_exog)
        mean_forecast = forecast.predicted_mean
        mean_forecast = round(inv_boxcox(mean_forecast, lambda_).iloc[0], 5)

        return render_template('forecast.html', predicted_value=mean_forecast)
    except Exception as e:
        flash(f"An error occurred: {e}", "error")
        return redirect(url_for('index_2'))

def forecast_catboost_with_additional_vars(dpm_value, aes_value):
    try:
        data = pd.read_sql("SELECT * FROM knal;", engine)
        flash(f"Data loaded successfully with {len(data)} rows", "info")

        X = data[['ppp', 'dpm', 'aes']]
        y = data['value']

        model = CatBoostRegressor(iterations=1000, learning_rate=0.1, depth=5, verbose=0)
        model.fit(X, y)
        flash("Model fitted successfully", "info")

        predicted_value = model.predict([[last_ppp_value, dpm_value, aes_value]])[0]
        predicted_value = round(predicted_value, 5)

        return render_template('forecast.html', predicted_value=predicted_value)
    except Exception as e:
        flash(f"An error occurred: {e}", "error")
        return redirect(url_for('index_2'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

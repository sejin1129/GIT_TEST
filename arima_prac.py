import pandas as pd
pd.options.plotting.backend = "plotly"

import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
import numpy as np

from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import pmdarima as pm
from pmdarima.arima import auto_arima
from pmdarima.arima import ndiffs

import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import itertools
from tqdm import tqdm

def Setting():
    # 전역으로 그래프 사이즈 고정
    plt.rcParams["figure.figsize"] = (12,5)

    # 유니코드 깨짐현상 해결
    plt.rcParams['axes.unicode_minus'] = False
    
    # 나눔고딕 폰트 적용
    plt.rcParams["font.family"] = 'NanumGothic'
Setting()

cycle_df = pd.read_csv('C:/Users/user/Desktop/학기별 문서/현장실습/데이터자료/debt_cycle.csv')
# start = "1790-01-01" # 최소 1790-01-01
# end = "2023-01-01" # 최대 2023-01-01
# cycle_df = cycle_df[cycle_df['Date'].between(start,end)]
cycle_df.loc[:,'Date'] = pd.to_datetime(cycle_df.Date)
cycle_df = cycle_df.set_index('Date')
cycle_df = cycle_df.drop(['Debt', 'y'], axis = 1)
# cycle_df = cycle_df.drop(['records_count'], axis = 1)

def DrawCycle():
    plt.title('original cycle', fontsize=20)
    plt.plot(cycle_df.index, cycle_df.Cycle)
    plt.ylabel('Cycle(%)', fontsize=12)
    plt.xlabel('Date', fontsize=12)
    plt.legend(['Cycle'], fontsize=12, loc='best')
    plt.grid()
    plt.show()
# DrawCycle()

### 추세, 계절성, 잔차 등을 파악
# result = seasonal_decompose(cycle_df.Cycle, model='additive', two_sided=True, # 주기성 확인
#                             period=75, extrapolate_trend='freq') 
# result.plot()
# plt.show()

### ACF / PACF 함수 그래프
# plot_acf(cycle_df.diff().dropna(), lags = 40)
# plot_pacf(cycle_df.diff().dropna(), lags = 40)
# plt.show()

### 최적의 차분 회수 구해주는 함수
# kpss_diffs = pm.arima.ndiffs(cycle_df, alpha=0.05, test='kpss', max_d=5)
# adf_diffs = pm.arima.ndiffs(cycle_df, alpha=0.05, test='adf', max_d=5)
# n_diffs = max(kpss_diffs, adf_diffs)
# print(f"추정된 차수 d = {n_diffs}") # 결과

### split data
forecast_year = 75
df_train = cycle_df.iloc[:-forecast_year] # 마지막 forecast_year 년만큼 예측
df_test = cycle_df.iloc[len(df_train):]


min_rmse_arima = float("inf")
min_rmse_sarimax = float("inf")
### input data의 길이를 forecast_year 만큼 5번 감소(2003~2023 / 1983~2023 / 1963~2023...)
for i in range(1):
    # df_train = cycle_df.iloc[:-forecast_year] # 마지막 forecast_year 년만큼 예측
    # df_test = cycle_df.iloc[len(df_train):]
    ### ARIMA
    # p = range(0,3)
    # d = range(1,2)
    # q = range(0,6)

    # pdq = list(itertools.product(p,d,q))

    # aic = []
    # params = []

    # with tqdm(total = len(pdq)) as pg:
    #     for i in pdq:
    #         pg.update(1)
    #         try:
    #             model = SARIMAX(df_train["Cycle"], order=(i))
    #             model_fit = model.fit()
    #             aic.append(round(model_fit.aic,2))
    #             params.append((i))
    #         except:
    #             continue

    # optimal = [(params[i],j) for i,j in enumerate(aic) if j == min(aic)]
    # model_opt = ARIMA(df_train["Cycle"], order = optimal[0][0])
    # model_opt_fit = model_opt.fit()
    # # print(model_opt_fit.summary())


    # model = ARIMA(df_train["Cycle"], order=(1,1,2))
    # model_fit = model.fit()
    # print(model_fit.summary())
    # forecast_a = model_fit.forecast(steps=forecast_year, typ = 'levels')

    # plt.figure(figsize=(20,5)) 
    # plt.title("ARIMA PREDICT", fontsize = 20)
    # plt.plot(df_train, label = "train")
    # plt.plot(df_test, label = "real")
    # plt.plot(forecast_a, label = "predict")
    # plt.legend()
    # plt.grid()
    # plt.show()
    # rmse_arima = np.sqrt(mean_squared_error(df_test, forecast_a))
    # mape_arima = np.mean(np.abs((df_test.Cycle - forecast_a) / df_test.Cycle)) * 100

    ## SARIMAX
    # p = range(0,3)
    # d = range(1,2)
    # q = range(0,6)
    # m = 75
    # pdq = list(itertools.product(p,d,q))
    # seasonal_pdq = [(x[0],x[1], x[2], m) for x in list(itertools.product(p,d,q))]

    # aic = []
    # params = []

    # with tqdm(total = len(pdq) * len(seasonal_pdq)) as pg:
    #     for i in pdq:
    #         for j in seasonal_pdq:
    #             pg.update(1)
    #             try:
    #                 model = SARIMAX(df_train["Cycle"], order=(i), season_order = (j))
    #                 model_fit = model.fit()
    #                 aic.append(round(model_fit.aic,2))
    #                 params.append((i,j))
    #             except:
    #                 continue
    # optimal = [(params[i],j) for i,j in enumerate(aic) if j == min(aic)]
    # model_opt = SARIMAX(df_train["Cycle"], order = (optimal[0][0][0]), seasonal_order = optimal[0][0][1])
    # model_opt_fit = model_opt.fit()
    # print(model_opt_fit.summary())

    model = SARIMAX(df_train, order=(2,1,1), seasonal_order=(1,1,1,75))
    model_fit = model.fit(disp=0)
    print(model_fit.summary())
    forecast_s = model_fit.forecast(steps=forecast_year + 75)
    print(forecast_s)

    plt.figure(figsize=(20,5))
    plt.title("SARIMAX PREDICT", fontsize = 20)
    plt.plot(df_train)
    plt.plot(df_test, label="real", color = 'r')
    plt.plot(forecast_s, label="forecast", color = 'b')
    plt.xlabel("Date")
    plt.ylabel("%")
    plt.title("SARIMAX FORECASTED")
    plt.legend()
    plt.grid()
    plt.show()
    # rmse_sarimax = np.sqrt(mean_squared_error(df_test, forecast_s))
    # mape_sarimax = np.mean(np.abs((df_test.Cycle - forecast_s) / df_test.Cycle)) * 100

#     print(f"추정된 ARIMA RMSE = {forecast_year}년 예측, {rmse_arima}")
#     print(f"추정된 SARIMAX RMSE = {forecast_year}년 예측, {rmse_sarimax}") 
#     print(f"추정된 ARIMA MAPE = {forecast_year}년 예측, {mape_arima}")
#     print(f"추정된 SARIMAX MAPE = {forecast_year}년 예측, {mape_sarimax}") 
    
#     if min_rmse_arima > rmse_arima:
#         min_rmse_arima = rmse_arima
#         forecast_arima = forecast_a
#         df_train_arima = df_train
#         df_test_arima = df_test
#         forecast_year_arima = forecast_year
        
#     if min_rmse_sarimax > rmse_sarimax:
#         min_rmse_sarimax = rmse_sarimax
#         forecast_sarimax = forecast_s
#         df_train_sarimax = df_train
#         df_test_sarimax = df_test
#         forecast_year_sarimax = forecast_year
        
#     forecast_year = forecast_year + 20


# plt.figure(figsize=(20,5))
# plt.title("ARIMA PREDICT", fontsize = 20)
# plt.plot(df_train_arima, label="train")
# plt.plot(df_test_arima, label="real")
# plt.plot(forecast_arima, label="predict")
# plt.legend()
# plt.grid()
# plt.show()


# plt.figure(figsize=(20,5))
# plt.title("SARIMAX PREDICT", fontsize = 20)
# plt.plot(df_train_sarimax, label="train")
# plt.plot(df_test_sarimax, label="real")
# plt.plot(forecast_sarimax, label="predict")
# plt.legend()
# plt.grid()
# plt.show()
# print(f"ARIMA 최적의 예측기간 및 RMSE = {forecast_year_arima}, {min_rmse_arima}")
# print(f"SARIMAX 최적의 예측기간 및 RMSE = {forecast_year_sarimax}, {min_rmse_sarimax}") 
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller
import copy
import seaborn as sns

# scaling
sc = MinMaxScaler(feature_range=(0, 1))

# Testing for stationary and non-stationary
def stationary_test(data):
    i = 'CALIFORNIA'
    dftest = adfuller(data[i], autolag='AIC')
    print("1. ADF : ", dftest[0])
    print("2. P-Value : ", dftest[1])
    print("3. Num Of Lags : ", dftest[2])
    print("4. Num Of Observations Used For ADF Regression and Critical Values Calculation :", dftest[3])
    print("5. Critical Values :")
    for key, val in dftest[4].items():
        print("\t", key, ": ", val)


# Removing seasonality and trend
def removing_seasonality_trend(dataset):
    # 1- seasonality
    order = 2
    coef = np.polyfit(np.arange(len(dataset['CALIFORNIA'])),
                      dataset['CALIFORNIA'].values.ravel(),
                      order)
    poly_mdl = np.poly1d(coef)
    poly_mdl
    trend = pd.Series(data=poly_mdl(np.arange(len(dataset['CALIFORNIA']))),
                      index=dataset.index)
    detrended = dataset['CALIFORNIA'] - trend
    seasonal = detrended.groupby(by=detrended.index.month).mean()
    col = 'CALIFORNIA'
    seasonal_component = copy.deepcopy(dataset)
    for i in seasonal.index:
        seasonal_component.loc[seasonal_component.index.month == i, col] = seasonal.loc[i]
    deseasonal = dataset - seasonal_component

    # 2- Removing trend
    coef = np.polyfit(np.arange(len(deseasonal)), deseasonal['CALIFORNIA'], order)
    poly_mdl = np.poly1d(coef)
    trend_comp = pd.DataFrame(data=poly_mdl(np.arange(len(dataset['CALIFORNIA']))),
                              index=dataset.index,
                              columns=['CALIFORNIA'])

    residual = dataset - seasonal_component - trend_comp
    trend_comp_test = trend_comp.iloc[-16:310, 0:1]
    seasonal_component_test = seasonal_component.iloc[-16:310, 0:1]
    print(seasonal_component_test.shape)
    return residual, trend_comp_test, seasonal_component_test


# Visualize learning
def plot_losses(training_losses, validation_losses):
    fig, axs = plt.subplots(ncols=2)
    for num, losses in enumerate[training_losses, validation_losses]:
        losses_float = [float(loss_.cpu().detach().numpy()) for loss_ in losses]
        loss_indices = [i for i, l in enumerate(losses_float)]
        axs[num] = sns.lineplot(loss_indices, losses_float)
    plt.savefig(f"Fold__losses.jpg")


def splitting(data, percentage):
    num_data = len(data)
    num_train = int(percentage * num_data)
    first_set = data[0:num_train]
    second_set = data[num_train:]
    return first_set, second_set

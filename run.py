import rnn_fw as rf
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from matplotlib import pyplot
import matplotlib.pyplot as plt

residual, trend_comp_test, seasonal_component_test = rf.models.removing_seasonality_trend(rf.utils.dataset)
training_data = residual.iloc[:, 0:1].values

# Defining training set and test set
num_data = len(training_data)
train_split = 0.95
num_train = int(train_split * num_data)
num_test = num_data - num_train
training_set = training_data[0:num_train]
test_set = training_data[num_train:]
print(training_set.shape)
print(test_set.shape)

training_set_scaled = rf.models.sc.fit_transform(training_set)

# For this problem, the model is trained based on data from last 14 days to predict the next day.
x_train = []
y_train = []
for i in range(14, 294):
    x_train.append(training_set_scaled[i-14:i, 0:1])
    y_train.append(training_set_scaled[i, 0:1])
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],x_train.shape[2]))
print(x_train.shape)
print(y_train.shape)

nfolds = 5
EPOCHS = 2000
MAPE_TOTAL = []
for fold in range(nfolds):
    regressor = Sequential()
    regressor.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units=50))
    regressor.add(Dropout(0.2))
    regressor.add(Dense(units=1))
    regressor.compile(optimizer='adam', loss='mean_squared_error')
    regressor.summary()

    history = regressor.fit(x_train, y_train, epochs=2000, batch_size=20, validation_split=0.33)

    pyplot.plot(history.history['loss'])
    pyplot.plot(history.history['val_loss'])
    pyplot.title('model train vs validation loss')
    pyplot.ylabel('loss')
    pyplot.xlabel('epoch')
    pyplot.legend(['train', 'validation'], loc='upper right')
    pyplot.show()

    # Building the test data
    dataset_total = training_data
    inputs = dataset_total[len(dataset_total) - len(test_set) - 14:]
    inputs = np.reshape(inputs, (inputs.shape[0], inputs.shape[1]))
    inputs = rf.models.sc.transform(inputs)
    x_test = []
    for i in range(14, 30):
        x_test.append(inputs[i - 14:i, 0:1])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2]))

    print(x_test.shape)

    # Predicting and reversing the data
    predicted_set = regressor.predict(x_test)
    predicted_set = rf.models.sc.inverse_transform(predicted_set)

    # Adding seasonality and trend to predicted set
    predicted_set_f = []
    trend_comp_test = np.array(trend_comp_test)
    seasonal_component_test = np.array(seasonal_component_test)
    predicted_set_f = np.array([[predicted_set[i][j] + trend_comp_test[i][j] + seasonal_component_test[i][j]
                                 for j in range(len(predicted_set[0]))] for i in range(len(predicted_set))])

    test_dataframe = pd.DataFrame(rf.utils.test_set_f)
    predict_dataframe = pd.DataFrame((predicted_set_f))

    ax = test_dataframe.plot()
    predict_dataframe.plot(ax=ax)
    plt.legend(['actual', 'predict'])
    plt.show()

    MAPE = rf.utils.mean_absolute_percentage_error(y_true=test_dataframe, y_pred=predict_dataframe)
    print(MAPE)
    MAPE_TOTAL.append(MAPE)

print(MAPE_TOTAL)



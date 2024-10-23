# Milk Price Prediction in the US Using Historical Dataset

## Overview
This project aims to predict milk prices in the United States using historical data. It utilizes time series analysis techniques and LSTM/GRU models to make accurate predictions. The data is loaded, preprocessed, and fed into neural network models to train and evaluate the predictions.

## Prerequisites
- Python 3.x
- Required libraries:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `tensorflow` (Keras)
  - `sklearn`
- Data file: `SeriesReport-20221211034635_cdcc19.xlsx` (must be in the specified directory)

## Data Loading and Preparation
1. Import the necessary libraries:
    ```python
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    plt.style.use('seaborn')
    plt.rcParams['figure.figsize'] = (20, 15)
    import seaborn as sns
    import math
    import os
    import warnings
    warnings.filterwarnings('ignore')

    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    from sklearn.metrics import mean_squared_error

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, GRU
    ```

2. Load the dataset:
    ```python
    dataframe = pd.ExcelFile('/Users/rajesh/Downloads/SeriesReport-20221211034635_cdcc19.xlsx')
    dataframe = pd.read_excel(dataframe, 'BLS Data Series')
    ```

3. Data Cleaning:
    - Remove unwanted characters from the `Period` column.
    - Convert `Year` and `Period` to strings and create a `Date` column.
    - Set the `Date` column as the index.
    ```python
    dataframe['Period'] = dataframe['Period'].map(lambda x: x.lstrip('M').rstrip('M'))
    dataframe['Period'] = dataframe['Period'].astype(str)
    dataframe['Year'] = dataframe['Year'].astype(str)
    dataframe['Date'] = dataframe[['Year', 'Period']].agg('-'.join, axis=1)
    dataframe.index = pd.to_datetime(dataframe.Date)
    ```

4. Select the target variable for prediction:
    ```python
    dataframe = dataframe.filter(items=['Value'], axis=1)
    ```

## Data Visualization
5. Plot the time series data:
    ```python
    dataframe.plot()
    ```

## Train-Test Split
6. Split the data into training and testing sets (80-20 split):
    ```python
    train = int(len(dataframe) * 0.8)
    Xtrain, Xtest = dataframe.iloc[:train, :], dataframe.iloc[train:, :]
    ```

## Prepare Data for LSTM/GRU Model
7. Define a function to create datasets:
    ```python
    def make_dataset(data, window=1):
        X, y = [], []
        for i in range(len(data) - window - 1):
            X.append(data.iloc[i:i + window, :])
            y.append(data.iloc[i + window, :])
        return np.array(X), np.array(y)
    
    Xtrain, ytrain = make_dataset(Xtrain, 5)
    Xtest, ytest = make_dataset(Xtest, 5)
    ```

## LSTM Model Training
8. Define and train an LSTM model with early stopping:
    ```python
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    model = Sequential()
    model.add(LSTM(60, return_sequences=True, input_shape=(5, 1)))
    model.add(LSTM(60))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(Xtrain, ytrain, validation_split=0.2, epochs=100, verbose=1, callbacks=[stop_early])
    ```

9. Evaluate the LSTM model:
    ```python
    prediction = model.predict(Xtest)
    mean_squared_error(ytest, prediction)
    ```

10. Visualize LSTM model predictions:
    ```python
    plt.plot(ytest, label='Original Test')
    plt.plot(prediction, label='Test Predictions')
    plt.legend()
    ```

## GRU Model Training
11. Define and train a GRU model:
    ```python
    model = Sequential()
    model.add(GRU(60, return_sequences=True, input_shape=(5, 1)))
    model.add(GRU(60))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(Xtrain, ytrain, validation_data=(Xtest, ytest), epochs=100, batch_size=1, verbose=1)
    ```

12. Visualize GRU model predictions:
    ```python
    plt.plot(ytest, label='Original Training')
    plt.plot(model.predict(Xtest), label='Train Predictions')
    plt.legend()
    ```

## Results
- Compare the Mean Squared Error (MSE) for LSTM and GRU models.
- Analyze the plots to understand the prediction accuracy.

## Conclusion
This project demonstrates how to use LSTM and GRU models for time series forecasting. By using historical data of milk prices, we can predict future trends and make data-driven decisions. Further improvements can be made by experimenting with different model architectures or using more advanced feature engineering techniques.

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv("yahoo_stock.csv")
print(df.shape)
df.head()

df = df.sort_values('Date')

# Use the 'Close' column for prediction
data = df[['Close']].values
df.head()

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Normalize the data to the range [0, 1] for better performance
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

# using the past n days to predict the next day(sliding window)
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])  # Using past 'time_step' days
        y.append(data[i + time_step, 0])  # Predicting the next day's closing price
    return np.array(X), np.array(y)

# Prepare the data for training
time_step = 60  
X, y = create_dataset(data_scaled, time_step)

# Reshape X to be compatible with CNN input: (samples, time steps, features)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = Sequential()

# Adding a 1D Convolutional layer
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))

# Add MaxPooling layer to reduce dimensionality
model.add(MaxPooling1D(pool_size=2))

# Flatten the output from the convolutional layers
model.add(Flatten())

# Fully connected layer
model.add(Dense(units=50, activation='relu'))

# Output layer with 1 neuron (for regression)
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Make predictions on the test set
predictions = model.predict(X_test)

# Invert the scaling of predictions and actual values
predictions = scaler.inverse_transform(predictions)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot the predictions vs actual values
plt.figure(figsize=(10,6))
plt.plot(y_test_actual, label='Actual Prices')
plt.plot(predictions, label='Predicted Prices')
plt.legend()
plt.show()

# redictions on the test set
predictions = model.predict(X_test)
predictions_actual = scaler.inverse_transform(predictions)  # Inverse transform predictions
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))  # Inverse transform actual values

#the first few predictions and actual values
print("Predicted values (first 5):", predictions_actual[:5])
print("Actual values (first 5):", y_test_actual[:5])

from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test_actual, predictions)
print(f'Mean Squared Error: {mse}')
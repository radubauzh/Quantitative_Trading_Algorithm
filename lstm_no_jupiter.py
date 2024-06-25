import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, BatchNormalization
from keras.optimizers import Adam, RMSprop
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping

# Load the CSV data
data = pd.read_csv('/Users/felix/Desktop/Quantitative_Trading_Algorithm/output_date.csv')

# Print the columns to understand the data structure
print("Data Columns:")
print(data.columns)

# Based on the legend, we should map these columns to lowercase
selected_columns = [
    'emp', 'pe', 'cape', 'dy', 'rho', 'mov', 'ir', 'rr',
    'y02', 'y10', 'stp', 'cf', 'mg', 'rv', 'ed', 'un',
    'gdp', 'm2', 'cpi', 'dil', 'yss', 'nyf',
    '_au', '_dxy', '_lcp', '_ty', '_oil',
    '_mkt', '_va', '_gr'
]

# Check if the selected columns are in the data
available_columns = [col for col in selected_columns if col in data.columns]
print("Selected Available Columns:")
print(available_columns)

# Extracting the relevant columns
data = data[available_columns + ['snp']]  # Including 'snp' as target

# Handle missing values
data = data.dropna()

# Extract features and target
features = data.drop(columns=['snp'])
target = data['snp']

# Normalize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

print("Feature Engineering and Data Preprocessing Complete.")

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape data for LSTM (samples, timesteps, features)
X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# Define a function to create the LSTM model
def create_model(units, dropout_rate, optimizer):
    model = Sequential()
    model.add(LSTM(units=units, activation='relu', input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))
    model.add(Dropout(dropout_rate))
    model.add(BatchNormalization())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=1))
    
    if optimizer == 'adam':
        opt = Adam()
    elif optimizer == 'rmsprop':
        opt = RMSprop()
    
    model.compile(optimizer=opt, loss='mean_squared_error')
    return model

# Hyperparameters to try
units_list = [10, 50, 100]
dropout_rate_list = [0.2, 0.5]
batch_size_list = [16, 32, 64]
epochs_list = [50, 100, 200]
optimizer_list = ['adam', 'rmsprop']

# Variables to store the best hyperparameters and lowest loss
best_params = {}
best_loss = float('inf')

# Iterate through all combinations of hyperparameters
for units in units_list:
    for dropout_rate in dropout_rate_list:
        for batch_size in batch_size_list:
            for epochs in epochs_list:
                for optimizer in optimizer_list:
                    print(f'Training model with units={units}, dropout_rate={dropout_rate}, batch_size={batch_size}, epochs={epochs}, optimizer={optimizer}')
                    
                    model = create_model(units, dropout_rate, optimizer)
                    
                    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                    
                    history = model.fit(X_train_lstm, y_train, epochs=epochs, batch_size=batch_size, 
                                        validation_split=0.2, verbose=0, callbacks=[early_stopping])
                    
                    val_loss = history.history['val_loss'][-1]
                    print(f'Validation Loss: {val_loss}')
                    
                    if val_loss < best_loss:
                        best_loss = val_loss
                        best_params = {
                            'units': units,
                            'dropout_rate': dropout_rate,
                            'batch_size': batch_size,
                            'epochs': epochs,
                            'optimizer': optimizer
                        }
                        best_model = model

# Print the best parameters and the best score
print(f"Best Parameters: {best_params}")
print(f"Best Validation Loss: {best_loss}")

# Evaluate the best model on the test set
test_loss = best_model.evaluate(X_test_lstm, y_test)
print(f"LSTM Model Test Loss: {test_loss}")

# Make predictions with the best model
lstm_predictions = best_model.predict(X_test_lstm)

print("LSTM Model Training and Evaluation with Manual Hyperparameter Tuning Complete.")

# Plot the training and validation loss
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
plt.title('LSTM Model Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
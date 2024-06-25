import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

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

# Reshape the input data to fit the Conv1D layer (samples, time_steps, features)
X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

print(f"X_train_scaled shape: {X_train_scaled.shape}")
print(f"X_test_scaled shape: {X_test_scaled.shape}")

# Define a function to create the CNN model with hyperparameters
def create_cnn_model(filters_1, filters_2, kernel_size_1, kernel_size_2, pool_size, dense_units, dropout_rate, optimizer):
    model = Sequential()
    model.add(Conv1D(filters=filters_1, kernel_size=kernel_size_1, activation='relu', input_shape=(X_train_scaled.shape[1], 1)))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(Dropout(dropout_rate))
    model.add(Conv1D(filters=filters_2, kernel_size=kernel_size_2, activation='relu'))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(Flatten())
    model.add(Dense(dense_units, activation='relu'))
    model.add(Dense(1))
    
    if optimizer == 'adam':
        opt = Adam()
    elif optimizer == 'rmsprop':
        opt = RMSprop()
    
    model.compile(optimizer=opt, loss='mean_squared_error')
    return model

# Hyperparameters to try
filters_1_list = [32, 64, 128]
filters_2_list = [32, 64, 128]
kernel_size_1_list = [2, 3, 4]
kernel_size_2_list = [2, 3, 4]
pool_size_list = [2, 3]
dense_units_list = [50, 100]
dropout_rate_list = [0.25, 0.5]
optimizer_list = ['adam', 'rmsprop']
batch_size_list = [16, 32]
epochs_list = [20, 40]

# Variables to store the best hyperparameters and lowest loss
best_params = {}
best_loss = float('inf')

# Iterate through all combinations of hyperparameters
for filters_1 in filters_1_list:
    for filters_2 in filters_2_list:
        for kernel_size_1 in kernel_size_1_list:
            for kernel_size_2 in kernel_size_2_list:
                for pool_size in pool_size_list:
                    for dense_units in dense_units_list:
                        for dropout_rate in dropout_rate_list:
                            for optimizer in optimizer_list:
                                for batch_size in batch_size_list:
                                    for epochs in epochs_list:
                                        try:
                                            print(f'Training model with filters_1={filters_1}, filters_2={filters_2}, kernel_size_1={kernel_size_1}, kernel_size_2={kernel_size_2}, pool_size={pool_size}, dense_units={dense_units}, dropout_rate={dropout_rate}, optimizer={optimizer}, batch_size={batch_size}, epochs={epochs}')
                                            
                                            model = create_cnn_model(filters_1, filters_2, kernel_size_1, kernel_size_2, pool_size, dense_units, dropout_rate, optimizer)
                                            
                                            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                                            
                                            history = model.fit(X_train_scaled, y_train, epochs=epochs, batch_size=batch_size, 
                                                                validation_split=0.2, verbose=1, callbacks=[early_stopping])
                                            
                                            val_loss = history.history['val_loss'][-1]
                                            print(f'Validation Loss: {val_loss}')
                                            
                                            if val_loss < best_loss:
                                                best_loss = val_loss
                                                best_params = {
                                                    'filters_1': filters_1,
                                                    'filters_2': filters_2,
                                                    'kernel_size_1': kernel_size_1,
                                                    'kernel_size_2': kernel_size_2,
                                                    'pool_size': pool_size,
                                                    'dense_units': dense_units,
                                                    'dropout_rate': dropout_rate,
                                                    'optimizer': optimizer,
                                                    'batch_size': batch_size,
                                                    'epochs': epochs
                                                }
                                                best_model = model
                                        except Exception as e:
                                            print(f"Error occurred: {e}")

# Print the best parameters and the best score
print(f"Best Parameters: {best_params}")
print(f"Best Validation Loss: {best_loss}")

# Evaluate the best model on the test set
cnn_evaluation = best_model.evaluate(X_test_scaled, y_test)
print(f"CNN Model Test Loss: {cnn_evaluation}")

# Make predictions with the best model
cnn_predictions = best_model.predict(X_test_scaled)

print("CNN Model Training and Evaluation with Manual Hyperparameter Tuning Complete.")

# Plot the training and validation loss of the best model
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
plt.title('CNN Model Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
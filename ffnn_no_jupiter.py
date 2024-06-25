import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, BatchNormalization, Conv1D, MaxPooling1D, Flatten, Dense, Reshape, Dropout
from keras.optimizers import Adam, RMSprop
#from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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

# Define a function to create the FFNN model with hyperparameters
def create_ffnn_model(units_1, units_2, dropout_rate, optimizer, learning_rate):
    model = Sequential()
    model.add(Dense(units_1, input_dim=X_train_scaled.shape[1], activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units_2, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='linear'))
    
    if optimizer == 'adam':
        opt = Adam(learning_rate=learning_rate)
    elif optimizer == 'rmsprop':
        opt = RMSprop(learning_rate=learning_rate)
    
    model.compile(optimizer=opt, loss='mean_squared_error')
    return model

# Hyperparameters to try
units_1_list = [32, 64, 128]
units_2_list = [16, 32, 64]
dropout_rate_list = [0.2, 0.4]
optimizer_list = ['adam', 'rmsprop']
learning_rate_list = [0.001, 0.01]
batch_size_list = [16, 32]
epochs_list = [100, 200, 400]

# Variables to store the best hyperparameters and lowest loss
best_params = {}
best_loss = float('inf')

# Iterate through all combinations of hyperparameters
for units_1 in units_1_list:
    for units_2 in units_2_list:
        for dropout_rate in dropout_rate_list:
            for optimizer in optimizer_list:
                for learning_rate in learning_rate_list:
                    for batch_size in batch_size_list:
                        for epochs in epochs_list:
                            print(f'Training model with units_1={units_1}, units_2={units_2}, dropout_rate={dropout_rate}, optimizer={optimizer}, learning_rate={learning_rate}, batch_size={batch_size}, epochs={epochs}')
                            
                            model = create_ffnn_model(units_1, units_2, dropout_rate, optimizer, learning_rate)
                            
                            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                            
                            history = model.fit(X_train_scaled, y_train, epochs=epochs, batch_size=batch_size, 
                                                validation_split=0.2, verbose=1, callbacks=[early_stopping])
                            
                            val_loss = history.history['val_loss'][-1]
                            print(f'Validation Loss: {val_loss}')
                            
                            if val_loss < best_loss:
                                best_loss = val_loss
                                best_params = {
                                    'units_1': units_1,
                                    'units_2': units_2,
                                    'dropout_rate': dropout_rate,
                                    'optimizer': optimizer,
                                    'learning_rate': learning_rate,
                                    'batch_size': batch_size,
                                    'epochs': epochs
                                }
                                best_model = model

# Print the best parameters and the best score
print(f"Best Parameters: {best_params}")
print(f"Best Validation Loss: {best_loss}")

# Evaluate the best model on the test set
ffnn_evaluation = best_model.evaluate(X_test_scaled, y_test)
print(f"FFNN Model Test Loss: {ffnn_evaluation}")

# Make predictions with the best model
ffnn_predictions = best_model.predict(X_test_scaled)

print("FFNN Model Training and Evaluation with Manual Hyperparameter Tuning Complete.")

# Plot the training and validation loss of the best model
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
plt.title('FFNN Model Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
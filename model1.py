import numpy as np
from sklearn.feature_selection import RFE, f_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import tensorflow as tf
from keras.models import Model, Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, Input, GlobalMaxPooling1D, LSTM, Dense
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# data_list = [Data(emp=..., pe=..., ...), Data(emp=..., pe=..., ...), ...]

class DataPoint:
    def __init__(self, emp=None, pe=None, cape=None, dy=None, rho=None, mov=None, ir=None, rr=None, y02=None, y10=None, stp=None, cf=None, mg=None, rv=None, ed=None, un=None, gdp=None, m2=None, cpi=None, dil=None, yss=None, nyf=None, _au=None, _dxy=None, _lcp=None, _ty=None, _oil=None, _mkt=None, _va=None, _gr=None, snp=None, label=None):
        self.emp = emp
        self.pe = pe
        self.cape = cape
        self.dy = dy
        self.rho = rho
        self.mov = mov
        self.ir = ir
        self.rr = rr
        self.y02 = y02
        self.y10 = y10
        self.stp = stp
        self.cf = cf
        self.mg = mg
        self.rv = rv
        self.ed = ed
        self.un = un
        self.gdp = gdp
        self.m2 = m2
        self.cpi = cpi
        self.dil = dil
        self.yss = yss
        self.nyf = nyf
        self._au = _au
        self._dxy = _dxy
        self._lcp = _lcp
        self._ty = _ty
        self._oil = _oil
        self._mkt = _mkt
        self._va = _va
        self._gr = _gr
        self.snp = snp
        self.label = label

    def __repr__(self) -> str:
        return f"{self.emp}, {self._oil}, {self.label}"

#---------------------------------------------
# Time series analysis -> currently overfitted
#---------------------------------------------

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i + seq_length]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def time_series_analysis(self):
    # Assuming self.get_snp() returns a list of DataPointStock objects
    data_points = self.get_snp()

    # Extract date and value for S&P 500 index
    dates = [dp.get_date() for dp in data_points]
    values = [dp.get_value() for dp in data_points]

    # Create a DataFrame
    df = pd.DataFrame({'Date': dates, 'Value': values})
    df.set_index('Date', inplace=True)

    # Normalize the data (scaling between 0 and 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = scaler.fit_transform(df)

    # Train-test split
    train_size = int(len(df_scaled) * 0.8)
    train_data = df_scaled[:train_size]
    test_data = df_scaled[train_size:]

    # Choose sequence length (e.g., 30 days)
    seq_length = 30

    # Define a function to create sequences of data for LSTM training
    def create_sequences(data, seq_length):
        xs, ys = [], []
        for i in range(len(data) - seq_length):
            x = data[i:i + seq_length]
            y = data[i + seq_length]
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    # Create sequences for LSTM
    X_train, y_train = create_sequences(train_data, seq_length)
    X_test, y_test = create_sequences(test_data, seq_length)

    # Reshape input data to be 3-dimensional for LSTM (samples, time steps, features)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Build LSTM model with dropout
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=False, input_shape=(seq_length, 1)))
    model.add(Dropout(0.2))  # Adding dropout layer with 20% dropout rate
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Define early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train the model with early stopping
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

    # Evaluate the model on test data
    test_loss = model.evaluate(X_test, y_test)
    print(f'Test Loss: {test_loss}')

    # Evaluate the model on training data
    train_loss = model.evaluate(X_train, y_train)
    print(f'Training Loss: {train_loss}')

    # Make predictions
    predictions = model.predict(X_test)

    # Inverse transform predictions and actual values for plotting
    predictions = scaler.inverse_transform(predictions)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Print the first and last dates in the dataset
    print(f"Data starts at: {df.index.min()}")
    print(f"Data ends at: {df.index.max()}")

    # Plotting
    plt.figure(figsize=(14, 7))
    plt.plot(df.index[train_size + seq_length:], y_test, label='Actual')
    plt.plot(df.index[train_size + seq_length:], predictions, label='Predicted')
    plt.title('S&P 500 Index Prediction with LSTM')
    plt.xlabel('Date')
    plt.ylabel('S&P 500 Index')
    plt.legend()
    plt.show()

    # Plot training & validation loss values
    plt.figure(figsize=(14, 7))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

#-------------------------------------------------------------
# Convolutional net attempt
#-------------------------------------------------------------


def prediction_model(self, selected_features):
    features, X, y = selected_features

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Feature selection with Recursive Feature Elimination (RFE)
    estimator = RandomForestClassifier(n_estimators=100, random_state=42)
    selector = RFE(estimator, n_features_to_select=15, step=1)
    selector = selector.fit(X_train, y_train)
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)

    # Autoencoder for dimensionality reduction
    input_dim = X_train_selected.shape[1]
    encoding_dim = 8

    input_layer = Input(shape=(input_dim,))
    encoder = Dense(encoding_dim, activation="relu")(input_layer)
    encoder = BatchNormalization()(encoder)
    decoder = Dense(input_dim, activation="sigmoid")(encoder)

    autoencoder = Model(inputs=input_layer, outputs=decoder)
    autoencoder.compile(optimizer='adam', loss='mse')

    autoencoder.fit(X_train_selected, X_train_selected, epochs=100, batch_size=32, shuffle=True, validation_data=(X_test_selected, X_test_selected), callbacks=[EarlyStopping(patience=10)])

    # Encode the data
    encoder_model = Model(inputs=input_layer, outputs=encoder)
    X_train_encoded = encoder_model.predict(X_train_selected)
    X_test_encoded = encoder_model.predict(X_test_selected)

    # Reshape data for CNN
    X_train_cnn = X_train_encoded.reshape((X_train_encoded.shape[0], X_train_encoded.shape[1], 1))
    X_test_cnn = X_test_encoded.reshape((X_test_encoded.shape[0], X_test_encoded.shape[1], 1))

    # Build and train the CNN
    cnn_model = Sequential([
        Input(shape=(encoding_dim, 1)),
        Conv1D(32, kernel_size=2, activation='relu'),
        BatchNormalization(),
        Conv1D(64, kernel_size=2, activation='relu'),
        BatchNormalization(),
        GlobalMaxPooling1D(),
        Dense(100, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    cnn_model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    # Train the CNN model and save history for plotting
    history = cnn_model.fit(X_train_cnn, y_train, epochs=100, batch_size=32, validation_data=(X_test_cnn, y_test), callbacks=[EarlyStopping(patience=10)])

    # Evaluate the model
    loss, accuracy = cnn_model.evaluate(X_test_cnn, y_test)
    print(f'Test Accuracy: {accuracy:.4f}')

    # Plot training & validation loss values
    plt.figure(figsize=(14, 7))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Plot training & validation accuracy values
    plt.figure(figsize=(14, 7))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


#------------------------------------------------------------------------------

# # Für den Fall du willst diese selber direct accessen und zusammenschustern
# # Beispiel
# emp = self.get_emp()# [List full of Data Objects]
# data_emp = [obj.get_value() for obj in self.get_emp()]
# Alle Werte sind erreichbar mit .get_value() ausser bei label, dass ist eh direkt nur der Integer.

def your_prediciton_model(self, features):
    
    # Für den Fall du willst features direkt übergeben.
    selected_features, X, y = features

    #print(selected_features, X, y)
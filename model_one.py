import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

# Example architecture
model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(input_width, num_features)),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),
    Conv1D(filters=128, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),
    Flatten(),
    Dense(50, activation='relu'),
    Dropout(0.2),
    Dense(1)  # Single output for regression (predicting the next value)
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Fit the model
history = model.fit(train_X, train_y, epochs=50, batch_size=32, validation_data=(val_X, val_y))




#----------

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization

# Define the model
model = Sequential()

# Add convolutional layers
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(30, 30)))
model.add(MaxPooling1D(pool_size=2))
model.add(BatchNormalization())
model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(BatchNormalization())

# Flatten the output from convolutional layers
model.add(Flatten())

# Add fully connected layers
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))

# Output layer
model.add(Dense(1, activation='linear'))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Model summary
model.summary()


# training and validation

# Early stopping and learning rate reduction
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, min_lr=0.00001)

# Fit the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping, lr_reduction])

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')


# making predicitons
predictions = model.predict(X_new)

#-------------------------------------------
# 1. Recursive Feature Elimination (RFE)
# RFE ist eine iterative Methode, die mit allen Merkmalen beginnt und schrittweise die am wenigsten wichtigen Merkmale eliminiert, bis nur noch eine vorgegebene Anzahl von Merkmalen übrig bleibt. Es wird häufig in Verbindung mit maschinellen Lernmodellen wie linearen Regressionsmodellen, Support Vector Machines oder Entscheidungsbäumen verwendet.

# Schritt-für-Schritt-Anleitung für RFE:

# Modell wählen: Wählen Sie ein Basismodell aus, z.B. eine lineare Regression oder einen Entscheidungsbaum.
# RFE anwenden: Verwenden Sie die RFE-Klasse aus Scikit-Learn, um die Merkmale schrittweise zu eliminieren.
# Modell trainieren und evaluieren: Trainieren und evaluieren Sie das Modell mit den ausgewählten Merkmalen.
# Hier ist ein Beispielcode in Python mit Scikit-Learn:

import numpy as np
from sklearn.datasets import make_classification
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# Beispieldatensatz erstellen
X, y = make_classification(n_samples=1000, n_features=30, n_informative=10, random_state=42)

# Basismodell auswählen
model = LogisticRegression()

# RFE-Objekt erstellen und anpassen
rfe = RFE(model, n_features_to_select=10)
fit = rfe.fit(X, y)

# Ergebnisse anzeigen
print("Num Features: %s" % (fit.n_features_))
print("Selected Features: %s" % (fit.support_))
print("Feature Ranking: %s" % (fit.ranking_))

#--------------------

# 2. Genetische Algorithmen (GA)
# Genetische Algorithmen sind eine Metaheuristik, die durch natürliche Selektion inspiriert ist. Sie sind besonders nützlich, wenn die Merkmalsauswahl komplex und nichtlinear ist. GA verwendet eine Population von Lösungen, die sich über Generationen hinweg entwickeln, um eine optimale Lösung zu finden.

# Schritt-für-Schritt-Anleitung für GA:

# Initialisierung: Erstellen Sie eine Population zufälliger Merkmalsauswahl-Lösungen.
# Fitnessbewertung: Bewerten Sie jede Lösung basierend auf der Leistung eines maschinellen Lernmodells.
# Selektion: Wählen Sie die besten Lösungen zur Reproduktion.
# Kreuzung und Mutation: Kombinieren und verändern Sie die ausgewählten Lösungen, um neue Lösungen zu erstellen.
# Iteration: Wiederholen Sie die Schritte 2-4 über mehrere Generationen.
# Hier ist ein Beispielcode in Python mit der DEAP-Bibliothek:


import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from deap import base, creator, tools, algorithms

# Beispieldatensatz erstellen
X, y = make_classification(n_samples=1000, n_features=30, n_informative=10, random_state=42)

# Train/Test-Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fitness-Funktion definieren
def evaluate(individual):
    selected_features = [i for i in range(len(individual)) if individual[i] == 1]
    if len(selected_features) == 0:
        return 0,
    model = LogisticRegression()
    model.fit(X_train[:, selected_features], y_train)
    predictions = model.predict(X_test[:, selected_features])
    return accuracy_score(y_test, predictions),

# GA-Setup
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_bool", np.random.randint, 2)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=30)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

# Evolution durchführen
population = toolbox.population(n=50)
ngen = 20
cxpb = 0.5
mutpb = 0.2

result_population, logbook = algorithms.eaSimple(population, toolbox, cxpb, mutpb, ngen, verbose=False)

# Beste Lösung anzeigen
best_individual = tools.selBest(result_population, k=1)[0]
print('Best Individual:', best_individual)
selected_features = [i for i in range(len(best_individual)) if best_individual[i] == 1]
print('Selected Features:', selected_features)

# zu berücksichtigen gilt es ebenfalls noch Ordergebühren, als auch Handelsplatzgebühren.
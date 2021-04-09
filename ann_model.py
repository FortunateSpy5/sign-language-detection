from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from pickle import dump
# from sklearn.metrics import accuracy_score

df = pd.read_csv("connections.csv", index_col=0)

# Feature selection
X = df.drop('SIGN', axis=1)
y = df[['SIGN']]

# One hot encoding
y = pd.get_dummies(y)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# Scaling
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save scaler for later use
dump(scaler, open('scaler.pkl', 'wb'))

# ANN Model
model = Sequential()
model.add(Dense(units=len(X.columns), activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(units=60, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(units=120, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(units=df['SIGN'].nunique(), activation='softmax'))
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

# Early Stopping Callback
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)

# Training the model
model.fit(
    x=X_train,
    y=y_train.values,
    epochs=500,
    validation_data=(X_test, y_test),
    verbose=1,
    callbacks=[early_stop]
)

# # Prediction
# pred = model.predict(X_test)

# # Reverse One Hot Encoding
# def get_sign(row):
#     for column in y.columns:
#         if row[column] == row.max():
#             return column[-1]
# y_test = y_test.apply(get_sign, axis=1)
# pred = pd.DataFrame(pred, columns=y.columns).apply(get_sign, axis=1)

print(f"Accuracy: {model.evaluate(X_test, y_test)[1] * 100:.2f}")
# Accuracy: 99.78

# Save model
model.save("ann_model.h5")

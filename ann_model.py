from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

df = pd.read_csv("my_train.csv", index_col=0)

# Feature selection
X = df.drop('Sign', axis=1)
y = df[['Sign']]

y['Sign'] = y['Sign'].apply(str)

print(y.groupby('Sign').count())

print(y['Sign'].unique())

# One hot encoding
y = pd.get_dummies(y)

# Scaling X
# Such a method is implemented to make sure that the 
# scale of the hand is independent of the image size

def scale_x(row):
    return row / row.max()

# X = X.apply(scale_x, axis=1)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

# ANN Model
model = Sequential()
model.add(Dense(units=len(X.columns)-1, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(units=60, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.1))
#model.add(Dense(units=280, activation='relu'))
#model.add(Dropout(0.1))
#model.add(Dense(units=580, activation='relu'))
#model.add(Dropout(0.1))
model.add(Dense(units=df['Sign'].nunique(), activation='softmax'))
model.compile(loss='categorical_crossentropy',
              metrics=['accuracy'], optimizer='adam')

# Early Stopping Callback
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100)

# Training the model
model.fit(
    x=X_train,
    y=y_train.values,
    epochs=5000,
    validation_data=(X_test, y_test),
    verbose=1,
    # callbacks=[early_stop]
)

print(f"Accuracy: {model.evaluate(X_test, y_test)[1] * 100:.2f}")
# Accuracy: 98.65

# Save model
model.save("ann_model1.h5")

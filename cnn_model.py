from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import cv2
import os
from generate_csv import get_image_list
from sklearn.model_selection import train_test_split

# Get the processed image directory tree
image_dict = get_image_list("./processed_images/")

# Get the total no. of images
c = 0
for folder, values in image_dict.items():
  c += len(values)

# Initialize X & y
X = np.empty((c, 120, 160, 3))
y = []

# Store images in numpy array
k = 0
for folder, image_names in image_dict.items():
    for image_name in image_names:
        image = cv2.imread(f"./processed_images/{folder}/{image_name}")
        image = cv2.flip(image, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        X[k] = cv2.resize(image, (160, 120))
        print(k)
        y.append(folder)
        k += 1

# One Hot Encoding
y = pd.get_dummies(pd.DataFrame(y, columns=['SIGN']))

# Scaling
X = X / 255

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=101)

# CNN Model
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(4, 4),
          input_shape=(120, 160, 3), activation='relu'))
model.add(Dropout(0.1))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(filters=64, kernel_size=(4, 4), activation='relu'))
model.add(Dropout(0.1))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(120, activation='relu'))
model.add(Dense(24, activation='softmax'))

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Early Stopping
early_stop = EarlyStopping(monitor='val_loss', patience=3)

# Training CNN Model
model.fit(
    X_train, y_train, epochs=100,
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
)

print(f"Accuracy: {model.evaluate(X_test, y_test)[1] * 100:.2f}")
# Accuracy: 99.55

# Save model
model.save("cnn_model.h5")


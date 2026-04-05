# Experiment No. 8
# Feedforward Neural Network on MNIST Dataset

# Step 1: Import libraries
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

# Step 2: Load MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Step 3: Normalize data (0 to 1)
X_train = X_train / 255.0
X_test = X_test / 255.0

# Step 4: One-hot encoding for labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Step 5: Build Feedforward Neural Network
model = Sequential()

# Flatten layer (28x28 → 784)
model.add(Flatten(input_shape=(28, 28)))

# Hidden layers
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))

# Output layer (10 classes)
model.add(Dense(10, activation='softmax'))

# Step 6: Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Step 7: Train model
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# Step 8: Evaluate model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

print("\nModel Performance:")
print(f"Test Accuracy: {accuracy:.2f}")
print(f"Test Loss: {loss:.2f}")

# Step 9: Plot Accuracy and Loss

# Accuracy Plot
plt.figure()
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Loss Plot
plt.figure()
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
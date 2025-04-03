import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
import numpy as np

def create_cnn_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    # For demonstration: simulate spectrogram image data
    input_shape = (128, 128, 1)  # Adjust based on your spectrogram dimensions
    num_classes = 4              # Adjust for your number of emotion classes

    # Simulate dummy data (replace with your actual data loader)
    X = np.random.rand(100, 128, 128, 1)
    y = tf.keras.utils.to_categorical(np.random.randint(0, num_classes, 100), num_classes)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = create_cnn_model(input_shape, num_classes)
    model.summary()
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from google.colab import files

# Step 2: Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Step 3: Preprocess the dataset
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')
x_train /= 255.0
x_test /= 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Step 4: Define the neural network architecture
model = Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(10, activation='softmax')
])

# Step 5: Compile the model
model.compile(optimizer=Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Step 6: Train the model
history = model.fit(x_train, y_train, epochs=50, batch_size=128, validation_data=(x_test, y_test))

# Step 7: Upload an image (using Google Colab uploader)
uploaded = files.upload()

# Step 8: Load the uploaded image
for fn in uploaded.keys():
    img = Image.open(fn)
    img = img.convert('L')  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to 28x28 pixels
    img = np.array(img).astype('float32')  # Convert to numpy array
    img = img.reshape(1, 28, 28, 1)  # Reshape for model input
    img /= 255.0  # Normalize the image data
    prediction = model.predict(img)
    predicted_label = np.argmax(prediction)
    print(f"Predicted class label: {predicted_label}")
    plt.imshow(img.reshape(28, 28), cmap='gray')
    plt.show()

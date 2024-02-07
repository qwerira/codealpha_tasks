
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
data = pd.read_csv("/kaggle/input/az-handwritten-alphabets-in-csv-format/A_Z Handwritten Data.csv").astype('float32')#download the datadet from kaggle as 
                                                                                                                     #it was too large to add to github

# Displaying the first 10 rows of the dataset
data.head(10)

# Separating features and target variable
X = data.drop('0', axis=1)
y = data['0']

# Reshaping the data
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Splitting the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Reshaping the features for compatibility with Convolutional Neural Network (CNN)
x_train = np.reshape(x_train.values, (x_train.shape[0], 28, 28))
x_test = np.reshape(x_test.values, (x_test.shape[0], 28, 28))

# Displaying the shape of the training and testing data
print("Shape of Training data: ", x_train.shape)
print("Shape of Testing data: ", x_test.shape)

# Thresholding and shuffling the training data
import cv2

shuffle_data = shuffle(x_train)

# Displaying a sample of shuffled data
fig, axes = plt.subplots(3, 3, figsize=(10, 10))
axes = axes.flatten()

for i in range(9):
    _, shu = cv2.threshold(shuffle_data[i], 30, 200, cv2.THRESH_BINARY)
    axes[i].imshow(np.reshape(shuffle_data[i], (28, 28)), cmap="Greys")
plt.show()

# Reshaping the data for compatibility with CNN
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

# Displaying the new shape of the training and testing data
print("New shape of training data: ", x_train.shape)
print("New shape of testing data: ", x_test.shape)

# Converting target variables to categorical format
from tensorflow.keras.utils import to_categorical

y_training = to_categorical(y_train, num_classes=26, dtype='int')
y_testing = to_categorical(y_test, num_classes=26, dtype='int')

# Displaying the new shape of the training and testing labels
print("New shape of training labels: ", y_training.shape)
print("New shape of testing labels: ", y_testing.shape)

# Building the Convolutional Neural Network (CNN) model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from tensorflow.keras.optimizers import Adam

model = Sequential()

model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPool2D(2, 2))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPool2D(2, 2))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPool2D(2, 2))

model.add(Flatten())

model.add(Dense(128, activation="relu"))
model.add(Dense(256, activation="relu"))

model.add(Dense(26, activation="softmax"))

# Compiling the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Training the model
history = model.fit(x_train, y_training, epochs=5, validation_data=(x_test, y_testing))

# Displaying the summary of the model
model.summary()

# Creating a dictionary for character labels
words = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

# Displaying predictions on a sample of testing data
fig, axes = plt.subplots(3, 3, figsize=(8, 9))
axes = axes.flatten()

for i, ax in enumerate(axes):
    image = np.reshape(x_test[i], (28, 28))
    ax.imshow(image, cmap="Greys")
    
    pred = words[np.argmax(y_testing[i])]
    ax.set_title("Prediction: " + pred)
    ax.grid()

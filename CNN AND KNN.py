import os
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Data Preparation
def preprocess_image(image_path, target_size=(64, 64)):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    image = cv2.resize(image, target_size)  # Resize the image to the target size
    return image

# Define directory paths for train, test, and validation sets
train_fire_directory = '/content/drive/MyDrive/spilt_dataset/train/fire_images'
train_non_fire_directory = '/content/drive/MyDrive/spilt_dataset/train/non_fire_images'
test_fire_directory = '/content/drive/MyDrive/spilt_dataset/test/fire_images'
test_non_fire_directory = '/content/drive/MyDrive/spilt_dataset/test/non_fire_images'
validation_fire_directory = '/content/drive/MyDrive/spilt_dataset/validation/fire_images'
validation_non_fire_directory = '/content/drive/MyDrive/spilt_dataset/validation/non_fire_images'

# Load and preprocess the dataset
fire_images = [preprocess_image(os.path.join(train_fire_directory, filename)) for filename in os.listdir(train_fire_directory)]
non_fire_images = [preprocess_image(os.path.join(train_non_fire_directory, filename)) for filename in os.listdir(train_non_fire_directory)]

# Validation data
validation_fire_images = [preprocess_image(os.path.join(validation_fire_directory, filename)) for filename in os.listdir(validation_fire_directory)]
validation_non_fire_images = [preprocess_image(os.path.join(validation_non_fire_directory, filename)) for filename in os.listdir(validation_non_fire_directory)]

# Create labels for the data (1 for fire, 0 for non-fire)
fire_labels = [1] * len(fire_images)
non_fire_labels = [0] * len(non_fire_images)

validation_fire_labels = [1] * len(validation_fire_images)
validation_non_fire_labels = [0] * len(validation_non_fire_images)

# Combine fire and non-fire data
X = np.array(fire_images + non_fire_images)
y = np.array(fire_labels + non_fire_labels)

# Validation data
X_validation = np.array(validation_fire_images + validation_non_fire_images)
y_validation = np.array(validation_fire_labels + validation_non_fire_labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Flatten the images to use as feature vectors
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)
X_validation = X_validation.reshape(X_validation.shape[0], -1)

# K-Nearest Neighbors Model
clf = KNeighborsClassifier(n_neighbors=5)

# Train the K-NN model
clf.fit(X_train, y_train)

# Test the model
y_pred = clf.predict(X_test)

# Validate the model
y_val_pred = clf.predict(X_validation)

# Calculate accuracy and reports for training, test, and validation sets
training_accuracy = accuracy_score(y_train, clf.predict(X_train))
test_accuracy = accuracy_score(y_test, y_pred)
validation_accuracy = accuracy_score(y_validation, y_val_pred)

training_report = classification_report(y_train, clf.predict(X_train), target_names=['Non-Fire', 'Fire'])
test_report = classification_report(y_test, y_pred, target_names=['Non-Fire', 'Fire'])
validation_report = classification_report(y_validation, y_val_pred, target_names=['Non-Fire', 'Fire'])

print("Training Accuracy: {:.2f}%".format(training_accuracy * 100))
print(training_report)
print("Test Accuracy: {:.2f}%".format(test_accuracy * 100))
print(test_report)
print("Validation Accuracy: {:.2f}%".format(validation_accuracy * 100))
print(validation_report)

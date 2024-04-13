import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Parameters
image_width, image_height = 64, 64  # Desired dimensions of the images
number_of_classes =36  # Total number of classes
images_per_class = 100  # Number of images per class

# Directories
DATA_DIR = './data'
PROCESSED_DATA_DIR = './processed_data'

# Create a directory for processed data if it does not exist
if not os.path.exists(PROCESSED_DATA_DIR):
    os.makedirs(PROCESSED_DATA_DIR)

# Initialize arrays for storing image data and labels
X = []  # Array for storing image data
y = []  # Array for storing labels

# Preprocessing
for class_label in range(number_of_classes):
    class_folder = os.path.join(DATA_DIR, str(class_label))
    for img_index in range(images_per_class):
        img_path = os.path.join(class_folder, f'{img_index}.jpg')
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Error loading image: {img_path}")
            continue

        # Convert to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Resize image
        img = cv2.resize(img, (image_width, image_height))
        
        # Normalize pixel values
        img = img / 255.0
        
        # Reshape grayscale image to add channel dimension (64, 64) -> (64, 64, 1)
        img = img.reshape((image_width, image_height, 1))
        
        # Append to dataset
        X.append(img)
        y.append(class_label)

# Convert to numpy arrays and one-hot encode labels
X = np.array(X)
y = to_categorical(y, number_of_classes)

# Split the data into training, validation, and test sets
X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=42)

# Save the data
np.save(os.path.join(PROCESSED_DATA_DIR, 'X_train.npy'), X_train)
np.save(os.path.join(PROCESSED_DATA_DIR, 'y_train.npy'), y_train)
np.save(os.path.join(PROCESSED_DATA_DIR, 'X_val.npy'), X_val)
np.save(os.path.join(PROCESSED_DATA_DIR, 'y_val.npy'), y_val)
np.save(os.path.join(PROCESSED_DATA_DIR, 'X_test.npy'), X_test)
np.save(os.path.join(PROCESSED_DATA_DIR, 'y_test.npy'), y_test)

print("Data preprocessing completed and saved.")




# Replace with your actual file path
file_path = './processed_data/X_test.npy'

# Load the file
data = np.load(file_path)

# Print the shape of the array
print("Shape of the array:", data.shape)

# Print the entire array
print("Array contents:\n", data)

# Print the first item in the array to see what it looks like
print(data)
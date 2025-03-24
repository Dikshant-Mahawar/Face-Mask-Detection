import os
import cv2
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight

# Constants
IMG_SIZE = 96
BATCH_SIZE = 32
LEARNING_RATE = 0.0005
EPOCHS = 50

# Load data
def load_data(paths, labels):
    data, target = [], []
    for path, label in zip(paths, labels):
        for file in os.listdir(path):
            img = cv2.imread(os.path.join(path, file))
            if img is not None:
                data.append(cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 127.5 - 1.0)
                target.append(label)
    return np.array(data), to_categorical(target)

X, y = load_data(["with_mask", "without_mask"], [1, 0])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Data Augmentation
datagen = ImageDataGenerator(rotation_range=15, width_shift_range=0.1,
                             height_shift_range=0.1, shear_range=0.1,
                             zoom_range=0.1, horizontal_flip=True,
                             fill_mode='nearest')
datagen.fit(X_train)

# Model
base_model = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet')
for layer in base_model.layers[:-30]:  # Freeze lower layers
    layer.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(2, activation='softmax')
])

model.compile(optimizer=keras.optimizers.Adam(LEARNING_RATE),
              loss='categorical_crossentropy', metrics=['accuracy'])

# Compute class weights
class_weight_dict = dict(enumerate(compute_class_weight('balanced', 
                            classes=np.unique(np.argmax(y, axis=1)), 
                            y=np.argmax(y, axis=1))))

# Callbacks
callbacks = [
    EarlyStopping(patience=7, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.2, patience=3, min_lr=1e-6)
]

# Training
history = model.fit(datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
                    epochs=EPOCHS,
                    validation_data=(X_test, y_test),
                    class_weight=class_weight_dict,
                    callbacks=callbacks)

# Evaluation
_, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")

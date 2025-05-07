import numpy as np
import pandas as pd
from keras.applications.mobilenet import preprocess_input
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense, BatchNormalization, Flatten, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from tqdm.notebook import tqdm_notebook as tqdm
import os
from tensorflow.keras.models import load_model
from sklearn.utils import class_weight 

# Define paths
real = "real_and_fake_face_detection/real_and_fake_face/training_real/"
fake = "real_and_fake_face_detection/real_and_fake_face/training_fake/"

# Load image paths
real_path = os.listdir(real)
fake_path = os.listdir(fake)

# Visualizing real and fake faces
def load_img(path):
    image = cv2.imread(path)
    image = cv2.resize(image, (224, 224))
    return image[..., ::-1]

fig = plt.figure(figsize=(10, 10))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(load_img(real + real_path[i]), cmap='gray')
    plt.suptitle("Real faces", fontsize=20)
    plt.axis('off')
plt.show()

fig = plt.figure(figsize=(10, 10))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(load_img(fake + fake_path[i]), cmap='gray')
    plt.suptitle("Fake faces", fontsize=20)
    plt.title(fake_path[i][:4])
    plt.axis('off')
plt.show()

# Data augmentation
# dataset_path = "real_and_fake_face"
dataset_path = "real_and_fake_face_detection/real_and_fake_face"

data_with_aug = ImageDataGenerator(horizontal_flip=True,
                                   vertical_flip=False,
                                   rescale=1./255,
                                   validation_split=0.2)
train = data_with_aug.flow_from_directory(dataset_path,
                                          class_mode="binary",
                                          target_size=(96, 96),
                                          batch_size=32,
                                          subset="training")
val = data_with_aug.flow_from_directory(dataset_path,
                                          class_mode="binary",
                                          target_size=(96, 96),
                                          batch_size=32,
                                          subset="validation")

# calculate class weight                                    #   
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(train.classes), y=train.classes)
class_weights = dict(zip(np.unique(train.classes), class_weights))                                          




# MobileNetV2 model
mnet = MobileNetV2(include_top=False, weights="imagenet", input_shape=(96, 96, 3))
tf.keras.backend.clear_session()

model = Sequential([mnet,
                    GlobalAveragePooling2D(),
                    Dense(512, activation="relu"),
                    BatchNormalization(),
                    Dropout(0.3),
                    Dense(128, activation="relu"),
                    Dropout(0.1),
                    Dense(2, activation="softmax")])

model.layers[0].trainable = False

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.summary()

# training the model
# Callbacks
def scheduler(epoch):
    if epoch <= 2:
        return 0.001
    elif epoch > 2 and epoch <= 15:
        return 0.0001
    else:
        return 0.00001

lr_callbacks = tf.keras.callbacks.LearningRateScheduler(scheduler)
hist = model.fit(train,
                 epochs=20,
                 callbacks=[lr_callbacks],
                 validation_data=val,
                  class_weight=class_weights
                 )

# Save model
# model.save('deepfake_detection_model.h5')
model.save('deepfake_detection_model.h5')
import os
print("Model saved at:", os.path.abspath("deepfake_detection_model.h5"))
print("Exists?", os.path.exists("deepfake_detection_model.h5"))

# model.save("my_deepfake_model.keras")
# test_loss, test_acc = model.evaluate(test_data)
# st.write(f"Test accuracy: {test_acc}, Test loss: {test_loss}")
# print(f'Test Loss: {test_loss}, Test Accuracy: {test_acc}')
# print the model summary to check its structure
model.summary()

# Visualizing accuracy and loss
epochs = 20
train_loss = hist.history['loss']
val_loss = hist.history['val_loss']
train_acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']
xc = range(epochs)

plt.figure(1, figsize=(7, 5))
plt.plot(xc, train_loss)
plt.plot(xc, val_loss)
plt.xlabel('Number of Epochs')
plt.ylabel('Loss')
plt.title('Train Loss vs Validation Loss')
plt.grid(True)
plt.legend(['Train', 'Validation'])
plt.style.use(['classic'])

plt.figure(2, figsize=(7, 5))
plt.plot(xc, train_acc)
plt.plot(xc, val_acc)
plt.xlabel('Number of Epochs')
plt.ylabel('Accuracy')
plt.title('Train Accuracy vs Validation Accuracy')
plt.grid(True)
plt.legend(['Train', 'Validation'], loc=4)
plt.style.use(['classic'])
plt.show()

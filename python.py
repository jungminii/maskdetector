import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
import numpy as np

import matplotlib.pyplot as plt

def add_random_noise(x):
  x = x + np.random.normal(size=x.shape) * np.random.uniform(1,5)
  x = x - x.min()
  x = x / x.max()

  return x * 255.0

TRAINING_DIR = "/content/drive/MyDrive/forif_tf_dataset/train-set"
VALIDATION_DIR = "/content/drive/MyDrive/forif_tf_dataset/test-set"

batch_size = 8

training_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    brightness_range=(0.5, 1.3),
    horizontal_flip=True,
    fill_mode='nearest',
    preprocessing_function=add_random_noise)

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = training_datagen.flow_from_directory(
    TRAINING_DIR,
    batch_size=batch_size,
    target_size=(224, 224),
    class_mode='categorical',
    shuffle=True
)

validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    batch_size=batch_size,
    target_size=(224, 224),
    class_mode='categorical'
)

img, label = next(train_generator)
plt.figure(figsize=(20, 20))

for i in range(8):
  plt.subplot(3, 3, i+1)
  plt.imshow(img[i])
  plt.title(label[i])
  plt.axis('off')

plt.show()

base_model = tf.keras.applications.VGG16(input_shape=(244, 244, 3),
                                         include_top=False, weights='imagenet')

base_model.trainable = False

out_layer = tf.keras.layers.Conv2D(128, (1, 1), padding='SAME', activation=None)(base_model.output)
out_layer = tf.keras.layers.BatchNormalization()(out_layer)
out_layer = tf.keras.layers.ReLU()(out_layer) # 7x7x128

out_layer = tf.keras.layers.GlobalAveragePooling2D()(out_layer) #128

out_layer = tf.keras.layers.Dense(2, activation='softmax')(out_layer)

model = tf.keras.models.Model(base_model.input, out_layer)

model.summary()

model.compile(loss='categorical_crossentropy', 
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              metrics=['accuracy'])

history = model.fit(train_generator, epochs=25,
                    validation_data=validation_generator, verbose=1)

model.save("forif_hackerton.h5")

import cv2
import matplotlib.image as img

input_size= (244, 244)

test_image = img.imread("/content/test1.jpg")
model = tf.keras.models.load_model("/content/forif_hackerton.h5")


# Resize the frame for the model
model_frame = cv2.resize(test_image, input_size, test_image)
# Expand Dimension (224, 224, 3) -> (1, 224, 224, 3) and Normalize the data
model_frame = np.expand_dims(model_frame, axis=0) / 255.0

# Predict
is_mask_prob = model.predict(model_frame)
is_mask = np.argmax(is_mask_prob)

if is_mask == 0:
    msg_mask = "Mask Off"
else:
    msg_mask = "Mask On"

plt.imshow(test_image)
plt.show()
print("Result : ", msg_mask)

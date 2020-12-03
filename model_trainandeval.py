#model architecture in keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras import models, layers

model = models.Sequential()
model.add(layers.Conv2D(64, (3, 3), activation='relu', name = 'conv1_1', padding = 'same', input_shape=(224, 224, 3)))
model.add(layers.Conv2D(64, (3, 3), activation='relu', name = 'conv1_2', padding = 'same'))
model.add(layers.MaxPooling2D(pool_size=2, name = 'pool1'))

model.add(layers.Conv2D(128, (3, 3), activation='relu', name = 'conv2_1', padding = 'same'))
model.add(layers.Conv2D(128, (3, 3), activation='relu', name = 'conv2_2', padding = 'same'))
model.add(layers.MaxPooling2D(pool_size=2, name = 'pool2'))

model.add(layers.Conv2D(256, (3, 3), activation='relu', name = 'conv3_1', padding = 'same'))
model.add(layers.Conv2D(256, (3, 3), activation='relu', name = 'conv3_2', padding = 'same'))
model.add(layers.Conv2D(256, (3, 3), activation='relu', name = 'conv3_3', padding = 'same'))
model.add(layers.MaxPooling2D(pool_size=2, name = 'pool3'))

model.add(layers.Conv2D(512, (3, 3), activation='relu', name = 'conv4_1', padding = 'same'))
model.add(layers.Conv2D(512, (3, 3), activation='relu', name = 'conv4_2', padding = 'same'))
model.add(layers.Conv2D(512, (3, 3), activation='relu', name = 'conv4_3', padding = 'same'))
model.add(layers.MaxPooling2D(pool_size=2, name = 'pool4'))

model.add(layers.Conv2D(512, (3, 3), activation='relu', name = 'conv5_1', padding = 'same'))
model.add(layers.Conv2D(512, (3, 3), activation='relu', name = 'conv5_2', padding = 'same'))
model.add(layers.Conv2D(512, (3, 3), activation='relu', name = 'conv5_3', padding = 'same'))
model.add(layers.MaxPooling2D(pool_size=2, name = 'pool5'))

model.add(Flatten())

model.add(layers.Dense(4096, activation='relu', name = 'fc6'))
model.add(layers.Dropout(0.5, name = 'drop6'))

model.add(layers.Dense(4096, activation='relu', name = 'fc7'))
model.add(layers.Dropout(0.5, name = 'drop7'))

#load pretrained weights to the model (obtained from caffe model)
model.load_weights('keras_weights.h5', skip_mismatch=True, by_name=True)

#Make the layers in the model non-trainable
for i in range(len(model.layers)):
  model.layers[i].trainable = False
  
#Add the classifier layer to the model
model.add(layers.Dense(2, activation= 'softmax', name = 'prob', trainable=True))

#print the model summary to visualize teh model architecture
print(model.summary())

######################################################################################

# load data to keras

import tensorflow as tf
from tensorflow.keras import preprocessing

data_train = tf.keras.preprocessing.image_dataset_from_directory(
    'data',
    labels= "inferred",
    label_mode="int",
    color_mode="rgb",
    batch_size=128,
    image_size=(224, 224),
    shuffle=True,
    subset= 'training',
    seed=40,
    validation_split=0.2,
    interpolation="bilinear"
)

data_val = tf.keras.preprocessing.image_dataset_from_directory(
    'data',
    labels= "inferred",
    label_mode="int",
    color_mode="rgb",
    batch_size=128,
    image_size=(224, 224),
    shuffle=True,
    subset= 'validation',
    seed=40,
    validation_split=0.2,
    interpolation="bilinear"
)

# compile the model

from tensorflow.keras import metrics

model.compile(
  optimizer='adam',
  loss= tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

# fit the model over 20 epochs

history = model.fit(
    data_train, 
    validation_data = data_val, 
    epochs= 15
    )

# save the model
model.save('final_tensorflow_model')

# evaluate model on validation data
results = model.evaluate(data_val, batch_size = 128)
print(results)

# plot for accuracy vs. no. of epochs for training and validation
import matplotlib.pyplot as plt
import numpy as np

plt.figure(1)
plt.plot(np.arange(15), history.history['val_accuracy'], label= 'validation accuracy')
plt.plot(np.arange(15), history.history['accuracy'], label = 'training accuracy')
plt.xlabel('no. of epochs')
plt.ylabel('accuracy')
plt.title('Accuracy vs no. of epochs')
plt.legend()
plt.show()

# plot for loss vs. no. of epochs for training and validation
plt.figure(2)
plt.plot(np.arange(15), history.history['val_loss'], label= 'validation loss')
plt.plot(np.arange(15), history.history['loss'], label = 'training loss')
plt.xlabel('no. of epochs')
plt.ylabel('loss')
plt.title('Loss vs no. of epochs')
plt.legend()
plt.show()

# get true labels (y_val) and predicted labels (y_pred)
y_val = []
y_pred = []
for image_batch, labels_batch in data_val:
  for label in labels_batch:
    y_val.append(label)
  for image in image_batch:
    y_pred.append(np.argmax(model.predict(image.numpy().reshape(1,224,224,3))))
    
# confusion matrix and classificaton report
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

print(classification_report(y_val,y_pred))
print(confusion_matrix(y_val,y_pred))

# visualize some of the wrong classifications

i = 0
j = 0
plt.figure(figsize=(10,10))
for images, labels in data_val.take(1000):
    if i == 9:
      break
    pred = model.predict(images[j].numpy().reshape(1,224,224,3))
    pred = np.argmax(pred)
    if pred != labels[j]:
      ax = plt.subplot(3, 3, i + 1)
      plt.imshow(images[j].numpy().astype("uint8"))
      plt.title(data_val.class_names[pred])
      plt.axis("off")
      i+=1
    j+=1

import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

train_dir = "train"
test_dir = "test"

labels = ("angry", "not_angry")

# image batch generators

data_train = tf.keras.utils.image_dataset_from_directory(train_dir,batch_size=16,image_size =(48,48))
data_gen_train = data_train.as_numpy_iterator()
batch_train = data_gen_train.next()

data_test = tf.keras.utils.image_dataset_from_directory(test_dir,batch_size=32,image_size =(48,48))
data_gen_test = data_test.as_numpy_iterator()
batch_test = data_gen_test.next()

# show a full batch of labeled train images

images, cls = batch_train
fig, axs = plt.subplots(4, 4, figsize=(12, 12))
axs = axs.flatten()

for i in range(len(images)):
    axs[i].imshow(images[i].astype("uint8"))
    axs[i].set_title(f"Label: {labels[cls[i]]}")
    axs[i].axis("off")

plt.show()

# image preprocessing & splitting

data_train = data_train.map(lambda x, y: (x/255, y))
data_train.as_numpy_iterator().next()

data_test = data_test.map(lambda x, y: (x/255, y))
data_test.as_numpy_iterator().next()

train_dataset_size = int(len(data_train) * 0.8)
validation_dataset_size = int(len(data_train) * 0.2)

train_dataset = data_train.take(train_dataset_size)
validation_dataset = data_train.skip(train_dataset_size).take(validation_dataset_size)

# CNN architecture design

CNN_model = Sequential()

CNN_model.add(Conv2D(16,(3,3),1, activation='relu', input_shape=(48,48,3)))
CNN_model.add(MaxPooling2D(pool_size=(2,2)))
CNN_model.add(Dropout(0.1))

CNN_model.add(Conv2D(32, (3,3),1,activation='relu'))
CNN_model.add(MaxPooling2D(pool_size=(2,2)))
CNN_model.add(Dropout(0.2))

CNN_model.add(Conv2D(16, (3,3),1,activation='relu'))
CNN_model.add(MaxPooling2D(pool_size=(2,2)))
CNN_model.add(Dropout(0.1))

CNN_model.add(Flatten())

CNN_model.add(Dense(256, activation='relu'))
CNN_model.add(Dense(1, activation='sigmoid'))

CNN_model.compile(optimizer=Adam(learning_rate=0.0001),loss=BinaryCrossentropy(), metrics=['accuracy'])
history = CNN_model.fit(train_dataset, epochs=50, validation_data=validation_dataset)

#version 1.0.1
#accuracy = 85.13% la validare

plt.subplot(121)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')

plt.subplot(122)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')

plt.show()

CNN_model.save('emotion_detection_cnn_50epochs_Dropout_v102.h5', include_optimizer=True)

# saved_model = load_model('emotion_detection_cnn_50epochs_Dropout_v102.h5',compile=False)


predicted_labels = []
batch = data_test.as_numpy_iterator().next()
test_images, test_labels = batch
predictions = CNN_model.predict(test_images)

for p in predictions:
    if p < 0.5:
        predicted_labels.append(0)
    else:
        predicted_labels.append(1)

print(CNN_model.evaluate(data_test))
print(classification_report(test_labels,predicted_labels))

cm = confusion_matrix(test_labels, predicted_labels)
plt.figure()
plt.title('version 1.0.2')
sns.heatmap(cm,annot=True,fmt='g', xticklabels=['angry','not_angry'], yticklabels=['angry','not_angry'])
plt.show()



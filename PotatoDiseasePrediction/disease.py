import tensorflow as tf
from keras.src.layers import Reshape, SimpleRNN, LSTM, Dropout
from tensorflow.keras import models,layers
import matplotlib.pyplot as plt
from keras.layers import SimpleRNN, Dense, Flatten, Reshape
from keras.models import Sequential


dataset= tf.keras.preprocessing.image_dataset_from_directory(
    "PlantVillage",
    shuffle=True,
    image_size=(256,256),
    batch_size=32
)
class_names=dataset.class_names
print(class_names)
for image_batch,label_batch in dataset.take(1):
    # print(image_batch.numpy())
    n=12

    for i in range (n):
        ax = plt.subplot(3, 4, i+1)
        plt.imshow(image_batch[i].numpy().astype("uint8"))
        plt.title(class_names[label_batch[i]])
        plt.axis("off")
    plt.show()
# train_ds=0.8*len(dataset)
# print(train_ds)
train_ds=dataset.take(54)
validation_ds=dataset.skip(54).take(6)
test_ds=dataset.skip(60).take(8)

# size=len(train_ds)+len(validation_ds)+len(test_ds)
# print(size) ::::::::::: 68
# ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
#
train_ds=train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
validation_ds=validation_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds=test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

resize_and_rescale=tf.keras.Sequential([
    layers.experimental.preprocessing.Resizing(256,256),
    layers.experimental.preprocessing.Rescaling(1.0/255)
])
data_augmentation=tf.keras.Sequential([
    layers.experimental.preprocessing.RandomRotation(0.2),
    layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical")
])
input_shape=(32,256,256,3)

# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------
#  ANN  model Building
# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------

model_ann=models.Sequential([
    resize_and_rescale,
    data_augmentation,
    layers.Flatten(),
    layers.Dense(256,activation='tanh'),
    layers.Dense(128,activation='tanh'),
    layers.Dense(64, activation='tanh'),
    layers.Dense(32,activation='tanh'),
    layers.Dense(len(class_names),activation='softmax')
])


# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------
#  CNN model Building using sgd
# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------
model=models.Sequential([
    resize_and_rescale,
    data_augmentation,
    layers.Conv2D(32,3,(2,2),padding="valid",activation='relu'),
    layers.MaxPooling2D(),
    # layers.VGG16(weights='imagenet', include_top=False, input_shape=input_shape),
    layers.Dropout(0.05),
    layers.Flatten(),
    layers.Dense(64,activation='relu'),
    layers.Dense(len(class_names),activation='softmax')
])

# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------
#  CNN model Building using adam
# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------

model1=models.Sequential([
    resize_and_rescale,
    data_augmentation,
    layers.Conv2D(32,3,(2,2),padding="valid",activation='relu'),
    layers.MaxPooling2D(),
    # layers.VGG16(weights='imagenet', include_top=False, input_shape=input_shape),
    layers.Dropout(0.05),
    layers.Flatten(),
    layers.Dense(64,activation='relu'),
    layers.Dense(len(class_names),activation='softmax')
])

# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------
#  RNN model Building
# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------


input_shapes = (256, 256, 3)
model2 = Sequential()
model2.add(Reshape((input_shapes[0] * input_shapes[1], input_shapes[2]), input_shape=input_shapes))
model2.add(SimpleRNN(128))
model2.add(Dropout(0.20))
model2.add(Dense(64, activation='relu'))
model2.add(Dense(len(class_names), activation='softmax'))

model2.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(),metrics=['accuracy'])

# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------
#  LSTM model Building
# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------

input_shapes = (256, 256, 3)
model_lstm= Sequential()
model_lstm.add(Reshape((input_shapes[0] * input_shapes[1], input_shapes[2]), input_shape=input_shapes))
model_lstm.add(Dropout(0.40))
model_lstm.add(LSTM(16))
model_lstm.add(Dense(16, activation='relu'))
model_lstm.add(Dense(len(class_names), activation='softmax'))

model_lstm.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(),metrics=['accuracy'])



# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------
#  Autoencoders model Building
# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------
# def create_autoencoder(input_shape=(256, 256, 3)):
#     input_img = tf.keras.Input(shape=input_shape)
#     x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
#     x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
#     x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
#     x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
#     encoded = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
#
#     x = tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=2, activation='relu', padding='same')(encoded)
#     x = tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same')(x)
#     decoded = tf.keras.layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
#
#     autoencoder = tf.keras.Model(input_img, decoded)
#     autoencoder.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(),metrics=['accuracy'])
#
#     return autoencoder
#
# autoencoder = create_autoencoder()
# autoencoder.summary()
#
# history = autoencoder.fit(
#     train_ds,
#     epochs=5,
#     validation_data=validation_ds,
#     batch_size=32,verbose=1
# )


# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------
#  ANN  model Building using adam optimizer
# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------

model_ann.build(input_shape=input_shape)

model_ann.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(),metrics=['accuracy'])
print(model_ann.summary())
history_ann=model_ann.fit(train_ds,epochs=7,batch_size=32,verbose=1,validation_data=validation_ds)

plt.plot(history_ann.history['loss'], label='train_loss')
plt.plot(history_ann.history['val_loss'], label='val_loss')
plt.title('training and validation loss using ANN')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show(block=True)

# plot accuracy
plt.plot(history_ann.history['accuracy'], label='train_acc')
plt.plot(history_ann.history['val_accuracy'], label='val_acc')
plt.title('training and validation accuracy using ANN ')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.show(block=True)


# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------
#  CNN model using SGD optimizer
# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------


model.build(input_shape=input_shape)

model.compile(optimizer='sgd',loss=tf.keras.losses.SparseCategoricalCrossentropy(),metrics=['accuracy'])
print(model.summary())
history=model.fit(train_ds,epochs=5,batch_size=32,verbose=1,validation_data=validation_ds)
print(history.history)
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('training and validation loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show(block=True)

# plot accuracy
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.title('training and validation accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.show(block=True)


# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------
#  CNN model using adam optimizer
# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------



model1.build(input_shape=input_shape)
model1.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(),metrics=['accuracy'])
print(model.summary())
history1=model1.fit(train_ds,epochs=5,batch_size=32,verbose=1,validation_data=validation_ds)
print(history1.history)

plt.plot(history1.history['loss'], label='train_loss')
plt.plot(history1.history['val_loss'], label='val_loss')
plt.title('training and validation loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show(block=True)

# plot accuracy
plt.plot(history1.history['accuracy'], label='train_acc')
plt.plot(history1.history['val_accuracy'], label='val_acc')
plt.title('training and validation accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.show(block=True)


# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------
#  RNN model using adam optimizer
# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------


model2.build(input_shape=input_shape)

history2=model2.fit(train_ds,epochs=5,batch_size=32,verbose=1,validation_data=validation_ds)
print(history2.history)

plt.plot(history2.history['loss'], label='train_loss')
plt.plot(history2.history['val_loss'], label='val_loss')
plt.title('training and validation loss for RNN using adam optimizer')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show(block=True)

# plot accuracy
plt.plot(history2.history['accuracy'], label='train_acc')
plt.plot(history2.history['val_accuracy'], label='val_acc')
plt.title('training and validation accuracy for RNN using adam optimizer')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.show(block=True)


# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------
#  LSTM model using adam optimizer
# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------

model_lstm.build(input_shape=input_shape)
history_lstm=model_lstm.fit(train_ds,epochs=5,batch_size=32,verbose=1,validation_data=validation_ds)
print(history_lstm.history)

plt.plot(history_lstm.history['loss'], label='train_loss')
plt.plot(history_lstm.history['val_loss'], label='val_loss')
plt.title('training and validation loss for LSTM using adam optimizer')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show(block=True)

# plot accuracy
plt.plot(history_lstm.history['accuracy'], label='train_acc')
plt.plot(history_lstm.history['val_accuracy'], label='val_acc')
plt.title('training and validation accuracy for LSTM using adam optimizer')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.show(block=True)


# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------
# Different Models Accuracy Comparision
# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------

histories = []
histories.append(history_ann) #history is for Adam optimizer---> ANN model
histories.append(history)  # history is for SGD optimizer---> CNN model
histories.append(history1) # history is for Adam optimizer---> CNN model
histories.append(history2) # history is for Adam optimizer---> RNN model
histories.append(history_lstm) #history is for adam optimizer -------> LSTM model
titles=["ANN","CNN_SGD","CNN_ADAM","RNN","LSTM"]
for i, history in enumerate(histories):
    plt.plot(history.history['accuracy'], label=f'Model {titles[i]}')
plt.title('Different Models Accuracy Comparision')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show(block=True)
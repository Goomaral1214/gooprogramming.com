import tensorflow as tf
# from tensorflow.keras.models import load_model
import os
from matplotlib import pyplot as plt
import numpy as np
#setting GPU memory consumption growth
# gpus = tf.config.experimental.list_physical_devices('CPU')

# gpus = tf.config.exprimental.list_physical_devices('GPU')

# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)


import cv2
import imghdr

data_dir = 'data'
os.listdir(data_dir)
os.listdir(os.path.join(data_dir, 'happy_people'))
#put picture in a number
img = cv2.imread(os.path.join('data', 'happy_people', 'business-people-succesful-celebrating-group-successful-39416686-800x500.jpg'))
# print(img.shape) #size of the image
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# plt.imshow(img)
# plt.show()

image_exts = ['jpeg', 'jpg', 'bmp', 'png' ]
image_exts[2]

for image_class in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)
        try:
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)
            if tip not in image_exts:
                print('Image not in ext list')
                os.remove(image_path)
        except Exception as e:
            print('Issue with image []'.format(image_path))

#Load data deep learning model
data = tf.keras.utils.image_dataset_from_directory('data')

# data_iterator = data.as_numpy_iterator()

# # print(data_iterator)
# batch = data_iterator.next()
# print(len(batch[0]))
#image represented as numpy arrays
# print(batch[0].shape)
#1 stands for sad, 0 stands for happy
# print(batch[1]) 


#Preprocess Data
#scale data
data = data.map(lambda x, y: (x/255, y))
# print(data)
scaled_iterator = data.as_numpy_iterator()
batch = scaled_iterator.next()
fig, ax = plt.subplots(ncols = 4, figsize = (20, 20))
# for idx, img in enumerate(batch[0][:4]):
#     ax[idx].imshow(img)
#     ax[idx].title.set_text(batch[1][idx])
# print(batch[0].min())
# print(batch[0].max())


#Split data into training and testing
# print(len(data))
train_size = int(len(data)*.7)  
val_size = int(len(data)*.2)+1
test_size = int(len(data)*.1)+1
# print(train_size, val_size, test_size)
train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)
# print(len(train))
# print(len(val))
# print(len(test))


#Build deep learning Model CNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

model = Sequential()
model.add(Conv2D(16, (3, 3), 1, activation = 'relu', input_shape = (256, 256, 3)))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3, 3), 1, activation = 'relu'))
model.add(MaxPooling2D())
model.add(Conv2D(16, (3, 3), 1, activation = 'relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(256, activation = 'relu'))
model.add(Dense(1, activation ='sigmoid'))
model.compile('adam', loss = tf.losses.BinaryCrossentropy(), metrics = ['accuracy'])
# print(model.summary())  #to get main information

#Train the data
logdir = 'logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logdir)
hist = model.fit(train, epochs = 20, validation_data = val, callbacks=[tensorboard_callback])

#Plot performance
fig = plt.figure()
plt.plot(hist.history['accuracy'], color = 'teal', label = 'loss')
plt.plot(hist.history['val_accuracy'], color = 'orange', label = 'val_loss')
fig.suptitle('Accuracy', fontsize = 20)
plt.legend(loc ="upper left")
plt.show()

#Evaluate Performance

# pre = Precision()
# re = Recall()
# acc = BinaryAccuracy()

# print(len(test))
# for batch in test.as_numpy_iterator():
#     x, y = batch
#     yhat = model.predict(x)
#     pre.update_state(y, yhat)
#     re.update_state(y, yhat)
#     acc.update_state(y, yhat)
# print(f'Precision{pre.result().numpy()}, Recall: {re.result().numpy()}, Accuracy: {acc.result().numpy}')


#Test performance 

img = cv2.imread('sadtest.jpg')
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#change the size to 256 for neural networks
resize = tf.image.resize(img, (256, 256))
plt.imshow(resize.numpy().astype(int))
plt.show()



print(np.expand_dims(resize, 0).shape)
yhat = model.predict(np.expand_dims(resize/255, 0))
print(yhat)

if yhat > 0.5:
    print(f'Predicted class is sad')
else:
    print(f'Predicted class is happy')

from tensorflow.keras.models import load_model
model.save(os.path.join('models', 'happysadmodel_google.hS'))


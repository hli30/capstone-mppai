# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 15:01:53 2019

@author: Harry
"""

import math
import os.path
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from keras.losses import categorical_crossentropy

# =============================================================================
# import os
# import shutil
# 
# # Move files to subfolders to conform to Keras requirement for ImageDataGenerator
# 
# y_train = pd.read_csv('train_labels.csv')
# 
# infected = y_train.loc[y_train['infected'] == 1, 'filename']
# not_infected = y_train.loc[y_train['infected'] == 0, 'filename']
# 
# srcpath = './train'
# destpath = './train_split_by_class/not_infected'
# 
# for filename in not_infected:
#     if os.path.isfile(os.path.join(srcpath, filename)):
#         shutil.copy(os.path.join(srcpath, filename), destpath)
# 
# print('complete')
# =============================================================================

# =============================================================================
# Preprocess
# =============================================================================
y_train = pd.read_csv('train_labels.csv')
steps = math.ceil(y_train.shape[0]*0.8 / 32)
val_steps = math.ceil(y_train.shape[0]*0.2 / 32)

imgen = ImageDataGenerator(validation_split=0.2, rescale=1./255)
test_imgen = ImageDataGenerator(rescale=1./255)

train_generator = imgen.flow_from_directory(
        './train_split_by_class',
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical',
        subset='training')

validation_generator = imgen.flow_from_directory(
        './train_split_by_class',
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical',
        subset='validation')

test_generator = test_imgen.flow_from_directory(
        './test',
        target_size=(128, 128),
        batch_size=1,
        class_mode=None,
        shuffle=False)

# =============================================================================
# Model
# =============================================================================
input_dim = (128,128,3)

model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = input_dim))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(2, activation = "softmax"))

model.compile(loss=categorical_crossentropy, optimizer='adam', metrics=['accuracy'])

model.fit_generator(
        train_generator, 
        epochs=5, 
        validation_data=validation_generator,
        steps_per_epoch=steps,
        validation_steps=val_steps)

# =============================================================================
# Predict
# =============================================================================
pred = model.predict_generator(
        test_generator,
        verbose=1,
        steps=5793)

# =============================================================================
# Output
# =============================================================================
labels = (train_generator.class_indices)

filenames=test_generator.filenames

filenames = [name.replace(name, os.path.basename(name)) for name in filenames]

results=pd.DataFrame({"filename":filenames,
                      "infected":pred[:,0]})

results.to_csv('submit2.csv', index=False)


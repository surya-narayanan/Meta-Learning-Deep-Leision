
import zipfile
import sys
import util
import tqdm
import os
import tensorflow as tf
import numpy as np
import math
import timeit
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from skimage.measure import block_reduce

def extract_zips():
  
  directory_to_extract_to = '/home/Surya/images/Images_png'
  
  for i in tqdm.trange(1, 57):
    
    path_to_zip_file = '/home/Surya/images/Images_png_%02d.zip' %i
    
    print('extracting', path_to_zip_file)

    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(directory_to_extract_to)

    print('done extracting', path_to_zip_file)


def pre_process(f = 0.01):
  dl_info = util.read_dl_info() 

  json_labels = util.read_json_labels('/home/Surya/cs230/text_mined_labels_171_and_split.json')   

  dl_info_vector = util.read_dl_info_vector(
    image_dir = '../images/Images_png/',
    DL_INFO_PATH  = '/home/Surya/cs230/') 

  

  data = {}

  X_train = np.empty((0, 512, 512, 3), float)
  Y_train = np.empty((0, 1), float)
  X_val = np.empty((0, 512, 512, 3), float)
  Y_val = np.empty((0, 1), float)
  X_test = np.empty((0, 512, 512, 3), float)
  Y_test = np.empty((0, 1), float)

  downsample = True
  downsample_factor = 2
  
  if downsample:
    X_train = block_reduce(X_train, block_size=(1, downsample_factor, downsample_factor, 1), func=np.mean)
    X_val = block_reduce(X_val, block_size=(1, downsample_factor, downsample_factor, 1), func=np.mean)
    X_test = block_reduce(X_test, block_size=(1, downsample_factor, downsample_factor, 1), func=np.mean)

  for i in range(1, 9):
    x_train,y_train,x_val,y_val,x_test,y_test = util.data_load(dl_info, dl_info_vector, json_labels, i, f=f)
    
    #Balance the classes
    x_train = x_train[:200, :, :, :]
    x_val = x_val[:20, :, :, :]
    x_test = x_test[:20, :, :, :]

    #Downsample
    if downsample :
      x_train = block_reduce(x_train, block_size=(1, downsample_factor, downsample_factor, 1), func=np.mean)
      x_val = block_reduce(x_val, block_size=(1, downsample_factor, downsample_factor, 1), func=np.mean)
      x_test = block_reduce(x_test, block_size=(1, downsample_factor, downsample_factor, 1), func=np.mean)

    y_train = np.array([i for _ in range(len(y_train))]).reshape(-1,1)
    y_val = np.array([i for _ in range(len(y_val))]).reshape(-1, 1)
    y_test = np.array([i for _ in range(len(y_test))]).reshape(-1, 1)

    X_train =  np.append(X_train, x_train, axis=0)
    Y_train = np.append(Y_train, y_train, axis=0)
    X_val = np.append(X_val, x_val, axis=0)
    Y_val = np.append(Y_val, y_val, axis=0)
    X_test = np.append(X_test, x_test, axis=0)
    Y_test = np.append(Y_test, y_test, axis=0)

  shuffled_train_indices = list(range(len(X_train)))
  np.random.shuffle(shuffled_train_indices)
  
  X_train = np.squeeze(X_train[shuffled_train_indices] ) 
  Y_train = np.squeeze(Y_train[shuffled_train_indices] ) 

  shuffled_val_indices = list(range(len(X_val)))
  np.random.shuffle(shuffled_val_indices)
  
  X_val = np.squeeze(X_val[shuffled_val_indices] ) 
  Y_val = np.squeeze(Y_val[shuffled_val_indices] ) 

  # print(Y_val)

  shuffled_test_indices = list(range(len(X_test)))
  np.random.shuffle(shuffled_test_indices)
  
  X_test = np.squeeze(X_test[shuffled_test_indices] ) 
  Y_test = np.squeeze(Y_test[shuffled_test_indices] ) 

  # print(Y_test)

  # print(shuffled_train_indices)
  # print(shuffled_val_indices)
  # print(shuffled_test_indices)
  
  # Using only 1 slice to conserve RAM footprint

  # X_train = X_train[:, :, :, 1]
  # X_val = X_val[:, :, :, 1]
  # X_test = X_test[:, :, :, 1]

  #
  # x = np.array(X_train[0, :, :, 1])
  
  # print('size of sample image', sys.getsizeof(x))
  
  return X_train, Y_train, X_val, Y_val, X_test, Y_test
  
def train():
  X_train, Y_train, X_val, Y_val, X_test, Y_test = pre_process(f = .15)

  input_shape = (256, 256, 3)
  num_classes = 9

  initializer = tf.initializers.VarianceScaling(scale=2.0)

  model = tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet', input_shape=input_shape)

  # print(model.summary())

  model.layers.pop()
  model.outputs = [model.layers[-1].output]
  output = model.get_layer('Conv_1_bn').output #Conv_1_bn for Mobilenetv2, block5_pool for vgg19
  output = tf.keras.layers.Flatten()(output)
  output = tf.keras.layers.Dense(num_classes, activation='softmax',
                            kernel_initializer=initializer)(output)
  new_model = tf.keras.Model(model.input, output)

  # print(new_model.summary())

  optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
  
  new_model.compile(optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=[tf.keras.metrics.sparse_categorical_accuracy])
  
  new_model.fit(X_train, Y_train, batch_size=8, epochs=5, validation_data=(X_val, Y_val))
  
  new_model.evaluate(X_test, Y_test)

  preds = new_model.predict(X_test)

  preds = np.argmax(preds, axis = 1)
  
  confusion = confusion_matrix(Y_test, preds)
  print('Confusion Matrix\n')
  print(confusion)

  print(Y_test, preds)

if __name__ == '__main__':
  train()
  # pre_process(f = .05)
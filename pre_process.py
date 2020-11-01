
import zipfile
import sys
sys.path.insert(1, '/home/Surya/cs230/')
import util
import tqdm
import os
import tensorflow as tf
import numpy as np
import math
import timeit
import matplotlib.pyplot as plt

def extract_zips():
  
  directory_to_extract_to = '/home/Surya/images/Images_png'
  
  for i in tqdm.trange(1, 57):
    
    path_to_zip_file = '/home/Surya/images/Images_png_%02d.zip' %i
    
    print('extracting', path_to_zip_file)

    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(directory_to_extract_to)

    print('done extracting', path_to_zip_file)


def pre_process():
  dl_info = util.read_dl_info() 

  json_labels = util.read_json_labels('/home/Surya/cs230/text_mined_labels_171_and_split.json')   

  dl_info_vector = util.read_dl_info_vector(
    image_dir = '../images/Images_png/',
    DL_INFO_PATH  = '/home/Surya/cs230/') 

  num_classes_to_construct = 2

  data = {}

  X_train = np.empty((0, 512, 512, 3), float)
  Y_train = np.empty((0, ), float)
  X_val = np.empty((0, 512, 512, 3), float)
  Y_val = np.empty((0, ), float)
  X_test = np.empty((0, 512, 512, 3), float)
  Y_test = np.empty((0, ), float)

  for i in range(1, num_classes_to_construct):
    x_train,y_train,x_val,y_val,x_test,y_test = util.data_load(dl_info, dl_info_vector, json_labels, i, f=0.01)
    
    y_train = np.array([i for _ in range(len(y_train))])
    y_val = np.array([i for _ in range(len(y_train))])
    y_test = np.array([i for _ in range(len(y_test))])

    X_train =  np.append(X_train, x_train, axis=0)
    Y_train = np.append(Y_train, y_train, axis=0)
    X_val = np.append(X_val, x_val, axis=0)
    Y_val = np.append(Y_val, y_val, axis=0)
    X_test = np.append(X_test, x_test, axis=0)
    Y_test = np.append(Y_test, y_test, axis=0)

  shuffled_train_indices = np.random.shuffle(list(range(len(X_train))))

  # X_train = X_train[shuffled_train_indices] 
  # print(X_train.shape)

  # Y_train = Y_train[shuffled_train_indices] 

  return X_train, Y_train, X_val, Y_val, X_test, Y_test
  
def train():
  X_train, Y_train, X_val, Y_val, X_test, Y_test = pre_process()

  print(X_train.shape)

  input_shape = (512, 512, 3)
  num_classes = 2

  initializer = tf.initializers.VarianceScaling(scale=2.0)

  model = tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet', input_shape=input_shape)

  print(model.summary())

  model.layers.pop()
  model.outputs = [model.layers[-1].output]
  output = model.get_layer('Conv_1_bn').output
  output = tf.keras.layers.Flatten()(output)
  output = tf.keras.layers.Dense(num_classes, activation='softmax',
                            kernel_initializer=initializer)(output)
  new_model = tf.keras.Model(model.input, output)

  # print(new_model.summary())

  optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
  new_model.compile(optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=[tf.keras.metrics.sparse_categorical_accuracy])
  new_model.fit(X_train, Y_train, batch_size=512, epochs=5, validation_data=(X_val, Y_val))
  new_model.evaluate(X_test, Y_test)

if __name__ == '__main__':
  train()
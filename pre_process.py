
import zipfile
import sys
import util
from tqdm import tqdm, trange
import os
import numpy as np
import math
import timeit
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from skimage.measure import block_reduce
import pickle
import wandb

NUM_CLASSES = 8


def extract_zips():
  
  directory_to_extract_to = '/home/Surya/images/Images_png'
  
  for i in tqdm.trange(1, 57):
    
    path_to_zip_file = '/home/Surya/images/Images_png_%02d.zip' %i
    
    print('extracting', path_to_zip_file)

    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(directory_to_extract_to)

    print('done extracting', path_to_zip_file)


def pre_process(fs = [0.01 for _ in range(1, 9)], downsample = True, save = True, load = True):
  if load:    

      X_train = pickle.load( open( "X_train.p", "rb" ) )      
      print('loaded X_Train')

      Y_train = pickle.load( open( "Y_train.p", "rb" ) )      
      print('loaded Y_train')

      X_val = pickle.load( open( "X_val.p", "rb" ) )      
      print('loaded X_val')

      Y_val = pickle.load( open( "Y_val.p", "rb" ) )
      print('loaded Y_val')

      X_test = pickle.load( open( "X_test.p", "rb" ) )
      print('loaded X_test')

      Y_test = pickle.load( open( "Y_test.p", "rb" ) )
      print('loaded Y_test')

      return X_train, Y_train, X_val, Y_val, X_test, Y_test
    
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
  
  Y_train_bb = np.empty((0, 4), float)

  downsample_factor = 2
  
  if downsample:
    X_train = block_reduce(X_train, block_size=(1, downsample_factor, downsample_factor, 1), func=np.mean)
    X_val = block_reduce(X_val, block_size=(1, downsample_factor, downsample_factor, 1), func=np.mean)
    X_test = block_reduce(X_test, block_size=(1, downsample_factor, downsample_factor, 1), func=np.mean)

  # Use these fs to preserve same number of images across
  
  for i in range(1, 9):
    x_train,y_train_bb,x_val,y_val_bb,x_test,y_test_bb = util.data_load(dl_info, dl_info_vector, json_labels, i, f=fs[i-1])
    
    #Convert to float 32
    x_train, x_val, x_test = [np.float32(a) for a in [x_train, x_val, x_test]]  

    #Balance the classes
    # x_train = x_train[:200, :, :, :]
    # x_val = x_val[:20, :, :, :]
    # x_test = x_test[:20, :, :, :]

    # y_train = y_train[:200, :]
    # y_val = y_val[:20, :]
    # y_test = y_test[:20, :]

    #Downsample
    if downsample :
      x_train = block_reduce(x_train, block_size=(1, downsample_factor, downsample_factor, 1), func=np.mean)
      x_val = block_reduce(x_val, block_size=(1, downsample_factor, downsample_factor, 1), func=np.mean)
      x_test = block_reduce(x_test, block_size=(1, downsample_factor, downsample_factor, 1), func=np.mean)

    y_train = (i-1) * np.ones((len(x_train), 1)) #classes are 1 indexed
    y_val = (i-1) * np.ones((len(x_val), 1)) #classes are 1 indexed
    y_test = (i-1) * np.ones((len(x_test), 1)) #classes are 1 indexed

    X_train =  np.append(X_train, x_train, axis=0)
    Y_train = np.append(Y_train, y_train, axis=0)
    X_val = np.append(X_val, x_val, axis=0)
    Y_val = np.append(Y_val, y_val, axis=0)
    X_test = np.append(X_test, x_test, axis=0)
    Y_test = np.append(Y_test, y_test, axis=0)

    Y_train_bb = np.append(Y_train_bb, y_train_bb, axis=0)
    # print(Y_test.shape)

  shuffled_train_indices = list(range(len(X_train)))
  np.random.shuffle(shuffled_train_indices)
  
  X_train = np.squeeze(X_train[shuffled_train_indices] ) 
  Y_train = np.squeeze(Y_train[shuffled_train_indices] ) 
  Y_train_bb = np.squeeze(Y_train_bb[shuffled_train_indices] ) 

  shuffled_val_indices = list(range(len(X_val)))
  np.random.shuffle(shuffled_val_indices)
  
  X_val = np.squeeze(X_val[shuffled_val_indices] ) 
  Y_val = np.squeeze(Y_val[shuffled_val_indices] ) 

  # print(Y_val)

  shuffled_test_indices = list(range(len(X_test)))

  # print('before shuffling', shuffled_test_indices)

  np.random.shuffle(shuffled_test_indices)

  # print('after shuffling', shuffled_test_indices)
  
  # print('Y_test shape', Y_test[shuffled_test_indices].shape)

  # print(Y_test[shuffled_test_indices])

  X_test = np.squeeze(X_test[shuffled_test_indices] ) 
  Y_test = np.squeeze(Y_test[shuffled_test_indices] ) 

  
  # print('size of sample image', sys.getsizeof(x))
    
  
  if save:
    pickle.dump( X_train, open( "X_train.p", "wb" ) , protocol=4)
    pickle.dump( Y_train, open( "Y_train.p", "wb" ) , protocol=4)
    pickle.dump( X_val, open( "X_val.p", "wb" ) , protocol=4)
    pickle.dump( Y_val, open( "Y_val.p", "wb" ) , protocol=4)
    pickle.dump( X_test, open( "X_test.p", "wb" ) , protocol=4)
    pickle.dump( Y_test, open( "Y_test.p", "wb" ) , protocol=4)
    pickle.dump( Y_train_bb, open( "Y_train_bb.p", "wb" ) , protocol=4)
  
  return X_train, Y_train, X_val, Y_val, X_test, Y_test

def sample_from(x, y, k, classes, frac_data = 1):

  #returns n samples of x, y of class = class_num
  # X : N, D, D, 3, 
  # Y : N
  
  n_classes = len(classes)
  kn = k * n_classes
  
  D = x.shape[1]

  X = np.empty((0, D, D, 3), numpy.float32)
  Y = np.empty((0, NUM_CLASSES), numpy.float32)

  labels = np.eye(NUM_CLASSES)
  counter = 0
  
  print('in sample from')
  print('classes sampling: ', classes)
  
  for class_num in tqdm(classes):

    # print(X.shape)

    x_class = np.float32(x[y == class_num])
    
    indices = list(range(len(x_class)))
    np.random.shuffle(indices)
    
    # Error handling when you try to sample more images than you have.
    k_ = int(min(k, len(indices)) * frac_data)
    
    print('adding ', k_, 'images of class', class_num)

    selected_samples = indices[:k_]

    #testing overfitting
    # selected_samples = [0 for _ in range(k)]
    #

    X = np.append(X, x_class[selected_samples], axis = 0)
    Y = np.append(Y, np.tile(labels[class_num], (k_ , 1)), axis = 0)

    # print('labels[counter]', labels[counter])

  print('sampled data , sample sizes are ', X.shape, Y.shape )

  return X, Y

def shuffle(x, y):
    shuffled_train_indices = list(range(len(x)))
    np.random.shuffle(shuffled_train_indices)
    
    x = np.squeeze(x[shuffled_train_indices] ) 
    y = np.squeeze(y[shuffled_train_indices] ) 

    return x, y

if __name__ == '__main__':
  
  five_hundred_images = [0.10539629, 0.175315568, 0.12025012, 0.192233756, 0.201288245, 0.151975684, 0.257201646, 0.330033003]
  
  pre_process(fs = [0.75 * i for i in five_hundred_images], downsample = False, save = True, load = False)
  
  pass
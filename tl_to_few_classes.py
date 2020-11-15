from pre_process import *
import tensorflow as tf

hyperparameter_defaults = dict(
  loglearning_rate_lr = -2,    
  logepochs = 0,
  log_batch_size = 2,
  frac_data = .1,
  frac_classes = NUM_CLASSES//2
)

wandb.init(config=hyperparameter_defaults)
config = wandb.config

def train_vgg_tl_across_classes():

  print('train_vgg_tl_across_classes')

  print('getting', config.frac_data, 'fraction of data')
  
  X_train_big, Y_train, X_val, Y_val, X_test, Y_test = pre_process(load = True)

  # print('Y_test', Y_test)
  # print('Y_val', Y_val)
  
  print('len Y_Train', len(Y_train))
  print('len Y_test', len(Y_test))
  print('len Y_val', len(Y_val))
  
  #Choose certain classes to test on
  all_classes = list(range(NUM_CLASSES))  
  np.random.shuffle(all_classes)  
  print('number of classes sampled', config.frac_classes)  
  select_classes_test = all_classes[:config.frac_classes]
  
  #Sample from X and Y for a certain fraction of Test/Train data for a certain set of classes
  k = int(len(X_train_big)/NUM_CLASSES)  
  X_train, Y_train = sample_from(X_train_big, Y_train, k, all_classes, frac_data = config.frac_data) 
        
  del X_train_big
  
  k = int(len(X_val)/NUM_CLASSES)
  X_val, Y_val = sample_from(X_val, Y_val, k, select_classes_test, frac_data = config.frac_data) 

  k = int(len(X_test)/NUM_CLASSES)
  X_test, Y_test = sample_from(X_test, Y_test, k, select_classes_test, frac_data = config.frac_data) 
  
  #shuffle the data

  X_train, Y_train = shuffle(X_train, Y_train)
  X_val, Y_val = shuffle(X_val, Y_val)
  X_test, Y_test = shuffle(X_test, Y_test)

  X_train, Y_train, X_val, Y_val, X_test, Y_test = [np.float32(a) for a in [X_train, Y_train, X_val, Y_val, X_test, Y_test]]  

  print('distribution of Y_test', np.sum(Y_test, axis = 0))
  print('distribution of Y_val', np.sum(Y_val, axis = 0))
  
  """inference network
  """

  input_shape = (512, 512, 3)
  num_classes = 8

  initializer = tf.initializers.VarianceScaling(scale=2.0)

  model = tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet', input_shape=input_shape)

  # print(model.summary())

  #Make model not trainable
  model.trainable = True

  model.layers.pop()
  model.outputs = [model.layers[-1].output]
  output = model.get_layer('Conv_1_bn').output #Conv_1_bn for Mobilenetv2, block5_pool for vgg19  
  # Surya Layers
  output = tf.keras.layers.Conv2D(4, 3)(output)
  #
  output = tf.keras.layers.Flatten()(output)
  output = tf.keras.layers.Dense(num_classes, activation='softmax',
                            kernel_initializer=initializer)(output)
  new_model = tf.keras.Model(model.input, output)

  # print(new_model.summary())

  optimizer = tf.keras.optimizers.Adam(learning_rate=10**config.loglearning_rate_lr)
  
  # new_model.compile(optimizer=optimizer,
  #               loss='sparse_categorical_crossentropy',
  #               metrics=[tf.keras.metrics.sparse_categorical_accuracy])

  new_model.compile(optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=[tf.keras.metrics.categorical_accuracy])
  
  
  history = new_model.fit(X_train, Y_train, batch_size=2**config.log_batch_size, epochs=min(2**config.logepochs, 8), validation_data=(X_val, Y_val))
  
  metrics = {'val_acc': history.history['val_categorical_accuracy'][-1],
   'val_loss': history.history['val_loss'][-1],
   'loss': history.history['loss'][-1],
   'acc': history.history['categorical_accuracy'][-1]}

  wandb.log(metrics)

  new_model.evaluate(X_test, Y_test)

  preds = new_model.predict(X_test)
  
  # print('before the arg max', preds)

  preds = np.argmax(preds, axis = 1)
  
  print('after the arg max', preds)

  y_test_classes = np.argmax(Y_test, axis = 1)
  
  print('y_test_classes')
  print(y_test_classes)
  
  confusion = confusion_matrix(y_test_classes, preds)

  print('Confusion Matrix\n')

  print(confusion)

  # print(Y_test, preds)

if __name__ == '__main__':
  train_vgg_tl_across_classes()
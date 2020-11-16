from pre_process import *
import tensorflow as tf

def train_vgg_snail():
  
  hyperparameter_defaults = dict(
    log_lr = -2,
    log_meta_lr = -3,
    epochs = 10,
    log_LSTM_HIDDEN_UNITS = 7,
    n_mt_samples = 25,
    k_shot = 3,
    n_mt_classes = 3
    )

  # Pass your defaults to wandb.init
  wandb.init(config=hyperparameter_defaults)
  config = wandb.config

  print('train_vgg_snail')

  X_train, Y_train, X_val, Y_val, X_test, Y_test = pre_process(load = True, save = False, downsample = True)

  X_train, Y_train, X_val, Y_val, X_test, Y_test = [np.float32(a) for a in [X_train, Y_train, X_val, Y_val, X_test, Y_test]]  

  print(Y_test)

  input_shape = (512, 512, 3)
  num_classes = NUM_CLASSES 

  initializer = tf.initializers.VarianceScaling(scale=2.0)

  model = tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet', input_shape=input_shape)
  # print(model.summary())

  # print(model.summary())

  model.layers.pop()
  model.outputs = [model.layers[-1].output]
  output = model.get_layer('Conv_1_bn').output #Conv_1_bn for Mobilenetv2, block5_pool for vgg19
  output = tf.keras.layers.Flatten()(output)
  new_model = tf.keras.Model(model.input, output)

  # print(new_model.summary())

  new_optimizer = tf.keras.optimizers.Adam(learning_rate=10**config.log_lr)
  
  new_model.compile(optimizer=new_optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=[tf.keras.metrics.sparse_categorical_accuracy])
  
  #meta train
  n_mt_classes = config.n_mt_classes
  k_shot = config.k_shot
  n_mt_samples = config.n_mt_samples
  
  classes = list(range(NUM_CLASSES))
  np.random.shuffle(classes)
  shuffled_classes = classes[:n_mt_classes] #indexing to select n classes for meta training

  print('classes during meta train', shuffled_classes)

  x_mt, y_mt = sample_from(X_train, Y_train, (k_shot + 1) * n_mt_samples, shuffled_classes)

  print(x_mt.shape)

  x_mt = x_mt.reshape(((k_shot + 1) * n_mt_classes, n_mt_samples, 512, 512, 3))
  y_mt = y_mt.reshape(((k_shot + 1) * n_mt_classes, n_mt_samples, NUM_CLASSES))

  y_mt = np.transpose(y_mt, (1, 0, 2))
  x_mt = np.transpose(x_mt, (1, 0, 2, 3, 4))

  # print(x_mt.shape, 'after reshape')
  # print('expected')
  # print((n_mt_samples, (k_shot + 1) * n_mt_classes, 256, 256, 3))

  embeddings = model.predict(x_mt.reshape((-1, 512, 512, 3)))  

  print('got embeddings')
  embeddings = embeddings.reshape((n_mt_samples, n_mt_classes, (k_shot + 1), -1))
  
  y_mt = y_mt.reshape((n_mt_samples, n_mt_classes, (k_shot + 1), -1))

  x_mt_support = np.concatenate([embeddings[:,:,:k_shot,:], y_mt[:,:,:k_shot,:]], axis = 3) # Append labels to support set embeddings
  x_mt_support = x_mt_support.reshape(((n_mt_samples, k_shot * n_mt_classes, -1))) #(n_mt_samples, k_shot * n_mt_classes, embedding + n_mt_classes)
  x_mt_support = np.repeat(x_mt_support, n_mt_classes, axis = 0)

  x_mt_query = np.concatenate([embeddings[:,:,-1,:], np.zeros((n_mt_samples, n_mt_classes, NUM_CLASSES))], axis = 2) #(n_mt_samples, n_mt_classes, embedding + n_mt_classes)
  x_mt_query = x_mt_query.reshape(n_mt_samples * n_mt_classes, 1, -1) #(n_mt_samples * n_mt_classes, 1, embedding + n_mt_classes)

  x_mt = np.concatenate((x_mt_support, x_mt_query), axis = 1)

  #Get the labels for the query set
  y_mt_query = y_mt[:,:,-1,:].reshape((n_mt_samples, 1 * n_mt_classes, -1))
  y_mt_query = y_mt_query.reshape((n_mt_samples * n_mt_classes, -1))

  # print("Expecting [10000], [01000]")
  # print(y_mt[0])
  # print(y_mt[1])
  # print(x_mt.shape)
  # print(y_mt.shape)

  B, K, D = x_mt.shape
  
  layers = [
    tf.keras.layers.LSTM(2**config.log_LSTM_HIDDEN_UNITS, input_shape = (K,D), return_sequences=True),
    tf.keras.layers.LSTM(NUM_CLASSES)
    # tf.keras.layers.Softmax()
  ]

  meta_model = tf.keras.Sequential(layers)

  meta_optimizer = tf.keras.optimizers.Adam(learning_rate=10**config.log_meta_lr)

  meta_model.compile(optimizer=meta_optimizer,
                loss='categorical_crossentropy', 
                metrics=[tf.keras.metrics.categorical_accuracy])

  # print(meta_model.summary())
  
  history = meta_model.fit(x_mt, y_mt_query, epochs = config.epochs)

  metrics = {'categorical_accuracy': history.history['categorical_accuracy'][-1]}
  wandb.log(metrics)

  preds = meta_model.predict(x_mt)
  
  print('before the arg max')
  print(preds)

  preds = np.argmax(preds, axis = 1)
  
  print('after the arg max')
  print(preds)

  confusion = confusion_matrix(np.argmax(y_mt_query, axis = 1), preds)

  print('Confusion Matrix\n')

  print(confusion)
  
  print('y_mt_query')
  print(y_mt_query)

  print('y_mt')
  print(y_mt)

  # print(Y_test, preds)


  # print(y_mt[0, 0, :])
  # print(y_mt[0, k_shot, :])
  # print(y_mt[0, k_shot + 1, :])
  # print(y_mt[0])

  # preds = new_model.predict(X_test)

  # preds = np.argmax(preds, axis = 1)
   
  # confusion = confusion_matrix(Y_test, preds)
  # print('Confusion Matrix\n')
  # print(confusion)

  # print(Y_test, preds)

if __name__ == '__main__':
  train_vgg_snail()
  pass
  
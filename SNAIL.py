from pre_process import *
import tensorflow as tf

hyperparameter_defaults = dict(
    log_inference_lr = -3.5,
    log_meta_lr = -3.5,
    dense_ml_model_epochs = 10,
    base_10_log_snail_epochs = 3,
    log_LSTM_HIDDEN_UNITS = 7,
    meta_batch_size = 5,
    k_shot = 5,
    n_mt_classes = 2,
    inference_log_batch_size = 3,
    inference_logepochs = 2,
    frac_data = 1,
    num_layers_dn_ml = 5,
    log_num_hidden_dn_ml = 8,
    mobilenet_early_stop_patience = 1,
    dense_ml_early_stop_patience = 3,
    snail_early_stop_patience = 80
    )

wandb.init(config=hyperparameter_defaults)
config = wandb.config


def train_vgg_snail():
  
  print('train_vgg_snail')

  """Load the data
  """
  X_train, Y_train, X_val, Y_val, X_test, Y_test = pre_process(load = True, save = False, downsample = True)
  
  print(Y_test)

  input_shape = (512, 512, 3)
  num_classes = NUM_CLASSES 
  
  """#Prepare the classes and samples we want to use for meta-training
  """
  n_mt_classes = config.n_mt_classes
  k_shot = config.k_shot
  batch_size = config.meta_batch_size
  
  classes = list(range(NUM_CLASSES))
  np.random.shuffle(classes)
  shuffled_classes = classes[:n_mt_classes] #indexing to select n classes for meta training
  # shuffled_classes = [1, 4]

  print('classes during meta train', shuffled_classes)

  """#Prepare the models
  """
  initializer = tf.initializers.VarianceScaling(scale=2.0)

  model = tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet', input_shape=input_shape)
  #Fine tune the model
  model.layers.pop()
  model.outputs = [model.layers[-1].output]
  output = model.get_layer('Conv_1_bn').output #Conv_1_bn for Mobilenetv2, block5_pool for vgg19
  output = tf.keras.layers.Flatten()(output)
  output = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax',
                            kernel_initializer=initializer)(output)
  fine_tune_model = tf.keras.Model(model.input, output)

  # print(new_model.summary())

  fine_tune_model_optimizer = tf.keras.optimizers.Adam(learning_rate=10**config.log_inference_lr)

  fine_tune_model.compile(optimizer=fine_tune_model_optimizer,
                loss='categorical_crossentropy',
                metrics=[tf.keras.metrics.categorical_accuracy])
  
  
  """#Sample from X and Y for a certain fraction of Test/Train data for a certain set of classes
  """

  """

  """

  k = int(len(X_train)/NUM_CLASSES)  
  X_train, Y_train = sample_from(X_train, Y_train, k, shuffled_classes, frac_data = config.frac_data) 
        
  k = int(len(X_val)/NUM_CLASSES)
  X_val, Y_val = sample_from(X_val, Y_val, k, shuffled_classes, frac_data = config.frac_data) 

  k = int(len(X_test)/NUM_CLASSES)
  X_test, Y_test = sample_from(X_test, Y_test, k, shuffled_classes, frac_data = config.frac_data) 
  
  print('distribution of Y_test', np.sum(Y_test, axis = 0))
  print('distribution of Y_val', np.sum(Y_val, axis = 0))
  
  #shuffle
  X_train, Y_train = shuffle(X_train, Y_train)
  X_val, Y_val = shuffle(X_val, Y_val)
  X_test, Y_test = shuffle(X_test, Y_test)
  X_train, Y_train, X_val, Y_val, X_test, Y_test = [np.float32(a) for a in [X_train, Y_train, X_val, Y_val, X_test, Y_test]]  

  es = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0, patience=config.mobilenet_early_stop_patience, verbose=0, mode='auto',
    baseline=None, restore_best_weights=True
  )
  checkpoint_filepath = 'model_checkpoint'
  model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
      filepath=checkpoint_filepath,
      save_weights_only=True,
      monitor='val_categorical_accuracy',
      mode='max',
      save_best_only=True)
  mobilenet_history = fine_tune_model.fit(X_train, Y_train, batch_size=2**config.inference_log_batch_size, epochs=2**config.inference_logepochs, validation_data=(X_val, Y_val), callbacks=[es, model_checkpoint_callback])
  fine_tune_model.save('classification_model')
  
  """Evaluate the fine-tuning model
  """
  print(fine_tune_model.metrics_names)
  mobilenet_test_accuracy = fine_tune_model.evaluate(X_test, Y_test)
  mobilenet_test_accuracy = mobilenet_test_accuracy[fine_tune_model.metrics_names.index('categorical_accuracy')]
  print('mobilenet_test_accuracy', mobilenet_test_accuracy)
  preds = fine_tune_model.predict(X_test)
  preds = np.argmax(preds, axis = 1)  
  y_test_classes = np.argmax(Y_test, axis = 1)
  confusion = confusion_matrix(y_test_classes, preds)
  print('Confusion Matrix\n')
  print(confusion)

  print('infered preds')
  print(preds)

  
  """#prepare the embedding model
  """
  
  # fine_tune_model = tf.keras.models.load_model("model_checkpoint")
  #
  # print(fine_tune_model.summary())
  #
  fine_tune_model.layers.pop()
  fine_tune_model.outputs = [fine_tune_model.layers[-1].output]
  output = fine_tune_model.get_layer('flatten').output #Conv_1_bn for Mobilenetv2, block5_pool for vgg19
  output = tf.keras.layers.Flatten()(output)
  embedding_model = tf.keras.Model(fine_tune_model.input, output)
  embedding_model_optimizer = tf.keras.optimizers.Adam(learning_rate=10**config.log_inference_lr) 
  embedding_model.compile(optimizer=embedding_model_optimizer)
  # outerloop_model = tf.keras.models.load_model("classification_model")

  """#prepare the meta-training data
  """

  x_mt, y_mt = sample_from(X_train, np.argmax(Y_train, axis = 1), (k_shot + 1) * batch_size, shuffled_classes)

  pickle.dump( x_mt, open( "x_mt_images.p", "wb" ) )
  pickle.dump( y_mt, open( "y_mt.p", "wb" ) )

  print(x_mt.shape)

  x_mt = x_mt.reshape(((k_shot + 1) * n_mt_classes, batch_size, 512, 512, 3))
  y_mt = y_mt.reshape(((k_shot + 1) * n_mt_classes, batch_size, NUM_CLASSES))

  y_mt = np.transpose(y_mt, (1, 0, 2))
  x_mt = np.transpose(x_mt, (1, 0, 2, 3, 4))

  embeddings = embedding_model.predict(x_mt.reshape((-1, 512, 512, 3)))  
  x_mt_embeddings = embeddings

  # print(x_mt.shape, 'after reshape')
  # print('expected')
  # print((batch_size, (k_shot + 1) * n_mt_classes, 256, 256, 3))
  
  #predict using the inference model to get embeddings
  

  print('got embeddings')
  print('embeddings shape', embeddings.shape)


  """Test if a dense network can predict using the embeddings
  """
  D = (embeddings.shape[-1])
  print("D mobilenet", D)

  layers = []
  layers += [tf.keras.layers.Dense(2 ** config.log_num_hidden_dn_ml, activation='relu')]
  layers += [tf.keras.layers.Dense(2 ** config.log_num_hidden_dn_ml, activation='relu') for _ in range(config.num_layers_dn_ml - 1)] 
  layers += [tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')]

  dense_ml_model = tf.keras.Sequential(layers)

  meta_optimizer = tf.keras.optimizers.Adam(learning_rate=10**config.log_meta_lr)

  dense_ml_model.compile(optimizer=meta_optimizer,
                loss='categorical_crossentropy', 
                metrics=[tf.keras.metrics.categorical_accuracy])
  

  es = tf.keras.callbacks.EarlyStopping(
    monitor='val_categorical_accuracy', min_delta=0, patience=config.dense_ml_early_stop_patience, verbose=1, mode='auto',
    baseline=None, restore_best_weights=True
  )

  
  dense_ml_model_history = dense_ml_model.fit(x_mt_embeddings.reshape((batch_size * (k_shot + 1) * n_mt_classes, -1)), y_mt.reshape((batch_size * (k_shot + 1) * n_mt_classes, -1)), epochs = config.dense_ml_model_epochs, validation_split = 0.3, callbacks = [es])

  print(dense_ml_model.summary())
  print('dense meta learning model fit')

  preds = dense_ml_model.predict(x_mt_embeddings.reshape((batch_size * (k_shot + 1) * n_mt_classes, -1)))
  preds = np.argmax(preds, axis = 1)  
  y_test_classes = np.argmax(y_mt.reshape((batch_size * (k_shot + 1) * n_mt_classes, -1)), axis = 1)
  confusion = confusion_matrix(y_test_classes, preds)
  print('Confusion Matrix\n')
  print(confusion)

  print('infered preds')
  print(preds)

  
  
  """Meta Learning
  """
  
  
  #Use the dense network trained above for even more refined embeddings
  dense_ml_model.layers.pop()
  dense_ml_model.outputs = [dense_ml_model.layers[-1].output]
  output = dense_ml_model.get_layer('dense_' + str(config.num_layers_dn_ml)).output #Conv_1_bn for Mobilenetv2, block5_pool for vgg19
  embedding_model = tf.keras.Model(dense_ml_model.input, output)
  embedding_model_optimizer = tf.keras.optimizers.Adam(learning_rate=10**config.log_inference_lr) 
  embedding_model.compile(optimizer=embedding_model_optimizer)
  embeddings = embedding_model.predict(x_mt_embeddings)  

  print('Meta Learning') 
  print('got embeddings')
  print('embeddings shape', embeddings.shape)
  # embeddings = embeddings.reshape((batch_size, n_mt_classes, (k_shot + 1), -1)) #Toggle to use output of the second dense
  embeddings = x_mt_embeddings.reshape((batch_size, n_mt_classes, (k_shot + 1), -1)) 
  print('embeddings shape', embeddings.shape)
  
  y_mt = y_mt.reshape((batch_size, n_mt_classes, (k_shot + 1), -1))

  x_mt_support = np.concatenate([embeddings[:,:,:k_shot,:], y_mt[:,:,:k_shot,:]], axis = 3) # Append labels to support set embeddings
  x_mt_support = x_mt_support.reshape(((batch_size, k_shot * n_mt_classes, -1))) #(batch_size, k_shot * n_mt_classes, embedding + n_mt_classes)
  x_mt_support = np.repeat(x_mt_support, n_mt_classes, axis = 0) #(batch_size * n_mt_classes, k_shot * n_mt_classes, embedding + n_mt_classes)

  x_mt_query = np.concatenate([embeddings[:,:,-1,:], np.zeros((batch_size, n_mt_classes, NUM_CLASSES))], axis = 2) #(batch_size, n_mt_classes, embedding + n_mt_classes)
  x_mt_query = x_mt_query.reshape(batch_size * n_mt_classes, 1, -1) #(batch_size * n_mt_classes, 1, embedding + n_mt_classes)

  x_mt = np.concatenate((x_mt_support, x_mt_query), axis = 1) #(batch_size, (k_shot + 1) * n_mt_classes, embedding + n_mt_classes)

  #Get the labels for the query set
  y_mt_query = y_mt[:,:,-1,:].reshape((batch_size, n_mt_classes, -1))
  y_mt_query = y_mt_query.reshape((batch_size * n_mt_classes, -1))

  # print("Expecting [10000], [01000]")
  # print(y_mt[0])
  # print(y_mt[1])
  # print(x_mt.shape)
  # print(y_mt.shape)

  (B, K, D) = x_mt.shape
  
  
  layers = [
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(2**config.log_LSTM_HIDDEN_UNITS, input_shape = (K,D), return_sequences=True)),
    # tf.keras.layers.Dense(2**config.log_LSTM_HIDDEN_UNITS, activation='softmax', return_sequences=True),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(NUM_CLASSES, activation="softmax")),    
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
  ]

  meta_model = tf.keras.Sequential(layers)

  meta_optimizer = tf.keras.optimizers.Adam(learning_rate=10**config.log_meta_lr)

  meta_model.compile(optimizer=meta_optimizer,
                loss='categorical_crossentropy', 
                metrics=[tf.keras.metrics.categorical_accuracy])

  es = tf.keras.callbacks.EarlyStopping(
    monitor='categorical_accuracy', min_delta=0, patience=config.snail_early_stop_patience, verbose=1, mode='auto',
    baseline=None, restore_best_weights=True
  )

  print('going to train SNAIL')
  snail_history = meta_model.fit(x_mt, y_mt_query, epochs = int(10 ** config.base_10_log_snail_epochs), callbacks = [es], verbose = 1)
  
  preds = meta_model.predict(x_mt)
  
  print('before the arg max')
  print(preds)

  preds = np.argmax(preds, axis = 1)
  
  print('after the arg max')
  print(preds)
  print('y_mt_query')
  print(y_mt_query)

  confusion = confusion_matrix(np.argmax(y_mt_query, axis = 1), preds)

  print('Confusion Matrix\n')

  print(confusion)
  
  print(meta_model.summary())

  snail_loss, snail_acc = meta_model.evaluate(x_mt, y_mt_query)

  metrics ={
    'mobilenet_test_accuracy': mobilenet_test_accuracy,
    'dense_meta_model_acc' : dense_ml_model_history.history['val_categorical_accuracy'][-1],
    'meta_train_accuracy' : max(snail_history.history['categorical_accuracy'])
  }

  wandb.log(metrics)

  # print('y_mt')
  # print(y_mt)

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
  
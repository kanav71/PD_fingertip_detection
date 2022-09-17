import tensorflow as tf
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from net.vgg16 import model as md
from preprocess.generator import train_generator, valid_generator
from fingertip import Fingertips ## Kanav adding this so that we can use transfer learning

import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def loss_function(y_true, y_pred):
    square_diff = tf.squared_difference(y_true, y_pred)
    square_diff = tf.reduce_mean(square_diff, 1)
    loss = tf.reduce_mean(square_diff)
    return loss


# creating the model
model = md()
model.summary()

#hyperparameters
learning_rate = [1e-2,1e-3,1e-4,1e-5, 1e-6]
dr = 0.5
neu = 1024

counter = 0 
out_list = []
for kk in learning_rate:
  # compile
  #model = md() ## commenting this for dynamic LR in decreasing order
  adam = Adam(lr= kk, beta_1=0.9, beta_2=0.999, epsilon=1e-10, decay=0.0) #changing lr to 1e-2 from 1e-5
  model.compile(optimizer=adam, loss=loss_function)

  print(f"New model running for learning_rate = {kk} and 2 layers{neu} & dr({dr})" )
  # train
  epochs = 60 # kanav changed from 10 to 2
  train_gen = train_generator(sample_per_batch=35, batch_number=232) # kanav changed from 1690 to 5, sample_per_batch to 1
  #chaanged from bth = 25 & b_num = 325 to 2,25
  val_gen = valid_generator(sample_per_batch=60, batch_number=20) # kanav changed sample_per_batch from 50 to 3 & batch number from 10 to 2 

  
  path = f"/content/drive/My Drive/Colab_Notebooks/Finger_tip_coordinates/hypertune_dir/hyper_weights/"

  stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10) #increasing patience to 15
  checkpoints = ModelCheckpoint( path + "wts_vg16" + f"_lr_{kk}_dr_{dr}_ly_{neu}_Dynm"+".h5", 
  save_best_only=True,  save_weights_only=True, monitor = 'val_loss', period=1, verbose = 1) 

  history = model.fit_generator(train_gen, steps_per_epoch=232, epochs=epochs, verbose=1, shuffle=True,  # kanav changed from 1690 to 3
                                validation_data=val_gen, validation_steps=20, #changed from 10 to 2 - nto sure of original though
                                workers = 3, use_multiprocessing=True, # Kanav added
                                callbacks=[stop_early, checkpoints], max_queue_size=100)

  with open('/content/drive/My Drive/Colab_Notebooks/Finger_tip_coordinates/hypertune_dir/hyper_history.txt', 'a+') as f:
      print(history.history, file=f)
  out_list.append([f"The iteration ran for learning_rate = {kk}, 2 layers = {neu} & dr({dr})."]) 
  counter += 1

print(out_list)

print('All Done!')

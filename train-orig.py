import tensorflow as tf
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from net.vgg16 import model
from preprocess.generator import train_generator, valid_generator


def loss_function(y_true, y_pred):
    square_diff = tf.squared_difference(y_true, y_pred)
    square_diff = tf.reduce_mean(square_diff, 1)
    loss = tf.reduce_mean(square_diff)
    return loss


# creating the model
model = model()
model.summary()

# compile
adam = Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-10, decay=0.0)
model.compile(optimizer=adam, loss=loss_function)

# train
epochs = 3 # kanav changed from 10 to 1
train_gen = train_generator(sample_per_batch=2, batch_number=15) # kanav changed from 1690 to 15
val_gen = valid_generator(sample_per_batch=5, batch_number=2) # kanav changed sample_per_batch from 50 to 5 & batch number from 10 to 2 

checkpoints = ModelCheckpoint('weights/weights{epoch:03d}.h5', save_weights_only=True, period=1)
history = model.fit_generator(train_gen, steps_per_epoch=2, epochs=epochs, verbose=1, shuffle=True,  # kanav changed from 1690 to 2
                              validation_data=val_gen, validation_steps=10,
                              callbacks=[checkpoints], max_queue_size=100)

with open('history.txt', 'a+') as f:
    print(history.history, file=f)

print('All Done!')

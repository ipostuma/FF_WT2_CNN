from net.unet_2D import model as net_model
from net.metrics.dice import DiceLoss, Dice
from tensorflow import keras
import pandas as pd
import numpy as np

train_img = np.zeros((10,384,192,17))
train_seg = np.zeros((10,384,192,1))

val_img = np.zeros((10,384,192,17))
val_seg = np.zeros((10,384,192,1))

model = net_model(input_shape=(384,192,17),k1=8,reg=0.1)
model.summary(line_length=150)

adam = keras.optimizers.Adam(
    learning_rate=0.0001, 
    beta_1=0.9, 
    beta_2=0.999, 
    epsilon=1e-08, 
    amsgrad=True)

model.compile(
    loss=DiceLoss,
    optimizer=adam,
    metrics=[Dice])

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_Dice', 
    factor=0.5, patience=5, 
    min_lr=0.000005, verbose= True)

history = model.fit(
    train_img,
    train_seg,
    epochs=3,
    batch_size=1,
    validation_data=(val_img,val_seg),
    callbacks=[reduce_lr])

print("Your model is well defined")

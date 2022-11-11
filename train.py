from net.unet_2D import model as net_model
from net.metrics.dice import DiceLoss, Dice
from tensorflow import keras
import pandas as pd
from glob import glob
import os
import numpy as np

base_folder = "../DATA"
train_folder_i = "../DATA/TS_i"
train_input_files = glob(os.path.join(train_folder_i),"*")
train_output_files = [f.replace("TS_i","TS_o") for f in train_input_files]

val_folder_i = "../DATA/VS_i"
val_input_files = glob(os.path.join(val_folder_i),"*")
val_output_files = [f.replace("VS_i","VS_o") for f in val_input_files]

train_img = [np.load(f) for f in train_input_files]
train_seg = [np.load(f) for f in train_output_files]

val_img = [np.load(f) for f in train_input_files]
val_seg = [np.load(f) for f in train_output_files]

model = net_model()
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

model.save('../DATA/lung_unet')

# convert the history.history dict to a pandas DataFrame:     
hist_df = pd.DataFrame(history.history) 

# save to json:  
hist_json_file = '../DATA/history.json' 
with open(hist_json_file, mode='w') as f:
    hist_df.to_json(f)

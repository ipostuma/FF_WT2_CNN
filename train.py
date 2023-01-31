from net.ResUnet import model as net_model
import model_test
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras
import pandas as pd
from glob import glob
import os
import numpy as np
import matplotlib.pyplot as plt


def input_norm(x):
    scaler = MinMaxScaler()
    if len(x.shape) == 3:
        fit = [scaler.fit(x[:,:,echo]) for echo in range(x.shape[2])]
        x_scaled =  np.array([scaler.transform(x[:,:,echo]) for echo in range(x.shape[2])])
        x_scaled = np.rollaxis(x_scaled, 0, 3)
    else:
        fit = scaler.fit(x)
        x_scaled = scaler.transform(x)
    return x_scaled

base_folder = "../DATA_slice_wise"
train_folder_i = "../DATA_slice_wise/TS_i"
train_input_files = glob(os.path.join(train_folder_i,"*"))
train_output_files = [f.replace("TS_i","TS_o") for f in train_input_files]

ima = [np.load(f) for f in train_input_files]
gs = [np.load(f) for f in train_output_files]

X_train, X_test, y_train, y_test = train_test_split(ima, gs, test_size=0.1, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=1)

train_img = np.array(list(map(input_norm,X_train)))
train_seg = np.array(list(map(input_norm,y_train)))
val_img = np.array(list(map(input_norm,X_val)))
val_seg = np.array(list(map(input_norm,y_val)))
test_img = np.array(list(map(input_norm,X_test)))
test_seg = np.array(list(map(input_norm,y_test)))


model = net_model()
model.summary(line_length=150)

adam = keras.optimizers.Adam(
    learning_rate=0.0001, 
    beta_1=0.9, 
    beta_2=0.999, 
    epsilon=1e-08, 
    amsgrad=True)

model.compile(
    loss='mse',
    optimizer=adam,
    metrics="mean_absolute_error")

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.5, patience=5, 
    min_lr=0.000005, verbose= True)

history = model.fit(
    train_img,
    train_seg,
    batch_size = 1,
    epochs=600,
    validation_data=(val_img,val_seg),
    callbacks=[reduce_lr])

# plot training, validation losses
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model accuracy')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


model_dst = '../DATA_slice_wise/HISTORY/wt2_ResUnet_31012023'
model.save(model_dst)
# convert the history.history dict to a pandas DataFrame:     
hist_df = pd.DataFrame(history.history) 

# save to json:  
hist_json_file = '../DATA_slice_wise/history_ResUnet_31012023.json' 
with open(hist_json_file, mode='w') as f:
    hist_df.to_json(f)


# MODEL TEST
evaluation, test_pred = model_test.test(test_img, test_seg, model_dst)

# plot err%
pred = test_pred[60,:,:]
true = test_seg[60,:,:]
err = 100*(np.absolute((true - pred)/true))
plt.imshow(true, cmap='gray')
plt.figure()
plt.imshow(pred, cmap='gray')
plt.figure()
plt.imshow(err, cmap='gray')
    



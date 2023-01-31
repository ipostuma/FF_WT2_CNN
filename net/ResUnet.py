from tensorflow import keras
keras.backend.clear_session()

def BatchActivate(x):
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    return x

def convolution_block(x, filters, size, strides=(1,1), padding='same', activation=True):
    x = keras.layers.Conv2D(filters, size, strides=strides, padding=padding)(x)
    if activation==True: x = BatchActivate(x)
    return x

def residual_block(blockInput, num_filters, batch_activate=False):
    x = BatchActivate(blockInput)
    x = convolution_block(x, num_filters, (3,3))
    x = convolution_block(x, num_filters, (3,3), activation=False)
    x = keras.layers.Add()([x, blockInput])
    x = BatchActivate(x)
    return x

# down
def down(previous_layer,k, **kargs):
    d = keras.layers.Conv2D(k,3,padding="same",**kargs)(previous_layer)
    d = keras.layers.BatchNormalization()(d)
    d = keras.layers.Activation('relu')(d)
    d = keras.layers.Conv2D(k,3,padding="same",**kargs)(d)
    d = keras.layers.BatchNormalization()(d)
    d = keras.layers.Activation('relu')(d)
    d = keras.layers.Conv2D(k,3,padding="same",**kargs)(d)
    res = residual_block(d,k)
    res = residual_block(res,k)
    d = keras.layers.MaxPooling2D()(res)
    return d

# bottleneck
def bottleneck(previous_layer,k, **kargs):
    b = keras.layers.Conv2D(k,3,padding="same",**kargs)(previous_layer)
    b = residual_block(b, k)
    b = residual_block(b, k)
    return b

# up
def up(previous_layer,down_layer,k, **kargs):
    u = keras.layers.Conv2DTranspose(k,3,padding="same",**kargs)(previous_layer)
    u = keras.layers.BatchNormalization()(u)
    u = keras.layers.Activation('relu')(u)
    u = keras.layers.Concatenate(axis=-1)([u,down_layer])
    u = keras.layers.Conv2D(k,3,padding="same",**kargs)(u)
    u = keras.layers.BatchNormalization()(u)
    u = keras.layers.Activation('relu')(u)
    u = keras.layers.Conv2D(k,3,padding="same",**kargs)(u)
    res = residual_block(u,k)
    res = residual_block(res,k)
    u = keras.layers.UpSampling2D()(res)
    return u

def model(input_shape=(384,192,17),k1=16,reg=0.0001):
    inputs=keras.layers.Input(shape=(input_shape))
    
    params = {"kernel_regularizer" : keras.regularizers.l2(reg), 
              "kernel_initializer" : 'random_normal'}
        
    d1 = down(inputs,k1,**params)
    d2 = down(d1,k1*2,**params) 
    d3 = down(d2,k1*4,**params) 
    d4 = down(d3, k1*8, **params)
    
    b = bottleneck(d4,k1*2,**params)
    
    u1 = up(b,d4,k1*8,**params)    
    u2 = up(u1,d3,k1*4,**params)
    u3 = up(u2,d2,k1*2,**params)
    u4 = up(u3,d1,k1,**params)
    
    out = keras.layers.Conv2D(1,3,padding="same",**params)(u4) 
    out = keras.activations.sigmoid(out)
    
    model=keras.models.Model(inputs=inputs,outputs=out)
    
    return model



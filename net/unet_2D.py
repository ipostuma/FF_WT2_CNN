from tensorflow import keras
keras.backend.clear_session()

# down
def down(previous_layer,k, **kargs):
    d = keras.layers.Conv2D(k,3,padding="same",**kargs)(previous_layer)
    d = keras.layers.BatchNormalization()(d)
    d = keras.layers.Activation('relu')(d)
    d = keras.layers.Conv2D(k,3,padding="same",**kargs)(d)
    d = keras.layers.BatchNormalization()(d)
    d = keras.layers.Activation('relu')(d)
    d = keras.layers.Conv2D(k,3,padding="same",**kargs)(d)
    d = keras.layers.BatchNormalization()(d)
    d = keras.layers.MaxPooling2D()(d)

    res = keras.layers.Conv2D(k,3,strides=2,padding="same",**kargs)(previous_layer)
    d = keras.layers.Add()([res,d])
    d = keras.layers.Activation('relu')(d)
    return d

# bottleneck
def bottleneck(previous_layer,k, **kargs):
    b = keras.layers.Conv2D(k,3,padding="same",**kargs)(previous_layer)
    b = keras.layers.BatchNormalization()(b)
    b = keras.layers.Activation('relu')(b)
    b = keras.layers.Conv2D(k,3,padding="same",**kargs)(b)
    b = keras.layers.BatchNormalization()(b)
    b = keras.layers.Activation('relu')(b)

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
    u = keras.layers.BatchNormalization()(u)
    u = keras.layers.UpSampling2D()(u)

    res = keras.layers.UpSampling2D()(previous_layer)
    res = keras.layers.Conv2D(k,1,padding="same",**kargs)(res)
    u = keras.layers.Add()([u,res])
    u = keras.layers.Activation('relu')(u)
    return u

def model(input_shape=(384,192,17,1),k1=8,reg=0.1):
    inputs=keras.layers.Input(shape=(input_shape))
    
    params = {"kernel_regularizer" : keras.regularizers.l2(reg), 
              "kernel_initializer" : 'random_normal'}
        
    d1 = down(inputs,k1,**params)
    d2 = down(d1,k1*2,**params) 
    d3 = down(d2,k1*4,**params) 
    
    b = bottleneck(d3,k1*8,**params)
        
    u1 = up(b ,d3,k1*4,**params)
    u2 = up(u1,d2,k1*2,**params)
    u3 = up(u2,d1,k1,**params)
    
    out = keras.layers.Conv3D(padding="same",**params)(u3) 
    out = keras.activations.sigmoid(out)
    
    model=keras.models.Model(inputs=inputs,outputs=out)
    
    return model

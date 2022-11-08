from tensorflow import keras

def Dice(targets, inputs, smooth=1e-6):

    #flatten label and prediction tensors
    inputs = keras.backend.flatten(inputs)
    targets = keras.backend.flatten(targets)

    intersection = keras.backend.sum(targets*inputs)
    dice = (2*intersection + smooth) / (keras.backend.sum(targets) + keras.backend.sum(inputs) + smooth)
    
    return dice

def DiceLoss(targets, inputs, smooth=1e-6):
    return 1 - Dice(targets, inputs, smooth=1e-6)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 10:34:06 2023

@author: user1nrx
"""

from tensorflow import keras

def test(test_img, test_seg,model_dst):
    model = keras.models.load_model(model_dst)
    
    evaluation = model.evaluate(
        test_img,
        test_seg,
        batch_size=1,
        verbose="auto",
        steps=30
    )
    
    out = model.predict(test_img)
    test_pred = out.reshape(out.shape[0], out.shape[1], out.shape[2])

    return evaluation, test_pred

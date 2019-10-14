# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 19:44:02 2017
@author: user
"""
import argparse
from flyai.dataset import Dataset as FlyAI_Dataset
from path import MODEL_PATH
from flyai.utils import remote_helper
import numpy as np
import os
import cv2
import math
from mymodel import CenterNet52
from config import COCOConfig
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler

# 这里使用mask-rcnn方法，主要参考了链接：
# https://machinelearningmastery.com/how-to-train-an-object-detection-model-with-keras/

'''
自己通过模板重新写时，记得将样例代码中app.yaml复制到模板中

Tensorflow模版项目下载： https://www.flyai.com/python/tensorflow_template.zip
PyTorch模版项目下载： https://www.flyai.com/python/pytorch_template.zip
Keras模版项目下载： https://www.flyai.com/python/keras_template.zip
第一次使用请看项目中的：第一次使用请读我.html文件
常见问题请访问：https://www.flyai.com/question
意见和问题反馈有红包哦！添加客服微信：flyaixzs
'''


# 判断路径是否存在
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)
KERAS_MODEL_DIR = os.path.join(MODEL_PATH, 'model.h5')

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=2, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=1, type=int, help="batch size")
args = parser.parse_args()
print('epoch is %d, batch is %d'%(args.EPOCHS, args.BATCH))

flyai_dataset = FlyAI_Dataset()
# 获取全量数据
x_train, y_train, x_val, y_val = flyai_dataset.get_all_processor_data()
train_len = len(x_train)
val_len = len(x_val)
print('train len is %d, val len is %d'%(train_len, val_len))

config = COCOConfig()
config.STEPS_PER_EPOCH = math.ceil(x_train.shape[0]/config.BATCH_SIZE)
config.VAL_STEPS = math.ceil(x_val.shape[0]/config.BATCH_SIZE)

model = CenterNet52(mode="training", config=config,
                          model_dir=config.MODEL_DIR)
model.keras_model.metrics_tensors = []
# model.keras_model.summary()
print('CenterNet52 loaded!')

modelCheckpoint = ModelCheckpoint(KERAS_MODEL_DIR, verbose=1, save_best_only=True, save_weights_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1)
# reduce_lr = LearningRateScheduler(lr_schedule, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0)

# callbacks = [reduce_lr]
callbacks = [early_stopping]
model.train_flyai(x_train, y_train, x_val, y_val,
    learning_rate=1e-3, epochs=100, augment=True, custom_callbacks=callbacks, opt_mode='AdaBound')

# callbacks = [modelCheckpoint, early_stopping, reduce_lr]
# model.train_flyai(x_train, y_train, x_val, y_val,
#     learning_rate=1e-4, epochs=50, augment=True, custom_callbacks=callbacks, opt_mode='SGD')

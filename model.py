# -*- coding: utf-8 -*
import os
import cv2
from flyai.model.base import Base
from path import MODEL_PATH
import numpy as np
from config import COCOConfig
from mymodel import CenterNet52

# 模型路径
KERAS_MODEL_DIR = os.path.join(MODEL_PATH, 'model.h5')
# define the prediction configuration， 评估时的参数


class Model(Base):
    def __init__(self, dataset):
        self.dataset = dataset
    '''
    评估一条数据
    '''

    def predict(self, **data):
        x_data = self.dataset.predict_data(**data)
        data_path = x_data[0]
        image = skimage.io.imread(data_path)
        scaled_image = mold_image(image, cfg)
        sample = np.expand_dims(scaled_image, 0)
        yhat = model.detect(sample, verbose=0)[0]
        return yhat["class_ids"]-1, yhat['rois'][1], yhat['rois'][0], yhat['rois'][3], yhat['rois'][2]

    '''
    评估的时候会调用该方法实现评估得分
    '''

    def predict_all(self, datas):
        # create config
        cfg = COCOConfig()
        cfg.BATCH_SIZE = 1
        # define the model
        model = CenterNet52(mode='inference', model_dir=MODEL_PATH, config=cfg)
        # load model weights
        if os.path.exists(KERAS_MODEL_DIR):
            model.load_weights(KERAS_MODEL_DIR, by_name=True)
        else:
            print('MODEL NO FOUND!!!')
            exit()

        labels = []

        for data in datas:
            x_data = self.dataset.predict_data(**data)
            data_path = x_data[0]
            image = cv2.imread(data_path)
            sample = np.expand_dims(image, 0)
            sample = np.array(sample, dtype=np.float32)
            yhat = model.detect(sample, verbose=0)[0]
            # 按照得分进行排序
            indices = np.argsort(yhat["scores"])[::-1]
            label = []
            for i in range(len(indices)):
                label.append([yhat["class_ids"][i]-1, yhat['rois'][i][1], yhat['rois'][i][0], yhat['rois'][i][3], yhat['rois'][i][2]])
            label = np.array(label)
            label = label[indices]
            labels.append(label)
        '''
        关于返回说明：
        labels 中返回的是 list 数组， 每一条包含 label, x_min, y_min, x_max, y_max . 每条之间的顺序是按照得分进行排序之后的
        在评估中进行mAP计算中，IoU阈值取0.5
        '''
        return labels
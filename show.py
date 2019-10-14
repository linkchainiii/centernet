import cv2
import pandas as pd
import os

base_path = 'data/input'
df = pd.read_csv(os.path.join(base_path, 'dev.csv'))
all_img_path = df['img_path'].values
all_box = df['box'].values

for i in range(len(all_img_path)):
    img = cv2.imread(os.path.join(base_path, all_img_path[i]))
    print(img.shape)
    line = all_box[i]
    for box in line.split(' '):
        xmin = int(box.split(',')[0])
        ymin = int(box.split(',')[1])
        xmax = int(box.split(',')[2])
        ymax = int(box.split(',')[3])
        label = int(box.split(',')[4])
        if label == 0:
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(img, 'hat', (xmin, ymin-2), 1, 1, (0, 255, 0), 1)
        else:
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
            cv2.putText(img, 'person', (xmin, ymin-2), 1, 1, (0, 0, 255), 1)

    cv2.imshow('img', img)
    cv2.waitKey()
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
import random
import os
import re
import natsort

img_path = "/mnt/hdd_6tb/jh2020/processed_test/images/"
label_path = "/mnt/hdd_6tb/jh2020/processed_test/labels/"
plotted_path = "/mnt/hdd_6tb/jh2020/processed_original/plotted_image/"

img_name_lst = natsort.natsorted(os.listdir(img_path))
label_name_lst = natsort.natsorted(os.listdir(label_path))
name_zip_lst = list(zip(img_name_lst, label_name_lst))

plotted_lst = random.sample(name_zip_lst, 10)

cnt = 1
for img_name, label_name in plotted_lst:
    # label
    f = open(label_path+label_name)
    labels = f.readlines()
    
    bbox_lst = []
    for label in labels:
        temp_lst = []
        for i, data in enumerate(label.split(' ')):
            if i == 0:
                pass
            else:
                temp_lst.append(eval(data))
        bbox_lst.append(temp_lst)
    # img
    image = Image.open(img_path+img_name)
    img_size = image.size
    fig, ax = plt.subplots()
    plt.imshow(image)
    for bbox in bbox_lst:
        x = float(bbox[0]) * img_size[0]
        y = float(bbox[1]) * img_size[1]
        w = float(bbox[2]) * img_size[0]
        h = float(bbox[3]) * img_size[1]
        bbox = patches.Rectangle((x - w/2, y -h/2), 
                                 w,
                                 h, 
                                linewidth = 1, 
                                edgecolor='r',
                                facecolor="none")
        ax.add_patch(bbox)
    fig.savefig(plotted_path+f'plotted{cnt}.png')
    cnt +=1
import torch
import re
import os
import json
import glob
import natsort
import numpy as np
from PIL import Image

############### image ###############


def image_resize(img):
    img_size = img.size
    if img_size[0] > img_size[1]:
        img_resized = img.resize((640, round(640*img_size[1]/img_size[0]))) # size(tuple)
    else:
        img_resized = img.resize((round(640*img_size[0]/img_size[1]), 640))
    return img_resized


def image_preprocess(save_path, plate_path): # ..data/image/plate
    img_path_lst = glob.glob(plate_path+'/**/*.jpg', recursive=True)
    img_path_lst = natsort.natsorted(img_path_lst)
    cnt = 1
    for img_path in img_path_lst:
        img = Image.open(img_path)
        img_resized = image_resize(img)
        img_resized.save(f'{save_path}/images/image{cnt}.png', 'png')
        cnt += 1
    

############### label ###############
# img size에 맞게 normalize
def bbox_normalize(bbox, img_size):
    x, y, w, h = bbox
    size = 640
    scale_factor_l = size
    scale_factor_s = round(size*img_size[1]/img_size[0])
    
    if img_size[0]>img_size[1]: # img_width > img_height
        n_x = x* scale_factor_l/img_size[0]
        n_y = y* scale_factor_s/img_size[1]
        n_w = w* scale_factor_l/img_size[0]
        n_h = h* scale_factor_s/img_size[1]
    else:                       # img_width < img_height
        n_x = x* scale_factor_s/img_size[0]
        n_y = y* scale_factor_l/img_size[1]
        n_w = w* scale_factor_s/img_size[0]
        n_h = h* scale_factor_l/img_size[1]
    return n_x, n_y, n_w, n_h, scale_factor_l, scale_factor_s

# yolo format에 맞게 normalize
def bbox_center_normalize(bbox, img_size):
    x, y, w, h = bbox
    left, top, width, height, n_img_width, n_img_height = bbox_normalize(bbox, img_size)
    
    xcenter = (left + width / 2) / n_img_width
    ycenter = (top + height / 2) / n_img_height
    w = width / n_img_width
    h = height / n_img_height
    
    return xcenter, ycenter, w, h

def json_to_label(plate_json_path, phase):
    # bbox
    f = open(plate_json_path)
    data = json.load(f)
    bbox_lst = [x["bbox"] for x in data["Learning_Data_Info"]["annotations"][0]["license_plate"]] # bbox가 하나가 아님
    
    # img size
    if phase == "train":
        path = plate_json_path.replace("VL","VS")
        path = path.replace("label","image")
        img_path = path.replace(".json",".jpg")
        img = Image.open(img_path)
    elif phase == "test":
        path = plate_json_path.replace("02.labeling","01.source")
        img_path = path.replace(".json",".jpg")
        img = Image.open(img_path)
        
    img_size = img.size # img size가 다름
    
    # normalize
    n_bbox_lst = [bbox_center_normalize(x, img_size) for x in bbox_lst]

    return n_bbox_lst # list

def label_preprocess(save_path, plate_json_path, phase):
    cnt = 1
    class_num = 0
    label_path_lst = glob.glob(plate_json_path+"/*/*/*.json")
    for label_path in natsort.natsorted(label_path_lst):
        n_bbox_lst = json_to_label(label_path, phase)
        txt_path = save_path+f"/labels/image{cnt}.txt"
        if cnt ==1:
        with open(txt_path, "w") as file:
            for i, n_bbox in enumerate(n_bbox_lst):
                if i==len(n_bbox_lst)-1:
                    data = f"{class_num} {n_bbox[0]} {n_bbox[1]} {n_bbox[2]} {n_bbox[3]}" # 마지막 bbox는 enter 제거
                else:
                    data = f"{class_num} {n_bbox[0]} {n_bbox[1]} {n_bbox[2]} {n_bbox[3]}\n"
                file.write(data)
            cnt +=1


if __name__=="__main__":
    phase = "test"
    plate_image_path = "/home/yohanban/data/Sample_data/01.source_data/detect_plate/교차로"
    plate_json_path = "/home/yohanban/data/Sample_data/02.labeling_data/detect_plate/교차로"
    save_path = f"/mnt/hdd_6tb/jh2020/processed_{phase}"

    label_preprocess(save_path, plate_json_path, phase = phase)
    image_preprocess(save_path, plate_image_path)
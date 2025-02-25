from dataclasses import dataclass, field
from ultralytics import YOLO
from PIL import ImageFont, Image, ImageDraw
import torch
import data_utils
import os
'''
차량을 탐지한 후 crop한 이미지에 대해 plate탐지
todo:
    3. 최종 result에 대한 nms
'''
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

@dataclass
class Boxes:
    xyxy: torch.Tensor = field(default_factory=lambda: torch.empty((0,4)))
    xywh: torch.Tensor = field(default_factory=lambda: torch.empty((0,4)))
    
@dataclass
class Result:
    boxes: list = field(default_factory=lambda: Boxes())
    orig_shape: tuple = None# (w,h) != yolo format(h,w)
    image = None

class Forward:
    def __init__(self, model_car, model_crop):
        self.model_car = model_car
        self.model_crop = model_crop
        self.boxes = Boxes()
        self.img = None
    
    def detect_car(self, img):
        '''
        원본 이미지에서 차량을 검출
        Args:
            img: PIL이미지
        Returns:
            result_car객체를 출력
        '''
        output = self.model_car(img, save=True)# inference
        
        result_car = Result()
        
        result_car.boxes.xyxy = torch.cat((result_car.boxes.xyxy.to('cuda'), output[0].boxes.xyxy),dim=0)
        result_car.boxes.xywh = torch.cat((result_car.boxes.xywh.to('cuda'), output[0].boxes.xywh),dim=0)
        result_car.orig_shape = img.size# (w, h)
        
        return result_car
    
    def crop(self, result_car):
        '''
        원본 이미지에서 detect car를 crop해서 출력
        Args:
            result_car: 원본 이미지에 대한 yolo_car's output
        Returns:
            crop된 사이즈의 자동차 이미지
        '''
        orig_img = self.img.copy()# 원본 이미지
        
        crop_lst = []
        cnt=0
        for i in result_car.boxes.xyxy:
            crop_img = orig_img.crop(tuple(i.tolist()))
            crop_lst.append(crop_img)
            cnt +=1
        
        return crop_lst
    
    def detect_plate(self, crop_lst: list):
        '''
        crop된 이미지에서 plate를 검출
        Args:
            crop_lst: crop된 차량
        Returns:
            result_plate: 검출된 번호판
        '''
        result_plate = self.model_crop(crop_lst, save=True)# inference

        return result_plate# yolo's output
    
    def scaling(self, result_plate, car_xywh, imgsz=640):
        '''
        plate bbox를 원본 이미지에 맞게 scaling
        Args:
            result_plate: crop img에 대한 yolo's output
            car_xywh: crop detection bbox's xywh
            imgsz: yolo input size
        Returns:
            scaled plate bbox
        '''
        W = car_xywh[2]# car bbox width
        H = car_xywh[3]# car bbox height
        xc = result_plate.boxes.xywh[0][0]# plate bbox xc
        yc = result_plate.boxes.xywh[0][1]# plate bbox yc
        w = result_plate.boxes.xywh[0][2]# plate bbox width
        h = result_plate.boxes.xywh[0][3]# plate bbox height
        
        scale_factor = max(W/imgsz, H/imgsz)
        
        pad_W = (imgsz - W/scale_factor)# if W>H: 0
        pad_H = (imgsz - H/scale_factor)# if H>H: 0
        
        x1 = (xc - pad_W)*scale_factor + W - w/2
        y1 = (yc - pad_H)*scale_factor + H - h/2
        w = w*scale_factor
        h = h*scale_factor

        
        xyxy = torch.tensor([x1, y1, x1+w, y1+h])# scaled xyxy
        xywh = torch.tensor([x1+w/2, y1+h/2, w, h])# scaled xywh
        
        result_scaled = Result()
        
        result_scaled.boxes.xyxy = torch.cat((result_scaled.boxes.xyxy, xyxy.unsqueeze(0)), dim=0)
        result_scaled.boxes.xywh = torch.cat((result_scaled.boxes.xywh, xywh.unsqueeze(0)), dim=0)
        result_scaled.orig_shape = self.img.size
        
        return result_scaled

    def plot(self, results):
        img = self.img.copy()
        bboxes = results.boxes.xyxy
        bboxes = bboxes.to('cpu').numpy()
        font = ImageFont.load_default()
        
        draw = ImageDraw.Draw(img)
        for bbox in bboxes:
            draw.rectangle((bbox), outline="red", width=2)

        return img
    
    def __call__(self, img):
        self.img = img
        result_car = self.detect_car(img)
        crop_lst = self.crop(result_car)
        
        results_plate = self.detect_plate(crop_lst)
        
        results = Result()# final result
        for i in range(len(crop_lst)):
            car_xyxy = result_car.boxes.xyxy[i]
            car_xywh = result_car.boxes.xywh[i]
            result_scaled = self.scaling(results_plate[i], car_xywh, imgsz=640)
            
            results.boxes.xyxy = torch.cat((results.boxes.xyxy, result_scaled.boxes.xyxy), dim=0)
            results.boxes.xywh = torch.cat((results.boxes.xywh, result_scaled.boxes.xywh), dim=0)
        results.boxes.orig_shape = self.img.size
        results.image = self.plot(results)
        
        return results
    
if __name__ == "__main__":
    model_car = YOLO("/mnt/hdd_6tb/jh2020/ckpt/YOLOv11n_car.pt")
    model_crop = YOLO("/mnt/hdd_6tb/jh2020/ckpt/YOLOv11n_carcrop.pt")
    
    forward = Forward(model_car = model_car, model_crop = model_crop)
    
    orig_img_path = "/home/yohanban/data/Sample_data/01.source_data/detect_car/교차로/[cr06]호계사거리/02번/C-221005_13_CR06_02_A0062.jpg"
    img = Image.open(orig_img_path)
    result = forward(img)
    result.image.save('car_crop.png', 'png')
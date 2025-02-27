from dataclasses import dataclass, field
from ultralytics import YOLO
from PIL import ImageFont, Image, ImageDraw
import torch
import os

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
        output = self.model_car(img)# inference
        
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
        if len(crop_lst)==0:
            result_plate = Result()
        if len(crop_lst)>0:
            result_plate = self.model_crop(crop_lst)# inference

        return result_plate# yolo's output
    
    def scaling(self, result_plate, car_xyxy, imgsz=640):
        '''
        plate bbox를 원본 이미지에 맞게 scaling
        Args:
            result_plate: crop img에 대한 yolo's output
            car_xywh: crop detection bbox's xywh
            imgsz: yolo input size
        Returns:
            scaled plate bbox
        '''
        if not len(result_plate.boxes.xyxy) > 0:
            xyxy = torch.empty((0,4))
            xywh = torch.empty((0,4))
            
            result_scaled = Result()
            
            result_scaled.boxes.xyxy = torch.cat((result_scaled.boxes.xyxy, xyxy), dim=0)
            result_scaled.boxes.xywh = torch.cat((result_scaled.boxes.xywh, xywh), dim=0)
            result_scaled.orig_shape = self.img.size
        
        else:
            x1, y1, x2, y2 = result_plate.boxes.xyxy[0]# nms 필요성 O
            w = x2 - x1
            h = y2 - y1
            x1_car, y1_car, x2_car, y2_car = car_xyxy
            w_car = x2_car-x1_car
            h_car = y2_car-y1_car
            
            x1 = x1 + x1_car
            y1 = y1 + y1_car
            x2 = x1 + w
            y2 = y1 + h

            xyxy = torch.tensor([x1, y1, x2, y2])# scaled xyxy
            xywh = torch.tensor([x1+w/2, y1+h/2, w, h])# scaled xywh
        
            result_scaled = Result()
            
            result_scaled.boxes.xyxy = torch.cat((result_scaled.boxes.xyxy, xyxy.unsqueeze(0)), dim=0)
            result_scaled.boxes.xywh = torch.cat((result_scaled.boxes.xywh, xywh.unsqueeze(0)), dim=0)
            result_scaled.orig_shape = self.img.size
        
        return result_scaled
    
    @staticmethod
    def plot(image, results):
        bboxes = results.boxes.xyxy
        bboxes = bboxes.to('cpu').numpy()
        font = ImageFont.load_default()
        
        draw = ImageDraw.Draw(image)
        for bbox in bboxes:
            draw.rectangle((bbox), outline="red", width=2)

        return image
    
    def __call__(self, img):
        self.img = img
        result_car = self.detect_car(img)
        crop_lst = self.crop(result_car)
        
        results_plate = self.detect_plate(crop_lst)
        
        results = Result()# final result
        for i in range(len(crop_lst)):
            car_xyxy = result_car.boxes.xyxy[i]
            car_xywh = result_car.boxes.xywh[i]
            result_scaled = self.scaling(results_plate[i], car_xyxy, imgsz=640)
            
            results.boxes.xyxy = torch.cat((results.boxes.xyxy, result_scaled.boxes.xyxy), dim=0)
            results.boxes.xywh = torch.cat((results.boxes.xywh, result_scaled.boxes.xywh), dim=0)
        results.boxes.orig_shape = self.img.size
        results.image = Forward.plot(img, results)
        
        return results
    
if __name__ == "__main__":
    model_car = YOLO("../car_plate_od/ckpt/YOLOv11n_car.pt")
    model_crop = YOLO("../car_plate_od/ckpt/YOLOv11n_carcrop.pt")
    
    forward = Forward(model_car = model_car, model_crop = model_crop)
    
    orig_img_path = "../IMAGE_PATH"
    img = Image.open(orig_img_path)
    result = forward(img)
    #result.image.save('test.png', 'png')
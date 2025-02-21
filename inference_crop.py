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
            crop된 사이즈의 자동차 이미지지
        '''
        orig_img = self.img.copy()
        
        crop_lst = []
        for i in result_car.boxes.xyxy:
            crop_img = orig_img.crop(tuple(i.tolist()))
            crop_lst.append(crop_img)
            
        return crop_lst
    
    def resize_crop(self, crop_img):
        '''
        crop된 이미지를 yolo_crop's input에 맞게 resize
        Args:
            img: crop된 이미지
        Reuturn:
            yolo_crop's input에 맞는 이미지
        '''
        resized_img = Result()
        resized_img.orig_shape = crop_img.size# crop size
        resized_img.image = data_utils.image_resize(crop_img)
        
        return resized_img# resized crop img
    
    def detect_plate(self, resized_lst: list):
        '''
        crop된 이미지에서 plate를 검출
        Args:
            resized_img: resized crop 이미지 객체
        Returns:
            resized crop 이미지 상에서의 yolo output
        '''
        resized_img_lst = []
        for resized_img in resized_lst:
            resized_img_lst.append(resized_img.image)
            
        result_plate = self.model_crop(resized_img_lst)# inference

        return result_plate# yolo's output
    
    def scaling(self, result_plate):
        '''
        단일 output을 orig image의 좌표로 변환한 객체로 return (1 : 1)
        Args:
            result_plate: resized crop img에 대한 yolo's output
        Returns:
            orig이미지 좌표로 스케일링
        '''
        orig_width = self.img.size[0]
        orig_height = self.img.size[1]
        crop_width = result_plate.orig_shape[0]
        crop_height = result_plate.orig_shape[1]
        
        scaling_factor_r = crop_width/orig_width
        scaling_factor_c = crop_height/orig_height
        
        xyxy = torch.tensor(result_plate.boxes.xyxy)
        xywh = torch.tensor(result_plate.boxes.xywh)
        
        xyxy[:,0] *= scaling_factor_r
        xyxy[:,1] *= scaling_factor_c
        xyxy[:,2] *= scaling_factor_r
        xyxy[:,3] *= scaling_factor_c
        
        xywh[:,0] *= scaling_factor_r
        xywh[:,1] *= scaling_factor_c
        xywh[:,2] *= scaling_factor_r
        xywh[:,3] *= scaling_factor_c
        
        result_forward = Result()
        
        result_forward.boxes.xyxy = torch.cat((result_forward.boxes.xyxy.to('cuda'), xyxy), dim=0)
        result_forward.boxes.xywh = torch.cat((result_forward.boxes.xywh.to('cuda'), xywh), dim=0)
        result_forward.orig_shape = self.img.size
        
        return result_forward

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
        
        car_xyxy = result_car.boxes.xyxy
        car_xywh = result_car.boxes.xywh
        
        resized_img_lst = []
        for img in crop_lst:
            resized_img = self.resize_crop(img)
            resized_img_lst.append(resized_img)
        
        result_plate = self.detect_plate(resized_img_lst)
        
        results = Result()
        for i in range(len(resized_img_lst)):
            result_forward = self.scaling(result_plate[i])
            
            plate_xyxy = result_plate[i].boxes.xyxy.clone()
            plate_xywh = result_plate[i].boxes.xywh.clone()

            plate_xyxy[:,:2] += car_xyxy[i,:2]
            plate_xyxy[:,2:] += car_xyxy[i,:2]
            plate_xywh[:,:2] += car_xywh[i,:2]
            
            results.boxes.xyxy = torch.cat((result_forward.boxes.xyxy, plate_xyxy), dim=0)
            results.boxes.xywh = torch.cat((result_forward.boxes.xywh, plate_xywh), dim=0)
            results.boxes.orig_shape = None

        results.image = self.plot(results)
        
        return results
    
if __name__ == "__main__":
    model_car = YOLO("/mnt/hdd_6tb/jh2020/ckpt/YOLOv11n_car.pt")
    model_crop = YOLO("/mnt/hdd_6tb/jh2020/ckpt/YOLOv11n_carcrop.pt")
    
    forward = Forward(model_car = model_car, model_crop = model_crop)
    
    orig_img_path = "/mnt/hdd_6tb/jh2020/pred/image58.png"
    img = Image.open(orig_img_path)
    result = forward(img)
    result.image.save('car_crop.png', 'png')
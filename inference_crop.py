from dataclass import dataclass
from ultralytics import YOLO
from PIL import ImageFont, Image, ImageDraw
import torch
import data_utils
'''
차량을 탐지한 후 crop한 이미지에 대해 plate탐지
todo:
    1. 객체의 개수 조정
    2. __call__ 
    3. 최종 result에 대한 nms
'''

@dataclass
class Boxes:
    xyxy: torch.Tensor = None
    xywh: torch.Tensor = None
    
@dataclass
class Result:
    boxes = Boxes()
    orig_shape: tuple = None# (w,h) != yolo format(h,w)
    image = None

class Forward:
    def __init__(self, model_car, model_crop):
        self.model_car = model_car
        self.model_crop = model_crop
        self.boxes = Boxes()
        self.img = None
        
    def __call__(self, img):
        
        return result
    
    def detect_car(self, img):
        '''
        원본 이미지에서 차량을 검출
        Args:
            img: PIL이미지
        Returns:
            result_car객체를 출력
        '''
        output = self.model_car(img, save=True)
        
        result_car = Result()
        
        result_car.boxes.xyxy = output[0].boxes.xyxy
        result_car.boxes.xywh = output[0].boxes.xywh
        result_car.orig_shape = img.shape
        
        return result_car
    
    def crop(self, result_car):
        '''
        원본 이미지에서 detect car를 crop해서 출력
        Args:
            result_car: 탐지된 car
        Returns:
            crop된 사이즈의 자동차 이미지, 원본에서의 좌표
        '''
        crop_lst = []
        orig_img = self.img
        
        for bbox in result_car.boxes.xyxy:
            bbox
        return crop_lst
    
    def crop2input(self, img):
        '''
        crop된 이미지를 yolo_crop's input에 맞게 resize
        Args:
            img: crop된 이미지
        Reuturn:
            yolo_crop's input에 맞는 이미지
        '''
        crop_img = Result()
        crop_img.orig_shape = img.size# crop size
        crop_img.image = data_utils.image_resize(img)
        
        return crop_img# cropped img
    
    def detect_plate(self, crop_img):
        '''
        crop된 이미지에서 plate를 검출
        Args:
            crop_img: crop 이미지 객체
        Returns:
            crop 이미지 상에서의 yolo output
        '''
        output = self.model_crop(crop_img.image)
        
        result_plate = Result()
        result_plate.boxes.xyxy = output.boxes.xyxy
        result_plate.boxes.xywh = output.boxes.xyxy
        result_plate.orig_shape = crop_img.orig_shape# crop size
        
        return result_plate
    
    def plate2orig(self, result_plate):
        '''
        crop 이미지에서 탐지된 plate를 orig image의 좌표로 변환
        '''
        orig_width = self.img.size[0]
        orig_height = self.img.size[1]
        crop_width = result_plate.orig_shape[0]
        crop_height = result_plate.orig_shape[1]
        
        scalied_bbox = None
        
        result_forward = Result()
        result_forward.boxes.xyxy = None
        result_forward.boxes.xywh = None
        result_forward.orig_shape = self.img.size
        
        return result_forward
    
    def nms():
        pass
    
    def plot(self, bboxes):
        pass
    
    def __call__(self):
        pass
    
if __name__ == "__main__":
    model_car = YOLO("./ckpt/YOLOv11n_car.pt")
    model_crop = YOLO("./ckpt/YOLOv11n_crop.pt")
    
    forward = Forwad() ###
    
    orig_img_path = "/mnt/hdd_6tb/jh2020/pred/image58.png"
    img = PIL.open(orig_img_path)
    result = forward(img)
    result.image.save('car_crop.png', 'png')
    
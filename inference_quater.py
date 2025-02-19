from PIL import Image, ImageDraw
import torch
import re
import os
import glob
import natsort
import data_utils
from ultralytics import YOLO
from dataclasses import dataclass
from PIL import ImageFont


@dataclass
class Boxes:
    xyxy: torch.Tensor = None
    xywh: torch.Tensor = None

@dataclass
class Result:
    boxes = Boxes()
    image = None

class QuaterYOLO:
    def __init__(self, margin, model):
        self.margin = margin
        self.model = model
        self.x_margin = None
        self.y_margin = None
        self.img = None
        self.boxes = Boxes()
        
        self.resized_width = None
        self.resized_height = None

    def quater_crop(self, image_path):
        '''
        이미지에 마진을 고려한 4분할을 진행
        Args:
            image_path: 대상 이미지 경로
        Return:
            잘린 이미지 4장과 resize된 원본 이미지
        '''
        self.img = Image.open(image_path)
        w = self.img.size[0]
        h = self.img.size[1]
        self.x_margin = w*self.margin
        self.y_margin = h*self.margin
        
        w_half = w/2
        h_half = h/2
        quater1 = self.img.crop((0, 0, w_half+(self.x_margin/2), h_half+(self.y_margin/2)))
        quater2 = self.img.crop((w_half-(self.x_margin/2), 0, w, h_half+(self.y_margin/2)))
        quater3 = self.img.crop((0, h_half-(self.y_margin/2), w_half+(self.x_margin/2), h))
        quater4 = self.img.crop((w_half+(self.x_margin/2), h_half+(self.y_margin/2), w, h))

        resized_img = data_utils.image_resize(self.img)
        self.resized_width = resized_img.size[0]
        self.resized_height = resized_img.size[1]
        
        return quater1, quater2, quater3, quater4, resized_img

    def infer_batch(self, args):
        '''
        4개의 분할된 이미지와 resized 이미지에 대한 batch inference를 진행
        Args:
            *args: crop된 이미지 4장과 resized 이미지지
        Returns:
            각 inference를 담은 list
        '''
        lst = []
        for i in args:
            lst.append(i)
        results = self.model(lst, save=True)

        return results

    def extract_xywh(self, results):
        '''
        (x1, y1, w, h)좌표만 추출
        Args:
            results: YOLO's inference list
        '''
        bboxes = []
        for i in results:
            x1y1= i.boxes.xyxy[:,:2]
            wh = i.boxes.xywh[:,2:]
            label = i.boxes.cls.unsqueeze(1)
            conf = i.boxes.conf.unsqueeze(1)
            tmp = torch.cat([x1y1, wh, label, conf], dim=1)
            bboxes.append(tmp)

        return bboxes

    
    def scaling(self, args):
        '''
        원본 이미지 스케일에 맞게 조정
        Args:
            *args: bbox list [x1, y1, w, h, lb, conf]
        '''
        bboxes1, bboxes2, bboxes3, bboxes4, bboxes5 = args

        bboxes2[:,0] += self.img.size[0]/2 -self.x_margin/2
        
        bboxes3[:,1] += self.img.size[1]/2 -self.y_margin/2
        
        bboxes4[:,0] += self.img.size[0]/2 +self.x_margin/2
        bboxes4[:,1] += self.img.size[1]/2 +self.y_margin/2
        
        bboxes5[:,0] *= self.img.size[0]/self.resized_width
        bboxes5[:,1] *= self.img.size[1]/self.resized_height
        bboxes5[:,2] *= self.img.size[0]/self.resized_width
        bboxes5[:,3] *= self.img.size[1]/self.resized_height
        
        return torch.cat([bboxes1, bboxes2, bboxes3, bboxes4, bboxes5], dim=0)
    
    @staticmethod
    def batch_iou(box, boxes):
        '''
        배치boxes와 단일box간의 IoU 계산을 좌우(lr) 상하(tb)를 기준으로 계산함
        Args:
            box: (4,) (x1, y1, w, h) 형식의 단일 bbox
            boxes: (N,4) (x1, y1, w, h) 형식의 bbox tensor
        Return:
            IoU 텐서(N,)
        '''
        lr = torch.clamp(
            torch.min(box[0] + box[2], boxes[:, 0] + boxes[:, 2]) -
            torch.max(box[0], boxes[:, 0]),
            min=0
        )
        
        tb = torch. clamp(
                torch.min(box[1] + box[3], boxes[:, 1] + boxes[:, 3]) -
                torch.max(box[1], boxes[:, 1]),
                min=0
        )
        
        intersection = lr*tb
        union = box[2]*box[3] + boxes[:, 2]*boxes[:, 3] - intersection
        ious = intersection/union
        
        return ious
    
    @staticmethod
    def nms(bboxes, threshold=0.1):
        '''
        IoU를 기준으로 nms 적용
        '''
        if len(bboxes)==0:
            return bboxes
        
        scores = bboxes[:,5].to('cuda:0')
        order = scores.argsort(descending=True).to('cuda:0')# ex) [1, 0, 2, 3]
        keep = torch.ones(len(order), dtype=torch.bool).to('cuda:0')# True:keep, False:pass
        
        for i in range(len(order)-1):
            if not keep[order[i]]:# score가 높은 bbox부터 들어오도록 order[i] 설계
                continue
            
            # num_bbox -1 개의 1차원 iou tensor
            ious = QuaterYOLO.batch_iou(bboxes[order[i]], bboxes[order[i+1:]])
            keep[order[i+1:]] &= (ious < threshold)# 1 and True인 경우에 1 할당
        
        return bboxes[keep]
    
    @staticmethod
    def textsize(text, font):
        img = Image.new(mode="P", size=(0, 0))
        draw = ImageDraw.Draw(img)
        _, _, width, height = draw.textbbox((0, 0), text=text, font=font)
        return width, height

    def plot(self, bboxes):
        '''
        최종 bbox들을 이미지에 그리기
        Args:
            bboxes: bboxes tensor
        Returns:
            bbox가 그려진 이미지
        '''
        img = self.img.copy()
        bboxes = bboxes.to('cpu').numpy()
        font = ImageFont.load_default(size=20)
        
        draw = ImageDraw.Draw(img)
        for bbox in bboxes:
            label, score = bbox[4:]
            text = f'{int(label)}: {score:.3f}'
            t_w, t_h = QuaterYOLO.textsize(text, font)
            x1, y1, w, h = bbox[:4]
            x2 = x1+w
            y2 = y1+h
            draw.rectangle((x1, y1, x2, y2), outline="red", width = 2)
            draw.text((x1, y1-t_h), text, "red", font)
            
        return img
    
    def __call__(self, image_path):
        '''
        최종 처리된 이미지를 return함
        '''
        args = self.quater_crop(image_path)
        results = self.infer_batch(args)
        bboxes= self.extract_xywh(results)
        bboxes= self.scaling(bboxes)
        bboxes= QuaterYOLO.nms(bboxes)
        
        result = Result()
        
        result.boxes.xyxy = torch.concat(
            [
                bboxes[:, :2], 
                bboxes[:, :2] + bboxes[:, 2:4]
            ]
            , dim=1
        )
        result.boxes.xywh = torch.concat(
            [
                bboxes[:, :2] + bboxes[:, 2:4]/2,
                bboxes[:, 2:4]
            ]
            , dim=1
        )
        
        result.image = self.plot(bboxes)
        return result
    

if __name__=="__main__":
    model = YOLO("/mnt/hdd_6tb/jh2020/runs/detect/tune/weights/best.pt")
    custom_model = QuaterYOLO(margin=0.1, model=model)
    #org_img_path = "/mnt/hdd_6tb/jh2020/pred/image58.png"
    #img = custom_model.infer(org_img_path)
    #img.save('test_quater.png', 'png')

    img_path = "/mnt/hdd_6tb/jh2020/processed_test/images"
    save_path = "/mnt/hdd_6tb/jh2020/runs/detect/predict5"
    img_path_lst = glob.glob(img_path+'/**/*.png', recursive=True)
    for i,path in enumerate(natsort.natsorted(img_path_lst)):
        result = custom_model(path)
        result.image.save(f"{save_path}/image{i}.png", "png")
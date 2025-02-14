from PIL import Image, ImageDraw
import torch
import re
import os
from ultralytics import YOLO


class InferenceQuater:
    def __init__(self, margin, model):
        self.margin = margin
        self.model = model
        self.x_margin = None
        self.y_margin = None
        self.img = None

    def quater_crop(self, image_path):
        '''
        이미지에 마진을 고려한 4분할을 진행
        '''
        self.img = Image.open(image_path)
        w = self.img.size[0]
        h = self.img.size[1]
        self.x_margin = w*self.margin
        self.y_margin = h*self.margin
        
        w_half = w/2
        h_hlaf = h/2
        bboxes1 = self.img.crop((0, 0, w_half+(self.x_margin/2), h_hlaf+(self.y_margin/2)))
        bboxes2 = self.img.crop((w_half-(self.x_margin/2), 0, w, h_hlaf+(self.y_margin/2)))
        bboxes3 = self.img.crop((0, h_hlaf-(self.x_margin/2), w_half+(self.x_margin/2), h))
        bboxes4 = self.img.crop((w_half+(self.x_margin/2), h_hlaf+(self.y_margin/2), w, h))
        return bboxes1, bboxes2, bboxes3, bboxes4

    def infer_quater(self, *args):
        '''
        4개의 분할된 이미지에 대한 batch inference를 진행
        Args:
            *args: crop된 이미지 4장
        Returns:
            각 inference를 담은 list
        '''
        results = self.model(list(args),save=True)

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

    
    def scaling(self, *args):
        '''
        원본 이미지 스케일에 맞게 조정
        Args:
            *args: bbox list
        '''
        bboxes1, bboxes2, bboxes3, bboxes4 = args

        bboxes2[:,0] -= self.x_margin # x 방향 -margin
        bboxes3[:,1] -= self.y_margin # y 방향 -margin
        
        bboxes4[:,0] -= self.x_margin # x 방향 -margin
        bboxes4[:,1] -= self.y_margin # y 방향 -margin
            
        return torch.cat([bboxes1, bboxes2, bboxes3, bboxes4], dim=0)
    
    @staticmethod
    def batch_iou(box, boxes):
        '''
        배치boxes와 단일box간의 IoU 계산을 좌우(lr) 상하(tb)를 기준으로 계산함
        Args:
            box: (4,) (x1, x2, w, h) 형식의 단일 bbox
            boxes: (N,4) (x1, x2, w, h) 형식의 bbox tensor
        Return:
            IoU 텐서(N,)
        '''
        lr = torch.clamp(
            torch.min(box[0] + box[2], boxes[:, 0] + boxes[:, 2]) -\
            torch.max(box[0], boxes[:, 0]),
            min=0
        )
        
        tb = torch. clamp(
            torch.min(box[1] + box[3], boxes[:, 1] + boxes[:, 3]) -\
            torch.max(box[1], boxes[:, 1]),
            min=0
        )
        
        intersection = lr*tb
        union = box[2]*box[3] + boxes[:, 2]*boxes[:, 3]
        
        return intersection/union
    
    @staticmethod
    def nms(bboxes, threshold=0.2):
        '''
        IoU를 기준으로 nms 적용
        '''
        if len(bboxes)==0:
            return bboxes
        
        scores = bboxes[:,5] 
        order = scores.argsort(descending=True)# ex) [1, 0, 2, 3]
        keep = torch.ones(len(order), dtype=torch.bool)# True:keep, False:pass
        
        for i in range(len(order)-1):
            if not keep[order[i]]:# score가 높은 bbox부터 들어오도록 order[i] 설계
                continue
            
            # num_bbox -1 개의 1차원 iou tensor
            ious = InferenceQuater.batch_iou(bboxes[order[i]], bboxes[order[i+1,:]])
            keep[order[i+1:]] &= (ious < threshold)# 1 and True인 경우에 1 할당
        
        return bboxes[keep]

    def plot(self, bboxes):
        '''
        최종 bbox들을 이미지에 그리기
        '''
        
        return image
    
    def __call__(self, image_path):
        '''
        최종 처리된 이미지를 return함
        '''
        args = self.quater_crop(image_path)
        results = self.infer_quater(args)
        bboxes= self.extract_xywh(results)
        bboxes= self.scaling(bboxes)
        # bboxes= InferenceQuater.nms(bboxes)
        # image = self.plot(bboxes)
        return image
    

if __name__=="__main__":
    model = YOLO("/mnt/hdd_6tb/jh2020/runs/detect/train28/weights/best.pt")
    custom_model = InferenceQuater(margin=0.1, model=model)
    for i in image_path_lst:
        image = custom_model(i)
        image.save
    
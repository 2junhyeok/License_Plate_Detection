import torch
import numpy as np
from PIL import Image
import natsort
import glob
from ultralytics import YOLO
from inference_quater import QuaterYOLO

device = "cuda:0"

def matching(A: torch.tensor,B: torch.tensor, threshold=0.1):
    '''
    iou nms matching list
    Args:
        param 'A': YOLO's result
        param 'B': ground truth .txt
    '''
    if len(A)>0:# num pred > 0
        iou_mat = torch.stack([QuaterYOLO.batch_iou(a, B) for a in A], dim=0)
        '''
        for a in A:
            ious = QuaterYOLO.batch_iou(a, B)
            iou_mat.append(ious)
        iou_mat = torch.stack(iou_mat, dim=0)
        '''
        matched_A = []
        matched_B = []
        
        while iou_mat.max() > threshold:
            i,j = torch.where(iou_mat == iou_mat.max())
            i = i[0].item()
            j = j[0].item()
            
            matched_A.append(A[i])
            matched_B.append(B[j])
            
            iou_mat[i,:] = 0
            iou_mat[:,j] = 0
    elif len(A) == 0:
        matched_A = []
        matched_B = []
        
    return matched_A, matched_B

def eval(img_path,gt_path, model, mode='dev'):
    '''
    img_path, model -> pred
    gt, pred -> matching
    matcing -> recall score
    Args:
        mode: 'base', 'dev'
    '''
    img_path_lst = glob.glob(img_path+'/**/*.png', recursive=True)
    gt_path_lst = glob.glob(gt_path+'**/*.txt', recursive=True)
    img_path_lst = natsort.natsorted(img_path_lst)
    gt_path_lst = natsort.natsorted(gt_path_lst)
    
    TP = 0
    TP_FN = 0
    for img_path, gt_path in zip(img_path_lst, gt_path_lst):
        
        if mode=='base':
            img = Image.open(img_path)
            result = model(img, save=True)
        elif mode=='dev':
            result = model(img_path)
        
        with open(gt_path) as gt_lst:
            lst = [list(map(float, line.split()[1:])) for line in gt_lst]

            if mode=='base':
                A = result[0].boxes.xywh.to(device)
            elif mode=='dev':
                A = result.boxes.xywh.to(device)
            
            A = A[:, :4]
            B = torch.stack([torch.tensor(i[:4]) for i in lst], dim=0).to(device) # xywh

            matched_A, _ = matching(A, B)
            TP += len(matched_A)
            TP_FN += len(B)
            
    return TP/TP_FN


if __name__=="__main__":
    test_img_path = "/mnt/hdd_6tb/jh2020/processed_test/images"
    test_label_path = "/mnt/hdd_6tb/jh2020/processed_test/labels" # gt
    model = YOLO("/mnt/hdd_6tb/jh2020/runs/detect/tune/weights/best.pt")
    #model = QuaterYOLO(margin=0.1, model=model)
    print("recall score: ", eval(test_img_path, test_label_path, model, mode='base'))
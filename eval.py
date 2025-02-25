import torch
import numpy as np
from PIL import Image
import natsort
import glob
from PIL import Image, ImageDraw
from ultralytics import YOLO
from inference_quater import QuaterYOLO
from inference_crop import Forward

device = "cuda:0"

def matching(A: torch.tensor,B: torch.tensor, threshold=0.5):
    '''
    iou nms matching list
    Args:
        param 'A': YOLO's result
        param 'B': ground truth .txt
    '''
    if len(A)>0:# num pred > 0
        iou_mat = []
        for a in A:
            iou_mat.append(QuaterYOLO.batch_iou(a, B))
        iou_mat = torch.stack(iou_mat, dim=0)

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

def eval(img_path,gt_path, model, mode='PATH'):
    '''
    img_path, model -> pred
    gt, pred -> matching
    matcing -> recall score
    Args:
        mode: 'IMAGE', 'PATH'
    '''
    img_path_lst = glob.glob(img_path+'/**/*.png', recursive=True)
    gt_path_lst = glob.glob(gt_path+'**/*.txt', recursive=True)
    img_path_lst = natsort.natsorted(img_path_lst)
    gt_path_lst = natsort.natsorted(gt_path_lst)
    
    TP = 0
    TP_FN = 0
    for img_path, gt_path in zip(img_path_lst, gt_path_lst):
        
        if mode=='IMAGE':
            img = Image.open(img_path)
            result = model(img)
        elif mode=='PATH':
            result = model(img_path)
        
        with open(gt_path) as gt_lst:
            lst = [list(map(float, line.split()[1:])) for line in gt_lst]

            if mode=='IMAGE' and model == 'base':
                A = result[0].boxes.xywh.to(device)
                A = torch.cat([A[:,:2] - A[:,2:]/2, A[:, 2:]], dim=1)# [xc,yc,w,h]->[x1,y1,w,h]
            else:
                A = result.boxes.xywh.to(device)
                A = torch.cat([A[:,:2] - A[:,2:]/2, A[:, 2:]], dim=1)# [xc,yc,w,h]->[x1,y1,w,h]
            B = torch.stack([torch.tensor(i[:4]) for i in lst], dim=0).to(device)
            
            
            matched_A, _ = matching(A, B)
            TP += len(matched_A)
            TP_FN += len(B)
            
    return TP/TP_FN


if __name__=="__main__":
    test_img_path = "/mnt/hdd_6tb/jh2020/processed_test/images"
    test_label_path = "/mnt/hdd_6tb/jh2020/processed_test/labels" # gt
    model = YOLO("/mnt/hdd_6tb/jh2020/runs/detect/tune/weights/best.pt")
    model_car = YOLO("/mnt/hdd_6tb/jh2020/ckpt/YOLOv11n_car.pt")
    model_crop = YOLO("/mnt/hdd_6tb/jh2020/ckpt/YOLOv11n_carcrop.pt")
    
    mode = "IMAGE"# IMAGE, PATH
    model = "Crop"# Quater, Crop
    
    if mode=="PATH":
        if model == "Quater":
            model = QuaterYOLO(margin=0.1, model=model)
    elif mode=="IMAGE":
        if model == "Crop":
            model = Forward(model_car = model_car, model_crop = model_crop)
        if model =="base":
            pass
    
    print("recall score: ", eval(test_img_path, test_label_path, model, mode=mode))
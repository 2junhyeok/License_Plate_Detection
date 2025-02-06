import torch
import numpy as np
from PIL import Image
import natsort
import glob
from ultralytics import YOLO


def pred_to_xyxwyh(result):
    xywh_tensor = result.boxes.xywh
    xyxy_tensor = result.boxes.xyxy
    lst = []
    for i in zip(xywh_tensor, xyxy_tensor):
        new = torch.Tensor((i[0][0],i[0][1],i[0][2]+i[1][2],i[0][1]+i[1][3]))
        lst.append(new)
    return lst

def label_to_xyxwyh(label):
    lst = []
    for i in label:
        new = torch.Tensor((i[0],i[1],i[0]+i[2],i[1]+i[3]))
        lst.append(new)
    return lst

def cos_sim(A: torch.tensor,B: torch.tensor)->torch.tensor:
    dot_product = A @ B.T
    norm = np.linalg.norm(A, axis=1, keepdims=True)@np.linalg.norm(B, axis=1, keepdims=True).T
    return dot_product/norm

def matching():
    pass

def IoU():
    pass

def eval(img_path,gt_path, model):
    '''
    img_path, model -> pred
    pred, label -> cos sim
    cos sim -> IoU
    IoU -> recall score
    '''
    img_path_lst = glob.glob(img_path+'/**/*.png', recursive=True)
    gt_path_lst = glob.glob(gt_path+'**/*.txt', recursive=True)
    img_path_lst = natsort.natsorted(img_path_lst)
    gt_path_lst = natsort.natsorted(gt_path_lst)
    
    for img_path in img_path_lst:
        img = Image.open(img_path)
        result = model(img, save=True)
        
        

if __name__=="__main__":
    test_img_path = "/mnt/hdd_6tb/jh2020/processed_test/images"
    test_label_path = "/mnt/hdd_6tb/jh2020/processed_test/labels" # gt
    save_pred_bbox_path = "/mnt/hdd_6tb/jh2020/processed_test/pred"
    model = YOLO("/mnt/hdd_6tb/jh2020/runs/detect/train28/weights/best.pt")
    
    
    tst = "/mnt/hdd_6tb/jh2020/processed_test/images/image1.png"
    result = model(tst, save = True)
    
    
    pred_tensor = result[0]
    gt_tensor = torch.tensor(lst)

    
    A = pred_tensor.to('cpu')
    B = gt_tensor.to('cpu')
    
    
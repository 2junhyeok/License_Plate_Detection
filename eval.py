import torch
import numpy as np
from PIL import Image
import natsort
import glob
from ultralytics import YOLO


def pred_to_xyxyw(result):
    '''
    xyxy,xywh -> xyxyw
    '''
    xywh_tensor = result.boxes.xywh
    xyxy_tensor = result.boxes.xyxy
    lst = []
    for i in zip(xywh_tensor, xyxy_tensor):
        new = torch.Tensor((i[0][0],i[0][1],i[0][2]+i[1][2],i[0][1]+i[1][3],i[0][2])).to('cpu').numpy()
        lst.append(new)
    return np.array(lst)

def label_to_xyxyw(label):
    '''
    xywh -> xyxyw
    '''
    lst = []
    for i in label:
        new = torch.Tensor((i[0],i[1],i[0]+i[2],i[1]+i[3],i[2])).to('cpu').numpy()
        lst.append(new)
    return np.array(lst)

def cos_sim(A: np.ndarray,B: np.ndarray)->np.ndarray:
    dot_product = A @ B.T
    norm_A = np.linalg.norm(A, axis=1)[:,np.newaxis]
    norm_B = np.linalg.norm(B, axis=1)[np.newaxis,:]
    sim_mat = dot_product/(norm_A @ norm_B)
    return sim_mat


def matching(A: np.ndarray,B: np.ndarray, threshold=0.9)-> list:
    '''
    cosine similarity matrix to matching list
    :param 'A': YOLO's result
    :param 'B': ground truth .txt
    '''
    gt_lst = []
    pred_lst = []
    tmp_mat = cos_sim(A,B) # rank mat
    while tmp_mat.max() > threshold:
        i,j = np.where(tmp_mat==tmp_mat.max())
        i,j = i[0],j[0]
        
        tmp_mat[i,:] = 0
        tmp_mat[:,j] = 0
        gt_lst.append(A[i.item()].tolist())
        pred_lst.append(B[j.item()].tolist())
    return pred_lst, gt_lst# matching list



def eval(img_path,gt_path, model):
    '''
    img_path, model -> pred
    gt, pred -> matching
    matcing -> recall score
    '''
    img_path_lst = glob.glob(img_path+'/**/*.png', recursive=True)
    gt_path_lst = glob.glob(gt_path+'**/*.txt', recursive=True)
    img_path_lst = natsort.natsorted(img_path_lst)
    gt_path_lst = natsort.natsorted(gt_path_lst)
    
    TP = 0
    TP_FN = 0
    for img_path, gt_path in zip(img_path_lst, gt_path_lst):
        img = Image.open(img_path)
        result = model(img, save=True)
        
        with open(gt_path) as gt_lst:
            lst = [list(map(float, line.split()[1:])) for line in gt_lst]
        
        A = pred_to_xyxyw(result[0])
        B = label_to_xyxyw(lst)
        try:
            pred_lst, _ = matching(A, B)
            TP += len(pred_lst)
            TP_FN += len(B)
        except:# num_pred==0 or num_ground truth==0
            print(f"num_pred: {len(pred_lst)}")
            print(f"num_ground truth: {len(B)}")
            pass
    return TP/TP_FN
        
        
        
        

if __name__=="__main__":
    test_img_path = "/mnt/hdd_6tb/jh2020/processed_test/images"
    test_label_path = "/mnt/hdd_6tb/jh2020/processed_test/labels" # gt
    model = YOLO("/mnt/hdd_6tb/jh2020/runs/detect/train28/weights/best.pt")

    print("recall score: ", eval(test_img_path, test_label_path, model))
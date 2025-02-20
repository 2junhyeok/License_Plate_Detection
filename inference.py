import os
import glob
import cv2
from PIL import Image
from ultralytics import YOLO
from tqdm import tqdm
import argparse

'''
동영상을 대상으로 frame을 추출하여 inference를 한 뒤 저장하는 코드
'''

def get_pred_from_video(file_path, save_path, img_per_sec):
    
    # open video
    '''
    openCV는 streaming방식으로 video를 load
    '''
    video = cv2.VideoCapture(file_path)
    assert video.isOpened(), "could not open file path"
        
    len_video = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)# 30

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    cnt = 0
    with tqdm(total = len_video) as pbar:
        model = YOLO("/mnt/hdd_6tb/jh2020/runs/detect/train28/weights/best.pt")
        while video.isOpened():
            ret, image = video.read()# bool, array(1080,1920,3)

            
            if not ret:# read all
                break
            frame_idx = int(video.get(cv2.CAP_PROP_POS_FRAMES))# frame index
            
            assert 0<img_per_sec<=30, "out of range"
            
            if frame_idx % int(fps/img_per_sec) == 0:
                save_idx = str(cnt +1)
                save_img_path = os.path.join(save_path, f"image{save_idx}.png")
                cv2.imwrite(save_img_path, image)
                cnt +=1
            results = model(save_img_path, save=True)
    video.release()

def get_args():
    parser = argparse.ArgumentParser(description="inference")
    parser.add_argument('--fps', type=int, default=30, help="fps")
    parser.add_argument('--img_per_sec', type=int, default=30, help="saving images per second")

    return parser.parse_args()


if __name__ == "__main__":
    file_path = "/mnt/hdd_6tb/jh2020/sample_video/cctv50mm.mp4"
    save_path = "/mnt/hdd_6tb/jh2020/pred"
    args = get_args()
    get_pred_from_video(file_path,
                         save_path,
                         args.img_per_sec)
    
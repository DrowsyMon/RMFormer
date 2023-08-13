import numpy as np
import os 
import cv2 as cv2
from tqdm import tqdm

def edge_map_generate(m_dir, e_dir):
    mask_dir = m_dir
    edge_dir = e_dir
    mask_file_list = os.listdir(mask_dir)
    for files in tqdm(mask_file_list):
        mask_file = mask_dir + files
        out_file = edge_dir + files

        mask = cv2.imread(mask_file, 0)
        kernel_size = int(min(mask.shape) * 0.004)
        kernel = np.ones((kernel_size,kernel_size), np.int8)
        canny = cv2.Canny(mask, 0, 100)
        dilate  =cv2.dilate(canny, kernel, iterations=1)

        cv2.imwrite(out_file, dilate)


if __name__ == '__main__':

    # change to your own dir

    mask_dir = 'train_data/MIX-KUH/KUH_train/mask/'
    edge_dir = 'train_data/MIX-KUH/KUH_train/edge/'

    edge_map_generate(mask_dir, edge_dir)


    print('OK...........................')

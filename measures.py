# MAE, Precision, Recall, F-measure, IoU, Precision-Recall curves
import numpy as np
from skimage import io
import cv2

# import matplotlib.pyplot as plt

def mask_normalize(mask):
# input 'mask': HxW
# output: HxW [0,255]
    return mask/(np.amax(mask)+1e-8)

def compute_mae(mask1,mask2):
# input 'mask1': HxW or HxWxn (asumme that all the n channels are the same and only the first channel will be used)
#       'mask2': HxW or HxWxn
# output: a value MAE, Mean Absolute Error
    if(len(mask1.shape)<2 or len(mask2.shape)<2):
        print("ERROR: Mask1 or mask2 is not matrix!")
        exit()
    if(len(mask1.shape)>2):
        mask1 = mask1[:,:,0]
    if(len(mask2.shape)>2):
        mask2 = mask2[:,:,0]
    if(mask1.shape!=mask2.shape):
        print("ERROR: The shapes of mask1 and mask2 are different!")
        exit()

    h,w = mask1.shape[0],mask1.shape[1]
    mask1 = mask_normalize(mask1)
    mask2 = mask_normalize(mask2)
    sumError = np.sum(np.absolute((mask1.astype(float) - mask2.astype(float))))
    maeError = sumError/(float(h)*float(w)+1e-8)

    return maeError


def compute_ave_MAE_of_methods(gt_name_list,rs_dir_lists):
#input 'gt_name_list': ground truth name list
#input 'rs_dir_lists': to-be-evaluated mask directories (not the file names, just folder names)
#output average Mean Absolute Error, 1xN, N is the number of folders
#output 'gt2rs': numpy array with shape of (num_rs_dir)

    num_gt = len(gt_name_list) # number of ground truth files
    num_rs_dir = len(rs_dir_lists) # number of method folders
    if(num_gt==0):
        print("ERROR: The ground truth directory is empty!")
        exit()

    mae = np.zeros((num_gt,num_rs_dir)) # MAE of methods
    gt2rs = np.zeros((num_gt,num_rs_dir)) # indicate if the mask mae of methods is correctly computed
    for i in range(0,num_gt):
        print('-Processed %d/%d'%(i+1,num_gt),end='\r')
        #print("Completed {:2.0%}".format(i / num_gt), end="\r") # print percentile of processed, python 3.0 and newer version
        gt = io.imread(gt_name_list[i]) # read ground truth
        gt_name = gt_name_list[i].split('/')[-1] # get the file name of the ground truth
        for j in range(0,num_rs_dir):
            tmp_mae = 0.0
            try:
                rs = io.imread(rs_dir_lists[j]+gt_name) # read the corresponding mask of each method
            except IOError:
                #print('ERROR: Couldn\'t find the following file:',rs_dir_lists[j]+gt_name)
                continue
            try:
                tmp_mae = compute_mae(gt,rs) # compute the mae
            except IOError:
                #print('ERROR: Fails in compute_mae!')
                continue
            mae[i][j] = tmp_mae
            gt2rs[i][j] = 1.0
    mae_col_sum = np.sum(mae,0) # compute the sum of MAE of each method
    gt2rs = np.sum(gt2rs,0) # compute the number of correctly computed MAE of each method
    ave_maes = mae_col_sum/(gt2rs+1e-8) # compute the average MAE of each method
    return ave_maes, gt2rs


def compute_pre_rec(gt,mask,mybins=np.arange(0,256)):

    if(len(gt.shape)<2 or len(mask.shape)<2):
        print("ERROR: gt or mask is not matrix!")
        exit()
    if(len(gt.shape)>2): # convert to one channel
        gt = gt[:,:,0]
    if(len(mask.shape)>2): # convert to one channel
        mask = mask[:,:,0]
    if(gt.shape!=mask.shape):
        print("ERROR: The shapes of gt and mask are different!")
        exit()

    gtNum = gt[gt>128].size # pixel number of ground truth foreground regions
    pp = mask[gt>128] # mask predicted pixel values in the ground truth foreground region
    nn = mask[gt<=128] # mask predicted pixel values in the ground truth bacground region

    pp_hist,pp_edges = np.histogram(pp,bins=mybins) #count pixel numbers with values in each interval [0,1),[1,2),...,[mybins[i],mybins[i+1]),...,[254,255)
    nn_hist,nn_edges = np.histogram(nn,bins=mybins)

    pp_hist_flip = np.flipud(pp_hist) # reverse the histogram to the following order: (255,254],...,(mybins[i+1],mybins[i]],...,(2,1],(1,0]
    nn_hist_flip = np.flipud(nn_hist)

    pp_hist_flip_cum = np.cumsum(pp_hist_flip) # accumulate the pixel number in intervals: (255,254],(255,253],...,(255,mybins[i]],...,(255,0]
    nn_hist_flip_cum = np.cumsum(nn_hist_flip)

    precision = pp_hist_flip_cum/(pp_hist_flip_cum + nn_hist_flip_cum+1e-8) #TP/(TP+FP)
    recall = pp_hist_flip_cum/(gtNum+1e-8) #TP/(TP+FN)

    precision[np.isnan(precision)]= 0.0
    recall[np.isnan(recall)] = 0.0

    return np.reshape(precision,(len(precision))),np.reshape(recall,(len(recall)))


def compute_PRE_REC_FM_of_methods(gt_name_list,rs_dir_lists,beta=0.3):
#input 'gt_name_list': ground truth name list
#input 'rs_dir_lists': to-be-evaluated mask directories (not the file names, just folder names)
#output precision 'PRE': numpy array with shape of (num_rs_dir, 256)
#       recall    'REC': numpy array with shape of (num_rs_dir, 256)
#       F-measure (beta) 'FM': numpy array with shape of (num_rs_dir, 256)

    mybins = np.arange(0,256) # different thresholds to achieve binarized masks for pre, rec, Fm measures

    num_gt = len(gt_name_list) # number of ground truth files
    num_rs_dir = len(rs_dir_lists) # number of method folders
    if(num_gt==0):
        #print("ERROR: The ground truth directory is empty!")
        exit()

    PRE = np.zeros((num_gt,num_rs_dir,len(mybins)-1)) # PRE: with shape of (num_gt, num_rs_dir, 256)
    REC = np.zeros((num_gt,num_rs_dir,len(mybins)-1)) # REC: the same shape with PRE
    # FM = np.zeros((num_gt,num_rs_dir,len(mybins)-1)) # Fm: the same shape with PRE
    gt2rs = np.zeros((num_gt,num_rs_dir)) # indicate if the mask of methods is correctly computed

    for i in range(0,num_gt):
        print('>>Processed %d/%d'%(i+1,num_gt),end='\r')
        gt = io.imread(gt_name_list[i]) # read ground truth
        gt = mask_normalize(gt)*255.0 # convert gt to [0,255]
        gt_name = gt_name_list[i].split('/')[-1] # get the file name of the ground truth "xxx.png"

        for j in range(0,num_rs_dir):
            pre, rec, f = np.zeros(len(mybins)), np.zeros(len(mybins)), np.zeros(len(mybins)) # pre, rec, f or one mask w.r.t different thresholds
            try:
                rs = io.imread(rs_dir_lists[j]+gt_name) # read the corresponding mask from each method
                rs = mask_normalize(rs)*255.0 # convert rs to [0,255]
            except IOError:
                #print('ERROR: Couldn\'t find the following file:',rs_dir_lists[j]+gt_name)
                continue
            try:
                pre, rec = compute_pre_rec(gt,rs,mybins=np.arange(0,256))
            except IOError:
                #print('ERROR: Fails in compute_mae!')
                continue

            PRE[i,j,:] = pre
            REC[i,j,:] = rec
            gt2rs[i,j] = 1.0
    print('\n')
    gt2rs = np.sum(gt2rs,0) # num_rs_dir
    gt2rs = np.repeat(gt2rs[:, np.newaxis], 255, axis=1) #num_rs_dirx255

    PRE = np.sum(PRE,0)/(gt2rs+1e-8) # num_rs_dirx255, average PRE over the whole dataset at every threshold
    REC = np.sum(REC,0)/(gt2rs+1e-8) # num_rs_dirx255
    FM = (1+beta)*PRE*REC/(beta*PRE+REC+1e-8) # num_rs_dirx255

    return PRE, REC, FM, gt2rs


def compute_MAE_F_S(gt_name_list,rs_dir_lists,beta=0.3):
    mybins = np.arange(0,256) # different thresholds to achieve binarized masks for pre, rec, Fm measures

    num_gt = len(gt_name_list) # number of ground truth files
    num_rs_dir = len(rs_dir_lists) # number of method folders
    if(num_gt==0):
        print("ERROR: The ground truth directory is empty!")
        exit()

    mae = np.zeros((num_gt,num_rs_dir)) # MAE of methods
    PRE = np.zeros((num_gt,num_rs_dir,len(mybins)-1)) # PRE: with shape of (num_gt, num_rs_dir, 256)
    REC = np.zeros((num_gt,num_rs_dir,len(mybins)-1)) # REC: the same shape with PRE
    # FM = np.zeros((num_gt,num_rs_dir,len(mybins)-1)) # Fm: the same shape with PRE
    gt2rs = np.zeros((num_gt,num_rs_dir)) # indicate if the mask of methods is correctly computed
    badlist = []

    for i in range(0,num_gt):
        print('>>Processed %d/%d'%(i+1,num_gt),end='\r')
        try:
            gt = io.imread(gt_name_list[i])
        except ValueError:
            print(gt_name_list[i])
            continue
        # gt = io.imread(gt_name_list[i]) # read ground truth
        gt = mask_normalize(gt)*255.0 # convert gt to [0,255]
        gt_name = gt_name_list[i].split('/')[-1] # get the file name of the ground truth "xxx.png"

        for j in range(0,num_rs_dir):
            tmp_mae = 0.0
            pre, rec, f = np.zeros(len(mybins)), np.zeros(len(mybins)), np.zeros(len(mybins)) # pre, rec, f or one mask w.r.t different thresholds
            try:
                rs = io.imread(rs_dir_lists[j]+gt_name) # read the corresponding mask from each method
                rs = mask_normalize(rs)*255.0 # convert rs to [0,255]
            except ValueError:
                #print('ERROR: Couldn\'t find the following file:',rs_dir_lists[j]+gt_name)
                print(rs_dir_lists[j]+gt_name)
                continue
            try:
                tmp_mae = compute_mae(gt,rs)
                pre, rec = compute_pre_rec(gt,rs,mybins=np.arange(0,256))
            except IOError:
                #print('ERROR: Fails in compute_mae!')
                continue

            mae[i][j] = tmp_mae
            PRE[i,j,:] = pre
            REC[i,j,:] = rec
            gt2rs[i,j] = 1.0

            # bestl = np.where(pre>rec)[0]
            # if len(bestl) == 0:
            #     badlist.append(rs_dir_lists[j]+gt_name)
            # else:
            #     best_pr_loc = bestl[-1]
            #     if pre[best_pr_loc] < 0.65:
            #         badlist.append(rs_dir_lists[j]+gt_name)
            #     else:
            #         continue
            if tmp_mae > 0.30:
                badlist.append(rs_dir_lists[j]+gt_name)


    print('\n')

    mae_col_sum = np.sum(mae,0) # compute the sum of MAE of each method
    gt2rs = np.sum(gt2rs,0) # num_rs_dir
    ave_maes = mae_col_sum/(gt2rs+1e-8) # compute the average MAE of each method

    gt2rs = np.repeat(gt2rs[:, np.newaxis], 255, axis=1) #num_rs_dirx255

    PRE2 = np.sum(PRE,0)/(gt2rs+1e-8) # num_rs_dirx255, average PRE over the whole dataset at every threshold
    REC2 = np.sum(REC,0)/(gt2rs+1e-8) # num_rs_dirx255
    FM = (1+beta)*PRE2*REC2/(beta*PRE2+REC2+1e-8) # num_rs_dirx255

    return PRE2, REC2, FM, gt2rs,ave_maes, PRE, REC, badlist

def get_disk_kernel(radius):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius*2+1, radius*2+1))

def compute_boundary_acc(gt, mask):
    gt = gt > 128
    mask = mask > 128
    gt = gt.astype(np.uint8)
    mask = mask.astype(np.uint8)
    if(len(gt.shape)>2):
        gt = gt[:,:,0]
    if(len(mask.shape)>2):
        mask = mask[:,:,0]



    h, w = gt.shape

    min_radius = 1
    max_radius = (w+h)/300
    num_steps = 5

    mask_acc = [None] * num_steps

    for i in range(num_steps):
        curr_radius = min_radius + int((max_radius-min_radius)/num_steps*i)

        kernel = get_disk_kernel(curr_radius)
        boundary_region = cv2.morphologyEx(gt, cv2.MORPH_GRADIENT, kernel) > 0

        gt_in_bound = gt[boundary_region]
        mask_in_bound = mask[boundary_region]

        num_edge_pixels = (boundary_region).sum()
        num_mask_gd_pix = ((gt_in_bound) * (mask_in_bound) + (1-gt_in_bound) * (1-mask_in_bound)).sum()

        mask_acc[i] = num_mask_gd_pix / num_edge_pixels

    return sum(mask_acc)/num_steps



def compute_MAE_F_S_cv2(gt_name_list,rs_dir_lists,beta=0.3):
    mybins = np.arange(0,256) # different thresholds to achieve binarized masks for pre, rec, Fm measures

    num_gt = len(gt_name_list) # number of ground truth files
    num_rs_dir = len(rs_dir_lists) # number of method folders
    if(num_gt==0):
        print("ERROR: The ground truth directory is empty!")
        exit()

    mae = np.zeros((num_gt,num_rs_dir)) # MAE of methods
    PRE = np.zeros((num_gt,num_rs_dir,len(mybins)-1)) # PRE: with shape of (num_gt, num_rs_dir, 256)
    REC = np.zeros((num_gt,num_rs_dir,len(mybins)-1)) # REC: the same shape with PRE
    # FM = np.zeros((num_gt,num_rs_dir,len(mybins)-1)) # Fm: the same shape with PRE
    gt2rs = np.zeros((num_gt,num_rs_dir)) # indicate if the mask of methods is correctly computed
    badlist = []

    total_mask_acc = np.zeros((num_gt,num_rs_dir))

    for i in range(0,num_gt):
        print('>>Processed %d/%d'%(i+1,num_gt),end='\r')
        try:
            # gt = io.imread(gt_name_list[i])
            gt = cv2.imread(gt_name_list[i])
        except ValueError:
            print(gt_name_list[i])
            continue
        # gt = io.imread(gt_name_list[i]) # read ground truth
        gt = mask_normalize(gt)*255.0 # convert gt to [0,255]
        gt_name = gt_name_list[i].split('/')[-1] # get the file name of the ground truth "xxx.png"

        for j in range(0,num_rs_dir):
            tmp_mae = 0.0
            pre, rec, f = np.zeros(len(mybins)), np.zeros(len(mybins)), np.zeros(len(mybins)) # pre, rec, f or one mask w.r.t different thresholds
            try:
                # rs = io.imread(rs_dir_lists[j]+gt_name) # read the corresponding mask from each method
                rs = cv2.imread(rs_dir_lists[j]+gt_name)
                rs = mask_normalize(rs)*255.0 # convert rs to [0,255]
            except ValueError:
                #print('ERROR: Couldn\'t find the following file:',rs_dir_lists[j]+gt_name)
                print(rs_dir_lists[j]+gt_name)
                continue
            try:
                tmp_mae = compute_mae(gt,rs)
                pre, rec = compute_pre_rec(gt,rs,mybins=np.arange(0,256))

                # mba
                mask_acc = compute_boundary_acc(gt, rs)
            except IOError:
                #print('ERROR: Fails in compute_mae!')
                continue

            mae[i][j] = tmp_mae
            total_mask_acc[i][j] = mask_acc
            PRE[i,j,:] = pre
            REC[i,j,:] = rec
            gt2rs[i,j] = 1.0

            if tmp_mae >0.20:
                badlist.append(rs_dir_lists[j]+gt_name)
            
            
            # bestl = np.where(pre>rec)[0]



            # if len(bestl) == 0:
            #     badlist.append(rs_dir_lists[j]+gt_name)
            # else:
            #     best_pr_loc = bestl[-1]
            #     if pre[best_pr_loc] < 0.65:
            #         badlist.append(rs_dir_lists[j]+gt_name)
            #     else:
            #         continue


    print('\n')

    mae_col_sum = np.sum(mae,0) # compute the sum of MAE of each method
    gt2rs = np.sum(gt2rs,0) # num_rs_dir
    ave_maes = mae_col_sum/(gt2rs+1e-8) # compute the average MAE of each method

    new_mba = np.sum(total_mask_acc,0) / (gt2rs+1e-8)

    gt2rs = np.repeat(gt2rs[:, np.newaxis], 255, axis=1) #num_rs_dirx255

    PRE2 = np.sum(PRE,0)/(gt2rs+1e-8) # num_rs_dirx255, average PRE over the whole dataset at every threshold
    REC2 = np.sum(REC,0)/(gt2rs+1e-8) # num_rs_dirx255
    FM = (1+beta)*PRE2*REC2/(beta*PRE2+REC2+1e-8) # num_rs_dirx255

    return PRE2, REC2, FM, gt2rs,ave_maes, PRE, REC, badlist, new_mba
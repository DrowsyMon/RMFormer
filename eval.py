import numpy as np
import glob
from measures import compute_MAE_F_S_cv2

def qual_eval(lbl_dir='./lbl/',results_dir='./test/res/'):
    ## 0. =======set the data path=======
    # print("------0. set the data path------")

    # >>>>>>> Follows have to be manually configured <<<<<<< #
    gt_dir = lbl_dir
                        # ground truth and results to-be-evaluated should be in this directory
                        # the figures of PR and F-measure curves will be saved in this directory as well

    rs_dirs = [results_dir]

    # >>>>>>> Above have to be manually configured <<<<<<< #

    gt_name_list = glob.glob(gt_dir+'*.png') # get the ground truth file name list

    ## get directory list of predicted maps
    rs_dir_lists = []
    for i in range(len(rs_dirs)):
        rs_dir_lists.append(rs_dirs[i])
    # print('\n')
    # 1. =======compute the average MAE of methods=========
    print("Compute Segmentation Eval------")
    PRE, REC, FM, gt2rs_fm,aveMAE, PA, RA, badlist, mba = compute_MAE_F_S_cv2(gt_name_list,rs_dir_lists)
    # print('\n')
    for i in range(0,FM.shape[0]):
        print(">>", rs_dirs[i],":", "num_rs/num_gt-> %d/%d,"%(int(gt2rs_fm[i][0]),len(gt_name_list)),\
            "maxF->%.5f, "%(np.max(FM,1)[i]), "meanF->%.5f, "%(np.mean(FM,1)[i]),\
            "aveMAE-> %.5f"%(aveMAE[i]), "mba-> %.5f"%(mba[i]))
    
    print('Done!!!')

    return aveMAE[0], np.max(FM,1)[0], np.mean(FM,1)[0], mba[0]

if __name__ == "__main__":
    # HRSOD
    lbl_dir='train_data/HRSOD/HRSOD_test_mask/'
    results_dir='train_data/HRSOD/Results/hrsod_results/'


    ave1,maxF1,meamF1, mba1 = qual_eval(lbl_dir, results_dir)

    # # UHRSD
    # lbl_dir='train_data/UHRSD/UHRSD_TE_2K/mask/'
    # results_dir='train_data/UHRSD/Results/uhrsd_results/'

    # ave2,maxF2,meamF2, mba2 = qual_eval(lbl_dir, results_dir)

    print('HR-SOD result:')
    print('hrsod  : aveMAE-> %.5f | maxF->%.5f | meanF->%.5f | mba->%.5f' %(ave1,maxF1,meamF1,mba1))
    # print('uhrsd  : aveMAE-> %.5f | maxF->%.5f | meanF->%.5f | mba->%.5f' %(ave2,maxF2,meamF2,mba2))
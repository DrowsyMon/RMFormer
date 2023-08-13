from test import test_Net
from eval import qual_eval
from tensorboardX import SummaryWriter
from myconfig import myParser
import shutil

def test_loop(args):
    model_dir = args.model_dir
    writer = SummaryWriter(args.writer_path)

    # select which epoch_xx.pth to test
    epoch_num = [28, 29, 30, 31]
    # select which dataset to test
    dataset_list = ['HRSOD','UHRSD', 'HR10K'] # dataset_list = ['HRSOD','UHRSD','HR10K']

    model_list = []
    for tnum in epoch_num:
        model_list.append('epoch_'+str(tnum)+'.pth')

    img_list = []
    label_list = []
    prediction_list = []

    for dataname in dataset_list:

        if dataname == 'HRSOD':
            image_dir = 'train_data/HRSOD/HRSOD_test/'
            label_dir = 'train_data/HRSOD/HRSOD_test_mask/'
            prediction_dir = 'train_data/HRSOD/Results/hrsod_results/'

            img_list.append(image_dir)
            label_list.append(label_dir)
            prediction_list.append(prediction_dir)


        # # # #UHRSD
        if dataname == 'UHRSD':
            image_dir = 'train_data/UHRSD/UHRSD_TE_2K/image/'
            label_dir = 'train_data/UHRSD/UHRSD_TE_2K/mask/'
            prediction_dir = 'train_data/UHRSD/Results/uhrsd_results/'
            img_list.append(image_dir)
            label_list.append(label_dir)
            prediction_list.append(prediction_dir)

        # #HR10K
        if dataname == 'HR10K':
            image_dir = 'train_data/HR10K/test/img_test_2560max/'
            label_dir = 'train_data/HR10K/test/label_test_2560max/'
            prediction_dir = 'train_data/HR10K/Results/10k_results/'
            img_list.append(image_dir)
            label_list.append(label_dir)
            prediction_list.append(prediction_dir)


        # LR SOD

        if dataname == 'DAVIS-S':
            image_dir = 'train_data/OTHER/LowRes_SOD/DAVIS-SOD/Imgs/'
            label_dir = 'train_data/OTHER/LowRes_SOD/DAVIS-SOD/Mask/'
            prediction_dir = 'train_data/OTHER/LowRes_SOD/Results_collect/DAVIS-S/'
            img_list.append(image_dir)
            label_list.append(label_dir)
            prediction_list.append(prediction_dir)
        # DUT-S
        if dataname == 'DUTS':
            image_dir = 'train_data/OTHER/LowRes_SOD/DUTS/DUTS-TE/DUTS-TE-Image/'
            label_dir = 'train_data/OTHER/LowRes_SOD/DUTS/DUTS-TE/DUTS-TE-Mask/'
            prediction_dir = 'train_data/OTHER/LowRes_SOD/Results_collect/DUTS/'
            img_list.append(image_dir)
            label_list.append(label_dir)
            prediction_list.append(prediction_dir)
        # DUT-OMRON
        if dataname == 'DUT-OMRON':
            image_dir = 'train_data/OTHER/LowRes_SOD/DUT-OMRON/image/'
            label_dir = 'train_data/OTHER/LowRes_SOD/DUT-OMRON/mask/'
            prediction_dir = 'train_data/OTHER/LowRes_SOD/Results_collect/DUT-OMRON/'
            img_list.append(image_dir)
            label_list.append(label_dir)
            prediction_list.append(prediction_dir)


        # ECSSD
        if dataname == 'ECSSD':
            image_dir = 'train_data/OTHER/LowRes_SOD/ECSSD/img/'
            label_dir = 'train_data/OTHER/LowRes_SOD/ECSSD/gt/'
            prediction_dir = 'train_data/OTHER/LowRes_SOD/Results_collect/ECSSD/'
            img_list.append(image_dir)
            label_list.append(label_dir)
            prediction_list.append(prediction_dir)

        # HKU-IS
        if dataname == 'HKU-IS':
            image_dir = 'train_data/OTHER/LowRes_SOD/HKU-IS/imgs/'
            label_dir = 'train_data/OTHER/LowRes_SOD/HKU-IS/gt/'
            prediction_dir = 'train_data/OTHER/LowRes_SOD/Results_collect/HKU-IS/'
            img_list.append(image_dir)
            label_list.append(label_dir)
            prediction_list.append(prediction_dir)


    b_epoch, b_MAE, b_maxF, b_meanF, b_mba = 0,0,0,0,0

    # start testing ==============================
    for i, name in enumerate(dataset_list):
        image_dir = img_list[i]
        label_dir = label_list[i]
        prediction_dir = prediction_list[i]

        for j in range(len(model_list)):
            model_path = model_dir + model_list[j]
            e_num = epoch_num[j]
            test_Net(model_path, image_dir, prediction_dir,args)

            MAE, maxF, meanF, mba = qual_eval(label_dir,prediction_dir)

            writer.add_scalar('Eval' + '_'+ name +'/MAE',MAE, e_num)
            writer.add_scalar('Eval' + '_'+ name +'/maxF',maxF, e_num)
            writer.add_scalar('Eval' + '_'+ name +'/meanF',meanF, e_num)
            writer.add_scalar('Eval' + '_'+ name +'/mba',mba, e_num)

            if j == 0 :
                best = maxF + meanF - MAE
                filename = (args.model_dir + "bestone.pth")
                shutil.copyfile(model_path, filename)
                b_epoch = e_num
                b_MAE = MAE
                b_maxF = maxF
                b_meanF = meanF
                b_mba = mba
            else:  # save model with best performance
                if maxF + meanF - MAE > best:
                    best = maxF + meanF - MAE
                    filename = (args.model_dir + "bestone.pth")
                    shutil.copyfile(model_path, filename)
                    b_epoch = e_num
                    b_MAE = MAE
                    b_maxF = maxF
                    b_meanF = meanF
                    b_mba = mba

        print('Dataset: ' + name)
        print('b_epoch_%d_MAE_%3f_maxF_%3f_meanF_%3f_mba%3f'%(b_epoch,b_MAE,b_maxF,b_meanF,b_mba))


    print('all model tested')






if __name__ == '__main__':
	args = myParser()

	test_loop(args)

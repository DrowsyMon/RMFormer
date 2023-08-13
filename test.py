import os
import cv2
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

import glob
import tqdm

from dataloader_collect import RescaleT
from dataloader_collect import ToTensorLab
from dataloader_collect import SalObjDataset

from model import myNet1
from myconfig import myParser

def normPRED(d):
	ma = torch.max(d)
	mi = torch.min(d)

	dn = (d-mi)/(ma-mi)

	return dn

def save_output_cv2(image_name,pred,d_dir,origin_shape):
    predict = pred.cpu().numpy().transpose((1,2,0))
    # im = cv2.cvtColor((predict*255),cv2.COLOR_RGB2BGR)
    im = predict*255
    imo = cv2.resize(im,(int(origin_shape[1]),int(origin_shape[0])))

    aaa = image_name.split("/")[-1]
    imidx = os.path.splitext(aaa)[0]

    # print(d_dir+imidx+'.png')

    cv2.imwrite(d_dir+imidx+'.png',imo)

def test_Net(model_list,image_dir, prediction_dir,args,flag_exp=1):
	# --------- 1. get image path and name ---------
	
	model_dir = model_list
	print('interferce test img at: ' + image_dir)
	print('prediction saved at: ' + prediction_dir)
	img_name_list = glob.glob(image_dir + '*.jpg')

	
	# --------- 2. dataloader ---------
	#1. dataload
	test_salobj_dataset = SalObjDataset(img_name_list = img_name_list, lbl_name_list = [],edge_name_list=[],\
		transform=transforms.Compose([RescaleT(1536),ToTensorLab(flag=0)]))
	test_salobj_dataloader = DataLoader(test_salobj_dataset, batch_size=1,shuffle=False,num_workers=4)
	
	# --------- 3. model define ---------
	print("...load MyNet...")
	if flag_exp==1:
		net = myNet1(args)

	net.load_state_dict(torch.load(model_dir)['net'],strict= True)
	if torch.cuda.is_available():
		net.cuda()
	net.eval()
	
	# --------- 4. inference for each image ---------
	with torch.no_grad():
		for i_test, data_test in enumerate(tqdm.tqdm(test_salobj_dataloader,ncols=60)):
		
			inputs_test = data_test['image']
			inputs_test = inputs_test.type(torch.FloatTensor)
	
			if torch.cuda.is_available():
				inputs_test = Variable(inputs_test.cuda(), requires_grad=False)
			else:
				inputs_test = Variable(inputs_test)
		
			dout = net(inputs_test)[0]
			
			# normalization
			pred = dout[:,0,:,:]

			pred = normPRED(pred)

			save_output_cv2(img_name_list[i_test],pred,prediction_dir,data_test['original_shape'])

	del dout, pred,inputs_test,data_test


if __name__ == '__main__':
	args = myParser()

    # saved_model_dir
	model_dir = 'save_models/Atemp/bestone.pth'
	# test img dir
	image_dir = 'train_data/HRSOD/HRSOD_test/'
	# results dir
	prediction_dir = 'train_data/HRSOD/Results/hrsod_results/'


	
	test_Net(model_dir, image_dir, prediction_dir,args)


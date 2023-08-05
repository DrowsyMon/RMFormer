import os
from skimage import io, transform
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import argparse

import numpy as np
from PIL import Image
import glob
import tqdm

from dataloader_collect import RescaleT
from dataloader_collect import ToTensorLab
from dataloader_collect import SalObjDataset

from model import myNet1

# notice
# all conv basicblock use group norm
# change in helper_block.py



def normPRED(d):
	ma = torch.max(d)
	mi = torch.min(d)

	dn = (d-mi)/(ma-mi)

	return dn

def save_output(image_name,pred,d_dir):

	predict = pred
	predict_np = predict.squeeze()
	predict_np = predict_np.cpu().data.numpy()

	im = Image.fromarray(predict_np*255).convert('RGB')
	img_name = image_name.split("/")[-1]
	image = io.imread(image_name)
	imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)

	pb_np = np.array(imo)

	aaa = img_name.split(".")
	bbb = aaa[0:-1]
	imidx = bbb[0]
	for i in range(1,len(bbb)):
		imidx = imidx + "." + bbb[i]

	imo.save(d_dir+imidx+'.png')

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
	
	# de_dir = 'train_data/HRSOD/test-result-course/'

	# image_dir = 'train_data/DAVIS-SOD/Images/'
	# prediction_dir = 'train_data/DAVIS-SOD/test-result/'

	model_dir = model_list
	print('interferce test img at: ' + image_dir)
	print('prediction saved at: ' + prediction_dir)
	img_name_list = glob.glob(image_dir + '*.jpg')

	
	# --------- 2. dataloader ---------
	#1. dataload
	# test_salobj_dataset = SalObjDataset(img_name_list = img_name_list, lbl_name_list = [],\
	# 	transform=transforms.Compose([RescaleT(1024),ToTensorLab(flag=0)]))
	test_salobj_dataset = SalObjDataset(img_name_list = img_name_list, lbl_name_list = [],edge_name_list=[],\
		transform=transforms.Compose([RescaleT(1536),ToTensorLab(flag=0)]))
	test_salobj_dataloader = DataLoader(test_salobj_dataset, batch_size=1,shuffle=False,num_workers=4)
	
	# --------- 3. model define ---------
	# print('\n')
	print("...load MyNet...")
	if flag_exp==1:
		net = myNet1(args)

	# if args.use_swinU:
	# 	net = IDEA2(args)
	# else:
	# 	net = IDEA2(3)
	# net.load_state_dict(torch.load(model_dir))
	net.load_state_dict(torch.load(model_dir)['net'],strict= False)
	if torch.cuda.is_available():
		net.cuda()
	net.eval()
	
	# --------- 4. inference for each image ---------
	with torch.no_grad():
		for i_test, data_test in enumerate(tqdm.tqdm(test_salobj_dataloader,ncols=60)):
		
			# print("inferencing:",img_name_list[i_test].split("/")[-1])
		
			inputs_test = data_test['image']
			inputs_test = inputs_test.type(torch.FloatTensor)

		
			if torch.cuda.is_available():
				inputs_test = Variable(inputs_test.cuda(), requires_grad=False)
			else:
				inputs_test = Variable(inputs_test)
		
			# dout = net(inputs_test)
			# dout,dsideout,dsideout2, d1, de,de2, de3= net(inputs_test)
			# dout, dedge, _, _, _,\
			dout = net(inputs_test)[0]
			
			# normalization
			pred = dout[:,0,:,:]
			# dedge = F.interpolate(dedge, dout.shape[-1], mode='bilinear')
			# dedge = dedge[:,0,:,:]
			

			pred = normPRED(pred)
			# dedge = normPRED(dedge)

			# save results to test_results folder
			# save_output(img_name_list[i_test],pred,prediction_dir)
			save_output_cv2(img_name_list[i_test],pred,prediction_dir,data_test['original_shape'])

	del dout, pred,inputs_test,data_test

def test_Net2(model_list,image_dir, prediction_dir,image_dir2, prediction_dir2, args,flag_exp=1):
	# --------- 1. get image path and name ---------
	
	model_dir = model_list
	print('interferce test img at: ' + image_dir)
	print('prediction saved at: ' + prediction_dir)
	img_name_list = glob.glob(image_dir + '*.jpg')

	print('interferce test img at: ' + image_dir2)
	print('prediction saved at: ' + prediction_dir2)
	img_name_list2 = glob.glob(image_dir2 + '*.jpg')


	# --------- 2. dataloader ---------
	#1. dataload
	test_salobj_dataset = SalObjDataset(img_name_list = img_name_list, lbl_name_list = [],edge_name_list=[],\
		transform=transforms.Compose([RescaleT(1024),ToTensorLab(flag=0)]))
	test_salobj_dataloader = DataLoader(test_salobj_dataset, batch_size=1,shuffle=False,num_workers=4)

	test_salobj_dataset2 = SalObjDataset(img_name_list = img_name_list2, lbl_name_list = [],edge_name_list=[],\
		transform=transforms.Compose([RescaleT(1024),ToTensorLab(flag=0)]))
	test_salobj_dataloader2 = DataLoader(test_salobj_dataset2, batch_size=1,shuffle=False,num_workers=4)	
	# --------- 3. model define ---------
	# print('\n')
	print("...load MyNet...")
	if flag_exp==1:
		net = myNet1(args)

	# net.load_state_dict(torch.load(model_dir))
	net.load_state_dict(torch.load(model_dir)['net'])
	if torch.cuda.is_available():
		net.cuda()
	net.eval()
	
	# --------- 4. inference for each image ---------
	with torch.no_grad():
		for i_test, data_test in enumerate(tqdm.tqdm(test_salobj_dataloader,ncols=60)):
		
			# print("inferencing:",img_name_list[i_test].split("/")[-1])
		
			inputs_test = data_test['image']
			inputs_test = inputs_test.type(torch.FloatTensor)

		
			if torch.cuda.is_available():
				inputs_test = Variable(inputs_test.cuda(), requires_grad=False)
			else:
				inputs_test = Variable(inputs_test)
		
			dout, _, _,_,_= net(inputs_test)
			
			# normalization
			pred = dout[:,0,:,:]
	

			pred = normPRED(pred)

			# save results to test_results folder
			# save_output(img_name_list[i_test],pred,prediction_dir)
			save_output_cv2(img_name_list[i_test],pred,prediction_dir,data_test['original_shape'])

		for i_test2, data_test2 in enumerate(tqdm.tqdm(test_salobj_dataloader2,ncols=60)):

			inputs_test2 = data_test2['image']
			inputs_test2 = inputs_test2.type(torch.FloatTensor)

			if torch.cuda.is_available():
				inputs_test2 = Variable(inputs_test2.cuda(), requires_grad=False)
			else:
				inputs_test2 = Variable(inputs_test2)
		
			# dout2, d12, _, _, _ ,_,_,_,_= net(inputs_test2)
			dout2, _, _,_= net(inputs_test2)

			pred2 = dout2[:,0,:,:]
		
			pred2 = normPRED(pred2)


			save_output_cv2(img_name_list2[i_test2],pred2,prediction_dir2,data_test2['original_shape'])

	del dout,pred,inputs_test,data_test,dout2,pred2,inputs_test2,data_test2
	
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--use_swinU', type=bool,
					default=True, help='if model has swin transformer')
	# Transformer setting-----------------------------------------------

	parser.add_argument('--patch_size', type=int,
						default=4, help='output channel of network')
	parser.add_argument('--in_chans', type=int,
						default=3, help='output channel of network')                    
	parser.add_argument('--num_classes', type=int,
						default=1, help='output channel of network')
	parser.add_argument('--embed_dim', type=int,
						default=128, help='Swin setting')                    
	parser.add_argument('--depth', type=tuple,
						default=[2,2,18,2], help='Swin setting')
	parser.add_argument('--depth_decoder', type=tuple,
						default=[2,2,2,2], help='Swin setting') 
	parser.add_argument('--num_heads', type=tuple,
						default=[4,8,16,32], help='Swin setting')  
	parser.add_argument('--window_size', type=int,
						default=12, help='Swin setting')
	parser.add_argument('--mlp_ratio', type=float,
						default=4.0, help='Swin setting')                    
	parser.add_argument('--qkv_bias', type=bool,
						default=True, help='Swin setting')
	parser.add_argument('--qk_scale', type=float,
						default=None, help='Swin setting')
	parser.add_argument('--drop_rate', type=float,
						default=0.0, help='Model setting')
	parser.add_argument('--drop_path_rate', type=float,
						default=0.1, help='Model setting')                    
	parser.add_argument('--ape', type=bool,
						default=False, help='Swin setting')
	parser.add_argument('--use_pretrain', type=bool,
						default=True, help='Swin setting')
	parser.add_argument('--patch_norm', type=bool,
						default=True, help='Swin setting')
	parser.add_argument('--use_checkpoint', type=bool,
						default=False, help='half precise training')
	parser.add_argument('--pretrained_path', type=str,
						default='save_models/pretrain/swin_base_patch4_window12_384_22k.pth', 
						help='model path')

	args = parser.parse_args()



	model_dir = 'save_models/Atemp2/epoch_31.pth'
	#HRSOD
	image_dir = 'train_data/HRSOD/HRSOD_test/'
	prediction_dir = 'train_data/HRSOD/Results/hrsod_results/'
	# #HR10K
	# image_dir = 'train_data/HR10K/test/img_test_2560max/'
	# prediction_dir = 'train_data/HR10K/Results/10k_results/'
	# #UHRSD
	# image_dir = 'train_data/UHRSD/UHRSD_TE_2K/image/'
	# prediction_dir = 'train_data/UHRSD/Results/uhrsd_results/'

	
	test_Net(model_dir, image_dir, prediction_dir,args)


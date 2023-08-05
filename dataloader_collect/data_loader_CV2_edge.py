# data loader
from __future__ import print_function, division
import glob
import torch
from skimage import io, transform, color
import cv2
import numpy as np
import math
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
#==========================dataset load==========================

class RescaleT(object):

	def __init__(self,output_size):
		assert isinstance(output_size,(int,tuple))
		self.output_size = output_size

	def __call__(self,sample):
		image, label, edge = sample['image'],sample['label'],sample['edge']

		h, w = image.shape[:2]

		if isinstance(self.output_size,int):
			if h > w:
				new_h, new_w = self.output_size*h/w,self.output_size
			else:
				new_h, new_w = self.output_size,self.output_size*w/h
		else:
			new_h, new_w = self.output_size

		new_h, new_w = int(new_h), int(new_w)

		# #resize the image to new_h x new_w and convert image from range [0,255] to [0,1]
		# img = transform.resize(image,(new_h,new_w),mode='constant')
		# lbl = transform.resize(label,(new_h,new_w),mode='constant', order=0, preserve_range=True)

		# img = transform.resize(image,(self.output_size,self.output_size),mode='constant')
		# lbl = transform.resize(label,(self.output_size,self.output_size),mode='constant', order=0, preserve_range=True)
		img = cv2.resize(image,(self.output_size,self.output_size))
		lbl = cv2.resize(label,(self.output_size,self.output_size))
		edg = cv2.resize(edge,(self.output_size,self.output_size))

		return {'image':img,'label':lbl,'edge':edg}

class Rescale(object):

	def __init__(self,output_size):
		assert isinstance(output_size,(int,tuple))
		self.output_size = output_size

	def __call__(self,sample):
		image, label, edge = sample['image'],sample['label'],sample['edge']

		h, w = image.shape[:2]

		if isinstance(self.output_size,int):
			if h > w:
				new_h, new_w = self.output_size*h/w,self.output_size
			else:
				new_h, new_w = self.output_size,self.output_size*w/h
		else:
			new_h, new_w = self.output_size

		new_h, new_w = int(new_h), int(new_w)

		# #resize the image to new_h x new_w and convert image from range [0,255] to [0,1]
		img = transform.resize(image,(new_h,new_w),mode='constant')
		lbl = transform.resize(label,(new_h,new_w),mode='constant', order=0, preserve_range=True)
		edg = transform.resize(edge,(new_h,new_w),mode='constant', order=0, preserve_range=True)

		return {'image':img,'label':lbl,'edge':edg}

class CenterCrop(object):

	def __init__(self,output_size):
		assert isinstance(output_size, (int, tuple))
		if isinstance(output_size, int):
			self.output_size = (output_size, output_size)
		else:
			assert len(output_size) == 2
			self.output_size = output_size
	def __call__(self,sample):
		image, label = sample['image'], sample['label']

		h, w = image.shape[:2]
		new_h, new_w = self.output_size

		# print("h: %d, w: %d, new_h: %d, new_w: %d"%(h, w, new_h, new_w))
		assert((h >= new_h) and (w >= new_w))

		h_offset = int(math.floor((h - new_h)/2))
		w_offset = int(math.floor((w - new_w)/2))

		image = image[h_offset: h_offset + new_h, w_offset: w_offset + new_w]
		label = label[h_offset: h_offset + new_h, w_offset: w_offset + new_w]

		return {'image': image, 'label': label}

class RandomCrop(object):

	def __init__(self,output_size):
		assert isinstance(output_size, (int, tuple))
		if isinstance(output_size, int):
			self.output_size = (output_size, output_size)
		else:
			assert len(output_size) == 2
			self.output_size = output_size
	def __call__(self,sample):
		image, label, edge = sample['image'],sample['label'],sample['edge']

		tocorp = np.random.randint(0,100)
		h, w = image.shape[:2]
		new_h, new_w = self.output_size

		if tocorp < 50:
			top = np.random.randint(0, h - new_h)
			left = np.random.randint(0, w - new_w)

			image = image[top: top + new_h, left: left + new_w]
			label = label[top: top + new_h, left: left + new_w]
			edge = edge[top: top + new_h, left: left + new_w]

		else:
			image = cv2.resize(image,self.output_size)
			label = cv2.resize(label,self.output_size)
			edge = cv2.resize(edge,self.output_size)
		return {'image': image, 'label': label, 'edge':edge}
		# H,W,_   = image.shape
		# randw   = np.random.randint(W/8)
		# randh   = np.random.randint(H/8)
		# offseth = 0 if randh == 0 else np.random.randint(randh)
		# offsetw = 0 if randw == 0 else np.random.randint(randw)
		# p0, p1, p2, p3 = offseth, H+offseth-randh, offsetw, W+offsetw-randw
		# return {'image': image[p0:p1,p2:p3, :], 'label': label[p0:p1,p2:p3], 'edge':edge[p0:p1,p2:p3]}





class RandomFlip(object):
	def __init__(self, p):
		self.prob = p

	def __call__(self, sample):
		image, label, edge = sample['image'],sample['label'],sample['edge']

		toflip1 = np.random.randint(0,100)
		toflip2 = np.random.randint(0,100)
		# if toflip1 <=(100*self.prob):
		# 	image = image[::-1,:]
		# 	label = label[::-1,:] #上下反转
		# 	edge = edge[::-1,:]
		if toflip2<=(100*self.prob):
			image = image[:,::-1]
			label = label[:,::-1] #左右翻转
			edge = edge[:,::-1]
		return {'image': image, 'label': label, 'edge':edge}

class ToTensor(object):
	"""Convert ndarrays in sample to Tensors."""

	def __call__(self, sample):

		image, label, edge = sample['image'],sample['label'],sample['edge']

		tmpImg = np.zeros((image.shape[0],image.shape[1],3))
		tmpLbl = np.zeros(label.shape)
		tmpEdg = np.zeros(edge.shape)

		image = image/np.max(image)
		if(np.max(label)<1e-6):
			label = label
		elif(np.max(edge)<1e-6):
			edge = edge
		else:
			label = label/np.max(label)
			edge = edge/np.max(edge)

		if image.shape[2]==1:
			tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
			tmpImg[:,:,1] = (image[:,:,0]-0.485)/0.229
			tmpImg[:,:,2] = (image[:,:,0]-0.485)/0.229
		else:
			tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
			tmpImg[:,:,1] = (image[:,:,1]-0.456)/0.224
			tmpImg[:,:,2] = (image[:,:,2]-0.406)/0.225

		tmpLbl[:,:,0] = label[:,:,0]
		tmpEdg[:,:,0] = edge[:,:,0]

		# change the r,g,b to b,r,g from [0,255] to [0,1]
		#transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
		tmpImg = tmpImg.transpose((2, 0, 1))
		tmpLbl = label.transpose((2, 0, 1))
		tmpEdg = tmpEdg.transpose((2, 0, 1))

		return {'image': torch.from_numpy(tmpImg),
			'label': torch.from_numpy(tmpLbl),
			'edge': torch.from_numpy(tmpEdg)}

class ToTensorLab(object):
	"""Convert ndarrays in sample to Tensors."""
	def __init__(self,flag=0):
		self.flag = flag

	def __call__(self, sample):

		image, label, edge = sample['image'],sample['label'],sample['edge']
		tmpLbl = np.zeros(label.shape)
		tmpEdg = np.zeros(edge.shape)

		if(np.max(label)<1e-6):
			label = label
		elif(np.max(edge)<1e-6):
			edge = edge
		else:
			label = label/np.max(label)
			edge = edge/np.max(edge)

		# change the color space
		if self.flag == 2: # with rgb and Lab colors
			tmpImg = np.zeros((image.shape[0],image.shape[1],6))
			tmpImgt = np.zeros((image.shape[0],image.shape[1],3))
			if image.shape[2]==1:
				tmpImgt[:,:,0] = image[:,:,0]
				tmpImgt[:,:,1] = image[:,:,0]
				tmpImgt[:,:,2] = image[:,:,0]
			else:
				tmpImgt = image
			tmpImgtl = color.rgb2lab(tmpImgt)

			# nomalize image to range [0,1]
			tmpImg[:,:,0] = (tmpImgt[:,:,0]-np.min(tmpImgt[:,:,0]))/(np.max(tmpImgt[:,:,0])-np.min(tmpImgt[:,:,0]))
			tmpImg[:,:,1] = (tmpImgt[:,:,1]-np.min(tmpImgt[:,:,1]))/(np.max(tmpImgt[:,:,1])-np.min(tmpImgt[:,:,1]))
			tmpImg[:,:,2] = (tmpImgt[:,:,2]-np.min(tmpImgt[:,:,2]))/(np.max(tmpImgt[:,:,2])-np.min(tmpImgt[:,:,2]))
			tmpImg[:,:,3] = (tmpImgtl[:,:,0]-np.min(tmpImgtl[:,:,0]))/(np.max(tmpImgtl[:,:,0])-np.min(tmpImgtl[:,:,0]))
			tmpImg[:,:,4] = (tmpImgtl[:,:,1]-np.min(tmpImgtl[:,:,1]))/(np.max(tmpImgtl[:,:,1])-np.min(tmpImgtl[:,:,1]))
			tmpImg[:,:,5] = (tmpImgtl[:,:,2]-np.min(tmpImgtl[:,:,2]))/(np.max(tmpImgtl[:,:,2])-np.min(tmpImgtl[:,:,2]))

			# tmpImg = tmpImg/(np.max(tmpImg)-np.min(tmpImg))

			tmpImg[:,:,0] = (tmpImg[:,:,0]-np.mean(tmpImg[:,:,0]))/np.std(tmpImg[:,:,0])
			tmpImg[:,:,1] = (tmpImg[:,:,1]-np.mean(tmpImg[:,:,1]))/np.std(tmpImg[:,:,1])
			tmpImg[:,:,2] = (tmpImg[:,:,2]-np.mean(tmpImg[:,:,2]))/np.std(tmpImg[:,:,2])
			tmpImg[:,:,3] = (tmpImg[:,:,3]-np.mean(tmpImg[:,:,3]))/np.std(tmpImg[:,:,3])
			tmpImg[:,:,4] = (tmpImg[:,:,4]-np.mean(tmpImg[:,:,4]))/np.std(tmpImg[:,:,4])
			tmpImg[:,:,5] = (tmpImg[:,:,5]-np.mean(tmpImg[:,:,5]))/np.std(tmpImg[:,:,5])

		elif self.flag == 1: #with Lab color
			tmpImg = np.zeros((image.shape[0],image.shape[1],3))

			if image.shape[2]==1:
				tmpImg[:,:,0] = image[:,:,0]
				tmpImg[:,:,1] = image[:,:,0]
				tmpImg[:,:,2] = image[:,:,0]
			else:
				tmpImg = image

			tmpImg = color.rgb2lab(tmpImg)

			# tmpImg = tmpImg/(np.max(tmpImg)-np.min(tmpImg))

			tmpImg[:,:,0] = (tmpImg[:,:,0]-np.min(tmpImg[:,:,0]))/(np.max(tmpImg[:,:,0])-np.min(tmpImg[:,:,0]))
			tmpImg[:,:,1] = (tmpImg[:,:,1]-np.min(tmpImg[:,:,1]))/(np.max(tmpImg[:,:,1])-np.min(tmpImg[:,:,1]))
			tmpImg[:,:,2] = (tmpImg[:,:,2]-np.min(tmpImg[:,:,2]))/(np.max(tmpImg[:,:,2])-np.min(tmpImg[:,:,2]))

			tmpImg[:,:,0] = (tmpImg[:,:,0]-np.mean(tmpImg[:,:,0]))/np.std(tmpImg[:,:,0])
			tmpImg[:,:,1] = (tmpImg[:,:,1]-np.mean(tmpImg[:,:,1]))/np.std(tmpImg[:,:,1])
			tmpImg[:,:,2] = (tmpImg[:,:,2]-np.mean(tmpImg[:,:,2]))/np.std(tmpImg[:,:,2])

		else: # with rgb color
			tmpImg = np.zeros((image.shape[0],image.shape[1],3))
			image = image/np.max(image)
			if image.shape[2]==1:
				tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
				tmpImg[:,:,1] = (image[:,:,0]-0.485)/0.229
				tmpImg[:,:,2] = (image[:,:,0]-0.485)/0.229
			else:
				tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
				tmpImg[:,:,1] = (image[:,:,1]-0.456)/0.224
				tmpImg[:,:,2] = (image[:,:,2]-0.406)/0.225



		
		if len(tmpLbl.shape) == 2:
			tmpLbl[:,:] = label[:,:]
			tmpLbl = tmpLbl[:,:,np.newaxis]
		else:
			tmpLbl[:,:,0] = label[:,:,0]

		if len(tmpEdg.shape) == 2:
			tmpEdg[:,:] = edge[:,:]
			tmpEdg = tmpEdg[:,:,np.newaxis]
		else:
			tmpEdg[:,:,0] = edge[:,:,0]


		# change the r,g,b to b,r,g from [0,255] to [0,1]
		#transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
		tmpImg = tmpImg.transpose((2, 0, 1))
		tmpLbl = tmpLbl.transpose((2, 0, 1))
		tmpEdg = tmpEdg.transpose((2, 0, 1))

		return {'image': torch.from_numpy(tmpImg.copy()),
			'label': torch.from_numpy(tmpLbl.copy()),
			'edge': torch.from_numpy(tmpEdg.copy())}

class SalObjDataset(Dataset):
	def __init__(self,img_name_list,lbl_name_list,edge_name_list,transform=None):
		# self.root_dir = root_dir
		# self.image_name_list = glob.glob(image_dir+'*.png')
		# self.label_name_list = glob.glob(label_dir+'*.png')
		self.image_name_list = img_name_list
		self.label_name_list = lbl_name_list
		self.edge_name_list = edge_name_list
		self.transform = transform

	def __len__(self):
		return len(self.image_name_list)

	def __getitem__(self,idx):

		image = cv2.imread(self.image_name_list[idx])
		image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
		original_shape = image.shape

		if(0==len(self.label_name_list)):
			label_3 = np.zeros(image.shape)
			edge_3 = np.zeros(image.shape)
		else:
			label_3 = cv2.imread(self.label_name_list[idx])
			edge_3 = cv2.imread(self.edge_name_list[idx])

		#print("len of label3")
		#print(len(label_3.shape))
		#print(label_3.shape)

		label = np.zeros(label_3.shape[0:2])
		edge = np.zeros(edge_3.shape[0:2])
		if(3==len(label_3.shape)):
			label = label_3[:,:,0]
		elif(2==len(label_3.shape)):
			label = label_3

		if(3==len(edge_3.shape)):
			edge = edge_3[:,:,0]
		elif(2==len(edge_3.shape)):
			edge = edge_3

		if 2==len(edge.shape):
			edge = edge[:,:,np.newaxis]

		if(3==len(image.shape) and 2==len(label.shape)):
			label = label[:,:,np.newaxis]
		elif(2==len(image.shape) and 2==len(label.shape)):
			image = image[:,:,np.newaxis]
			label = label[:,:,np.newaxis] #np.newaxis 增加一维

		# #vertical flipping
		# # fliph = np.random.randn(1)
		# flipv = np.random.randn(1)
		#
		# if flipv>0:
		# 	image = image[::-1,:,:]
		# 	label = label[::-1,:,:]
		# #vertical flip

		sample = {'image':image, 'label':label, 'edge':edge,'original_shape':original_shape}

		if self.transform:
			sample = self.transform(sample)

		return {'image':sample['image'],'label':sample['label'],\
			'edge':sample['edge'],'original_shape':original_shape}

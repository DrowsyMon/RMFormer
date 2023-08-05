
# RMFormer

This Repo. is used for our ACM MM2023 paper: 


<!-- <p align="center">
  <img src="" width="85%">
</p> -->

> Recurrent Multi-scale Transformer for High-Resolution Salient Object Detection, ACM MM2023  

## HR10K Dataset

We contribute a new HRSOD dataset named HR10K, which contains a total of 10,500 images, dividing 8,400 images for training and 2,100 images for testing.

Download Link:


[Baidu](https://drive.google.com/drive/folders/1u3K65AaKh78P5qKXTsMjVI1SvBXNAPFk?usp=sharing) [Extracting code: ]


## Usage

### Directory
The directory should be like this:

````
-- loss (loss function)

-- model (model structure)
   |-- RMFormer.py

-- save_models (pretrained model)
   |-- Atemp (save training process models)
   |-- pretrain (pretrain swin model)
   |   |-- swin_base_patch4_window12_384_22k.pth

-- train_data (train/test datasets, change in ./myconfig.py)
   |-- HR10K
   |   |-- image
   |   |-- mask
   |   |-- Results
   |-- UHRSD
   |   |-- image
   |   |-- mask
   |   |-- Results
   ...
   
````



### Train
```
python train.py
```

Download Swin pretrain model, save them in ./save_models/pretrain:

[Swin-B-224](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth)


### Test
The trained model can be download here: [Baidu]()

```
python test.py
```
* After testing, saliency maps will be saved in the '--pred_results_dir' in myconfig.py



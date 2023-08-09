
# RMFormer

This Repo. is used for our ACM MM2023 paper: 


<!-- <p align="center">
  <img src="" width="85%">
</p> -->

> Recurrent Multi-scale Transformer for High-Resolution Salient Object Detection, ACM MM2023  

## HR10K Dataset

We contribute a new HRSOD dataset named HR10K, which contains a total of 10,500 images, dividing 8,400 images for training and 2,100 images for testing.

Download Link:


[Baidu](https://pan.baidu.com/s/1qOqVu-6QWlunua2FCw-hRw) [Extracting code: a750]


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

### Edge map generate
Modify the dir in ./gen_edgemap.py, then run:
```
python gen_edgemap.py
```

### Train


Download Swin pretrain model, save them in ./save_models/pretrain:

[Swin-B-224](https://pan.baidu.com/s/1vwJxnJcVqcLZAw9HaqiR6g) [Extracting code:swin]

Change the dataset paths in ./myconfig.py

run:

```
python train.py
```

* After training, models will be saved in ./save_models/Atemp


### Test
Our trained model can be download here: [baidu](https://pan.baidu.com/s/196Wi4L5-nTUdP4ov8BLLTA) [Extracting code:w6ew]

Change the paths in ./test.py, then run:
```
python test.py
```
* After testing, saliency maps will be saved in the '--pred_results_dir' in myconfig.py


123123testtest

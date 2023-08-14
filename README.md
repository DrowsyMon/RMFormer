
# RMFormer

This Repo. is used for our ACM MM2023 paper: 

> Recurrent Multi-scale Transformer for High-Resolution Salient Object Detection, ACM MM2023 (https://arxiv.org/abs/2308.03826)

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
Modify the dir in `./gen_edgemap.py`, then run:
```
python gen_edgemap.py
```

### Train


Download Swin pretrain model, save them in `./save_models/pretrain`:

[Swin-B-224](https://pan.baidu.com/s/1vwJxnJcVqcLZAw9HaqiR6g) [Extracting code:swin]

Change the dataset directory, training setting in `./myconfig.py`

then run:

```
bash train.sh
```

* After training, models will be saved in `./save_models/Atemp`
* Tensorboard log files are in `./runs1`


### Interference
We trained model in three different training setting: DH, UH and KUH 

These trained models be download here: [Baidu](https://pan.baidu.com/s/1IAkMLPxEw_27wGWYLnNNhA) [Extracting code:3xsr]

Change the paths in `./test.py`, then run:
```
python test.py
```
* After testing, saliency maps will be saved in the `'prediction_dir'`

### Evaluation

The code in `./eval.py` is from:
> https://github.com/xuebinqin/Binary-Segmentation-Evaluation-Tool


We also use the evalutation below to generate results in our paper:

> https://github.com/Jun-Pu/Evaluation-on-salient-object-detection


We have add with some minor modifications to generate mBA results.


---
---

If you have any problems. Please concat

zhpp@dlut.edu.cn or dengxh@mail.dlut.edu.cn

# RMFormer

This Repo. is used for our ACM MM2023 paper: 

> Recurrent Multi-scale Transformer for High-Resolution Salient Object Detection, ACM MM2023 (https://arxiv.org/abs/2308.03826)

## HRS10K Dataset

![Image](https://github.com/DrowsyMon/RMFormer/blob/main/pic/10k_1.png)

![Image](https://github.com/DrowsyMon/RMFormer/blob/main/pic/10k_2.png)

We contribute a new HRSOD dataset named HRS10K, which contains a total of 10,500 images, dividing 8,400 images for training and 2,100 images for testing.

Download Link:


>[Baidu](https://pan.baidu.com/s/1qOqVu-6QWlunua2FCw-hRw) [Extracting code: a750]

>[Google Drive](https://drive.google.com/drive/folders/1LpCkuTX2Efy2tKak3qVz_Uma2-6DEmaN?usp=sharing)

## Usage
### Conda Enviorment
* PyTorch 1.12.1
* python 3.8
* cudatoolkit 11.6
* OpenCV
* einops
* timm
* skimage
* tqdm
* tenosorboard



You can initialize a new conda enviroment as follow: 
```
# Create New Enviorment
conda create -n RMFormer python=3.8

# CUDA & PyTorch
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge

# Others
conda install tqdm tensorboard tensorboardX
pip install opencv-python einops timm scikit-image
```





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
   |-- HRS10K
   |   |-- image
   |   |-- mask
   |   |-- Results
   |-- UHRSD
   |   |-- image
   |   |-- mask
   |   |-- Results
   |-- HRSOD
   |   |-- image
   ...
````
---

### Edge Map Generate
Modify the dir in `./gen_edgemap.py`, then run:
```
python gen_edgemap.py
```

### Train


Download Swin pretrain model, save them in `./save_models/pretrain`

The link below is from https://github.com/microsoft/Swin-Transformer

>[Baidu](https://pan.baidu.com/s/1vwJxnJcVqcLZAw9HaqiR6g) [Extracting code:swin]

>[github](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth)

Change the dataset directory, training setting in `./myconfig.py`

Make sure `--itr_epoch` in `training_param` is set correctly

then run:

```
bash train.sh
```

* After training, models will be saved in `./save_models/Atemp`
* Tensorboard log files are in `./runs1`

---

### Interference
We trained model in three different training setting: DH, UH and KUH 

These trained models be download here: 

>[Baidu](https://pan.baidu.com/s/1h5hEpEdTHRpXp-QT-ys4dg) [Extracting code:iavg]
>[Google Drive](https://drive.google.com/drive/folders/1avHY7VASvLSsqvT5saU9OBAsbx2oJ0HD?usp=sharing)

The saliency maps of our RMFormer can be download here:

>[Baidu](https://pan.baidu.com/s/1BVj_BaaFX4vz7PlbSDFqcw) [Extracting code:4h2g]
>[Google Drive](https://drive.google.com/file/d/1MZj3Nzz3NSbTWPLSutKcgRXypTdZu_5h/view?usp=sharing)



Notice:
We have reorganized the code and achieved enhanced performance by adjusting the parameter initialization before training. If you would like to access the improved models, please download them from the provided link below.


>Trained models:
>[Baidu](https://pan.baidu.com/s/1SjWXIALGAzG6et769mXIZQ) [Extracting code:a2r6]
>[Google Drive](https://drive.google.com/drive/folders/17LkT_7GHMiQ2Eqnj3aBTqUyUdVszd9x2?usp=drive_link)

>Saliency maps:
>[Baidu](https://pan.baidu.com/s/1-b1HYIja-692cG33lygleA) [Extracting code:akhg]
>[Google Drive](https://drive.google.com/drive/folders/1tqXm1qn7dgar6k8xSlmtYVEhzdWDNpaT?usp=drive_link)

Change the paths in `./test.py`, then run:
```
python test.py
```
* After testing, saliency maps will be saved in the `'prediction_dir'`

---

### Evaluation


The code in `./eval.py` is from:
> https://github.com/xuebinqin/Binary-Segmentation-Evaluation-Tool


We also use the evalutation below to generate results in our paper:

> https://github.com/Jun-Pu/Evaluation-on-salient-object-detection


Some modifications are made in our codes for generating mBA results.

You can refer to codes in `./evaluation_code.zip`

---
---

If you have any problems. Please concat

dengxh@mail.dlut.edu.cn or zhpp@dlut.edu.cn
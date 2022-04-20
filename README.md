# Self-supervised Transparent Liquid Segmentation for Robotic Pouring

[Video](https://youtu.be/uXGCSd3KVd8) | [Website](https://sites.google.com/view/transparentliquidpouring) | [Paper](https://www.google.com/url?q=https%3A%2F%2Farxiv.org%2Fabs%2F2203.01538&sa=D&sntz=1&usg=AOvVaw3d5pQfrbcL9HH5mwauGDwD)

![](pouring.gif)

**ABSTRACT:** Liquid state estimation is important for robotics tasks such as pouring; however, estimating the state of transparent liquids is a challenging problem. We propose a novel segmentation pipeline that can segment transparent liquids such as water from a static, RGB image without requiring any manual annotations or heating of the liquid for training. Instead, we use a generative model that is capable of translating unpaired images of colored liquids into synthetically generated transparent liquid images. Segmentation labels of colored liquids are obtained automatically using background subtraction. We use paired samples of synthetically generated transparent liquid images and background subtraction for our segmentation pipeline. Our experiments show that we are able to accurately predict a segmentation mask for transparent liquids without requiring any manual annotations. We demonstrate the utility of transparent liquid segmentation in a robotic pouring task that controls pouring by perceiving liquid height in a transparent cup.

**Gautham Narayan Narasimhan**, Kai Zhang, Ben Eisner, Xingyu Lin, David Held

International Conference of Robotics and Automation (ICRA), 2022

# Getting Started

- Clone this repo and install requirements

```
git clone https://github.com/gauthamnarayan/transparent-liquid-segmentation
pip install -r requirements.txt
cd Segment_Transparent_Objects
pip install -e .
```

- Download pre-trained weights for [Segment_Transparent_Objects](https://github.com/xieenze/Segment_Transparent_Objects) from [here](https://drive.google.com/drive/folders/1yJMEB4rNKIZt5IWL13Nn-YwckrvAPNuz)

```
cp 16.pth transparent-liquid-segmentation/Segment_Transparent_Objects/demo
```

Additional instructions can be found at contrastive-unpaired-translation[link] and Segment_Transparent_Objects[link]

## Dataset Generation

Please use the datasets and pre-trained models provided here to quickly get started.

However, we do not expect our pre-trained models to generalize beyond our lab due to complex lighting conditions; Our training procedure is easy to use and self-supervised so we advise for people who want to use our method to retrain it with data collected in their own labs

## Cup segmentation
TODO: Automate for every video

Run once for each video of opaque and colored liquids

```
python -u ./tools/test_demo.py --config-file configs/trans10K/translab.yaml TEST.TEST_MODEL_PATH ./demo/16.pth  DEMO_DIR ../data/datasets/colored2transparent_dataset/trainA/
```
```
python -u ./tools/test_demo.py --config-file configs/trans10K/translab.yaml TEST.TEST_MODEL_PATH ./demo/16.pth  DEMO_DIR ../data/datasets/colored2transparent_dataset/trainB/
```

## Prepare dataset

Prepares a dataset for the pipeline and also runs background subtraction for colored liquids.

```
cd transparent-liquid-segmentation/scripts

python prepare_dataset.py \
--pouring-videos_dir ../data/datasets/pouring_videos/opaque/ \
--output-dir ../data/datasets/pouring_dataset/ \
--mode opaque

python prepare_dataset.py \
--pouring-videos_dir ../data/datasets/pouring_videos/transparent/ \
--output-dir ../data/datasets/pouring_dataset/ \
--mode transparent
```

Pouring video dataset directory structure:

```
pouring_dataset
├── fakeB
├── trainA
├── trainA_cup_masks
├── trainA_liquid_masks
├── trainB
└── trainB_cup_masks
```

# Image Translation


To train the image translation model, please use the following command.

```
cd contrastive-unpaired-translation/

python train.py --dataroot ../data/datasets/pouring_dataset/ \
                --name pouring_CUT \
                --CUT_mode CUT \
                --preprocess none
```
        
Run inference using CUT model to obtain paired transparent liquid data
``` 
python test.py --dataroot ../data/datasets/pouring_dataset 
               --name pouring_CUT 
               --CUT_mode CUT 
               --phase train 
               --preprocess none
```

# Image Segmentation

To train the segmentation model (UNet), please use the following command.
```
python train_unet_transparent_pouring.py --train-data-dir ../data/datasets/pouring_dataset --save-dir ../data/saved_models
```

To test the segmentation model (UNet), please use the following command. Results from segmentation will be written out at ```--results-dir```.
```
python test_unet_transparent_pouring.py --model-path ../data/saved_models/<model_name> --test-videos-path ../data/datasets/pouring_testset_videos --results-dir ../results
```


# Notes:

- We use [labelme](https://github.com/wkentaro/labelme) to generate manual ground truth for evaluating the performance of our image segmentation method. 

- Translab might not produce good cup segmentation masks. In this case, please obtain the cup mask when the cup is empty (first few frames of pouring) and copy it to all subsequent frames where the cup position has not changed.

# Bibtex

```

@inproceedings{icra2022pouring,
title={Self-supervised Transparent Liquid Segmentation for Robotic Pouring},
author={Gautham Narayan Narasimhan, Kai Zhang, Ben Eisner, Xingyu Lin, David Held},
booktitle={International Conference on Robotics and Automation (ICRA)},
year={2022}}

```

# Acknowledgement

This material is based upon work supported by LG Electronics and National Science Foundation under Grant No. IIS-2046491.


ToDo
Update requirements.txt

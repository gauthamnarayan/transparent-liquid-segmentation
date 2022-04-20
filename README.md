# Self-supervised Transparent Liquid Segmentation for Robotic Pouring - ICRA 2022


# Setup
Git clone project
    Should automatically clone CUT
    Should automatically clone TransLab
        Copy 16.pth to project_root/Segment_Transparent_Objects/demo/

pip install using requirements.txt
pip install Segment_Transparent_Objects

Additional instructions can be found at contrastive-unpaired-translation[link] and Segment_Transparent_Objects[link]

## Datasets

# Dataset Generation

Pouring videos directory structure

pouring_dataset
├── fakeB
├── trainA
├── trainA_cup_masks
├── trainA_liquid_masks
├── trainB
└── trainB_cup_masks


# Cup segmentation
TODO: Automate for every video
```python -u ./tools/test_demo.py --config-file configs/trans10K/translab.yaml TEST.TEST_MODEL_PATH ./demo/16.pth  DEMO_DIR ../data/datasets/colored2transparent_dataset/trainA/```
```python -u ./tools/test_demo.py --config-file configs/trans10K/translab.yaml TEST.TEST_MODEL_PATH ./demo/16.pth  DEMO_DIR ../data/datasets/colored2transparent_dataset/trainB/```

# Prepare dataset

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

# Image Translation

```
cd transparent-liquid-segmentation_temp/contrastive-unpaired-translation/
```

    Dataset structure
    datasets/colored2transparent_dataset/

    Command
        Train CUT model
        ```python train.py --dataroot ../data/datasets/pouring_dataset/ --name pouring_CUT --CUT_mode CUT --preprocess none```
        
        Run inference using CUT model to obtain paired transparent liquid data
        ``` python test.py --dataroot ../data/datasets/pouring_dataset --name pouring_CUT --CUT_mode CUT --phase train --preprocess none```
        
        <!-- copy
        ```cp -r contrastive-unpaired-translation/results/opaque2transparent_pouring_videos_colored_background_2_090321/train_latest/images/fake_B/ data/datasets/colored2transparent_dataset/``` -->

# Image Segmentation

```python train_unet_transparent_pouring.py --train-data-dir /home/gautham/programs/RPAD/pouring/pouring_git/transparent-liquid-segmentation_temp/data/datasets/pouring_dataset --save-dir /home/gautham/programs/RPAD/pouring/pouring_git/transparent-liquid-segmentation_temp/data/saved_models```

```python test_unet_transparent_pouring.py --model-path ../data/saved_models/transparent_liquid_segmentation_unet_150x300_epoch_10_20220420_15_29_55 --test-videos-path ../data/datasets/pouring_testset_videos```

# To obtain ground truth
Include labelme instructions and dataset structure

# Trouble shooting:

- Translab might not produce good cup segmentation masks. In this case, please obtain the cup mask when the cup is empty (first few frames of pouring) and copy it to all subsequent frames where the cup position has not changed.


ToDo
Add requirements.txt
Add MIT licence
Add citation

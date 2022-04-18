#!/bin/bash
# Basic range in for loop
for idx in {0..17}
do
    #echo $value
    #mv $idx.txt $idx.log
    
    #python3 -u ./tools/test_demo.py --config-file configs/trans10K/translab.yaml TEST.TEST_MODEL_PATH ./demo/16.pth  DEMO_DIR /home/gautham/programs/RPAD/pouring/unet_trials/datasets/pouring_videos_082721/opaque_5/video_$idx
    #cp /home/gautham/programs/RPAD/pouring/unet_trials/datasets/pouring_videos_082721/result/* /home/gautham/programs/RPAD/pouring/unet_trials/datasets/pouring_videos_082721/opaque_5/video_$idx

    #python3 -u ./tools/test_demo.py --config-file configs/trans10K/translab.yaml TEST.TEST_MODEL_PATH ./demo/16.pth  DEMO_DIR /home/gautham/programs/RPAD/pouring/unet_trials/datasets/pouring_videos_082721/transparent_5/video_$idx
    #cp /home/gautham/programs/RPAD/pouring/unet_trials/datasets/pouring_videos_082721/result/* /home/gautham/programs/RPAD/pouring/unet_trials/datasets/pouring_videos_082721/transparent_5/video_$idx
    
    python3 -u ./tools/test_demo.py --config-file configs/trans10K/translab.yaml TEST.TEST_MODEL_PATH ./demo/16.pth  DEMO_DIR /home/gautham/programs/RPAD/pouring/unet_trials/datasets/pouring_videos_082721/test_pouring_videos/video_$idx
    cp /home/gautham/programs/RPAD/pouring/unet_trials/datasets/pouring_videos_082721/result/* /home/gautham/programs/RPAD/pouring/unet_trials/datasets/pouring_videos_082721/test_pouring_videos/video_$idx

    rm -rf /home/gautham/programs/RPAD/pouring/unet_trials/datasets/pouring_videos_082721/result/
done
echo All done


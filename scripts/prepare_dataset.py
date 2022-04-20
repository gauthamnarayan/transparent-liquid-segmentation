import os
import sys
import cv2
import copy
import numpy as np
import argparse

import utils

def trainBagroundSubtraction(backsub, path, files, cup_mask):
    for file in files:
        rgb_img = cv2.imread(os.path.join(path, file))
        # rgb_img[cup_mask == 0] = 0
        backsub.apply(rgb_img, learningRate=-1)


def main(args):
    video_dirs = os.listdir(args.pouring_videos_dir)
    video_dirs = [v for v in video_dirs if "mask" not in v]
    video_dirs.sort()

    if args.mode == "opaque":
        if not os.path.exists(os.path.join(args.output_dir, "trainA")):
            os.makedirs(os.path.join(args.output_dir, "trainA"))
        if not os.path.exists(os.path.join(args.output_dir, "trainA_cup_masks")):
            os.makedirs(os.path.join(args.output_dir, "trainA_cup_masks"))
        if not os.path.exists(os.path.join(args.output_dir, "trainA_liquid_masks")):
            os.makedirs(os.path.join(args.output_dir, "trainA_liquid_masks"))

    elif args.mode == "transparent":
        if not os.path.exists(os.path.join(args.output_dir, "trainB")):
            os.makedirs(os.path.join(args.output_dir, "trainB"))
        if not os.path.exists(os.path.join(args.output_dir, "trainB_cup_masks")):
            os.makedirs(os.path.join(args.output_dir, "trainB_cup_masks"))

    img_counter = 0    
    for dir in video_dirs:
        files = os.listdir(os.path.join(args.pouring_videos_dir, dir))
        empty_cup_imgs = [f for f in files if "empty" in f and "glass" not in f and "mask" not in f]
        filled_cup_imgs = [f for f in files if "rgb" in f and "glass" not in f and "mask" not in f]
        filled_cup_inds = [f.split(".")[0] for f in filled_cup_imgs]
        filled_cup_inds = [f.split("_")[1] for f in filled_cup_inds]
        filled_cup_inds.sort()
        
        # Use cup mask from empty cup, since Translab is more stable with empty cups.
        cup_mask = cv2.imread(os.path.join(args.pouring_videos_dir, dir + "_cup_masks", "empty_cup_0_cup_mask.png"), cv2.IMREAD_GRAYSCALE)
        print("Processing: ", os.path.join(args.pouring_videos_dir, dir))
        img_corners, contour_img = utils.get_cup_corners_contours(cup_mask, cropPad=5)
        
        cup_mask_resized = cup_mask[img_corners[0]: img_corners[1], img_corners[2]: img_corners[3]]
        cup_mask_resized = cv2.resize(cup_mask_resized, (150, 300))            
        
        if args.mode == "opaque":
            # Train background subtraction
            backsub = cv2.createBackgroundSubtractorMOG2(history=20, varThreshold=100)
            trainBagroundSubtraction(backsub, os.path.join(args.pouring_videos_dir, dir), empty_cup_imgs, cup_mask)
        
        # Segment colored liquids
        for ind in filled_cup_inds:
            rgb_img = cv2.imread(os.path.join(args.pouring_videos_dir, dir, "rgb_" + str(ind) + ".jpg"))
            # rgb_img[cup_mask == 0] = 0

            if args.mode == "opaque":
                liquid_mask = backsub.apply(rgb_img, learningRate=0)            
                liquid_mask[liquid_mask<255] = 0 # Remove shadows
                liquid_mask[cup_mask == 0] = 0 # Remove regions exterior to cup
                liquid_mask = liquid_mask[img_corners[0]: img_corners[1], img_corners[2]: img_corners[3]]
                liquid_mask = cv2.resize(liquid_mask, (150, 300))
                liquid_mask = liquid_mask / liquid_mask.max()

                rgb_img = rgb_img[img_corners[0]: img_corners[1], img_corners[2]: img_corners[3]]
                rgb_img = cv2.resize(rgb_img, (150, 300))

                cv2.imwrite(os.path.join(args.output_dir, "trainA", "rgb_" + str(img_counter) + ".jpg"), rgb_img)
                cv2.imwrite(os.path.join(args.output_dir, "trainA_liquid_masks", "liquid_mask_" + str(img_counter) + ".jpg"), liquid_mask)
                cv2.imwrite(os.path.join(args.output_dir, "trainA_cup_masks", "cup_mask_" + str(img_counter) + ".jpg"), cup_mask_resized)
            
            if args.mode == "transparent":
                rgb_img = rgb_img[img_corners[0]: img_corners[1], img_corners[2]: img_corners[3]]
                rgb_img = cv2.resize(rgb_img, (150, 300))
                cv2.imwrite(os.path.join(args.output_dir, "trainB", "rgb_" + str(img_counter) + ".jpg"), rgb_img)
                cv2.imwrite(os.path.join(args.output_dir, "trainB_cup_masks", "cup_mask_" + str(img_counter) + ".jpg"), cup_mask_resized)
            
            img_counter += 1
            
            # For debugging
            # print(rgb_img.shape)
            # cv2.imshow("rgb_img", rgb_img)
            # cv2.imshow("liquid_mask", liquid_mask)
            # cv2.imshow("cup_mask_resized", cup_mask_resized)
            # cv2.waitKey()
        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--pouring-videos_dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--mode', help="opaque / transparent", required=True)
    args = parser.parse_args()
    
    main(args)

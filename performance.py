### Updating the performance.py for my study
import os
import cv2
import time
import numpy as np
import pandas as pd
import imgaug as ia
from utils.iou import iou
from imgaug import augmenters as iaa
from yolo.detector import YOLO
from fingertip import Fingertips

#hand = YOLO('weights/yolo.h5', threshold=0.5)
fingertip = Fingertips(model='vgg', weights='weights/weights025.h5')
resolution = 1


# def flip_horizontal(img, keys):
#     """ Flipping """
#     aug = iaa.Sequential([iaa.Fliplr(1.0)])
#     seq_det = aug.to_deterministic()
#     keys = ia.KeypointsOnImage([ia.Keypoint(x=keys[0], y=keys[1]),
#                                 ia.Keypoint(x=keys[2], y=keys[3]),
#                                 ia.Keypoint(x=keys[4], y=keys[5]),
#                                 ia.Keypoint(x=keys[6], y=keys[7])], shape=img.shape)

#     image_aug = seq_det.augment_images([img])[0]
#     keys_aug = seq_det.augment_keypoints([keys])[0]
#     k = keys_aug.keypoints
#     keys_aug = [k[0].x, k[0].y, k[1].x, k[1].y, k[2].x, k[2].y, k[3].x, k[3].y]

#     return image_aug, keys_aug


image_directory = 'C:/Users/Kanav/Documents/Dissertation/Parkinsons_Disease/Codes/Fingertip-Mixed-Reality/Dataset/Valid/' 
label_directory = 'C:/Users/Kanav/Documents/Dissertation/Parkinsons_Disease/Codes/Fingertip-Mixed-Reality/Dataset/label/'
image_files = os.listdir(image_directory)

""" Ground truth label file for Parkinson's data """
labels_data = pd.read_csv(label_directory + "combined_labels_v3_cleaned.csv")
labels_data["filename"] = labels_data["video_name"] + "@" + labels_data.seq.str[3:6].astype(int).astype(str)

# file = open(label_directory + 'TI1K.txt')
# lines = file.readlines()
# file.close()

# """ Ground truth label file for SingleEight dataset """
# file = open(label_directory + 'SingleEight.txt')
# ego_lines = file.readlines()
# file.close()

total_error = np.zeros([1, 4])
avg_hand_detect_time = 0
avg_fingertip_detect_time = 0
avg_time = 0
avg_iou = 0
count = 0
distance_error = []

# height = int(480 * resolution)
# width = int(640 * resolution)

for image_name in image_files:
    """ Getting the ground truth labels """
    image = cv2.imread(image_directory + image_name)
    cols, rows, _ = image.shape
    if count == 0:
        print("row dim", rows)
        print("col dim", cols)
    filename = image_name.split(".jpg")[0]
    gt = []

    #fetching the groundtruth labels (based on cropped image of dim 780*910)
    xi = int(labels_data[labels_data["filename"] == filename]["new_finger_x"]) 
    yi = int(labels_data[labels_data["filename"] == filename]["new_finger_y"]) 
    xt = int(labels_data[labels_data["filename"] == filename]["new_thumb_x"]) 
    yt = int(labels_data[labels_data["filename"] == filename]["new_thumb_y"]) 
    

    
    gt = [xi, yi, xt, yt] #first index and then thumb
    
    """ Predictions for the test images """
    # Resizing the image to a square dimension - Kanav
    old_size = image.shape[:2]
    desired_size = max(old_size)
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    new_size
    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [0,0,0]
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)

    #resizing image as per model requirements - change this when you try some other size - Kanav        
    image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_LINEAR)

    tic3 = time.time()
    position = fingertip.classify(image=image)
    toc3 = time.time()
    avg_fingertip_detect_time = avg_fingertip_detect_time + (toc3 - tic3)

    for i in range(0, len(position), 2):
        position[i] = (position[i] * max(old_size)) - left
        position[i + 1] = (position[i + 1] * max(old_size)) - top

        # Commenting the below adjustment to keep labels within bounding box. But may use in future
#     for i in range(0, len(position), 2):
#         position[i] = (position[i] + tl[0])
#         position[i + 1] = (position[i + 1] + tl[1])

    pr = [position[0], position[1], position[2], position[3]]

    """ Calculating error for fingertips only """ 
    gt = np.asarray(gt)
    pr = np.asarray(pr)
    abs_err = abs(gt - pr)
    total_error = total_error + abs_err
    #D = np.sqrt((gt[0] - gt[2]) ** 2 + (gt[1] - gt[3]) ** 2)
    #D_hat = np.sqrt((pr[0] - pr[2]) ** 2 + (pr[1] - pr[3]) ** 2)
    #distance_error.append(abs(D - D_hat))
    count = count + 1
    print('Detected Image: {0}'.format(count))

er = total_error / count
#avg_iou = avg_iou / count
er = er[0]
er = np.round(er, 4)
# distance_error = np.array(distance_error)
# distance_error = np.mean(distance_error)
# distance_error = np.round(distance_error, 4)

print('Total Detected Image: {0}'.format(count))
#print('Average IOU: {0}'.format(avg_iou))
print('Pixel errors: xi = {0}, yi = {1}, xt = {2}, yt = {3}'.format(er[0], er[1],
                                                                                   er[2], er[3],
                                                                                   distance_error))

# avg_time = avg_time / 1000
# avg_hand_detect_time = avg_hand_detect_time / count
avg_fingertip_detect_time = avg_fingertip_detect_time / count

#print('Average execution time: {0:1.5f} ms'.format(avg_time * 1000))
#print('Average hand detection time: {0:1.5f} ms'.format(avg_hand_detect_time * 1000))
print('Average fingertip detection time: {0:1.5f} ms'.format(avg_fingertip_detect_time * 1000))

print('{0} & {1} & {2} & {3} '.format(er[0], er[1], er[2], er[3]))



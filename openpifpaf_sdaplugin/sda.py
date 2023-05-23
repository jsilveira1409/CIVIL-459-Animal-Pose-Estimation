import openpifpaf.transforms as transforms
from argparse import ArgumentParser
import random
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import torch
from PIL import Image

KP_DIST_THRESHOLD = 10
NB_BODY_PARTS = 5
IMG_TO_BODYPART_RATION = 4

# cow sheep horse cat dog
labels = {'dog':1, 'cat':2, 'sheep':3, 'horse':4, 'cow':5} 

face_color = (100, 120, 5)
limb_color =(100, 200, 5)
other_color = (100, 200, 5)
kp_color = (100, 200, 5)

segm_colors = [face_color] * 5 + [other_color] * 2 + [limb_color] * 4 + [other_color] * 4

def random_color():
    levels = range(32,256,32)
    return tuple(random.choice(levels) for _ in range(3))


def draw_keypoint(image, keypoints):
    '''
        order: 
        0-4 (face): left eye, right eye, nose, left earbase, right earbase
        5-16 (limbs):   L_F_elbow, R_F_elbow, L_B_elbow, R_B_elbow
                        L_F_knee, R_F_knee, L_B_knee, R_B_knee
                        L_F_paw, R_F_paw, L_B_paw, R_B_paw
        17-19 (others): throat, withers, tailbase
    '''

    segmts = [  (0,1), (0,2), (1,2), (0,3), (1,4),
                (2,17), (18,19),
                (5,9), (6,10), (7,11), (8,12),
                (9,13), (10,14), (11,15), (12,16)]
    
    im = image.copy()
    for i in range(len(segmts)):

        segm = segmts[i]
        kp1_idx = segm[0] * 3    
        kp2_idx = segm[1] * 3
        kp1 = keypoints[kp1_idx:kp1_idx + 3]
        kp2 = keypoints[kp2_idx:kp2_idx + 3]
        
        if kp1[2] == 0 or kp2[2] == 0:
            continue

        cv2.line(im, (int(kp1[0]), int(kp1[1])), (int(kp2[0]), int(kp2[1])), face_color, thickness=2)
    

    for i in range(0, len(keypoints), 3):
        if keypoints[i + 2] == 0:
            continue
        cv2.circle(im, (int(keypoints[i]), int(keypoints[i + 1])), radius=4, color=kp_color, thickness=-1)

    return im

output_folder = 'data-animalpose/bodyparts/'
bodypart_file = 'data-animalpose/bodyparts/cropped_bodyparts.json'
traim_ann = 'data-animalpose/annotations/animal_keypoints_20_train.json'
train_img = 'data-animalpose/images/train/'
train_ann = 'data-animalpose/annotations/animal_keypoints_20_train.json'
val_img = 'data-animalpose/images/val/'
val_ann = 'data-animalpose/annotations/animal_keypoints_20_val.json'


class SDA(transforms.Preprocess):
    def __init__(self, probability=0.5, tolerance=5):
        super().__init__()
        self.probability = probability
        self.tolerance = tolerance
        print("sdaplugin init")

    def apply(self, image, ann):
        # get the keypoints from all the annotations
        keypoints = []
        for kp in ann:
            keypoints.extend(kp['keypoints'])

        # Implement the SDA logic here
        # 1. Check if the augmentation should be applied based on the probability
        #if random.random() > self.probability:
        #    return image, anns, meta

        # 2. Perform the augmentation:
        #   - choose random number of body parts to add to the image
        #   - choose random body parts from the pool of body parts
        #   - add them to the image
        
        nb_bodyparts = random.randint(1, NB_BODY_PARTS)
        #print("nb_bodyparts ",nb_bodyparts)
        augmented_image = np.asarray(image, dtype=np.uint8).copy()
        # get the image dimensions
        image_height, image_width = augmented_image.shape[:2]
        # load the body parts pool
        bodyparts = json.load(open(bodypart_file))
        for i in range(nb_bodyparts):
            # choose a random body part from the pool
            bodypart = random.choice(bodyparts)
            # load the body part
            bodypart = plt.imread(bodypart)
            # get the body part dimensions
            bodypart_height, bodypart_width = bodypart.shape[:2]
            # ensure the body part is not too big compared to the image
            if image_height/bodypart_height > IMG_TO_BODYPART_RATION or image_width/bodypart_width > IMG_TO_BODYPART_RATION:
                # choose a random position to add the body part
                # ensure image_width - bodypart_width > 0
                # ensure image_height - bodypart_height > 0
                if image_width - bodypart_width > 0 and image_height - bodypart_height > 0:
                    # choose a random position to add the body part not directly on top of keypoints
                    x = random.randint(0, image_width - bodypart_width)
                    y = random.randint(0, image_height - bodypart_height)
                    nb_retries = 0
                    not_on_kp = True
                    # to avoid infinite loop
                    while nb_retries < 30:
                        # check if the body part is not on top of keypoints
                        for i in range(0, len(keypoints), 3):
                            if  x < keypoints[i] - KP_DIST_THRESHOLD and \
                                keypoints[i] + KP_DIST_THRESHOLD < x + bodypart_width and \
                                y < keypoints[i + 1] - KP_DIST_THRESHOLD and \
                                keypoints[i + 1] + KP_DIST_THRESHOLD < y + bodypart_height:
                                    continue
                            else:
                                x = random.randint(0, image_width - bodypart_width)
                                y = random.randint(0, image_height - bodypart_height)
                                nb_retries += 1
                                not_on_kp = False
                        if not_on_kp:
                            break
                        
                else:
                    x = 0
                    y = 0
                # TODO:rotate image ? 
                # add the body part to the image
                augmented_image[y : y+bodypart_height, x : x+bodypart_width] = bodypart
                augmented_image = draw_keypoint(augmented_image, keypoints)

        # 3. Return the augmented image
            #transform to pil image
        augmented_image = Image.fromarray(augmented_image)

        return augmented_image
   
    def crop(self,image, keypoints):
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        # draw the keypoints on the mask
        mask = draw_keypoint(mask, keypoints)
        # thicken the mask
        kernel = np.ones((10,10), np.uint8)
        mask = cv2.dilate(mask,kernel,iterations = 3)
        #find the contours in the mask
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP , cv2.CHAIN_APPROX_SIMPLE)
        # get contour area and centroid
        moments = [] 
        areas = []
        for contour in contours:
            moment = cv2.moments(contour)
            area = cv2.contourArea(contour)
#            # centroid
            cx = int(moment['m10']/moment['m00'])
            if moment['m00'] != 0:
                cx = int(moment['m10'] / moment['m00'])
                cy = int(moment['m01'] / moment['m00'])
            else:
                cx, cy = 0, 0
            moments.append([cx, cy])
            areas.append(area)
          # remove contours that are too close to each other, keeping only the biggest one
        indices_to_remove = set()
        for i in range(len(contours)):
            for j in range(len(contours)):
                if i == j or j in indices_to_remove:
                    continue
                dist = np.sqrt((moments[i][0] - moments[j][0])**2 + (moments[i][1] - moments[j][1])**2)
                if dist < 50:
                    if areas[i] > areas[j]:
                        indices_to_remove.add(j)
                    else:
                        indices_to_remove.add(i)
                        
        # remove contours, moments, and areas using list comprehension
        contours = [contour for i, contour in enumerate(contours) if i not in indices_to_remove]
        moments = [moment for i, moment in enumerate(moments) if i not in indices_to_remove]
        areas = [area for i, area in enumerate(areas) if i not in indices_to_remove]
#        # crop the different body parts and store them
        bodyparts = []
        for i in range(len(contours)):
            x, y, w, h = cv2.boundingRect(contours[i])
            bodyparts.append(image[y : y+h+self.tolerance, 
                                   x : x+w+self.tolerance])
        # return the image with the body parts and the keypoints
        return mask, keypoints, bodyparts

    def __call__(self, image, anns=None, meta=None):
        #img = self.apply(image, anns['keypoints'])
        img = self.apply(image, anns)
        return img, anns, meta
    
    @classmethod
    def configure(cls, args: argparse.Namespace):
        pass

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        pass

    def test_instance(self, image_id):
        annotations = json.load(open(train_ann))
        # find the unique image in images with id equal to image_id
        img_index = [i for i, x in enumerate(annotations['images']) if x['id'] == image_id][0]
        ann_index = [i for i, x in enumerate(annotations['annotations']) if x['image_id'] == image_id]
        img_file = annotations['images'][img_index]['file_name']
        category = annotations['annotations'][ann_index[0]]['category_id']

        print("\n")
        print("img index ",img_index)
        print("image_id ",image_id)
        print("ann_index ",ann_index)
        print(img_file)
        print("category ",category)
        print("\n")
        
        image = os.path.join(train_img, img_file )
        image = plt.imread(image)
    
        masks = []
        keypoints = []
        cropped_images = []
        print("ann_len ",len(ann_index))    
        for i in range(len(ann_index)):
            kp = annotations['annotations'][ann_index[i]]['keypoints']
            print("category ",annotations['annotations'][ann_index[i]]['category_id'])
            mask, keypoint, cropped_image = sda.crop(image, kp)
            masks.append(mask)
            keypoints.append(keypoint)
            cropped_images.append(cropped_image)

        os.makedirs('output', exist_ok=True)
        image_annotated = image.copy()

        for an in ann_index:
            image_annotated = draw_keypoint(image_annotated, annotations['annotations'][an]['keypoints'])
        

        plt.subplot(1,2,1)
        plt.imshow(image_annotated)
        plt.subplot(1,2,2)
        plt.imshow(masks[-1])
        plt.show()
        i = 0

        for cropped_image in cropped_images:
            for crop in cropped_image:
                plt.imsave(output_folder+str(i)+'.jpg', crop)
                i += 1
        return
    
    def crop_dataset(self):
        # make output folder and child folders
        os.makedirs(output_folder, exist_ok=True)
        annotations = json.load(open(train_ann))
        # iterate over the unique ids in the images
        pool = []
        
        for key in annotations['images']:
            #print(key['id'])
            # find all the keypoints image_id associated with this image id            
            ann_index = [i for i, x in enumerate(annotations['annotations']) if x['image_id'] == int(key['id'])]
            file = os.path.join(train_img, key['file_name'])
            image = plt.imread(file)
            cropped_images = []
            for ann in ann_index:
                kp = annotations['annotations'][ann]['keypoints']
                _, _, cropped_image = self.crop(image, kp)
                cropped_images.append(cropped_image)
            

            # save the cropped images
            i = 0
            for cropped_image in cropped_images:
                for crop in cropped_image:
                    if len(crop) == 0:
                        continue
                    #file_name = 'output/cropped_'+str(key['id'])+ '_'+str(i)+'.jpg'
                    file_path = os.path.join(output_folder, 'cropped_'+str(key['id'])+ '_'+str(i)+'.jpg')
                    plt.imsave(file_path, crop)
                    pool.append(file_path)
                    i += 1   
        
        text_file = os.path.join(output_folder, 'cropped_bodyparts.json')
        with open(text_file, 'w') as file:
            json.dump(pool, file)
        return 
        

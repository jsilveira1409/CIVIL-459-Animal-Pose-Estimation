import openpifpaf.transforms as transforms
from argparse import ArgumentParser
import random
from scipy import ndimage
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import torch
from PIL import Image

KP_DIST_THRESHOLD = 5
NB_BODY_PARTS = 3
IMG_TO_BODYPART_RATION = 4
CONTOUR_DIST_THRESHOLD = 20

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

    #segmts = [  (0,1), (0,2), (1,2), (0,3), (1,4), (3,4),
    #            (2,17), (18,19),
    #            (5,9), (6,10), (7,11), (8,12),
    #            (9,13), (10,14), (11,15), (12,16)]

    # everything is more connected so the mask takes more of the image
    segmts = [  (0,1), (0,2), (1,2), (0,3), (1,4), (3,4),
                (1,3), (0,4), (2,17),          
                (2,17), (18,19),
                (5,9), (6,10), (7,11), (8,12),
                (9,13), (10,14), (11,15), (12,16),
                (17,18), (18,19)]
    im = image.copy()
    for i in range(len(segmts)):

        segm = segmts[i]
        kp1_idx = segm[0] * 3    
        kp2_idx = segm[1] * 3
        kp1 = keypoints[kp1_idx:kp1_idx + 3]
        kp2 = keypoints[kp2_idx:kp2_idx + 3]
        
        if kp1[2] == 0 or kp2[2] == 0:
            continue
        # red line
        cv2.line(im, (int(kp1[0]), int(kp1[1])), (int(kp2[0]), int(kp2[1])), (120, 0,0), thickness=2)
    
    for i in range(0, len(keypoints), 3):
        if keypoints[i + 2] == 0:
            continue
        cv2.circle(im, (int(keypoints[i]), int(keypoints[i + 1])), radius=4, color=(255, 0,0), thickness=-1)

    return im

output_folder = 'data-animalpose/bodyparts/'
bodypart_file = 'data-animalpose/bodyparts/cropped_bodyparts.json'
all_bodypart_file = 'data-animalpose/bodyparts/all_bodyparts_kp.json'   
masks_file = 'data-animalpose/bodyparts/mask_bodyparts.json'
keypoints_file = 'data-animalpose/bodyparts/keypoints_bodyparts.json'
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
        self.all_bodypart_dict = json.load(open(all_bodypart_file))
        print("sdaplugin init")

    @classmethod
    def configure(cls, args: argparse.Namespace):
        pass

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        pass

    def __call__(self, image, anns=None, meta=None):
        #img = self.apply(image, anns['keypoints'])
        img,_, bodypart_keypoints = self.apply(image, anns)
        augmented_image = Image.fromarray(img)
        new_keypoints = []
        for kps in bodypart_keypoints:
            print("FLAG",kps)
            for kp in kps:
                new_keypoints.append(kp)
            new_anns = {'id': anns[0]['id'],
                        'image_id': anns[0]['image_id'],
                        'category_id': anns[0]['category_id'],
                        'keypoints': new_keypoints,
                        'bbox': anns[0]['bbox'],
                        'num_keypoints':20,
                        'iscrowd': 0,
                        'visible': 1}
            anns.append(new_anns)
            new_keypoints = []
        print("new_anns ",anns)

        return augmented_image, anns, meta

    def apply(self, image, ann):
        # TODO change this
        nb_bodyparts = NB_BODY_PARTS
        augmented_image = np.asarray(image, dtype=np.uint8).copy()
        mask = []
        # get the image dimensions
        image_height, image_width = augmented_image.shape[:2]     
        # load the body parts pool
        #bodyparts = json.load(open(bodypart_file))
        # load the masks pool
        #masks_path = json.load(open(masks_file))
        new_keypoints = []
        
        masks = []
        for i in range(nb_bodyparts):
            # choose a random body part from the pool and get the index
            index = random.randint(0, len(self.all_bodypart_dict) - 1)
            # get the body part path
            bodypart = cv2.imread(self.all_bodypart_dict[index]['bodypart'])
            #bodypart = bodyparts[index] 
            # get the mask path
            mask = cv2.imread(self.all_bodypart_dict[index]['mask'])
            # get the keypoints from the mask and save them
            print("bodypart index",self.all_bodypart_dict[index]['keypoints'])
            new_keypoints.append(self.all_bodypart_dict[index]['keypoints'])
            print("new_keypoints ",new_keypoints)
            # fill the holes in the mask      
            #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            # load the body part
            #bodypart = cv2.imread(bodypart)
            # randomly rotate the body part and the mask (same angle for both of course)
            angle = random.randint(0, 360)
            bodypart = ndimage.rotate(bodypart, angle)
            mask = ndimage.rotate(mask, angle)
            # get the body part dimensions
            bodypart_height, bodypart_width = bodypart.shape[:2]
            # resize the body part to fit the image
            if image_height/bodypart_height < IMG_TO_BODYPART_RATION or image_width/bodypart_width < IMG_TO_BODYPART_RATION:
                ratio = min(image_height/bodypart_height, image_width/bodypart_width)
                bodypart = cv2.resize(bodypart, (int(bodypart_width * ratio), int(bodypart_height * ratio)))
                mask = cv2.resize(mask, (int(bodypart_width * ratio), int(bodypart_height * ratio)))
                bodypart_height, bodypart_width = bodypart.shape[:2]
            # ensure the body part is not too big compared to the image
            if image_height/bodypart_height > IMG_TO_BODYPART_RATION or image_width/bodypart_width > IMG_TO_BODYPART_RATION:
                # apply a random scale to the body part, between 0.5 and 1 of the Image to body part ratio
                scale = random.uniform(0.1, 1) * IMG_TO_BODYPART_RATION
                bodypart = cv2.resize(bodypart, (int(bodypart_width * scale), int(bodypart_height * scale)))
                mask = cv2.resize(mask, (int(bodypart_width * scale), int(bodypart_height * scale)))
                bodypart_height, bodypart_width = bodypart.shape[:2]
                # choose a random position to add the body part
                # ensure image_width - bodypart_width > 0
                # ensure image_height - bodypart_height > 0
                if image_width - bodypart_width > 0 and image_height - bodypart_height > 0:
                    # choose a random position to add the body part not directly on top of keypoints
                    x = random.randint(0, image_width - bodypart_width)
                    y = random.randint(0, image_height - bodypart_height)
                    # save the keypoints of the body part
                    # add the pixels of the cropped body part to the image if the mask is 1 in that position
                    for i in range(bodypart_height):
                        for j in range(bodypart_width):
                            if mask[i][j].sum() >= 200:
                                augmented_image[y + i][x + j] = bodypart[i][j]
                    
                    # TODO should remove this for training
                    #augmented_image = draw_keypoint(augmented_image, new_keypoints)
                    # save the mask
                    masks.append(mask)
        # update the annotations with the new keypoints from the body parts added
        # return the augmented image
        # TODO: check this during training, for now I moved this to apply
        return augmented_image, masks, new_keypoints
   
    def crop(self,image, keypoints):
        # create an rgb mask
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        # draw the keypoints on the mask
        mask = draw_keypoint(mask, keypoints)
        # transform into a binary mask
        ret, bin_mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
        # thicken the mask
        kernel = np.ones((10,10), np.uint8)
        bin_mask = cv2.dilate(bin_mask,kernel, iterations=4)
        #find the contours in the mask
        contours, hierarchy = cv2.findContours(bin_mask, cv2.RETR_CCOMP , cv2.CHAIN_APPROX_SIMPLE)
        
        # draw the contours on the mask
        cv2.drawContours(bin_mask, contours, -1, (255,255,255), thickness=cv2.FILLED)
        # get contour area and centroid
        moments = [] 
        areas = []
        for contour in contours:
            moment = cv2.moments(contour)
            area = cv2.contourArea(contour)
            # get the centroid
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
                if dist < CONTOUR_DIST_THRESHOLD:
                    if areas[i] < areas[j]:
                        indices_to_remove.add(j)
                    else:
                        indices_to_remove.add(i)
                        
        # remove contours, moments, and areas using list comprehension
        contours = [contour for i, contour in enumerate(contours) if i not in indices_to_remove]
        moments = [moment for i, moment in enumerate(moments) if i not in indices_to_remove]
        areas = [area for i, area in enumerate(areas) if i not in indices_to_remove]
        # crop the different body parts and store them
        bodyparts = []
        bin_masks = []
        bodyparts_kp = []
        bodypart_kp = []
        for i in range(len(contours)):
            x, y, w, h = cv2.boundingRect(contours[i])
            bodyparts.append(image[y : y+h+self.tolerance, 
                                   x : x+w+self.tolerance])
            bin_masks.append(bin_mask[y : y+h+self.tolerance,
                                x : x+w+self.tolerance])
            for i in range(0, len(keypoints), 3):
                if keypoints[i] > x and keypoints[i] < x+w+self.tolerance and keypoints[i+1] > y and keypoints[i+1] < y+h+self.tolerance:
                    bodypart_kp.append(keypoints[i] - x)
                    bodypart_kp.append(keypoints[i+1] - y)
                    bodypart_kp.append(keypoints[i+2])
                else:
                    bodypart_kp.append(0)
                    bodypart_kp.append(0)
                    bodypart_kp.append(0)                   
            bodyparts_kp.append(bodypart_kp)
            bodypart_kp = []
            
            
        # return the image with the body parts and the keypoints
        return mask, bin_masks, bodyparts, bodyparts_kp

    def crop_dataset(self):
        # make output folder and child folders
        os.makedirs(output_folder, exist_ok=True)
        annotations = json.load(open(train_ann))
        # iterate over the unique ids in the images
        body_pool = []
        mask_pool = []
        all_bodyparts_kp = []
        output = []
        print("len ",len(annotations['images']) )
        for key in annotations['images']:
            # find all the keypoints image_id associated with this image id            
            ann_index = [i for i, x in enumerate(annotations['annotations']) if x['image_id'] == int(key['id'])]
            file = os.path.join(train_img, key['file_name'])
            image = plt.imread(file)
            cropped_images = []
            crop_masks = []
            
            for ann in ann_index:
                kp = annotations['annotations'][ann]['keypoints']
                _, crop_mask, cropped_image, crop_bodyparts_kp = self.crop(image, kp)
                cropped_images.append(cropped_image)
                crop_masks.append(crop_mask)
                all_bodyparts_kp.append(crop_bodyparts_kp)
            # save the cropped images
            i = 0
            for cropped_image in cropped_images:
                for crop in cropped_image:
                    if len(crop) == 0:
                        continue
                    file_path = os.path.join(output_folder, 'cropped_'+str(key['id'])+ '_'+str(i)+'.jpg')
                    plt.imsave(file_path, crop)
                    body_pool.append(file_path)
                    i += 1   
            # save the masks
            i = 0
            for mask in crop_masks:
                for m in mask:
                    if len(m) == 0:
                        continue
                    #file_name = 'output/mask_'+str(key['id'])+ '_'+str(i)+'.jpg'
                    file_path = os.path.join(output_folder, 'mask_'+str(key['id'])+ '_'+str(i)+'.jpg')
                    plt.imsave(file_path, m)
                    mask_pool.append(file_path)
                    i += 1
        # save the keypoints, the path to the cropped bodypart and the mask in a list of dictionaries
        # unravel to have a list of list of 20 points
        output_list = []
        
        for list in all_bodyparts_kp:
            for sublist in list:
                output_list.append(sublist)
        print(len(body_pool), len(mask_pool), len(output_list))
        

        for i in range(len(body_pool)):
            output.append({'bodypart':body_pool[i], 'mask':mask_pool[i], 'keypoints':output_list[i]})
            
        # save the list of dictionaries in a json file
        file_path = os.path.join(output_folder, 'all_bodyparts_kp'+'.json')
        with open(file_path, 'w') as file:
            json.dump(output, file)




        #file_path = os.path.join(output_folder, 'all_bodyparts_kp'+'.json')
        #with open(file_path, 'w') as file:
        #    json.dump(all_bodyparts_kp, file)
        #
        #text_file = os.path.join(output_folder, 'cropped_bodyparts.json')
        #with open(text_file, 'w') as file:
        #    json.dump(body_pool, file)
        #
        #text_file = os.path.join(output_folder, 'mask_bodyparts.json')
        #with open(text_file, 'w') as file:
        #    json.dump(mask_pool, file)

        return 
    
    def get_kp_from_masks(self, mask):
        # get the keypoint locations from the mask
        # the keypoints are green circles in the mask
        # the keypoints are the center of the green circles
        keypoints_pool = []
        lower_threshold = 210
        upper_threshold = 255
        
        for m in mask:
            for i in range(m.shape[0]):
                for j in range(m.shape[1]):
                    if m[i][j] > lower_threshold and m[i][j] < upper_threshold:
                        keypoints_pool.append((i, j))
        return keypoints_pool

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
    


import openpifpaf
from openpifpaf_sdaplugin import SDA
import numpy as np

#from openpifpaf.plugins.animalpose import AnimalKp
from openpifpaf_animalpose2.animal_kp_custom import AnimalKpCustom
import matplotlib.pyplot as plt
import torch
import torch.utils.data
from PIL import Image
import numpy
import gdown
import subprocess
import argparse
import json
import copy
import random
import os
import subprocess


openpifpaf.show.Canvas.show = True

args = argparse.Namespace(
    debug=False,
    pin_memory=False,
    animal_train_annotations='data-animalpose/annotations/animal_keypoints_20_train.json',
    animal_val_annotations='data-animalpose/annotations/animal_keypoints_20_val.json',
    animal_eval_annotations='data-animalpose/annotations/animal_keypoints_20_val.json',
    animal_train_image_dir='data-animalpose/images/train/',
    animal_val_image_dir='data-animalpose/images/val/',
    animal_eval_image_dir='data-animalpose/images/val/',
    animal_n_images=None,
    animal_square_edge=513,
    animal_extended_scale=False,
    animal_orientation_invariant=0.0,
    animal_blur=0.0,
    animal_augmentation=True,
    animal_rescale_images=1.0,
    animal_upsample=1,
    animal_min_kp_anns=1,
    animal_bmin=1,
    animal_eval_annotation_filter=True,
    animal_eval_long_edge=0,
    animal_eval_orientation_invariant=0.0,
    animal_eval_extended_scale=False,
    animal_eval_test2017=False,
    animal_eval_testdev2017=False
)

pred_cmd = 'python3 -m openpifpaf.predict \
    test.jpg \
    --checkpoint shufflenetv2k16 \
    --image-output \
    --json-output\
    --dataset=custom_animal'

train_cmd = 'python3 -m openpifpaf.train \
    --dataset custom_animal \
    --basenet=shufflenetv2k16 \
    --lr=0.00002 \
    --momentum=0.95 \
    --epochs=1 \
    --lr-decay 160 260 \
    --lr-decay-epochs=10  \
    --weight-decay=1e-5 \
    --weight-decay=1e-5 \
    --val-interval 10 \
    --loader-workers 1 \
    --batch-size 8'



train_annotations = 'data-animalpose/annotations/animal_keypoints_20_train.json'
val_annotations = 'data-animalpose/annotations/animal_keypoints_20_val.json'
eval_annotations = val_annotations
train_image_dir = 'data-animalpose/images/train/'
val_image_dir = 'data-animalpose/images/val/'
eval_image_dir = val_image_dir

keypoint_file = 'data-animalpose/keypoints.json'
images_folder = 'data-animalpose/images/'

def download_dataset():
    cmd = 'rm -rf data-animalpose'
    subprocess.call(cmd, shell=True)
    cmd = 'mkdir data-animalpose'
    subprocess.call(cmd, shell=True)
    cmd = 'gdown "https://drive.google.com/drive/folders/1xxm6ZjfsDSmv6C9JvbgiGrmHktrUjV5x" -O data-animalpose --folder'
    subprocess.call(cmd, shell=True)
    cmd = 'mkdir data-animalpose/output'
    subprocess.call(cmd, shell=True)
    cmd = 'unzip data-animalpose/images.zip -d data-animalpose/'
    subprocess.call(cmd, shell=True)
    cmd = 'rm data-animalpose/images.zip'
    subprocess.call(cmd, shell=True)
    print("Downloaded dataset")


def convert_keypoints_format(keypoints_list):
    keypoints_flat = []
    for keypoint in keypoints_list:
        keypoints_flat.extend([k for k in keypoint])
    return keypoints_flat

def adapt_to_coco():
    # Load the input JSON file
    with open(keypoint_file, 'r') as f:
        input_dict = json.load(f)

    # Create a copy of the input dictionary
    output_dict = copy.deepcopy(input_dict)

    # Update images list format
    images_list = []
    for image_id, image_filename in input_dict['images'].items():
        images_list.append({'id': int(image_id), 'file_name': image_filename})
        #images_list.append({'id': image_id, image_id: image_filename})
    output_dict['images'] = images_list

    # Update annotations keypoints format and add missing fields
    annotations_list = []
    for i, annotation in enumerate(input_dict['annotations']):
        new_annotation = {
            'id': i + 1,
            'image_id': annotation['image_id'],
            'category_id': annotation['category_id'],
            'bbox': annotation['bbox'],
            'keypoints': convert_keypoints_format(annotation['keypoints']),
            'num_keypoints': annotation['num_keypoints'],
            #'iscrowd': 0,
        }
        annotations_list.append(new_annotation)
    output_dict['annotations'] = annotations_list

    # Save the converted data to a JSON file
    with open(keypoint_file, 'w') as f:
        json.dump(output_dict, f, indent=4)
    print("Converted to COCO format")


def split_data():
    # Load the input JSON file
    with open(keypoint_file, 'r') as f:
        input_dict = json.load(f)

    # Create a copy of the input dictionary
    output_dict = copy.deepcopy(input_dict)

    # Split the data into train and val
    train_dict = copy.deepcopy(output_dict)
    val_dict = copy.deepcopy(output_dict)

    # Update images list format
    train_images_list = []
    val_images_list = []
    for image in output_dict['images']:
        if random.random() < 0.8:  # 80% probability for train, 20% for validation
            train_images_list.append(image)
        else:
            val_images_list.append(image)
    train_dict['images'] = train_images_list
    val_dict['images'] = val_images_list

    # Update annotations keypoints format and add missing fields
    train_annotations_list = []
    val_annotations_list = []
    for annotation in output_dict['annotations']:
        if annotation['image_id'] in [img['id'] for img in train_images_list]:
            train_annotations_list.append(annotation)
        elif annotation['image_id'] in [img['id'] for img in val_images_list]:
            val_annotations_list.append(annotation)
    train_dict['annotations'] = train_annotations_list
    val_dict['annotations'] = val_annotations_list
    
    # create files if not already existant
    if not os.path.exists(train_image_dir):
        os.makedirs(train_image_dir)
    if not os.path.exists(val_image_dir):
        os.makedirs(val_image_dir)
    if not os.path.exists(eval_image_dir):
        os.makedirs(eval_image_dir)
    if not os.path.exists('data-animalpose/annotations/'):
        os.makedirs('data-animalpose/annotations/')

    # create train_annotations file
    if not os.path.exists(train_annotations):
        open(train_annotations, 'w').close()
    # create val_annotations file
    if not os.path.exists(val_annotations):
        open(val_annotations, 'w').close()
    # create eval_annotations file
    if not os.path.exists(eval_annotations):
        open(eval_annotations, 'w').close()

        
    # Save the converted data to a JSON file
    
    with open(train_annotations, 'w') as f:
        json.dump(train_dict, f, indent=4)
    with open(eval_annotations, 'w') as f:
        json.dump(val_dict, f, indent=4)


    # move images to train and val folders
    for image in train_images_list:
        #file_name = image[str(image['id'])]
        file_name = image['file_name']
        os.rename(images_folder + file_name, train_image_dir + file_name)
    for image in val_images_list:
        #file_name = image[str(image['id'])]
        file_name = image['file_name']
        os.rename(images_folder + file_name, val_image_dir + file_name)
    print("Split data into train and val")

def test_sda (sda, img_id=0):
    #img = Image.open('data-animalpose/images/train/2007_001397.jpg')
    # get image id from annotations
    with open(train_annotations, 'r') as f:
        input_dict = json.load(f)
    img_name = input_dict['images'][img_id]['file_name']
    keypoints = input_dict['annotations'][img_id]['keypoints']
    img = Image.open(train_image_dir + img_name)
    tensor_img = np.array(img)
    print(keypoints)

    img1 = sda.apply(tensor_img, keypoints)
    plt.imshow(tensor_img)
    plt.imshow(img1)
    plt.show()

def main():
    # 1. Download dataset
    download_dataset()
    # 2. Convert to COCO format 
    adapt_to_coco()
    # 3. Split data into train and val
    split_data()
    
    # 4. Initialize SDA and crop the dataset, creating a body part pool
    sda = SDA()
    sda.crop_dataset()
    print("Cropped dataset")
    
    # 5. Configure plugins
    config = openpifpaf.plugin.register()
    print(openpifpaf.DATAMODULES)

    # 6. Train the model
    subprocess.run(train_cmd, shell=True)
    pass

from multiprocessing import freeze_support

if __name__ == '__main__':
    #freeze_support()
    #main()
    sda = SDA()
    test_sda(sda)

import json
import copy
import random
import os
import subprocess


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
    cmd = 'unzip data-animalpose/images.zip -d data-animalpose/'
    subprocess.call(cmd, shell=True)
    cmd = 'rm data-animalpose/images.zip'
    subprocess.call(cmd, shell=True)
    print("Downloaded dataset")


def convert_keypoints_format(keypoints_list):
    keypoints_flat = []
    for keypoint in keypoints_list:
        keypoints_flat.extend([float(k) for k in keypoint])
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
        images_list.append({'id': int(image_id), image_id: image_filename})
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
        os.rename(images_folder + image['file_name'], train_image_dir + image['file_name'])
    for image in val_images_list:
        os.rename(images_folder + image['file_name'], val_image_dir + image['file_name'])
    print("Split data into train and val")


if __name__ == '__main__':
    download_dataset()
    adapt_to_coco()
    split_data()
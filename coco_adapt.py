import json
import copy

keypoint_file = 'data-animalpose/keypoints.json'
images_folder = '/../data-animalpose/images/'

def convert_keypoints_format(keypoints_list):
    keypoints_flat = []
    for keypoint in keypoints_list:
        keypoints_flat.extend(keypoint)
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
        if image['id'] % 5 == 0:
            val_images_list.append(image)
        else:
            train_images_list.append(image)
    train_dict['images'] = train_images_list
    val_dict['images'] = val_images_list

    # Update annotations keypoints format and add missing fields
    train_annotations_list = []
    val_annotations_list = []
    for i, annotation in enumerate(output_dict['annotations']):
        if annotation['image_id'] % 5 == 0:
            val_annotations_list.append(annotation)
        else:
            train_annotations_list.append(annotation)
    train_dict['annotations'] = train_annotations_list
    val_dict['annotations'] = val_annotations_list

    # Save the converted data to a JSON file
    with open('data-animalpose/train.json', 'w') as f:
        json.dump(train_dict, f, indent=4)
    with open('data-animalpose/val.json', 'w') as f:
        json.dump(val_dict, f, indent=4)


if __name__ == '__main__':
    adapt_to_coco()
    #split_data()
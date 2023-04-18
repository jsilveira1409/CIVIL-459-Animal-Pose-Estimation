
def format_bbox (bbox, image_size):
    bbox[0] = bbox[0]/image_size[1]
    bbox[1] = bbox[1]/image_size[0]
    bbox[2] = bbox[2]/image_size[1]
    bbox[3] = bbox[3]/image_size[0]
    # convert xmax, ymax to w, h
    bbox[2] = bbox[2] - bbox[0]
    bbox[3] = bbox[3] - bbox[1]
    # convert xmin, ymin to x_center, y_center
    bbox[0] = bbox[0] + bbox[2]/2
    bbox[1] = bbox[1] + bbox[3]/2
    bbox = [round(num, 6) for num in bbox]
    return bbox

def extract_info(annotations, image_id, root = root_images):
    filtered_list = [d for d in annotations['annotations'] if d['image_id'] == image_id+1]
    # get image name 
    image_name = annotations['images'][str(image_id+1)]
    category_list = []
    bbox_list = []
    for element in filtered_list:
        # get category id, needs to start from 0
        category_list.append(element['category_id']-1)
        # get bounding box
        bbox  = element['bbox']
        # get image size
        image_path = os.path.join(root, image_name)
        image = cv2.imread(image_path)
        image_size = image.shape
        # normalize bounding box
        bbox_list.append(format_bbox(bbox, image_size))
    return image_name, category_list, bbox_list

def write_info(image_name, category_list, bbox_list, root = root_annotations):
    # write to file with same name as image, removing the extension and replacing it with .txt
    image_name = image_name.split('.')[0] + '.txt'
    with open(os.path.join(root, image_name), 'w') as f:
        # write category id and bounding box
        for i in range(len(category_list)):
            f.write(str(category_list[i]) + ' ' + ' '.join([str(num) for num in bbox_list[i]]) + '\n')

def move_files_to_folder(list_of_files, destination_folder):
    for f in list_of_files:
        try:
            shutil.move(f, destination_folder)
        except:
            print(f)
            assert False



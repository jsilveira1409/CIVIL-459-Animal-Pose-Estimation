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
    --loader-workers 2 \
    --batch-size 1'

def test_sda (sda):
    img = Image.open('data-animalpose/images/train/2007_001397.jpg')
    tensor_img = np.array(img)
    img1 = sda.apply(tensor_img)
    plt.imshow(tensor_img)
    plt.show()

def main():
    config = openpifpaf.plugin.register()
    #print(openpifpaf.DATAMODULES)
    dataset = AnimalKpCustom()
    sda = SDA(train_img='data-animalpose/images/train/', 
              val_img='data-animalpose/images/val/', 
              train_ann='data-animalpose/annotations/animal_keypoints_20_train.json',
              val_ann='data-animalpose/annotations/animal_keypoints_20_val.json')
    
    #test_sda(sda)
    #dataset.configure

    subprocess.run(train_cmd, shell=True)

    pass

from multiprocessing import freeze_support

if __name__ == '__main__':
    freeze_support()
    main()




#pred_cmd = 'python3 -m openpifpaf.predict \
#    test.jpg \
#    --checkpoint shufflenetv2k16 \
#    --image-output \
#    --json-output\
#    --dataset=custom_animal'

 


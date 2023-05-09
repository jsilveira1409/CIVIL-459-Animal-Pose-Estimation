
import openpifpaf
from openpifpaf_sdaplugin import SDA

from openpifpaf.plugins.animalpose import AnimalKp
#from openpifpaf_animalplugin import AnimalPoseEstimation
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




def main():
    config = openpifpaf.plugin.register()
    dataset = AnimalKp()
    sda = SDA(train_img='data-animalpose/images/train/', 
              val_img='data-animalpose/images/val/', 
              train_ann='data-animalpose/annotations/animal_keypoints_20_train.json',
              val_ann='data-animalpose/annotations/animal_keypoints_20_val.json')
    

    #arg = argparse.ArgumentParser()
    #dataset.cli(arg)
    #a = arg.parse_args()
    #dataset.configure(a)
    #dataloader = dataset.train_loader()
    #sample = next(iter(dataloader))
    #image_batch, anns_batch, meta_batch = sample
    ## show image with matplotlib
    #image = image_batch[0].numpy().transpose((2, 1, 0))
    #image = numpy.clip(image, 0, 1)
    #plt.imshow(image)
    #plt.show()

    # send dataset to SDA to create a bodypart database
    sda.crop_dataset()

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
#train_cmd = 'python3 -m openpifpaf.train \
#    --dataset custom_animal \
#    --basenet=shufflenetv2k16 \
#    --lr=0.00002 \
#    --momentum=0.95 \
#    --epochs=10 \
#    --lr-decay 160 260 \
#    --lr-decay-epochs=10  \
#    --weight-decay=1e-5 \
#    --weight-decay=1e-5 \
#    --val-interval 10 \
#    --loader-workers 2 \
#    --batch-size 1'
# 

#subprocess.run(train_cmd, shell=True)

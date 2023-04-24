import openpifpaf
import IPython
from openpifpaf.transforms import Crop
from openpifpaf_sdaplugin import SDA
from openpifpaf_animalplugin import AnimalPoseEstimation
from PIL import Image
import numpy
import gdown
import subprocess
openpifpaf.show.Canvas.show = True

config = openpifpaf.plugin.register()
print(openpifpaf.DATAMODULES)
dataset = AnimalPoseEstimation()


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
    --epochs=10 \
    --lr-decay 160 260 \
    --lr-decay-epochs=10  \
    --weight-decay=1e-5 \
    --weight-decay=1e-5 \
    --val-interval 10 \
    --loader-workers 2 \
    --batch-size 1'
 

#subprocess.run(train_cmd, shell=True)

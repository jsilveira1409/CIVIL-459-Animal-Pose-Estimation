import json
import torch
import argparse
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Resize, Compose

from openpifpaf.datasets import DataModule
from openpifpaf import encoder, headmeta, metric, transforms
from openpifpaf.datasets import collate_images_anns_meta, collate_images_targets_meta

from PIL import Image

from .constants import ANIMAL_KEYPOINTS, ANIMAL_SKELETON, HFLIP, \
    ANIMAL_SIGMAS, ANIMAL_POSE, ANIMAL_CATEGORIES, ANIMAL_SCORE_WEIGHTS



class AnimalPoseEstimation (DataModule):
    debug = False
    pin_memory = False

    n_images = None
    square_edge = 513
    extended_scale = False
    orientation_invariant = 0.0
    blur = 0.0
    augmentation = True
    rescale_images = 1.0
    upsample_stride = 1
    min_kp_anns = 1
    b_min = 1  # 1 pixel

    eval_annotation_filter = True
    eval_long_edge = 0  # set to zero to deactivate rescaling
    eval_orientation_invariant = 0.0
    eval_extended_scale = False


    def __init__(self,image_dir='data-animalpose/images', annotation_file='data-animalpose/keypoints.json', preprocess=None):
        super().__init__()
        # load json file
        #self.annotation = json.load(open(annotation_file))
        self.annotation_file = annotation_file
        self.annotation = json.load(open(annotation_file))
        self.image_dir = image_dir
        self.preprocess = preprocess

        cif = headmeta.Cif('cif', 'custom-animal',
                           keypoints=ANIMAL_KEYPOINTS,
                           sigmas=ANIMAL_SIGMAS,
                           pose=ANIMAL_POSE,
                           draw_skeleton=ANIMAL_SKELETON,
                           score_weights=ANIMAL_SCORE_WEIGHTS)
        caf = headmeta.Caf('caf', 'custom-animal',
                           keypoints=ANIMAL_KEYPOINTS,
                           sigmas=ANIMAL_SIGMAS,
                           pose=ANIMAL_POSE,
                           skeleton=ANIMAL_SKELETON)

        cif.upsample_stride = self.upsample_stride
        caf.upsample_stride = self.upsample_stride
        self.head_metas = [cif, caf]

    @classmethod
    def cli(cls, parser:argparse.ArgumentParser):
        group = parser.add_argument_group('AnimalPoseEstimation')
        group.add_argument('--animalpose-image-dir', default='data-animalpose/images')
        group.add_argument('--animalpose-annotation-file', default='data-animalpose/keypoints.json')
    
    # TODO check
    @classmethod
    def configure(cls, args):
        return cls(args.animalpose_image_dir, args.animalpose_annotation_file)
    
    def _preprocess(self):
        encoders = (encoder.Cif(self.head_metas[0], bmin=self.b_min),
                    encoder.Caf(self.head_metas[1], bmin=self.b_min))

        if not self.augmentation:
            return transforms.Compose([
                transforms.NormalizeAnnotations(),
                transforms.RescaleAbsolute(self.square_edge),
                transforms.CenterPad(self.square_edge),
                transforms.EVAL_TRANSFORM,
                transforms.Encoders(encoders),
            ])

        if self.extended_scale:
            rescale_t = transforms.RescaleRelative(
                scale_range=(0.2 * self.rescale_images,
                                2.5 * self.rescale_images),
                power_law=True, stretch_range=(0.75, 1.33))
        else:
            rescale_t = transforms.RescaleRelative(
                scale_range=(0.3 * self.rescale_images,
                                2.0 * self.rescale_images),
                power_law=True, stretch_range=(0.75, 1.33))

        return transforms.Compose([
            transforms.NormalizeAnnotations(),
            transforms.RandomApply(
                transforms.HFlip(ANIMAL_KEYPOINTS, HFLIP), 0.5),
            rescale_t,
            transforms.RandomApply(
                transforms.Blur(), self.blur),
            transforms.RandomChoice(
                [transforms.RotateBy90(),
                    transforms.RotateUniform(30.0)],
                [self.orientation_invariant, 0.4],
            ),
            transforms.Crop(self.square_edge, use_area_of_interest=True),
            transforms.CenterPad(self.square_edge),
            transforms.TRAIN_TRANSFORM,
            transforms.Encoders(encoders),
        ])

    def train_loader(self):
        # Return the training data loader
        train_dataset = AnimalPoseEstimation(image_dir=self.image_dir, annotation_file=self.annotation_file, preprocess = self._preprocess())
        return torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.loader_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_images_targets_meta,
        )

    def val_loader(self):
        # Return the training data loader
        val_dataset = AnimalPoseEstimation(image_dir=self.image_dir, annotation_file=self.annotation_file, preprocess=self._preprocess())
        return torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.loader_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_images_targets_meta,
        )
        

    def image_file(self, index):
        # Return the file path of the image with the given index
        img_id = self.annotation['annotations'][index]['image_id']
        img = self.annotation['images'][str(img_id)]
        return '{}/{}'.format(self.image_dir, img)
       

    def anns(self, index):
        # Return the annotations for the image with the given index
        instances = [ann for ann in self.annotation['annotations'] if ann['image_id'] == index]
        ann = []

        for instance in instances:  
            # Return the annotations for the image with the given index
            image_id = instance['image_id']
            category_id = instance['category_id']
            keypoints = instance['keypoints']
            #bbox = self.annotation['annotations'][index]['bbox']
            ann.append({'image_id': image_id, 'category_id': category_id, 'keypoints': keypoints})
    
        return ann
    
    @classmethod
    def common_eval_preprocess(cls):
        rescale_t = None
        if cls.eval_extended_scale:
            assert cls.eval_long_edge
            rescale_t = [
                transforms.DeterministicEqualChoice([
                    transforms.RescaleAbsolute(cls.eval_long_edge),
                    transforms.RescaleAbsolute((cls.eval_long_edge - 1) // 2 + 1),
                ], salt=1)
            ]
        elif cls.eval_long_edge:
            rescale_t = transforms.RescaleAbsolute(cls.eval_long_edge)

        if cls.batch_size == 1:
            padding_t = transforms.CenterPadTight(16)
        else:
            assert cls.eval_long_edge
            padding_t = transforms.CenterPad(cls.eval_long_edge)

        orientation_t = None
        if cls.eval_orientation_invariant:
            orientation_t = transforms.DeterministicEqualChoice([
                None,
                transforms.RotateBy90(fixed_angle=90),
                transforms.RotateBy90(fixed_angle=180),
                transforms.RotateBy90(fixed_angle=270),
            ], salt=3)

        return [
            transforms.NormalizeAnnotations(),
            rescale_t,
            padding_t,
            orientation_t,
        ]

    def _eval_preprocess(self):
        return transforms.Compose([
            *self.common_eval_preprocess(),
            transforms.ToAnnotations([
                transforms.ToKpAnnotations(
                    ANIMAL_CATEGORIES,
                    keypoints_by_category={1: self.head_metas[0].keypoints},
                    skeleton_by_category={1: self.head_metas[1].skeleton},
                ),
                transforms.ToCrowdAnnotations(ANIMAL_CATEGORIES),
            ]),
            transforms.EVAL_TRANSFORM,
        ])

    def eval_loader(self):
        eval_data = CocoLoader(
            image_dir=self.eval_image_dir,
            ann_file=self.eval_annotations,
            preprocess=self._eval_preprocess(),
            annotation_filter=self.eval_annotation_filter,
            min_kp_anns=self.min_kp_anns if self.eval_annotation_filter else 0,
            category_ids=[1] if self.eval_annotation_filter else [],
        )
        return torch.utils.data.DataLoader(
            eval_data, batch_size=self.batch_size, shuffle=False,
            pin_memory=self.pin_memory, num_workers=self.loader_workers, drop_last=False,
            collate_fn=collate_images_anns_meta)

    def metrics(self):
        return [metric.Coco(
            COCO(self.eval_annotations),
            max_per_image=20,
            category_ids=[1],
            iou_type='keypoints',
            keypoint_oks_sigmas=ANIMAL_SIGMAS,
        )]

    def meta(self, index):
        # Return metadata for the image with the given index
        meta = {
            'file_name': self.image_file(index),
            # Add any other metadata you need here
        }
        return meta

    def __len__(self):
        # Return the number of images in the dataset
        return len(self.annotation['images'])
    
    def __getitem__(self, index):
        # Return the image and annotations for the image with the given index
        image = Image.open(self.image_file(index)).convert('RGB')
        anns = self.anns(index)
        meta = self.meta(index)

        if self.preprocess is not None:
            image, anns, meta = self.preprocess(image, anns, meta)

        return image, anns, meta
        
    



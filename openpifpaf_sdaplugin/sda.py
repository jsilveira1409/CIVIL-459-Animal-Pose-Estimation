from openpifpaf.transforms import Compose
from argparse import ArgumentParser
import os

class SDA(Compose):
    def __init__(self, output_dir):
        self.bodypart_pool = []
        self.output_dir = output_dir
        super().__init__([])

    def __call__(self, image, annotation, *, image_info=None):
        image, annotation = super().__call__(image, annotation, image_info=image_info)
    
        # save the body parts to the image
        for ann in annotation:
            for i, bbox in enumerate(ann.data):
                body_part_image = image.crop(bbox)
                filename = f"{ann.id}_part_{i}.jpg"
                body_part_image.save(os.path.join(self.output_dir, filename))

        return image, annotation
    
    @staticmethod
    def cli(parser: ArgumentParser):
        parser.add_argument('--sda-option', type=str, default='default_value')
        

        
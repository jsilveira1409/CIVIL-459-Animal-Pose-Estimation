from openpifpaf.transforms import Compose
from argparse import ArgumentParser
import os

class SDA(Compose):
    def __init__(self, output_dir):
        self.bodypart_pool = []
        self.output_dir = output_dir
        super().__init__([])

    def __call__(self, data):
        # crop bodypart images from keypoint location
        keypoints = data['keypoints']
        for i in range(len(keypoints)):
            if keypoints[i][2] > 0.5:
                bodypart = data['keypoint_names'][i]
                if bodypart not in self.bodypart_pool:
                    self.bodypart_pool.append(bodypart)
                x = int(keypoints[i][0])
                y = int(keypoints[i][1])
                image = data['image']
                crop = image.crop((x-50, y-50, x+50, y+50))
                crop.save(os.path.join(self.output_dir, bodypart + '.jpg'))
        return data
    
        

    @staticmethod
    def cli(parser: ArgumentParser):
        parser.add_argument('--sda-option', type=str, default='default_value')
        

        
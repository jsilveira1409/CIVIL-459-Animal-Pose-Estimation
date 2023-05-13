
import openpifpaf

from . import animal_kp_custom

from .animal_kp_custom import AnimalKpCustom


def register():
    openpifpaf.DATAMODULES['custom_animal'] = animal_kp_custom.AnimalKpCustom
    openpifpaf.CHECKPOINT_URLS['shufflenetv2k30-animalpose'] = \
        "http://github.com/vita-epfl/openpifpaf-torchhub/releases/" \
        "download/v0.12.9/shufflenetv2k30-210511-120906-animal.pkl.epoch400"

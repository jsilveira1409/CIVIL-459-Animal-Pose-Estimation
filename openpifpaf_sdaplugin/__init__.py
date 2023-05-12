import openpifpaf
from .sda import SDA

def register():
    openpifpaf.DATAMODULES['sda'] = SDA
    pass
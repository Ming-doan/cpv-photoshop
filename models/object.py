import enum
import numpy as np
from logic.transform import *
from logic.image_processing import *


class ObjectType(enum.Enum):
    SHAPE = 'shape'
    IMAGE = 'image'


class Object:
    def __init__(self, name: str, img: np.ndarray, object_type: ObjectType):
        self.name = name
        self.img = img
        self.type = object_type
        self.transform = Transform()
        self.filter = Filters()

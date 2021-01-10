from enum import Enum


class NetType(Enum):
    Fitting = 1,
    BinaryClassifier = 2,
    MultipleClassifier = 3


class InitMethod(Enum):
    Zero = 0,
    Normal = 1,
    Xavier = 2

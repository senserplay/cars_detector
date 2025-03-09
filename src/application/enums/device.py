from enum import Enum


class Device(Enum):
    cuda = "cuda"
    cpu = "cpu"
    mps = "mps"

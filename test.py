class Transformation:
    def __init__(self, name, **params):
        self.name = name
        self.params = params

TRANSFORMS_SUPPORTED = [
    Transformation("Grayscale"),
    Transformation("ColorJitter", brightness=float, contrast=float, saturation=float, hue=float),
]
import pdb; pdb.set_trace()

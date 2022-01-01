from .pixelbert import (
    pixelbert_transform,
    pixelbert_transform_randaug,
)
from .wit import (
    wit_transform
)

_transforms = {
    "pixelbert": pixelbert_transform,
    "pixelbert_randaug": pixelbert_transform_randaug,
}

_wit_transform = {
    "wit_default": wit_transform,
}


def keys_to_transforms(keys: list, size=224):
    return [_transforms[key](size=size) for key in keys]


def keys_to_wit_transforms(keys: list):
    return [_wit_transform[key]() for key in keys]

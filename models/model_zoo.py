"""Model store which handles pretrained models """
from .deeplabv3 import *
from .deeplabv3_mobile import *
from .pspnet import *
from .psp_mobile import *

__all__ = ['get_segmentation_model']


def get_segmentation_model(model, **kwargs):
    models = {
        'psp': get_psp,
        'psp_mobile': get_psp_mobile,
        'deeplabv3': get_deeplabv3,
        'deeplabv3_mobile': get_deeplabv3_mobile,
    }
    return models[model](**kwargs)

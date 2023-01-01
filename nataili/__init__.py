from .blip import Caption
from .clip import Interrogator
from .codeformers import codeformers
from .esrgan import esrgan
from .gfpgan import gfpgan
from .model_manager import (
    AITemplateModelManager,
    BlipModelManager,
    ClipModelManager,
    CodeFormerModelManager,
    CompVisModelManager,
    EsrganModelManager,
    GfpganModelManager,
    ModelManager,
)
from .stable_diffusion import CompVis
from .util import Switch, logger

disable_xformers = Switch()
disable_voodoo = Switch()
disable_local_ray_temp = Switch()

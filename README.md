# Nataili: Multimodal AI Python Library

Nataili is a Python library that provides a set of tools to build multimodal AI applications. It provides a set of tools to build multimodal AI applications, including:

* BLIP: Image captioning
* CLIP: Image interrogation
* ESRGAN: Image super-resolution
* GFPGAN: Post-processing for faces
* CodeFormer: Post-processing and super-resolution for faces
* Stable Diffusion: Image generation

It is designed to be modular, so you can use the tools you need and ignore the rest.

## Installation

`pip install nataili`

From source:

```
git clone https://github.com/Sygil-Dev/nataili
cd nataili
pip install -e .
```

## Usage

### General information

#### Model manager

Nataili has a model manager that allows you to download and manage models. You can use it to download models, list models, and remove models.

Each service has a set of models that can be used. You can use the model manager to download the models you need.

The models for a service are stored in a .json file. These files are included in the package.

Models will be downloaded to the `~/.cache/nataili` directory. The models are stored in a directory named after the service.

Structure of a model file:

```
{
  "": {
    "name": "",
    "type": "",
    "config": {
      "files": [
        {
          "path": ""
        }
      ],
      "download": [
        {
          "file_name": "",
          "file_path": "",
          "file_url": ""
        }
      ]
    },
    "available": false
  }
}
```

* `key`: Key used to identify the model by the model manager
* `name`: Name of the model, can be descriptive
* `type`: Type of model, used by the model manager to identify the type of model
* `config`: Configuration of the model
  * `files`: List of files that are required for the model to work
    * `path`: Path to the file
  * `download`: Download information for the files
    * `file_name`: Name of the file
    * `file_path`: Path to the file
    * `file_url`: URL to download the file
* `available`: Whether the model is available or not, model manager will set this to true if the model is downloaded

There is a BaseModelManager class that has the basic functionality to manage models. You can use this class to create your own model manager for a service.

BaseModelManager functionality includes:

* Downloading models
* Listing models
* Validating models
* CUDA support detection

Service ModelManager classes inherit from BaseModelManager and add functionality specific to the service. Generally speaking, Service ModelManager classes will have functionality to load the model.

This is done with a `load()` function and a `load_<service>()` function. The `load()` function will call the `load_<service>()` function.
`load()` is the external function that is called by the user, and `load_<service>()` is the internal function that is called by `load()`.

`load_<service>()` will load the model and return it. `load()` will then store the model in `self.loaded_models` under the model key.

`nataili/model_manager/new.py` can be used as a template for creating a new model manager.

There is a Super ModelManager class that wraps the service ModelManager classes. This class is used to load models from multiple services.
The individual service ModelManager classes are stored in `self.<service>`.

Accessing the service ModelManager classes is done through `self.<service>.<function>`.

#### Model manager usage
Single service model manager usage:

```
from nataili import <service>ModelManager

model_manager = <service>ModelManager()

model_manager.load(<model_key>)
```

Super model manager usage:

```
from nataili import ModelManager

model_manager = ModelManager()

model_manager.<service>.load(<model_key>)
```

#### Service Classes

Service classes are used to interact with the models.

Service classes accept a loaded model as an argument. The model can be loaded with the model manager.

Service classes are used with the `__call__` function. The `__call__` function accepts the input and returns the output.

#### Utils

Nataili has a set of utils that are used by the service classes and the model manager.

### BLIP

BLIP is a model that generates image captions. [arxiv](https://arxiv.org/abs/2201.12086).

Models:
* BLIP
* BLIP Large

Usage:

```
import PIL

from nataili import Caption, ModelManager, logger

image = PIL.Image.open("01.png").convert("RGB")

mm = ModelManager()

mm.blip.load("BLIP")

blip = Caption(mm.blip.loaded_models["BLIP"])

logger.generation(f"caption: {blip(image, sample=False)} - sample: False")
```

Options:

```
image: The image to caption. This can be a PIL.Image.Image or a path to an image.
sample: Whether to sample or not. If False, the model will return the most likely caption.
num_beams: The number of beams to use. This is only used if sample is False.
max_length: The maximum length of the caption.
min_length: The minimum length of the caption.
top_p: The top p to use. This is only used if sample is True.
repetition_penalty: The repetition penalty to use. This is only used if sample is True.
```

### CLIP

CLIP (Contrastive Language-Image Pre-training). [arxiv](https://arxiv.org/abs/2103.00020).

CLIP can be used to interrogate images. This means you can compare text to images and get a score for how similar they are.

Text embeds are cached to speed up the process.

Image embeds are cached to speed up the process.

Models:
* ViT-L/14
* ViT-H-14
* ViT-g-14

Usage:

```
import PIL
import os

from nataili import Interrogator, ModelManager, logger

image = PIL.Image.open("01.png").convert("RGB")

mm = ModelManager()

mm.clip.load("ViT-L/14")

interrogator = Interrogator(mm.clip.loaded_models["ViT-L/14"])

results = interrogator(image)
logger.generation(results)
```

### CodeFormer

CodeFormer is a model that can be used for face restoration. [arxiv](https://arxiv.org/abs/2206.11253).

Models:
* CodeFormer

Usage:

```
import PIL
import time

from nataili import codeformers, ModelManager, logger

image = PIL.Image.open("01.png").convert("RGB")

mm = ModelManager()

mm.codeformer.load("CodeFormers")

upscaler = codeformers(mm.codeformer.loaded_models["CodeFormers"])

results = upscaler(input_image=image)
```

### ESRGAN

ESRGAN is a model that can be used for image super resolution. [arxiv](https://arxiv.org/abs/2107.10833).

Models:
* RealESRGAN_x4plus
* RealESRGAN_x4plus_anime_6B
* RealESRGAN_x2plus

Usage:

```
import PIL
import time

from nataili import esrgan, ModelManager, logger

image = PIL.Image.open("01.png").convert("RGB")

mm = ModelManager()

mm.esrgan.load("RealESRGAN_x4plus")

upscaler = esrgan(mm.esrgan.loaded_models["RealESRGAN_x4plus"])

results = upscaler(input_image=image)
```

### GFPGAN

GFPGAN is a model that can be used for face restoration. [arxiv](https://arxiv.org/abs/2101.04061).

Models:
* GFPGAN v1.4

Usage:

```
import PIL
import time

from nataili import gfpgan, ModelManager, logger

image = PIL.Image.open("01.png").convert("RGB")

mm = ModelManager()

mm.gfpgan.load("GFPGAN")

facefixer = gfpgan(mm.gfpgan.loaded_models["GFPGAN"])

results = facefixer(input_image=image, strength=1.0)
```

### Stable Diffusion

Stable Diffusion is a model that can be used for image generation. [arxiv](https://arxiv.org/abs/2112.10752).

Models:
* many

Usage:

```
from nataili import CompVis, ModelManager, logger

mm = ModelManager()
mm.compvis.load("stable_diffusion")

compvis = CompVis(mm.compvis.loaded_models["stable_diffusion"])
compvis = CompVis(
  model=mm.compvis.loaded_models["stable_diffusion"],
  model_name="stable_diffusion",
  output_dir="./output"
)

compvis.generate(
  "a dog"
)
```

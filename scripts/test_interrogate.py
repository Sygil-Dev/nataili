import os

import PIL

from nataili import Interrogator, ModelManager, logger

images = []

for file in os.listdir("test_images"):
    pil_image = PIL.Image.open(f"test_images/{file}").convert("RGB")
    images.append(pil_image)

mm = ModelManager()

mm.clip.load("ViT-L/14")

interrogator = Interrogator(
    mm.clip.loaded_models["ViT-L/14"],
)

for image in images:
    results = interrogator(image)
    logger.generation(results)

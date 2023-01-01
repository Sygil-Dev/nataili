import PIL

from nataili import Caption, ModelManager, logger

image = PIL.Image.open("01.png").convert("RGB")

mm = ModelManager()

mm.blip.load("BLIP")

blip = Caption(mm.blip.loaded_models["BLIP"])

logger.generation(f"caption: {blip(image, sample=False)} - sample: False")

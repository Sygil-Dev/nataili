import PIL

from nataili.model_manager import ModelManager
from nataili.upscalers.codeformers import codeformers
from nataili.util.logger import logger

image = PIL.Image.open("./01.png").convert("RGB")

mm = ModelManager()

mm.init()

model = "CodeFormers"

if model not in mm.available_models:
    logger.error(f"Model {model} not available", status=False)
    logger.init(f"Downloading {model}", status="Downloading")
    mm.download_model(model)
    logger.init_ok(f"Downloaded {model}", status=True)

logger.init(f"Model: {model}", status="Loading")
success = mm.load_model(model)
logger.init_ok(f"Loading {model}", status=success)

upscaler = codeformers(
    mm.loaded_models[model]["model"],
    mm.loaded_models[model]["device"],
    "./",
)

results = upscaler(input_image=image)
images = upscaler.output_images
output_image = PIL.Image.open(images[0])
images.PIL.save("./01_postprocessed.png")
logger.init_ok(f"Job Completed", status="Success")
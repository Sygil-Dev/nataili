import time

import PIL

from nataili import ModelManager, esrgan, logger

image = PIL.Image.open("01.png").convert("RGB")

mm = ModelManager()

mm.esrgan.load("RealESRGAN_x4plus")

upscaler = esrgan(mm.esrgan.loaded_models["RealESRGAN_x4plus"])

tick = time.time()
results = upscaler(input_image=image)
logger.init_ok(f"Job Completed. Took {time.time() - tick} seconds", status="Success")

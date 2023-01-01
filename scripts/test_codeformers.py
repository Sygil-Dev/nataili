import time

import PIL

from nataili import ModelManager, codeformers, logger

image = PIL.Image.open("01.png").convert("RGB")

mm = ModelManager()

mm.codeformer.load("CodeFormers")

upscaler = codeformers(
    mm.codeformer.loaded_models["CodeFormers"],
)

for iter in range(5):
    tick = time.time()
    results = upscaler(input_image=image)
    logger.init_ok(f"Job Completed. Took {time.time() - tick} seconds", status="Success")
